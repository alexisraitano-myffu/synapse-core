//! Device pairing channel (SYN-128) — authenticated ECDH so a new device can
//! join an existing memory space without the user copying a 48-char token.
//!
//! Threat model. The two devices are on the same LAN; a passive or active
//! network attacker may see every byte on the wire but did NOT see the QR
//! code. The QR carries a fresh 32-byte **secret** that is mixed into the key
//! derivation, so only a device that actually scanned the QR can derive the
//! channel key. A man-in-the-middle who swaps public keys still cannot: it
//! never learns the secret, so its derived key differs from the offerer's and
//! the AEAD open fails. This is the standard "short-authenticated-string over
//! ECDH" shape (à la Signal/WhatsApp device linking), written once in the core
//! and consumed by every platform.
//!
//! Flow (transport lives in the host — this module is pure crypto):
//! 1. The device that SHOWS the QR calls [`PairingSession::offer`]. It keeps
//!    the session and encodes [`PairingOffer`] into the QR.
//! 2. The device that SCANS calls [`accept`] with the decoded offer. It gets
//!    its own ephemeral public key (to send back over the transport) and the
//!    derived channel key.
//! 3. The offerer feeds the scanner's returned key into
//!    [`PairingSession::channel_key`]. Both sides now hold the same key.
//! 4. The device that HOLDS the payload (the existing member) — only after the
//!    user approves — calls [`seal`]; the joiner calls [`open`]. The AAD binds
//!    both public keys so a key from a different exchange can't be replayed.
//!
//! The payload (space_id, sync token, peer addresses, and the API key IFF the
//! user opted in) never touches the base — it is a one-shot transfer that
//! lands directly in the joiner's keychain/settings.

use chacha20poly1305::aead::{Aead, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, KeyInit, Nonce};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use spake2::{Ed25519Group, Identity, Password, Spake2};
use x25519_dalek::{PublicKey, StaticSecret};

use crate::embedder::CoreError;

const PAIRING_VERSION: u8 = 1;
const SECRET_LEN: usize = 32;
const NONCE_LEN: usize = 12;
const HKDF_INFO: &[u8] = b"synapse-pairing-v1";

fn b64(bytes: &[u8]) -> String {
    use base64ct::{Base64, Encoding};
    Base64::encode_string(bytes)
}

fn unb64(s: &str) -> Result<Vec<u8>, CoreError> {
    use base64ct::{Base64, Encoding};
    Base64::decode_vec(s).map_err(|_| CoreError::Storage("pairing: bad base64".into()))
}

/// The QR payload: version, the offerer's ephemeral public key, the shared
/// secret that authenticates the channel, and where to reach the offerer.
/// Encoded as compact JSON → base64 for the QR.
#[derive(Debug, Clone, PartialEq)]
pub struct PairingOffer {
    pub offer_pub: [u8; 32],
    pub secret: [u8; SECRET_LEN],
    /// Reachability hints for the transport, e.g. `["http://192.168.1.10:8000"]`.
    pub addrs: Vec<String>,
}

impl PairingOffer {
    /// Compact wire form for the QR: `v.offer_pub.secret.addr,addr` in base64
    /// fields joined by `|` (QR-friendly, no JSON braces to escape).
    pub fn encode(&self) -> String {
        format!(
            "{}|{}|{}|{}",
            PAIRING_VERSION,
            b64(&self.offer_pub),
            b64(&self.secret),
            self.addrs.join(",")
        )
    }

    pub fn decode(s: &str) -> Result<Self, CoreError> {
        let parts: Vec<&str> = s.splitn(4, '|').collect();
        if parts.len() != 4 {
            return Err(CoreError::Storage("pairing: malformed offer".into()));
        }
        if parts[0] != PAIRING_VERSION.to_string() {
            return Err(CoreError::Storage(format!(
                "pairing: unsupported version {}",
                parts[0]
            )));
        }
        let offer_pub: [u8; 32] = unb64(parts[1])?
            .try_into()
            .map_err(|_| CoreError::Storage("pairing: bad offer key".into()))?;
        let secret: [u8; SECRET_LEN] = unb64(parts[2])?
            .try_into()
            .map_err(|_| CoreError::Storage("pairing: bad secret".into()))?;
        let addrs = if parts[3].is_empty() {
            Vec::new()
        } else {
            parts[3].split(',').map(|s| s.to_string()).collect()
        };
        Ok(Self {
            offer_pub,
            secret,
            addrs,
        })
    }
}

/// Derive the channel key from a completed ECDH plus the QR secret. Both sides
/// feed the SAME (offer_pub, accept_pub, secret) in the SAME order, so the
/// salt/info are identical and the keys match.
fn derive_key(shared: &[u8; 32], secret: &[u8], offer_pub: &[u8], accept_pub: &[u8]) -> Key {
    // Bind the transcript into the salt so the derived key is unique to this
    // exchange (defends against key reuse / unknown-key-share).
    let mut salt = Vec::with_capacity(SECRET_LEN + 64);
    salt.extend_from_slice(secret);
    salt.extend_from_slice(offer_pub);
    salt.extend_from_slice(accept_pub);
    let hk = Hkdf::<Sha256>::new(Some(&salt), shared);
    let mut okm = [0u8; 32];
    hk.expand(HKDF_INFO, &mut okm)
        .expect("32 is a valid HKDF-SHA256 length");
    Key::clone_from_slice(&okm)
}

/// AAD binding both public keys — a ciphertext from one exchange can't be
/// opened as belonging to another.
fn aad(offer_pub: &[u8], accept_pub: &[u8]) -> Vec<u8> {
    let mut a = Vec::with_capacity(64);
    a.extend_from_slice(offer_pub);
    a.extend_from_slice(accept_pub);
    a
}

/// The offerer's half of a pairing: holds the ephemeral secret between showing
/// the QR and receiving the scanner's public key. Send + Sync for UniFFI.
pub struct PairingSession {
    secret_scalar: StaticSecret,
    offer_pub: [u8; 32],
    secret: [u8; SECRET_LEN],
}

impl PairingSession {
    /// Start a pairing: fresh ephemeral keypair + fresh secret. Returns the
    /// session (keep it) and the offer to render as a QR.
    pub fn offer(addrs: Vec<String>) -> Result<(Self, PairingOffer), CoreError> {
        let mut scalar_bytes = [0u8; 32];
        let mut secret = [0u8; SECRET_LEN];
        getrandom::getrandom(&mut scalar_bytes)
            .map_err(|e| CoreError::Storage(format!("pairing: rng: {e}")))?;
        getrandom::getrandom(&mut secret)
            .map_err(|e| CoreError::Storage(format!("pairing: rng: {e}")))?;
        let secret_scalar = StaticSecret::from(scalar_bytes);
        let offer_pub = PublicKey::from(&secret_scalar).to_bytes();
        let offer = PairingOffer {
            offer_pub,
            secret,
            addrs,
        };
        Ok((
            Self {
                secret_scalar,
                offer_pub,
                secret,
            },
            offer,
        ))
    }

    pub fn offer_public(&self) -> [u8; 32] {
        self.offer_pub
    }

    /// Complete the exchange with the scanner's returned public key → the
    /// channel key. Returns raw bytes so the FFI can pass them around; use
    /// [`seal`]/[`open`] with (offer_pub, accept_pub) for the AEAD.
    pub fn channel_key(&self, accept_pub: &[u8; 32]) -> [u8; 32] {
        let peer = PublicKey::from(*accept_pub);
        let shared = self.secret_scalar.diffie_hellman(&peer).to_bytes();
        derive_key(&shared, &self.secret, &self.offer_pub, accept_pub)
            .as_slice()
            .try_into()
            .expect("32-byte key")
    }
}

/// The scanner's side: given the decoded offer, produce our ephemeral public
/// key (to send back) and the derived channel key.
pub fn accept(offer: &PairingOffer) -> Result<([u8; 32], [u8; 32]), CoreError> {
    let mut scalar_bytes = [0u8; 32];
    getrandom::getrandom(&mut scalar_bytes)
        .map_err(|e| CoreError::Storage(format!("pairing: rng: {e}")))?;
    let scalar = StaticSecret::from(scalar_bytes);
    let accept_pub = PublicKey::from(&scalar).to_bytes();
    let peer = PublicKey::from(offer.offer_pub);
    let shared = scalar.diffie_hellman(&peer).to_bytes();
    let key = derive_key(&shared, &offer.secret, &offer.offer_pub, &accept_pub)
        .as_slice()
        .try_into()
        .expect("32-byte key");
    Ok((accept_pub, key))
}

/// AEAD-seal `plaintext` under the channel key, binding both handshake
/// messages as AAD (X25519 public keys for the QR channel, SPAKE2 messages
/// for the code channel — SYN-137). Output = nonce ‖ ciphertext, base64.
/// Called by the payload holder ONLY after the user approves the join.
pub fn seal(
    channel_key: &[u8; 32],
    offer_pub: &[u8],
    accept_pub: &[u8],
    plaintext: &[u8],
) -> Result<String, CoreError> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(channel_key));
    let mut nonce_bytes = [0u8; NONCE_LEN];
    getrandom::getrandom(&mut nonce_bytes)
        .map_err(|e| CoreError::Storage(format!("pairing: rng: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ad = aad(offer_pub, accept_pub);
    let ct = cipher
        .encrypt(
            nonce,
            Payload {
                msg: plaintext,
                aad: &ad,
            },
        )
        .map_err(|_| CoreError::Storage("pairing: seal failed".into()))?;
    let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ct);
    Ok(b64(&out))
}

/// Open what [`seal`] produced. Fails (AEAD auth error) if the key differs —
/// which is exactly what a MITM without the QR secret ends up with.
pub fn open(
    channel_key: &[u8; 32],
    offer_pub: &[u8],
    accept_pub: &[u8],
    sealed_b64: &str,
) -> Result<Vec<u8>, CoreError> {
    let raw = unb64(sealed_b64)?;
    if raw.len() < NONCE_LEN {
        return Err(CoreError::Storage("pairing: short ciphertext".into()));
    }
    let (nonce_bytes, ct) = raw.split_at(NONCE_LEN);
    let cipher = ChaCha20Poly1305::new(Key::from_slice(channel_key));
    let ad = aad(offer_pub, accept_pub);
    cipher
        .decrypt(
            Nonce::from_slice(nonce_bytes),
            Payload { msg: ct, aad: &ad },
        )
        .map_err(|_| CoreError::Storage("pairing: open failed (wrong channel / tampered)".into()))
}

// ── code pairing (SYN-137) — Mac↔Mac, no camera ──────────────────────────────
//
// A 6-digit code cannot ride the QR scheme above: mixed into HKDF it would be
// offline-brute-forceable (10^6 tries against one observed exchange). SPAKE2
// makes the code safe by construction — each guess requires a fresh ACTIVE
// handshake with the member, who counts and caps the attempts. A wrong code
// does not error inside SPAKE2: both sides simply derive different keys, so
// the joiner proves knowledge with a key-confirmation MAC over the transcript
// before the request may enter the human-approval queue. The payload transfer
// then reuses `seal`/`open` with the two SPAKE2 messages as AAD.
//
// Never log the code, the messages or the derived key.

const CODE_IDENTITY: &[u8] = b"synapse-pair-code-v1";
const CONFIRM_INFO: &[u8] = b"synapse-pair-code-confirm-v1";

type HmacSha256 = Hmac<Sha256>;

/// One side of the PAKE on the short code (symmetric — member and joiner run
/// the same role). Keep it between sending our message and receiving the
/// peer's; `finish` consumes it (SPAKE2 states are one-shot).
pub struct CodePairing {
    state: Spake2<Ed25519Group>,
}

impl CodePairing {
    /// Start the handshake on `code`. Returns the session and our message
    /// (~33 bytes) to send to the peer.
    pub fn start(code: &str) -> (Self, Vec<u8>) {
        let (state, msg) = Spake2::<Ed25519Group>::start_symmetric(
            &Password::new(code.as_bytes()),
            &Identity::new(CODE_IDENTITY),
        );
        (Self { state }, msg)
    }

    /// Complete with the peer's message → the 32-byte channel key. With a
    /// wrong code this still SUCCEEDS but yields a different key — detect it
    /// with [`code_confirm_mac`]/[`code_confirm_verify`].
    pub fn finish(self, peer_msg: &[u8]) -> Result<[u8; 32], CoreError> {
        let key = self
            .state
            .finish(peer_msg)
            .map_err(|_| CoreError::Storage("pairing: code handshake failed".into()))?;
        key.as_slice()
            .try_into()
            .map_err(|_| CoreError::Storage("pairing: bad code key length".into()))
    }
}

/// Joiner side: key-confirmation MAC over the transcript (member msg first),
/// proving — one online attempt at a time — that the joiner knew the code.
pub fn code_confirm_mac(key: &[u8; 32], member_msg: &[u8], joiner_msg: &[u8]) -> [u8; 32] {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(CONFIRM_INFO);
    mac.update(member_msg);
    mac.update(joiner_msg);
    mac.finalize().into_bytes().into()
}

/// Member side: constant-time verification of the joiner's confirmation MAC.
/// A mismatch burns one attempt; the caller caps them.
pub fn code_confirm_verify(
    key: &[u8; 32],
    member_msg: &[u8],
    joiner_msg: &[u8],
    mac_bytes: &[u8],
) -> bool {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(CONFIRM_INFO);
    mac.update(member_msg);
    mac.update(joiner_msg);
    mac.verify_slice(mac_bytes).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full happy path: offer → accept → both derive the same key → seal/open
    /// round-trips the payload.
    #[test]
    fn honest_pairing_transfers_payload() {
        let (session, offer) = PairingSession::offer(vec!["http://10.0.0.2:8000".into()]).unwrap();
        // The QR survives an encode/decode round trip.
        let offer = PairingOffer::decode(&offer.encode()).unwrap();

        let (accept_pub, joiner_key) = accept(&offer).unwrap();
        let offerer_key = session.channel_key(&accept_pub);
        assert_eq!(offerer_key, joiner_key, "both sides derive one key");

        let payload = br#"{"space_id":"uuid-s","token":"secret-token","key":"sk-ant-xxx"}"#;
        let sealed = seal(&offerer_key, &offer.offer_pub, &accept_pub, payload).unwrap();
        let opened = open(&joiner_key, &offer.offer_pub, &accept_pub, &sealed).unwrap();
        assert_eq!(opened, payload);
    }

    /// A MITM that swaps the scanner's public key (never having seen the QR
    /// secret) derives a different key: the offerer's ciphertext won't open.
    #[test]
    fn mitm_without_qr_secret_cannot_open() {
        let (session, offer) = PairingSession::offer(vec![]).unwrap();
        // Honest joiner accepts; MITM substitutes its OWN accept key on the
        // wire back to the offerer.
        let (_honest_pub, _honest_key) = accept(&offer).unwrap();
        let mut mitm_scalar = [7u8; 32];
        mitm_scalar[0] &= 248;
        let mitm = StaticSecret::from(mitm_scalar);
        let mitm_pub = PublicKey::from(&mitm).to_bytes();

        // Offerer completes with the MITM's key (it thinks that's the joiner).
        let offerer_key = session.channel_key(&mitm_pub);
        let payload = b"top secret";
        let sealed = seal(&offerer_key, &offer.offer_pub, &mitm_pub, payload).unwrap();

        // The MITM computes ECDH but lacks `secret`, so its HKDF salt differs.
        let shared = mitm.diffie_hellman(&PublicKey::from(offer.offer_pub)).to_bytes();
        let wrong_secret = [0u8; SECRET_LEN];
        let mitm_key: [u8; 32] = derive_key(&shared, &wrong_secret, &offer.offer_pub, &mitm_pub)
            .as_slice()
            .try_into()
            .unwrap();
        assert!(
            open(&mitm_key, &offer.offer_pub, &mitm_pub, &sealed).is_err(),
            "MITM without the QR secret must not open the payload"
        );
    }

    /// A ciphertext bound to one exchange can't be opened under another
    /// exchange's public keys (AAD mismatch), even with the right key.
    #[test]
    fn aad_binds_ciphertext_to_its_public_keys() {
        let (session, offer) = PairingSession::offer(vec![]).unwrap();
        let (accept_pub, key) = accept(&offer).unwrap();
        let _ = session.channel_key(&accept_pub);
        let sealed = seal(&key, &offer.offer_pub, &accept_pub, b"payload").unwrap();

        let other_pub = [9u8; 32];
        assert!(open(&key, &offer.offer_pub, &other_pub, &sealed).is_err());
    }

    #[test]
    fn tampered_ciphertext_is_rejected() {
        let (session, offer) = PairingSession::offer(vec![]).unwrap();
        let (accept_pub, key) = accept(&offer).unwrap();
        let _ = session.channel_key(&accept_pub);
        let mut sealed = seal(&key, &offer.offer_pub, &accept_pub, b"payload").unwrap();
        // Flip a character in the base64 body.
        let mut bytes = sealed.into_bytes();
        let last = bytes.len() - 1;
        bytes[last] = if bytes[last] == b'A' { b'B' } else { b'A' };
        sealed = String::from_utf8(bytes).unwrap();
        assert!(open(&key, &offer.offer_pub, &accept_pub, &sealed).is_err());
    }

    #[test]
    fn offer_decode_rejects_bad_version_and_shape() {
        assert!(PairingOffer::decode("2|a|b|c").is_err());
        assert!(PairingOffer::decode("1|only-three|parts").is_err());
    }

    #[test]
    fn two_offers_use_distinct_secrets_and_keys() {
        let (_s1, o1) = PairingSession::offer(vec![]).unwrap();
        let (_s2, o2) = PairingSession::offer(vec![]).unwrap();
        assert_ne!(o1.secret, o2.secret);
        assert_ne!(o1.offer_pub, o2.offer_pub);
    }

    /// SYN-137 happy path: same code → same key, MAC confirms, seal/open
    /// round-trips the payload with the SPAKE2 messages as AAD.
    #[test]
    fn code_pairing_same_code_transfers_payload() {
        let (member, msg_m) = CodePairing::start("483921");
        let (joiner, msg_j) = CodePairing::start("483921");
        let key_m = member.finish(&msg_j).unwrap();
        let key_j = joiner.finish(&msg_m).unwrap();
        assert_eq!(key_m, key_j, "same code, same transcript → one key");

        let mac = code_confirm_mac(&key_j, &msg_m, &msg_j);
        assert!(code_confirm_verify(&key_m, &msg_m, &msg_j, &mac));

        let payload = br#"{"space_id":"uuid-s","token":"secret"}"#;
        let sealed = seal(&key_m, &msg_m, &msg_j, payload).unwrap();
        assert_eq!(open(&key_j, &msg_m, &msg_j, &sealed).unwrap(), payload);
    }

    /// A wrong code never errors inside SPAKE2 — it yields a different key.
    /// The confirmation MAC is what catches it (and the AEAD stays shut).
    #[test]
    fn code_pairing_wrong_code_fails_confirmation_and_open() {
        let (member, msg_m) = CodePairing::start("483921");
        let (guesser, msg_g) = CodePairing::start("000000");
        let key_m = member.finish(&msg_g).unwrap();
        let key_g = guesser.finish(&msg_m).unwrap();
        assert_ne!(key_m, key_g);

        let mac = code_confirm_mac(&key_g, &msg_m, &msg_g);
        assert!(
            !code_confirm_verify(&key_m, &msg_m, &msg_g, &mac),
            "a guessed code must not pass confirmation"
        );

        let sealed = seal(&key_m, &msg_m, &msg_g, b"payload").unwrap();
        assert!(open(&key_g, &msg_m, &msg_g, &sealed).is_err());
    }

    /// The MAC binds the transcript: swapping messages or truncating rejects.
    #[test]
    fn code_confirmation_binds_the_transcript() {
        let (member, msg_m) = CodePairing::start("112233");
        let (joiner, msg_j) = CodePairing::start("112233");
        let key = member.finish(&msg_j).unwrap();
        let _ = joiner;

        let mac = code_confirm_mac(&key, &msg_m, &msg_j);
        assert!(!code_confirm_verify(&key, &msg_j, &msg_m, &mac), "order matters");
        assert!(!code_confirm_verify(&key, &msg_m, &msg_j, &mac[..16]), "no truncation");
    }
}
