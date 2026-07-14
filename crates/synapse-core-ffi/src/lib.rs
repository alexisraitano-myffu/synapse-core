//! UniFFI surface of synapse-core, consumed by Kotlin (Android) and Swift (iOS).

use std::sync::Arc;

uniffi::setup_scaffolding!();

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum CoreError {
    #[error("model load failed: {msg}")]
    ModelLoad { msg: String },
    #[error("embedding failed: {msg}")]
    Embedding { msg: String },
    #[error("storage error: {msg}")]
    Storage { msg: String },
    #[error("llm http error: {msg}")]
    LlmHttp { msg: String },
    #[error("llm content error: {msg}")]
    LlmContent { msg: String },
}

impl From<synapse_core::CoreError> for CoreError {
    fn from(e: synapse_core::CoreError) -> Self {
        match e {
            synapse_core::CoreError::ModelLoad(msg) => CoreError::ModelLoad { msg },
            synapse_core::CoreError::Embedding(msg) => CoreError::Embedding { msg },
            synapse_core::CoreError::Storage(msg) => CoreError::Storage { msg },
            synapse_core::CoreError::LlmHttp(msg) => CoreError::LlmHttp { msg },
            synapse_core::CoreError::LlmContent(msg) => CoreError::LlmContent { msg },
        }
    }
}

#[uniffi::export]
pub fn embedding_dim() -> u32 {
    synapse_core::EMBEDDING_DIM as u32
}

/// Text embedder. `model_dir` must contain the bundled ONNX + tokenizer files
/// (same files as the desktop backend, shipped as app assets).
#[derive(uniffi::Object)]
pub struct Embedder {
    inner: synapse_core::Embedder,
}

#[uniffi::export]
impl Embedder {
    #[uniffi::constructor]
    pub fn new(model_dir: String) -> Result<Arc<Self>, CoreError> {
        let inner = synapse_core::Embedder::new(&model_dir)?;
        Ok(Arc::new(Self { inner }))
    }

    /// L2-normalized 384-d vector, bit-parity with the desktop core.
    pub fn embed(&self, text: String) -> Result<Vec<f32>, CoreError> {
        Ok(self.inner.embed(&text)?)
    }

    /// One vector per ~128-token window (SYN-118); short text = one vector.
    pub fn embed_chunks(&self, text: String) -> Result<Vec<Vec<f32>>, CoreError> {
        Ok(self.inner.embed_chunks(&text)?)
    }
}

/// A note KNN hit; `distance` is sqlite-vec's L2 on unit vectors ([0, 2]).
#[derive(uniffi::Record)]
pub struct NoteHit {
    pub note_id: String,
    pub distance: f64,
}

/// An entity similarity hit; `score` = `1 - distance/2`, rounded to 4 decimals.
#[derive(uniffi::Record)]
pub struct EntityHit {
    pub id: String,
    pub canonical_name: String,
    pub entity_type: Option<String>,
    pub summary: String,
    pub score: f64,
}

#[derive(uniffi::Record)]
pub struct ResourceHit {
    pub id: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub summary: String,
    pub score: f64,
}

/// Storage substrate (SYN-110 / T1): schema ownership + vector reads/writes.
/// Embedding blobs are the sqlite-vec serialized float32 format (what
/// `Embedder.embed` yields once packed little-endian).
#[derive(uniffi::Object)]
pub struct Storage {
    inner: synapse_core::Storage,
}

#[uniffi::export]
impl Storage {
    /// Open (creating if needed) the database and init/migrate the schema.
    #[uniffi::constructor]
    pub fn open(db_path: String) -> Result<Arc<Self>, CoreError> {
        let inner = synapse_core::Storage::open(&db_path)?;
        Ok(Arc::new(Self { inner }))
    }

    pub fn upsert_note_vector(&self, note_id: String, embedding: Vec<u8>) -> Result<(), CoreError> {
        Ok(self.inner.upsert_note_vector(&note_id, &embedding)?)
    }

    /// Chunked upsert (SYN-118): chunk 0 keyed by the note uuid, then `uuid#k`.
    pub fn upsert_note_vectors(
        &self,
        note_id: String,
        embeddings: Vec<Vec<u8>>,
    ) -> Result<(), CoreError> {
        Ok(self.inner.upsert_note_vectors(&note_id, &embeddings)?)
    }

    pub fn delete_note_vector(&self, note_id: String) -> Result<(), CoreError> {
        Ok(self.inner.delete_note_vector(&note_id)?)
    }

    pub fn get_note_vector(&self, note_id: String) -> Result<Option<Vec<u8>>, CoreError> {
        Ok(self.inner.get_note_vector(&note_id)?)
    }

    pub fn search_notes(&self, query: Vec<u8>, k: u32) -> Result<Vec<NoteHit>, CoreError> {
        Ok(self
            .inner
            .search_notes(&query, k)?
            .into_iter()
            .map(|h| NoteHit {
                note_id: h.note_id,
                distance: h.distance,
            })
            .collect())
    }

    pub fn set_entity_embedding(
        &self,
        entity_id: String,
        embedding: Vec<u8>,
    ) -> Result<(), CoreError> {
        Ok(self.inner.set_entity_embedding(&entity_id, &embedding)?)
    }

    pub fn set_resource_embedding(
        &self,
        resource_id: String,
        embedding: Vec<u8>,
    ) -> Result<(), CoreError> {
        Ok(self.inner.set_resource_embedding(&resource_id, &embedding)?)
    }

    pub fn search_entities(
        &self,
        query: Vec<u8>,
        limit: u32,
        min_score: f64,
        type_filter: Option<String>,
        exclude_ids: Vec<String>,
    ) -> Result<Vec<EntityHit>, CoreError> {
        Ok(self
            .inner
            .search_entities(&query, limit, min_score, type_filter.as_deref(), &exclude_ids)?
            .into_iter()
            .map(|h| EntityHit {
                id: h.id,
                canonical_name: h.canonical_name,
                entity_type: h.entity_type,
                summary: h.summary,
                score: h.score,
            })
            .collect())
    }

    pub fn search_resources(
        &self,
        query: Vec<u8>,
        limit: u32,
    ) -> Result<Vec<ResourceHit>, CoreError> {
        Ok(self
            .inner
            .search_resources(&query, limit)?
            .into_iter()
            .map(|h| ResourceHit {
                id: h.id,
                title: h.title,
                url: h.url,
                summary: h.summary,
                score: h.score,
            })
            .collect())
    }

    // ── P2P sync (SYN-112/SYN-113): engine surface for the mobile transport ──

    pub fn sync_device_id(&self) -> Result<String, CoreError> {
        Ok(self.inner.sync_device_id()?)
    }

    /// Changeset (protocol-v1 JSON string) of everything journaled after
    /// `since`; paginate with the returned `next` cursor while `has_more`.
    pub fn sync_changes_since(&self, since: i64, limit: i64) -> Result<String, CoreError> {
        Ok(self.inner.sync_changes_since(since, limit)?)
    }

    /// Merge a peer's changeset (per-column LWW) → JSON report. The caller
    /// re-embeds the notes listed under `notes_changed`.
    pub fn sync_apply(&self, changes_json: String) -> Result<String, CoreError> {
        Ok(self.inner.sync_apply(&changes_json)?)
    }
}

/// One SQLite value crossing the FFI boundary, in either direction.
#[derive(uniffi::Enum)]
pub enum SqlValue {
    Null,
    Integer { value: i64 },
    Real { value: f64 },
    Text { value: String },
    Blob { value: Vec<u8> },
}

impl From<SqlValue> for synapse_core::SqlValue {
    fn from(v: SqlValue) -> Self {
        match v {
            SqlValue::Null => Self::Null,
            SqlValue::Integer { value } => Self::Integer(value),
            SqlValue::Real { value } => Self::Real(value),
            SqlValue::Text { value } => Self::Text(value),
            SqlValue::Blob { value } => Self::Blob(value),
        }
    }
}

impl From<synapse_core::SqlValue> for SqlValue {
    fn from(v: synapse_core::SqlValue) -> Self {
        match v {
            synapse_core::SqlValue::Null => Self::Null,
            synapse_core::SqlValue::Integer(value) => Self::Integer { value },
            synapse_core::SqlValue::Real(value) => Self::Real { value },
            synapse_core::SqlValue::Text(value) => Self::Text { value },
            synapse_core::SqlValue::Blob(value) => Self::Blob { value },
        }
    }
}

/// Result of an `execute`: `columns` is None for statements returning no
/// result set (INSERT/UPDATE/DDL).
#[derive(uniffi::Record)]
pub struct SqlResult {
    pub columns: Option<Vec<String>>,
    pub rows: Vec<Vec<SqlValue>>,
}

/// LLM call settings shared by the cycle passes (mirror of the core's
/// LlmConfig; a fuel token routes through the proxy instead of a raw key).
#[derive(uniffi::Record)]
pub struct LlmSettings {
    pub model: String,
    pub api_key: String,
    pub prompts_dir: String,
    pub today: String,
    pub base_url: Option<String>,
    pub fuel_token: Option<String>,
}

impl From<LlmSettings> for synapse_core::LlmConfig {
    fn from(s: LlmSettings) -> Self {
        synapse_core::LlmConfig {
            model: s.model,
            api_key: s.api_key,
            base_url: s.base_url,
            fuel_token: s.fuel_token,
            prompts_dir: s.prompts_dir,
            today: s.today,
        }
    }
}

/// Generic SQL access to the core-owned database — the ONLY SQLite in the
/// process (mixing two SQLite libraries on one file corrupts it).
#[derive(uniffi::Object)]
pub struct SqlConnection {
    inner: synapse_core::SqlConnection,
}

#[uniffi::export]
impl SqlConnection {
    /// Open a SQL connection. No schema init — that's `Storage`'s job.
    #[uniffi::constructor]
    pub fn open(db_path: String) -> Result<Arc<Self>, CoreError> {
        let inner = synapse_core::connect(&db_path)?;
        Ok(Arc::new(Self { inner }))
    }

    pub fn execute(&self, sql: String, params: Vec<SqlValue>) -> Result<SqlResult, CoreError> {
        let values: Vec<synapse_core::SqlValue> = params.into_iter().map(Into::into).collect();
        let result = self.inner.execute(&sql, &values)?;
        Ok(SqlResult {
            columns: result.columns,
            rows: result
                .rows
                .into_iter()
                .map(|row| row.into_iter().map(Into::into).collect())
                .collect(),
        })
    }

    pub fn last_insert_rowid(&self) -> Result<i64, CoreError> {
        Ok(self.inner.last_insert_rowid()?)
    }

    /// SYN-132 — one-call read snapshot for the app's local replica: the same
    /// JSON shapes as the desktop backend's read endpoints, served from this
    /// local core db.
    pub fn read_snapshot(&self) -> Result<String, CoreError> {
        Ok(self.inner.read_snapshot()?.to_string())
    }

    /// SYN-132 — reverse provenance of one capture (`/capture/{id}/generated`).
    pub fn generated_for_capture(&self, capture_id: String) -> Result<String, CoreError> {
        Ok(self.inner.generated_for_capture(&capture_id)?.to_string())
    }

    // ── Full-cycle passes (SYN-130): the mobile host runs the same Dream
    // Cycle as the desktop backend, so the T5 surface crosses the FFI too. ──

    /// SYN-19 decay pass over atomic_notes; `now` = optional fixed clock
    /// 'YYYY-MM-DD HH:MM:SS' (tests inject it), None = system now.
    pub fn apply_decay(&self, tau_days: Option<f64>, now: Option<String>) -> Result<i64, CoreError> {
        Ok(self.inner.apply_decay(tau_days, now.as_deref())?)
    }

    /// SYN-68 decay pass over entities (anchor `last_mentioned`).
    pub fn apply_entity_decay(
        &self,
        tau_days: Option<f64>,
        now: Option<String>,
    ) -> Result<i64, CoreError> {
        Ok(self.inner.apply_entity_decay(tau_days, now.as_deref())?)
    }

    /// Move notes' reactivation anchor toward now (1.0 = full mention reset,
    /// <1 = light search bump). Returns the count touched.
    pub fn reactivate_notes(
        &self,
        note_ids: Vec<String>,
        factor: f64,
        now: Option<String>,
    ) -> Result<i64, CoreError> {
        Ok(self.inner.reactivate_notes(&note_ids, factor, now.as_deref())?)
    }

    /// Full reactivation of every note mentioning one of `entity_names`.
    pub fn reactivate_notes_for_entities(
        &self,
        entity_names: Vec<String>,
        now: Option<String>,
    ) -> Result<i64, CoreError> {
        Ok(self
            .inner
            .reactivate_notes_for_entities(&entity_names, now.as_deref())?)
    }

    /// SYN-23 — the digest's structured week as JSON (pure SQL on THIS
    /// connection, offline). `now` = optional fixed clock 'YYYY-MM-DD HH:MM:SS'.
    pub fn gather_week(&self, now: Option<String>, days: i64) -> Result<String, CoreError> {
        Ok(self.inner.gather_week(now.as_deref(), days)?.to_string())
    }
}

// ── Pairing channel (SYN-128): authenticated secret transfer at join time ──

/// Scanner-side result of accepting a QR offer: send `accept_pub` back to the
/// offerer over the transport; keep `channel_key` to open the sealed payload.
#[derive(uniffi::Record)]
pub struct PairingAccept {
    pub accept_pub: Vec<u8>,
    pub channel_key: Vec<u8>,
}

/// Pairing offerer session (SYN-128): the device that SHOWS the QR keeps this
/// between showing the offer and receiving the scanner's returned key.
#[derive(uniffi::Object)]
pub struct PairingSession {
    inner: synapse_core::PairingSession,
    qr: String,
}

#[uniffi::export]
impl PairingSession {
    /// Start a pairing. `addrs` = how the joiner can reach us (LAN URLs).
    /// Render `qr()` as a QR code.
    #[uniffi::constructor]
    pub fn offer(addrs: Vec<String>) -> Result<Arc<Self>, CoreError> {
        let (inner, offer) = synapse_core::PairingSession::offer(addrs)?;
        let qr = offer.encode();
        Ok(Arc::new(Self { inner, qr }))
    }

    /// The offer string to render as a QR code.
    pub fn qr(&self) -> String {
        self.qr.clone()
    }

    /// The offerer's ephemeral public key (32 bytes), for AAD in seal.
    pub fn offer_pub(&self) -> Vec<u8> {
        self.inner.offer_public().to_vec()
    }

    /// Complete with the scanner's returned public key → channel key.
    pub fn channel_key(&self, accept_pub: Vec<u8>) -> Result<Vec<u8>, CoreError> {
        let ap = key32(&accept_pub, "accept_pub")?;
        Ok(self.inner.channel_key(&ap).to_vec())
    }
}

/// Scanner side (SYN-128): decode the QR and derive the channel key.
#[uniffi::export]
pub fn pairing_accept(qr: String) -> Result<PairingAccept, CoreError> {
    let offer = synapse_core::PairingOffer::decode(&qr)?;
    let (accept_pub, channel_key) = synapse_core::pairing_accept(&offer)?;
    Ok(PairingAccept {
        accept_pub: accept_pub.to_vec(),
        channel_key: channel_key.to_vec(),
    })
}

/// The reachability hints embedded in a QR offer (so a joiner knows where to
/// call back) — decoded without completing the exchange.
#[uniffi::export]
pub fn pairing_offer_addrs(qr: String) -> Result<Vec<String>, CoreError> {
    Ok(synapse_core::PairingOffer::decode(&qr)?.addrs)
}

/// AEAD-seal a payload under the channel key (SYN-128). Returns base64.
#[uniffi::export]
pub fn pairing_seal(
    channel_key: Vec<u8>,
    offer_pub: Vec<u8>,
    accept_pub: Vec<u8>,
    plaintext: Vec<u8>,
) -> Result<String, CoreError> {
    let ck = key32(&channel_key, "channel_key")?;
    let op = key32(&offer_pub, "offer_pub")?;
    let ap = key32(&accept_pub, "accept_pub")?;
    Ok(synapse_core::pairing_seal(&ck, &op, &ap, &plaintext)?)
}

/// Open what `pairing_seal` produced (SYN-128) → the plaintext bytes.
#[uniffi::export]
pub fn pairing_open(
    channel_key: Vec<u8>,
    offer_pub: Vec<u8>,
    accept_pub: Vec<u8>,
    sealed_b64: String,
) -> Result<Vec<u8>, CoreError> {
    let ck = key32(&channel_key, "channel_key")?;
    let op = key32(&offer_pub, "offer_pub")?;
    let ap = key32(&accept_pub, "accept_pub")?;
    Ok(synapse_core::pairing_open(&ck, &op, &ap, &sealed_b64)?)
}

fn key32(bytes: &[u8], what: &str) -> Result<[u8; 32], CoreError> {
    bytes.try_into().map_err(|_| CoreError::Storage {
        msg: format!("pairing: {what} must be 32 bytes"),
    })
}

/// The Dream Cycle brain (SYN-111): deterministic routing + classifier
/// orchestration. JSON strings across the boundary, same shapes as PyO3.
#[derive(uniffi::Object)]
pub struct Brain {
    inner: synapse_core::Brain,
}

#[uniffi::export]
impl Brain {
    #[uniffi::constructor]
    pub fn open(db_path: String, model_dir: Option<String>) -> Result<Arc<Self>, CoreError> {
        let inner = synapse_core::Brain::open(&db_path, model_dir.as_deref())?;
        Ok(Arc::new(Self { inner }))
    }

    /// Route one capture; returns the RouteReport as JSON.
    #[allow(clippy::too_many_arguments)]
    pub fn route_capture(
        &self,
        entry_json: String,
        classified_json: String,
        now: String,
        today: String,
        intentions_cutoff: String,
        now_sql: String,
    ) -> Result<String, CoreError> {
        let entry: serde_json::Value = serde_json::from_str(&entry_json)
            .map_err(|e| CoreError::Storage { msg: e.to_string() })?;
        let classified: serde_json::Value = serde_json::from_str(&classified_json)
            .map_err(|e| CoreError::Storage { msg: e.to_string() })?;
        let ctx = synapse_core::RouteContext {
            now,
            today,
            intentions_cutoff,
            now_sql,
        };
        let report = self.inner.route_capture(&entry, &classified, &ctx)?;
        Ok(synapse_core::Brain::report_to_json(&report).to_string())
    }

    /// Embed with the Brain's already-loaded model (no second Embedder):
    /// the re-embed path after a sync apply on mobile hosts.
    pub fn embed(&self, text: String) -> Result<Vec<f32>, CoreError> {
        Ok(self.inner.embed_text(&text)?)
    }

    /// Chunked variant (SYN-118): one vector per ~128-token window, feeding
    /// `Storage.upsert_note_vectors` so mobile re-embeds match the desktop.
    pub fn embed_chunks(&self, text: String) -> Result<Vec<Vec<f32>>, CoreError> {
        Ok(self.inner.embed_text_chunks(&text)?)
    }

    // ── Full-cycle passes (SYN-130): total parity with the desktop host. ──

    /// SYN-89 re-summary pass (T5): entities touched by the run + stale ones,
    /// summaries rebuilt from active facts/relations via the LLM. Returns the
    /// regenerated entity ids as a JSON array. An HTTP failure stops the pass
    /// silently (stale flags survive).
    pub fn resummarize(
        &self,
        touched_ids: Vec<String>,
        config: LlmSettings,
    ) -> Result<String, CoreError> {
        let ids = self.inner.resummarize(&touched_ids, &config.into())?;
        Ok(serde_json::Value::from(ids).to_string())
    }

    /// SYN-43/44 living project synthesis (T5): append + threshold-triggered
    /// refinement. Returns the new summary_md or None (failures never block).
    pub fn synthesize_project(
        &self,
        project_id: String,
        project_name: String,
        new_entry_content: String,
        new_entry_count: i64,
        config: LlmSettings,
    ) -> Result<Option<String>, CoreError> {
        Ok(self.inner.synthesize_project(
            &project_id,
            &project_name,
            &new_entry_content,
            new_entry_count,
            &config.into(),
        )?)
    }

    /// Port of `step6_vectorize` (T5): embed each entity's composite text and
    /// store the vector; per-entity failures skip. Returns the count.
    pub fn vectorize_entities(&self, entity_ids: Vec<String>) -> Result<i64, CoreError> {
        Ok(self.inner.vectorize_entities(&entity_ids)?)
    }

    /// SYN-23 — render the gathered week (JSON string) into the digest
    /// markdown (prompt = data `digest.md`, LLM via the core HTTP path).
    pub fn summarize_digest(
        &self,
        week_json: String,
        config: LlmSettings,
    ) -> Result<String, CoreError> {
        let week: serde_json::Value = serde_json::from_str(&week_json)
            .map_err(|e| CoreError::Storage { msg: e.to_string() })?;
        Ok(self.inner.summarize_digest(&week, &config.into())?)
    }

    /// SYN-23 — store the digest note (idempotent per ISO week) + its vector,
    /// on the Brain's OWN connection: call outside host transactions. Returns
    /// the note id.
    pub fn write_digest_note(
        &self,
        week_json: String,
        markdown: String,
    ) -> Result<String, CoreError> {
        let week: serde_json::Value = serde_json::from_str(&week_json)
            .map_err(|e| CoreError::Storage { msg: e.to_string() })?;
        Ok(self.inner.write_digest_note(&week, &markdown)?)
    }

    /// SYN-21 — process every URL found in a capture (each independent, one
    /// failure never blocks the others). `config = None` → snippet-fallback
    /// summaries (no LLM). Returns the stored resource ids as a JSON array.
    pub fn process_capture_resources(
        &self,
        content: String,
        capture_id: Option<String>,
        config: Option<LlmSettings>,
    ) -> Result<String, CoreError> {
        let cfg: Option<synapse_core::LlmConfig> = config.map(Into::into);
        let ids = self
            .inner
            .process_capture_resources(&content, capture_id.as_deref(), cfg.as_ref())?;
        Ok(serde_json::Value::from(ids).to_string())
    }

    pub fn validate_pending(&self, new_facts_json: String) -> Result<i64, CoreError> {
        let new_facts: Vec<serde_json::Value> = serde_json::from_str(&new_facts_json)
            .map_err(|e| CoreError::Storage { msg: e.to_string() })?;
        Ok(self.inner.validate_pending(&new_facts)?)
    }

    /// Synchronous classification (prompt build + HTTP + parse) → JSON.
    #[allow(clippy::too_many_arguments)]
    pub fn classify(
        &self,
        content: String,
        day_context: Option<String>,
        model: String,
        api_key: String,
        prompts_dir: String,
        today: String,
        base_url: Option<String>,
        fuel_token: Option<String>,
    ) -> Result<String, CoreError> {
        let config = synapse_core::LlmConfig {
            model,
            api_key,
            base_url,
            fuel_token,
            prompts_dir,
            today,
        };
        Ok(self
            .inner
            .classify(&content, day_context.as_deref(), &config)?
            .to_string())
    }
}
