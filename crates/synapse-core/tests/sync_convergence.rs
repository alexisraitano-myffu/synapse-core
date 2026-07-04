//! SYN-112 (T3, phase 4) — convergence property tests for the P2P sync
//! engine, through the PUBLIC surface only (Storage + the SQL gateway),
//! i.e. exactly what the Python host and the transport layer see.
//!
//! Model: three devices (two "Macs" + one "phone") apply random writes in a
//! random interleaving — conflicting inserts on shared primary keys,
//! cross-device column updates, deletes racing updates, owner-lock claims —
//! with PARTIAL cursor-based syncs (short pages, so pagination and relaying
//! are exercised) woven between the writes. After a closing full-mesh round,
//! the property is strict equality of every replicated table on all three
//! devices, plus echo-safety (re-applying everyone's full history from
//! cursor 0 changes nothing).
//!
//! The RNG is a seeded xorshift: every failure prints its seed and replays
//! deterministically.

use std::collections::HashMap;

use synapse_core::{connect, SqlConnection, SqlValue, Storage};

// ── Deterministic RNG (xorshift64*) ──────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [0, n).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// Uniform in [lo, hi).
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo)
    }
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

// ── One simulated device ─────────────────────────────────────────────────────

struct Device {
    name: &'static str,
    store: Storage,
    gate: SqlConnection,
    /// Pull cursor per source device name (what the transport keeps).
    cursors: HashMap<&'static str, i64>,
}

impl Device {
    fn open(dir: &std::path::Path, name: &'static str) -> Device {
        let path = dir.join(format!("{name}.db"));
        let path = path.to_str().unwrap();
        Device {
            name,
            store: Storage::open(path).unwrap(),
            gate: connect(path).unwrap(),
            cursors: HashMap::new(),
        }
    }

    fn exec(&self, sql: &str, params: &[SqlValue]) {
        self.gate.execute(sql, params).unwrap();
    }

    fn one(&self, sql: &str) -> Option<SqlValue> {
        let res = self.gate.execute(sql, &[]).unwrap();
        res.rows.first().map(|r| r[0].clone())
    }
}

/// Pull src → dst with cursor tracking and randomized page sizes.
fn pull(rng: &mut Rng, src: &Device, dst: &mut Device) {
    let mut cursor = *dst.cursors.get(src.name).unwrap_or(&0);
    loop {
        let limit = rng.range(3, 60) as i64;
        let page = src.store.sync_changes_since(cursor, limit).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&page).unwrap();
        dst.store.sync_apply(&page).unwrap();
        cursor = parsed["next"].as_i64().unwrap();
        if !parsed["has_more"].as_bool().unwrap() {
            break;
        }
    }
    dst.cursors.insert(src.name, cursor);
}

// ── Random writes ────────────────────────────────────────────────────────────

const TABLES: &[&str] = &[
    "inbox",
    "atomic_notes",
    "entities",
    "facts",
    "relations",
    "resources",
    "pending_facts",
    "review_queue",
    "intentions",
    "validation_events",
    "cycle_runs",
    "project_entries",
    "project_state_versions",
    "project_state",
    "entity_merge_proposals",
    "active_entity_types",
    "entity_type_proposals",
    "project_attach_proposals",
    "sync_owner",
];

fn random_op(rng: &mut Rng, dev: &Device, op_no: usize) {
    // Shared id pools (identical on every device) force real conflicts:
    // the same pk gets inserted/updated/deleted on devices that have not
    // synced yet.
    let cap = format!("cap-{}", rng.below(12));
    let ent = format!("ent-{}", rng.below(8));
    let note = format!("note-{}", rng.below(8));
    let fact = format!("fact-{}", rng.below(10));
    let text = |what: &str| SqlValue::Text(format!("{what} by {} #{op_no}", dev.name));

    match rng.below(12) {
        0 | 1 => {
            // Capture upsert — the same client uuid can land on two devices
            // (double-POST from the phone) with different content.
            let exists = dev
                .one(&format!("SELECT 1 FROM inbox WHERE id = '{cap}'"))
                .is_some();
            if exists {
                dev.exec(
                    "UPDATE inbox SET status = ?1, processed_at = CURRENT_TIMESTAMP WHERE id = ?2",
                    &[text("processed"), SqlValue::Text(cap)],
                );
            } else {
                dev.exec(
                    "INSERT INTO inbox (id, content, source) VALUES (?1, ?2, 'prop')",
                    &[SqlValue::Text(cap), text("capture")],
                );
            }
        }
        2 => {
            dev.exec("DELETE FROM inbox WHERE id = ?1", &[SqlValue::Text(cap)]);
        }
        3 | 4 => {
            let exists = dev
                .one(&format!("SELECT 1 FROM entities WHERE id = '{ent}'"))
                .is_some();
            if exists {
                // Column-level edits: different devices touch different
                // (or the same) columns of the same entity.
                match rng.below(3) {
                    0 => dev.exec(
                        "UPDATE entities SET canonical_name = ?1 WHERE id = ?2",
                        &[text("renamed"), SqlValue::Text(ent)],
                    ),
                    1 => dev.exec(
                        "UPDATE entities SET summary = ?1 WHERE id = ?2",
                        &[text("summary"), SqlValue::Text(ent)],
                    ),
                    _ => dev.exec(
                        "UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?1",
                        &[SqlValue::Text(ent)],
                    ),
                }
            } else {
                dev.exec(
                    "INSERT INTO entities (id, canonical_name, type) VALUES (?1, ?2, 'concept')",
                    &[SqlValue::Text(ent), text("entity")],
                );
            }
        }
        5 => {
            dev.exec("DELETE FROM entities WHERE id = ?1", &[SqlValue::Text(ent)]);
        }
        6 | 7 => {
            let exists = dev
                .one(&format!("SELECT 1 FROM facts WHERE id = '{fact}'"))
                .is_some();
            if exists {
                dev.exec(
                    "UPDATE facts SET value = ?1, confidence = 0.9 WHERE id = ?2",
                    &[text("value"), SqlValue::Text(fact)],
                );
            } else {
                dev.exec(
                    "INSERT INTO facts (id, entity_id, predicate, value, provenance_capture_id) \
                     VALUES (?1, ?2, 'aime', ?3, ?4)",
                    &[
                        SqlValue::Text(fact),
                        SqlValue::Text(ent),
                        text("fact"),
                        SqlValue::Text(cap),
                    ],
                );
            }
        }
        8 => {
            let exists = dev
                .one(&format!("SELECT 1 FROM atomic_notes WHERE id = '{note}'"))
                .is_some();
            if exists {
                dev.exec(
                    "UPDATE atomic_notes SET content = ?1 WHERE id = ?2",
                    &[text("note-edit"), SqlValue::Text(note)],
                );
            } else {
                dev.exec(
                    "INSERT INTO atomic_notes (id, content, kind, provenance_capture_id) \
                     VALUES (?1, ?2, 'note', ?3)",
                    &[SqlValue::Text(note), text("note"), SqlValue::Text(cap)],
                );
            }
        }
        9 => {
            dev.exec(
                "DELETE FROM atomic_notes WHERE id = ?1",
                &[SqlValue::Text(note)],
            );
        }
        10 => {
            // Owner-lock claim — the replicated singleton, contested by
            // every device (INSERT OR REPLACE = one HLC for the whole row).
            let epoch = match dev.one("SELECT epoch FROM sync_owner WHERE id = 'owner'") {
                Some(SqlValue::Integer(e)) => e + 1,
                _ => 1,
            };
            dev.exec(
                "INSERT OR REPLACE INTO sync_owner (id, device_id, epoch, claimed_at) \
                 VALUES ('owner', ?1, ?2, CURRENT_TIMESTAMP)",
                &[SqlValue::Text(dev.name.into()), SqlValue::Integer(epoch)],
            );
        }
        _ => {
            // Validation event — append-only stream, unique per device+op.
            dev.exec(
                "INSERT INTO validation_events (id, predicate, value, confirmed) \
                 VALUES (?1, 'p', ?2, 1)",
                &[
                    SqlValue::Text(format!("ve-{}-{op_no}", dev.name)),
                    text("event"),
                ],
            );
        }
    }
}

// ── State comparison ─────────────────────────────────────────────────────────

fn dump(dev: &Device, table: &str) -> Vec<Vec<String>> {
    let res = dev
        .gate
        .execute(&format!("SELECT * FROM \"{table}\" ORDER BY 1"), &[])
        .unwrap();
    res.rows
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect()
}

fn assert_all_equal(devices: &[Device], seed: u64) {
    for table in TABLES {
        let reference = dump(&devices[0], table);
        for dev in &devices[1..] {
            let got = dump(dev, table);
            assert_eq!(
                reference, got,
                "seed {seed}: table `{table}` diverged between {} and {}",
                devices[0].name, dev.name
            );
        }
    }
}

// ── The property ─────────────────────────────────────────────────────────────

fn run_scenario(seed: u64) {
    let tmp = tempfile::tempdir().unwrap();
    let mut rng = Rng::new(seed);

    let mut devices = vec![
        Device::open(tmp.path(), "mac-a"),
        Device::open(tmp.path(), "mac-b"),
        Device::open(tmp.path(), "phone"),
    ];

    // Wall-clock skew: real machines drift. Push each device's HLC forward
    // by a random offset (up to ~28 h) — convergence must not depend on
    // devices agreeing on the time, only on the (hlc, device) total order
    // and on observed-HLC catch-up at apply time.
    for dev in &devices {
        let skew = (rng.below(100_000_000)) as i64;
        dev.exec(
            "UPDATE sync_meta SET v = v + ?1 WHERE k = 'hlc_last'",
            &[SqlValue::Integer(skew)],
        );
    }

    // Random writes with partial syncs woven in.
    let ops = rng.range(40, 90);
    for op_no in 0..ops {
        let d = rng.below(devices.len());
        random_op(&mut rng, &devices[d], op_no);

        if rng.chance(25) {
            let src = rng.below(devices.len());
            let mut dst = rng.below(devices.len());
            while dst == src {
                dst = rng.below(devices.len());
            }
            // Split-borrow via pointers: src and dst are distinct indices.
            let (s, d2) = if src < dst {
                let (a, b) = devices.split_at_mut(dst);
                (&a[src], &mut b[0])
            } else {
                let (a, b) = devices.split_at_mut(src);
                (&b[0], &mut a[dst])
            };
            pull(&mut rng, s, d2);
        }
    }

    // Closing full-mesh rounds: with 3 devices and relaying journals, two
    // rounds propagate everything; a third is cheap insurance.
    for _ in 0..3 {
        for src in 0..devices.len() {
            for dst in 0..devices.len() {
                if src == dst {
                    continue;
                }
                let (s, d2) = if src < dst {
                    let (a, b) = devices.split_at_mut(dst);
                    (&a[src], &mut b[0])
                } else {
                    let (a, b) = devices.split_at_mut(src);
                    (&b[0], &mut a[dst])
                };
                pull(&mut rng, s, d2);
            }
        }
    }

    assert_all_equal(&devices, seed);

    // Echo-safety: replaying anyone's FULL history from cursor 0 into anyone
    // else must be a strict no-op (idempotence after convergence).
    for src in 0..devices.len() {
        for dst in 0..devices.len() {
            if src == dst {
                continue;
            }
            let full = devices[src]
                .store
                .sync_changes_since(0, 1_000_000)
                .unwrap();
            let report: serde_json::Value =
                serde_json::from_str(&devices[dst].store.sync_apply(&full).unwrap()).unwrap();
            for k in ["rows_created", "rows_updated", "rows_deleted", "conflicts"] {
                assert_eq!(
                    report[k].as_i64().unwrap(),
                    0,
                    "seed {seed}: echo {} → {} not a no-op ({k}: {report})",
                    devices[src].name,
                    devices[dst].name
                );
            }
        }
    }
}

#[test]
fn random_interleavings_converge() {
    for seed in 1..=25u64 {
        run_scenario(seed);
    }
}

#[test]
fn delete_heavy_interleavings_converge() {
    // Seeds shifted into another region of the space; the op mix is already
    // delete-capable, larger seeds just explore different interleavings.
    for seed in 1_000..=1_015u64 {
        run_scenario(seed);
    }
}
