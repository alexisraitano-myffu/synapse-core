//! SYN-112 (T3) — the homemade P2P sync engine: change journal + hybrid
//! logical clock + per-column LWW merge + tombstones, protocol versioned.
//!
//! Decision context (recorded on SYN-112): cr-sqlite is dormant and rejects
//! this schema, Automerge is the wrong model for 20 SQL tables, and the
//! owner-lock (a single device runs the Dream Cycle) makes all derived
//! tables single-writer — so the remaining engine is deliberately small.
//!
//! Design:
//! - **Journal (`sync_log`)** — a version map, not an event log: one row per
//!   (table, pk, column) holding the HLC of its last local-or-merged write,
//!   plus a monotonic `seq` (the pull cursor). Values are NOT stored in the
//!   journal; they are read live from the table when producing a changeset.
//!   `col = '-'` is the row tombstone (kept forever, column entries pruned).
//! - **HLC in pure SQL** — a single hybrid integer `max(last + 1, wall_ms)`
//!   bumped in `sync_meta` by the triggers themselves, tie-broken by
//!   device_id. No custom SQL function: any writer (Python through the
//!   `sql.rs` gateway, the core, even a debugging `sqlite3` CLI) journals
//!   correctly with zero registration requirements.
//! - **Changesets carry whole rows** — whenever any column of a row is in
//!   the pulled window, ALL its columns ship (each with its own HLC), so a
//!   fresh peer bootstraps from cursor 0 and a tombstoned row can be
//!   resurrected in full by a later concurrent update (no partial-row
//!   inserts against NOT NULL columns).
//! - **Apply under the `applying` flag** — the triggers' WHEN clause skips
//!   journaling during a merge; the merge upserts the journal itself with
//!   the *incoming* HLCs (new seq → changes relay to third peers, and echo
//!   back to the sender dies on the equal-version check). The whole apply is
//!   one IMMEDIATE transaction; each row merges inside a savepoint so a
//!   local uniqueness conflict (e.g. `idx_resources_url` from two devices
//!   capturing the same URL) skips that row instead of aborting the batch.
//!
//! Out of scope here (phase 3): transport, owner-lock enforcement, peer
//! cursor storage, capture dedup by `provenance_capture_id`.

use std::collections::{HashMap, HashSet};

use rusqlite::types::{Value as SqlV, ValueRef};
use rusqlite::{params, Connection};
use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::embedder::CoreError;

pub const SYNC_PROTOCOL: i64 = 1;

/// Row tombstone marker in `sync_log.col`.
const TOMB: &str = "-";

/// Current wall clock in unix milliseconds, computed inside SQLite.
const NOW_MS_SQL: &str = "CAST((julianday('now') - 2440587.5) * 86400000.0 AS INTEGER)";

/// The replicated tables and their (TEXT) primary-key column.
///
/// Excluded on purpose: `knowledge_graph` (dead, no writers),
/// `atomic_notes_vec` (derived — receivers re-embed; virtual tables cannot
/// carry triggers anyway), `node_positions` + `cluster_labels` (projection
/// caches, recomputed locally), and the engine's own `sync_*` tables.
fn synced_tables() -> &'static [(&'static str, &'static str)] {
    &[
        ("inbox", "id"),
        ("atomic_notes", "id"),
        ("entities", "id"),
        ("facts", "id"),
        ("relations", "id"),
        ("resources", "id"),
        ("pending_facts", "id"),
        ("review_queue", "id"),
        ("intentions", "id"),
        ("validation_events", "id"),
        ("cycle_runs", "id"),
        ("project_entries", "id"),
        ("project_state_versions", "id"),
        ("project_state", "project_id"),
        ("entity_merge_proposals", "id"),
        ("active_entity_types", "type"),
        ("entity_type_proposals", "id"),
        ("project_attach_proposals", "id"),
        ("sync_owner", "id"),
    ]
}

fn pk_col(table: &str) -> Option<&'static str> {
    synced_tables()
        .iter()
        .find(|(t, _)| *t == table)
        .map(|(_, pk)| *pk)
}

fn table_columns(conn: &Connection, table: &str) -> Result<Vec<String>, CoreError> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info(\"{table}\")"))?;
    let cols = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(cols)
}

// ── Install (runs at every Storage::open, idempotent) ───────────────────────

pub(crate) fn install(conn: &Connection) -> Result<(), CoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sync_meta (k TEXT PRIMARY KEY, v);
         CREATE TABLE IF NOT EXISTS sync_log (
             seq INTEGER PRIMARY KEY AUTOINCREMENT,
             tbl TEXT NOT NULL,
             pk  TEXT NOT NULL,
             col TEXT NOT NULL,
             hlc INTEGER NOT NULL,
             dev TEXT NOT NULL,
             UNIQUE(tbl, pk, col) ON CONFLICT REPLACE
         );",
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO sync_meta (k, v) VALUES ('device_id', ?1)",
        params![Uuid::new_v4().simple().to_string()],
    )?;
    conn.execute_batch(
        "INSERT OR IGNORE INTO sync_meta (k, v) VALUES ('hlc_last', 0);
         INSERT OR IGNORE INTO sync_meta (k, v) VALUES ('applying', 0);
         -- The flag lives inside apply's transaction, so a crash rolls it
         -- back; this reset is belt-and-braces for exotic failure paths.
         UPDATE sync_meta SET v = 0 WHERE k = 'applying';",
    )?;

    let hlc = "(SELECT v FROM sync_meta WHERE k = 'hlc_last')";
    let dev = "(SELECT v FROM sync_meta WHERE k = 'device_id')";
    let guard = "(SELECT v FROM sync_meta WHERE k = 'applying') = 0";
    let bump = format!("UPDATE sync_meta SET v = max(v + 1, {NOW_MS_SQL}) WHERE k = 'hlc_last';");

    // Triggers are regenerated on every open so later ALTER TABLE ADD COLUMN
    // migrations start journaling their new column automatically.
    for (table, pk) in synced_tables() {
        let cols = table_columns(conn, table)?;

        let inserts: String = cols
            .iter()
            .map(|c| {
                format!(
                    "INSERT INTO sync_log (tbl, pk, col, hlc, dev) \
                     SELECT '{table}', NEW.\"{pk}\", '{c}', {hlc}, {dev};\n"
                )
            })
            .collect();
        let updates: String = cols
            .iter()
            .map(|c| {
                format!(
                    "INSERT INTO sync_log (tbl, pk, col, hlc, dev) \
                     SELECT '{table}', NEW.\"{pk}\", '{c}', {hlc}, {dev} \
                     WHERE NEW.\"{c}\" IS NOT OLD.\"{c}\";\n"
                )
            })
            .collect();

        conn.execute_batch(&format!(
            "DROP TRIGGER IF EXISTS \"__syn_sync_{table}_ai\";
             CREATE TRIGGER \"__syn_sync_{table}_ai\" AFTER INSERT ON \"{table}\"
             WHEN {guard}
             BEGIN
               {bump}
               DELETE FROM sync_log WHERE tbl = '{table}' AND pk = NEW.\"{pk}\" AND col = '{TOMB}';
               {inserts}
             END;
             DROP TRIGGER IF EXISTS \"__syn_sync_{table}_au\";
             CREATE TRIGGER \"__syn_sync_{table}_au\" AFTER UPDATE ON \"{table}\"
             WHEN {guard}
             BEGIN
               {bump}
               {updates}
             END;
             DROP TRIGGER IF EXISTS \"__syn_sync_{table}_ad\";
             CREATE TRIGGER \"__syn_sync_{table}_ad\" AFTER DELETE ON \"{table}\"
             WHEN {guard}
             BEGIN
               {bump}
               DELETE FROM sync_log WHERE tbl = '{table}' AND pk = OLD.\"{pk}\";
               INSERT INTO sync_log (tbl, pk, col, hlc, dev)
                 SELECT '{table}', OLD.\"{pk}\", '{TOMB}', {hlc}, {dev};
             END;"
        ))?;
    }

    seed_existing_rows(conn)?;
    Ok(())
}

/// One-shot: rows written before the journal existed (the pre-T3 production
/// database) get version entries at the current HLC, so a fresh peer pulling
/// from cursor 0 receives the full history. Guarded by "journal is empty".
fn seed_existing_rows(conn: &Connection) -> Result<(), CoreError> {
    let n: i64 = conn.query_row("SELECT count(*) FROM sync_log", [], |r| r.get(0))?;
    if n > 0 {
        return Ok(());
    }
    conn.execute(
        &format!("UPDATE sync_meta SET v = max(v + 1, {NOW_MS_SQL}) WHERE k = 'hlc_last'"),
        [],
    )?;
    for (table, pk) in synced_tables() {
        for c in table_columns(conn, table)? {
            conn.execute(
                &format!(
                    "INSERT INTO sync_log (tbl, pk, col, hlc, dev)
                     SELECT '{table}', \"{pk}\", '{c}',
                            (SELECT v FROM sync_meta WHERE k = 'hlc_last'),
                            (SELECT v FROM sync_meta WHERE k = 'device_id')
                     FROM \"{table}\""
                ),
                [],
            )?;
        }
    }
    Ok(())
}

pub(crate) fn device_id(conn: &Connection) -> Result<String, CoreError> {
    Ok(conn.query_row(
        "SELECT v FROM sync_meta WHERE k = 'device_id'",
        [],
        |r| r.get(0),
    )?)
}

// ── Wire encoding ────────────────────────────────────────────────────────────

fn to_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

fn sql_to_json(v: ValueRef<'_>) -> Value {
    match v {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => json!(i),
        ValueRef::Real(f) => json!(f),
        ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
        ValueRef::Blob(b) => json!({ "$blob": to_hex(b) }),
    }
}

fn json_to_sql(v: &Value) -> Result<SqlV, CoreError> {
    Ok(match v {
        Value::Null => SqlV::Null,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                SqlV::Integer(i)
            } else {
                SqlV::Real(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => SqlV::Text(s.clone()),
        Value::Object(o) => match o.get("$blob").and_then(Value::as_str).and_then(from_hex) {
            Some(b) => SqlV::Blob(b),
            None => return Err(CoreError::Storage("sync: unknown value object".into())),
        },
        _ => return Err(CoreError::Storage("sync: unsupported JSON value".into())),
    })
}

/// (hlc, device) — total order, device id breaks wall-clock ties.
type Ver = (i64, String);

fn local_versions(conn: &Connection, table: &str, pk: &str) -> Result<HashMap<String, Ver>, CoreError> {
    let mut stmt =
        conn.prepare_cached("SELECT col, hlc, dev FROM sync_log WHERE tbl = ?1 AND pk = ?2")?;
    let rows = stmt
        .query_map(params![table, pk], |r| {
            Ok((r.get::<_, String>(0)?, (r.get::<_, i64>(1)?, r.get::<_, String>(2)?)))
        })?
        .collect::<Result<HashMap<_, _>, _>>()?;
    Ok(rows)
}

// ── Producing a changeset ────────────────────────────────────────────────────

/// Everything journaled after `since`, as protocol-v1 JSON. Whole rows: any
/// row touched in the window ships all its columns with their versions.
/// `next` is the cursor for the following pull; `has_more` signals a full
/// page (call again from `next`).
pub(crate) fn changes_since(
    conn: &Connection,
    since: i64,
    limit: i64,
) -> Result<String, CoreError> {
    let tx = conn.unchecked_transaction()?;
    let device = device_id(&tx)?;

    let mut stmt = tx.prepare(
        "SELECT seq, tbl, pk, col, hlc, dev FROM sync_log
         WHERE seq > ?1 ORDER BY seq ASC LIMIT ?2",
    )?;
    let entries = stmt
        .query_map(params![since, limit], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, String>(5)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    drop(stmt);

    let taken = entries.len() as i64;
    let next = entries.last().map(|e| e.0).unwrap_or(since);

    let mut tombstones = Vec::new();
    let mut row_keys: Vec<(String, String)> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for (_, tbl, pk, col, hlc, dev) in &entries {
        if col == TOMB {
            tombstones.push(json!({ "t": tbl, "pk": pk, "hlc": hlc, "dev": dev }));
        } else if seen.insert((tbl.clone(), pk.clone())) {
            row_keys.push((tbl.clone(), pk.clone()));
        }
    }

    let mut schema_cols: HashMap<String, Vec<String>> = HashMap::new();
    let mut rows = Vec::new();
    for (tbl, pk) in row_keys {
        let Some(pkc) = pk_col(&tbl) else { continue };
        let cols = match schema_cols.get(&tbl) {
            Some(c) => c.clone(),
            None => {
                let c = table_columns(&tx, &tbl)?;
                schema_cols.insert(tbl.clone(), c.clone());
                c
            }
        };
        let vers = local_versions(&tx, &tbl, &pk)?;

        // Only ship columns that exist in the schema AND have a version
        // (a column added by a later migration has no entry until written).
        let shipped: Vec<&String> = cols.iter().filter(|c| vers.contains_key(*c)).collect();
        if shipped.is_empty() {
            continue;
        }
        let select_list = shipped
            .iter()
            .map(|c| format!("\"{c}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let mut vstmt = tx.prepare(&format!(
            "SELECT {select_list} FROM \"{tbl}\" WHERE \"{pkc}\" = ?1"
        ))?;
        let values: Option<Vec<Value>> = vstmt
            .query_row(params![pk], |r| {
                Ok((0..shipped.len())
                    .map(|i| sql_to_json(r.get_ref(i).unwrap()))
                    .collect())
            })
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        // Row gone between journal read and value read (deleted by a
        // concurrent writer): its tombstone has a later seq, next pull wins.
        let Some(values) = values else { continue };

        let mut colmap = Map::new();
        for (c, v) in shipped.iter().zip(values) {
            let (hlc, dev) = &vers[*c];
            colmap.insert(
                (*c).clone(),
                json!({ "hlc": hlc, "dev": dev, "v": v }),
            );
        }
        rows.push(json!({ "t": tbl, "pk": pk, "cols": colmap }));
    }
    tx.commit()?;

    Ok(json!({
        "protocol": SYNC_PROTOCOL,
        "device": device,
        "since": since,
        "next": next,
        "has_more": taken == limit,
        "rows": rows,
        "tombstones": tombstones,
    })
    .to_string())
}

// ── Applying a changeset ─────────────────────────────────────────────────────

pub(crate) fn apply_changes(conn: &Connection, changes_json: &str) -> Result<String, CoreError> {
    let payload: Value = serde_json::from_str(changes_json)
        .map_err(|e| CoreError::Storage(format!("sync: bad changeset JSON: {e}")))?;
    let protocol = payload.get("protocol").and_then(Value::as_i64).unwrap_or(0);
    if protocol != SYNC_PROTOCOL {
        return Err(CoreError::Storage(format!(
            "sync: protocol {protocol} unsupported (expected {SYNC_PROTOCOL})"
        )));
    }
    let empty = Vec::new();
    let tombstones = payload
        .get("tombstones")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let rows = payload.get("rows").and_then(Value::as_array).unwrap_or(&empty);

    let ver_of = |o: &Value| -> Option<Ver> {
        Some((
            o.get("hlc")?.as_i64()?,
            o.get("dev")?.as_str()?.to_string(),
        ))
    };

    // Observe every incoming HLC so our next local write sorts after
    // everything we have seen, whatever the wall-clock skew between devices.
    let observed = tombstones
        .iter()
        .chain(rows.iter().flat_map(|r| {
            r.get("cols")
                .and_then(Value::as_object)
                .map(|c| c.values().collect::<Vec<_>>())
                .unwrap_or_default()
        }))
        .filter_map(|o| o.get("hlc").and_then(Value::as_i64))
        .max()
        .unwrap_or(0);

    let tx = conn.unchecked_transaction()?;
    tx.execute("UPDATE sync_meta SET v = 1 WHERE k = 'applying'", [])?;
    tx.execute(
        "UPDATE sync_meta SET v = max(v, ?1) WHERE k = 'hlc_last'",
        params![observed],
    )?;

    let mut deleted = 0u64;
    let mut created = 0u64;
    let mut updated = 0u64;
    let mut skipped = 0u64;
    let mut conflicts = 0u64;
    let mut notes_changed: Vec<String> = Vec::new();
    let mut note_set: HashSet<String> = HashSet::new();
    let note_touched = |id: &str, set: &mut HashSet<String>, list: &mut Vec<String>| {
        if set.insert(id.to_string()) {
            list.push(id.to_string());
        }
    };

    fn upsert_log(
        conn: &Connection,
        tbl: &str,
        pk: &str,
        col: &str,
        ver: &Ver,
    ) -> Result<(), CoreError> {
        // Plain INSERT: the UNIQUE(tbl, pk, col) ON CONFLICT REPLACE clause
        // makes it an upsert that also refreshes seq (relays downstream).
        conn.execute(
            "INSERT INTO sync_log (tbl, pk, col, hlc, dev) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![tbl, pk, col, ver.0, ver.1],
        )?;
        Ok(())
    }

    for t in tombstones {
        let (Some(tbl), Some(pk), Some(ver)) = (
            t.get("t").and_then(Value::as_str),
            t.get("pk").and_then(Value::as_str),
            ver_of(t),
        ) else {
            skipped += 1;
            continue;
        };
        let Some(pkc) = pk_col(tbl) else {
            skipped += 1;
            continue;
        };
        let locals = local_versions(&tx, tbl, pk)?;
        // Row-level LWW: the delete wins only over a row whose every column
        // (and any previous tombstone) is older.
        if locals.values().any(|v| *v >= ver) {
            skipped += 1;
            continue;
        }
        let n = tx.execute(&format!("DELETE FROM \"{tbl}\" WHERE \"{pkc}\" = ?1"), params![pk])?;
        if tbl == "atomic_notes" {
            tx.execute(
                "DELETE FROM atomic_notes_vec WHERE note_id = ?1",
                params![pk],
            )?;
            note_touched(pk, &mut note_set, &mut notes_changed);
        }
        tx.execute(
            "DELETE FROM sync_log WHERE tbl = ?1 AND pk = ?2",
            params![tbl, pk],
        )?;
        upsert_log(&tx, tbl, pk, TOMB, &ver)?;
        if n > 0 {
            deleted += 1;
        }
    }

    let mut schema_cols: HashMap<String, Vec<String>> = HashMap::new();
    for row in rows {
        let (Some(tbl), Some(pk), Some(cols)) = (
            row.get("t").and_then(Value::as_str),
            row.get("pk").and_then(Value::as_str),
            row.get("cols").and_then(Value::as_object),
        ) else {
            skipped += 1;
            continue;
        };
        let Some(pkc) = pk_col(tbl) else {
            skipped += 1;
            continue;
        };
        let known = match schema_cols.get(tbl) {
            Some(c) => c.clone(),
            None => {
                let c = table_columns(&tx, tbl)?;
                schema_cols.insert(tbl.to_string(), c.clone());
                c
            }
        };
        let locals = local_versions(&tx, tbl, pk)?;
        let tomb = locals.get(TOMB);

        // Parse incoming (col → version, value), schema-known columns only.
        let mut incoming: Vec<(&String, Ver, SqlV)> = Vec::new();
        let mut bad = false;
        for (c, o) in cols {
            if !known.contains(c) {
                continue;
            }
            match (ver_of(o), o.get("v").map(json_to_sql)) {
                (Some(ver), Some(Ok(v))) => incoming.push((c, ver, v)),
                _ => bad = true,
            }
        }
        if bad || incoming.is_empty() {
            skipped += 1;
            continue;
        }

        let winning: Vec<&(&String, Ver, SqlV)> = incoming
            .iter()
            .filter(|(c, ver, _)| {
                tomb.map_or(true, |t| ver > t)
                    && locals.get(c.as_str()).map_or(true, |l| ver > l)
            })
            .collect();
        if winning.is_empty() {
            skipped += 1;
            continue;
        }

        let exists = tx
            .query_row(
                &format!("SELECT 1 FROM \"{tbl}\" WHERE \"{pkc}\" = ?1"),
                params![pk],
                |_| Ok(()),
            )
            .map(|_| true)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(false),
                other => Err(other),
            })?;

        // A local uniqueness conflict (e.g. two devices captured the same
        // resource URL under different uuids) must not poison the batch.
        tx.execute("SAVEPOINT syn_apply_row", [])?;
        let merged: Result<(), CoreError> = (|| {
            if exists {
                let sets: Vec<&(&String, Ver, SqlV)> = winning
                    .iter()
                    .copied()
                    .filter(|(c, _, _)| c.as_str() != pkc)
                    .collect();
                if !sets.is_empty() {
                    let set_list = sets
                        .iter()
                        .enumerate()
                        .map(|(i, (c, _, _))| format!("\"{c}\" = ?{}", i + 2))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let mut p: Vec<&dyn rusqlite::ToSql> = vec![&pk];
                    for (_, _, v) in &sets {
                        p.push(v);
                    }
                    tx.execute(
                        &format!("UPDATE \"{tbl}\" SET {set_list} WHERE \"{pkc}\" = ?1"),
                        &p[..],
                    )?;
                }
                for (c, ver, _) in &winning {
                    upsert_log(&tx, tbl, pk, c, ver)?;
                }
            } else {
                // Fresh (or resurrected) row: insert every incoming column —
                // there is no local value to preserve, and partial inserts
                // would trip NOT NULL constraints.
                let data: Vec<&(&String, Ver, SqlV)> = incoming
                    .iter()
                    .filter(|(c, _, _)| c.as_str() != pkc)
                    .collect();
                let col_list = std::iter::once(format!("\"{pkc}\""))
                    .chain(data.iter().map(|(c, _, _)| format!("\"{c}\"")))
                    .collect::<Vec<_>>()
                    .join(", ");
                let ph = (1..=data.len() + 1)
                    .map(|i| format!("?{i}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut p: Vec<&dyn rusqlite::ToSql> = vec![&pk];
                for (_, _, v) in &data {
                    p.push(v);
                }
                tx.execute(
                    &format!("INSERT INTO \"{tbl}\" ({col_list}) VALUES ({ph})"),
                    &p[..],
                )?;
                tx.execute(
                    "DELETE FROM sync_log WHERE tbl = ?1 AND pk = ?2 AND col = ?3",
                    params![tbl, pk, TOMB],
                )?;
                for (c, ver, _) in &incoming {
                    upsert_log(&tx, tbl, pk, c, ver)?;
                }
            }
            Ok(())
        })();

        match merged {
            Ok(()) => {
                tx.execute("RELEASE syn_apply_row", [])?;
                if exists {
                    updated += 1;
                } else {
                    created += 1;
                }
                if tbl == "atomic_notes" {
                    note_touched(pk, &mut note_set, &mut notes_changed);
                }
            }
            Err(_) => {
                tx.execute_batch("ROLLBACK TO syn_apply_row; RELEASE syn_apply_row;")?;
                conflicts += 1;
            }
        }
    }

    tx.execute("UPDATE sync_meta SET v = 0 WHERE k = 'applying'", [])?;
    tx.commit()?;

    Ok(json!({
        "protocol": SYNC_PROTOCOL,
        "rows_created": created,
        "rows_updated": updated,
        "rows_deleted": deleted,
        "skipped": skipped,
        "conflicts": conflicts,
        "observed_hlc": observed,
        // atomic_notes whose content may have changed: the caller re-embeds
        // (the vec0 index is local and derived, never on the wire).
        "notes_changed": notes_changed,
    })
    .to_string())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::storage::Storage;
    use serde_json::Value;

    fn mem_store(dir: &tempfile::TempDir, name: &str) -> Storage {
        Storage::open(dir.path().join(name).to_str().unwrap()).unwrap()
    }

    fn exec(s: &Storage, sql: &str) {
        s.lock().unwrap().execute(sql, []).unwrap();
    }

    fn query_one(s: &Storage, sql: &str) -> Option<String> {
        let conn = s.lock().unwrap();
        conn.query_row(sql, [], |r| r.get::<_, Option<String>>(0))
            .ok()
            .flatten()
    }

    fn count(s: &Storage, sql: &str) -> i64 {
        let conn = s.lock().unwrap();
        conn.query_row(sql, [], |r| r.get(0)).unwrap()
    }

    /// Pull everything the peer has not seen yet and merge it, both cursors
    /// starting at 0 (tests re-pull from 0 every time: idempotence is part
    /// of the contract, equal versions are skipped).
    fn sync_once(from: &Storage, to: &Storage) -> Value {
        let changes = from.sync_changes_since(0, 100_000).unwrap();
        let report = to.sync_apply(&changes).unwrap();
        serde_json::from_str(&report).unwrap()
    }

    fn table_dump(s: &Storage, table: &str, pk: &str) -> Vec<Vec<String>> {
        let conn = s.lock().unwrap();
        let mut stmt = conn
            .prepare(&format!("SELECT * FROM \"{table}\" ORDER BY \"{pk}\""))
            .unwrap();
        let n = stmt.column_count();
        stmt.query_map([], |r| {
            Ok((0..n)
                .map(|i| format!("{:?}", r.get_ref(i).unwrap()))
                .collect::<Vec<_>>())
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
    }

    #[test]
    fn journal_captures_inserts_updates_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let s = mem_store(&dir, "a.db");

        exec(&s, "INSERT INTO entities (id, canonical_name, type) VALUES ('e1', 'Alexis', 'person')");
        let cols = count(&s, "SELECT count(*) FROM sync_log WHERE tbl='entities' AND pk='e1'");
        // Every column journaled on insert (entities has 20+ columns).
        assert!(cols > 10, "expected all columns journaled, got {cols}");

        let hlc_before: i64 = count(&s, "SELECT hlc FROM sync_log WHERE tbl='entities' AND pk='e1' AND col='summary'");
        exec(&s, "UPDATE entities SET summary = 'le boss' WHERE id = 'e1'");
        let hlc_after: i64 = count(&s, "SELECT hlc FROM sync_log WHERE tbl='entities' AND pk='e1' AND col='summary'");
        assert!(hlc_after > hlc_before, "update must bump the column version");
        // Untouched column keeps its version (only changed columns journal).
        assert_eq!(
            hlc_before,
            count(&s, "SELECT hlc FROM sync_log WHERE tbl='entities' AND pk='e1' AND col='type'")
        );

        exec(&s, "DELETE FROM entities WHERE id = 'e1'");
        assert_eq!(1, count(&s, "SELECT count(*) FROM sync_log WHERE tbl='entities' AND pk='e1'"));
        assert_eq!(
            Some("-".to_string()),
            query_one(&s, "SELECT col FROM sync_log WHERE tbl='entities' AND pk='e1'")
        );
    }

    #[test]
    fn fresh_peer_bootstraps_and_stores_converge() {
        let dir = tempfile::tempdir().unwrap();
        let a = mem_store(&dir, "a.db");
        let b = mem_store(&dir, "b.db");

        exec(&a, "INSERT INTO inbox (id, content, source) VALUES ('c1', 'hello', 'test')");
        exec(&a, "INSERT INTO entities (id, canonical_name, type) VALUES ('e1', 'Alexis', 'person')");
        exec(&a, "INSERT INTO atomic_notes (id, content) VALUES ('n1', 'une note')");

        let report = sync_once(&a, &b);
        // At least the 3 domain rows land as creations (b already holds its
        // own 6 builtin active_entity_types rows — merged, not created).
        assert!(report["rows_created"].as_i64().unwrap() >= 3);
        assert_eq!(1, count(&b, "SELECT count(*) FROM inbox WHERE id='c1'"));
        assert_eq!(
            Some("Alexis".into()),
            query_one(&b, "SELECT canonical_name FROM entities WHERE id='e1'")
        );
        assert_eq!(1, count(&b, "SELECT count(*) FROM atomic_notes WHERE id='n1'"));
        // The received note must be flagged for re-embedding.
        assert!(report["notes_changed"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "n1"));

        // Disjoint column edits merge column-wise.
        exec(&b, "UPDATE entities SET type = 'humain' WHERE id = 'e1'");
        std::thread::sleep(std::time::Duration::from_millis(3));
        exec(&a, "UPDATE entities SET summary = 'fondateur' WHERE id = 'e1'");
        sync_once(&a, &b);
        sync_once(&b, &a);
        for s in [&a, &b] {
            assert_eq!(Some("humain".into()), query_one(s, "SELECT type FROM entities WHERE id='e1'"));
            assert_eq!(Some("fondateur".into()), query_one(s, "SELECT summary FROM entities WHERE id='e1'"));
        }

        // Same-column conflict: the later write wins on both sides.
        exec(&a, "UPDATE entities SET summary = 'A said' WHERE id = 'e1'");
        std::thread::sleep(std::time::Duration::from_millis(3));
        exec(&b, "UPDATE entities SET summary = 'B said later' WHERE id = 'e1'");
        sync_once(&a, &b);
        sync_once(&b, &a);
        for s in [&a, &b] {
            assert_eq!(Some("B said later".into()), query_one(s, "SELECT summary FROM entities WHERE id='e1'"));
        }
        assert_eq!(table_dump(&a, "entities", "id"), table_dump(&b, "entities", "id"));
    }

    #[test]
    fn delete_vs_update_last_writer_wins() {
        let dir = tempfile::tempdir().unwrap();
        let a = mem_store(&dir, "a.db");
        let b = mem_store(&dir, "b.db");

        exec(&a, "INSERT INTO entities (id, canonical_name) VALUES ('e1', 'X')");
        sync_once(&a, &b);

        // Concurrent: B deletes, then (later clock) A edits → row survives
        // everywhere, fully resurrected on B.
        exec(&b, "DELETE FROM entities WHERE id = 'e1'");
        std::thread::sleep(std::time::Duration::from_millis(3));
        exec(&a, "UPDATE entities SET summary = 'survived' WHERE id = 'e1'");
        sync_once(&b, &a);
        sync_once(&a, &b);
        for s in [&a, &b] {
            assert_eq!(Some("survived".into()), query_one(s, "SELECT summary FROM entities WHERE id='e1'"));
            assert_eq!(Some("X".into()), query_one(s, "SELECT canonical_name FROM entities WHERE id='e1'"));
        }

        // Concurrent the other way: A edits, then (later) B deletes → gone
        // everywhere.
        exec(&a, "UPDATE entities SET summary = 'doomed' WHERE id = 'e1'");
        std::thread::sleep(std::time::Duration::from_millis(3));
        exec(&b, "DELETE FROM entities WHERE id = 'e1'");
        sync_once(&a, &b);
        sync_once(&b, &a);
        for s in [&a, &b] {
            assert_eq!(0, count(s, "SELECT count(*) FROM entities WHERE id='e1'"));
        }
    }

    #[test]
    fn apply_is_idempotent_and_echo_safe() {
        let dir = tempfile::tempdir().unwrap();
        let a = mem_store(&dir, "a.db");
        let b = mem_store(&dir, "b.db");

        exec(&a, "INSERT INTO inbox (id, content) VALUES ('c1', 'hello')");
        // First round-trip also converges the independently-seeded builtin
        // active_entity_types rows (each side seeded its own versions).
        sync_once(&a, &b);
        sync_once(&b, &a);

        let changes = a.sync_changes_since(0, 100_000).unwrap();
        let second: Value = serde_json::from_str(&b.sync_apply(&changes).unwrap()).unwrap();
        assert_eq!(0, second["rows_created"].as_i64().unwrap());
        assert_eq!(0, second["rows_updated"].as_i64().unwrap());

        // Echo: A applying what B now relays must change nothing on A.
        let echo = b.sync_changes_since(0, 100_000).unwrap();
        let report: Value = serde_json::from_str(&a.sync_apply(&echo).unwrap()).unwrap();
        assert_eq!(0, report["rows_created"].as_i64().unwrap());
        assert_eq!(0, report["rows_updated"].as_i64().unwrap());
        assert_eq!(0, report["rows_deleted"].as_i64().unwrap());
    }

    #[test]
    fn hlc_stays_ahead_of_observed_remote() {
        let dir = tempfile::tempdir().unwrap();
        let a = mem_store(&dir, "a.db");
        let b = mem_store(&dir, "b.db");

        // Fake a far-future clock on A (device skew).
        exec(&a, "UPDATE sync_meta SET v = v + 9999999 WHERE k = 'hlc_last'");
        exec(&a, "INSERT INTO inbox (id, content) VALUES ('c1', 'from the future')");
        sync_once(&a, &b);

        // B's next local write must sort after A's future HLC.
        exec(&b, "INSERT INTO inbox (id, content) VALUES ('c2', 'local after')");
        let h_a = count(&b, "SELECT hlc FROM sync_log WHERE tbl='inbox' AND pk='c1' AND col='content'");
        let h_b = count(&b, "SELECT hlc FROM sync_log WHERE tbl='inbox' AND pk='c2' AND col='content'");
        assert!(h_b > h_a, "local write ({h_b}) must sort after observed remote ({h_a})");
    }

    #[test]
    fn seeding_journals_pre_engine_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seed.db");
        {
            let s = Storage::open(path.to_str().unwrap()).unwrap();
            exec(&s, "INSERT INTO inbox (id, content) VALUES ('old1', 'ancienne')");
            // Simulate a pre-T3 database: wipe the journal.
            exec(&s, "DELETE FROM sync_log");
        }
        let s = Storage::open(path.to_str().unwrap()).unwrap();
        assert!(count(&s, "SELECT count(*) FROM sync_log WHERE tbl='inbox' AND pk='old1'") > 3);

        // And a fresh peer receives the seeded history.
        let b = mem_store(&dir, "peer.db");
        sync_once(&s, &b);
        assert_eq!(1, count(&b, "SELECT count(*) FROM inbox WHERE id='old1'"));
    }

    #[test]
    fn pagination_cursor_walks_the_journal() {
        let dir = tempfile::tempdir().unwrap();
        let a = mem_store(&dir, "a.db");
        for i in 0..5 {
            exec(&a, &format!("INSERT INTO inbox (id, content) VALUES ('c{i}', 'x')"));
        }
        let mut cursor = 0i64;
        let mut pulled = 0;
        loop {
            let page: Value =
                serde_json::from_str(&a.sync_changes_since(cursor, 7).unwrap()).unwrap();
            pulled += page["rows"].as_array().unwrap().len();
            cursor = page["next"].as_i64().unwrap();
            if !page["has_more"].as_bool().unwrap() {
                break;
            }
        }
        // 5 captures + 6 builtin active_entity_types rows, deduped by row.
        assert!(pulled >= 11, "expected all rows across pages, got {pulled}");
    }
}
