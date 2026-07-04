//! SYN-112 — one-shot migration of the legacy INTEGER AUTOINCREMENT ids
//! (`inbox`, `atomic_notes`) to TEXT uuids, the prerequisite of P2P sync:
//! auto-increment ids cannot give rows a cross-device identity.
//!
//! Runs from `Storage::open` right after `init_schema`, inside one
//! transaction. Detection is the declared type of `inbox.id`: a fresh
//! database (new DDL) is already TEXT and the migration is a no-op.
//!
//! Mapping rules:
//! - `inbox.id` = the row's `client_id` when present (the app's idempotency
//!   uuid is promoted to primary key), else a fresh uuid4.
//! - `atomic_notes.id` = fresh uuid4; the vec0 index is rebuilt keyed by the
//!   note uuid (`note_id TEXT PRIMARY KEY`).
//! - Every referencing column is rewritten through the mapping; a dangling
//!   reference keeps its old value rather than being nulled (COALESCE).
//! - `knowledge_graph` is dead (no writers) and deliberately left untouched.

use std::collections::HashMap;

use rusqlite::{params, Connection};
use uuid::Uuid;

use crate::embedder::{CoreError, EMBEDDING_DIM};

pub(crate) fn migrate_integer_ids(conn: &Connection) -> Result<(), CoreError> {
    let id_type: String = conn.query_row(
        "SELECT type FROM pragma_table_info('inbox') WHERE name = 'id'",
        [],
        |r| r.get(0),
    )?;
    if !id_type.eq_ignore_ascii_case("INTEGER") {
        return Ok(());
    }

    conn.execute_batch("BEGIN IMMEDIATE")?;
    match run(conn) {
        Ok(()) => {
            conn.execute_batch("COMMIT")?;
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e.into())
        }
    }
}

fn run(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TEMP TABLE map_inbox (old INTEGER PRIMARY KEY, new TEXT NOT NULL);
         CREATE TEMP TABLE map_notes (old INTEGER PRIMARY KEY, new TEXT NOT NULL);",
    )?;

    // ── Mappings ────────────────────────────────────────────────────────────
    {
        let mut read = conn.prepare("SELECT id, client_id FROM inbox")?;
        let mut write = conn.prepare("INSERT INTO map_inbox (old, new) VALUES (?1, ?2)")?;
        let rows: Vec<(i64, Option<String>)> = read
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        for (old, client_id) in rows {
            let new = match client_id {
                Some(c) if !c.is_empty() => c,
                _ => Uuid::new_v4().to_string(),
            };
            write.execute(params![old, new])?;
        }
    }
    {
        // uuid4 for every note: no natural uuid twin exists for notes.
        let mut read = conn.prepare("SELECT id FROM atomic_notes")?;
        let mut write = conn.prepare("INSERT INTO map_notes (old, new) VALUES (?1, ?2)")?;
        let olds: Vec<i64> =
            read.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
        for old in olds {
            write.execute(params![old, Uuid::new_v4().to_string()])?;
        }
    }

    // ── inbox: rebuild with TEXT pk ────────────────────────────────────────
    conn.execute_batch(
        "CREATE TABLE inbox_new (
            id           TEXT NOT NULL PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
            content      TEXT NOT NULL,
            source       TEXT NOT NULL DEFAULT 'manual',
            created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            client_id    TEXT,
            device_id    TEXT,
            captured_at  TIMESTAMP,
            status       TEXT DEFAULT 'queued',
            error        TEXT
         );
         INSERT INTO inbox_new (id, content, source, created_at, processed_at,
                                client_id, device_id, captured_at, status, error)
         SELECT m.new, i.content, i.source, i.created_at, i.processed_at,
                i.client_id, i.device_id, i.captured_at, i.status, i.error
         FROM inbox i JOIN map_inbox m ON m.old = i.id;
         DROP TABLE inbox;
         ALTER TABLE inbox_new RENAME TO inbox;
         CREATE UNIQUE INDEX IF NOT EXISTS idx_inbox_client_id
             ON inbox(client_id) WHERE client_id IS NOT NULL;",
    )?;

    // ── atomic_notes: rebuild with TEXT pk (provenance mapped inline) ──────
    conn.execute_batch(
        "CREATE TABLE atomic_notes_new (
            id                    TEXT NOT NULL PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
            title                 TEXT,
            content               TEXT NOT NULL,
            source_ids            TEXT,
            created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            summary               TEXT,
            entities_mentioned    TEXT DEFAULT '[]',
            memory_strength       REAL DEFAULT 1.0,
            last_reactivated_at   TIMESTAMP,
            provenance_capture_id TEXT REFERENCES inbox(id),
            kind                  TEXT NOT NULL DEFAULT 'note',
            event_date            TIMESTAMP,
            event_recurring       INTEGER NOT NULL DEFAULT 0,
            archived_at           TIMESTAMP,
            review_status         TEXT NOT NULL DEFAULT 'confirmed'
         );
         INSERT INTO atomic_notes_new
         SELECT m.new, n.title, n.content, n.source_ids, n.created_at,
                n.updated_at, n.summary, n.entities_mentioned,
                n.memory_strength, n.last_reactivated_at,
                COALESCE((SELECT mi.new FROM map_inbox mi
                          WHERE mi.old = n.provenance_capture_id),
                         n.provenance_capture_id),
                n.kind, n.event_date, n.event_recurring, n.archived_at,
                n.review_status
         FROM atomic_notes n JOIN map_notes m ON m.old = n.id;
         DROP TABLE atomic_notes;
         ALTER TABLE atomic_notes_new RENAME TO atomic_notes;",
    )?;

    // ── vec index: buffer, drop, recreate keyed by note uuid ───────────────
    let vectors: Vec<(i64, Vec<u8>)> = {
        let mut stmt = conn.prepare("SELECT rowid, embedding FROM atomic_notes_vec")?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        rows
    };
    conn.execute_batch("DROP TABLE atomic_notes_vec")?;
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE atomic_notes_vec \
         USING vec0(note_id TEXT PRIMARY KEY, embedding float[{EMBEDDING_DIM}])"
    ))?;
    {
        let mut lookup = conn.prepare("SELECT new FROM map_notes WHERE old = ?1")?;
        let mut insert =
            conn.prepare("INSERT INTO atomic_notes_vec (note_id, embedding) VALUES (?1, ?2)")?;
        for (old_rowid, blob) in vectors {
            // Orphan vectors (note deleted while its vector survived) are
            // dropped here rather than resurrected under a fake id.
            let new: Option<String> =
                lookup.query_row([old_rowid], |r| r.get(0)).map(Some).or_else(|e| {
                    if matches!(e, rusqlite::Error::QueryReturnedNoRows) { Ok(None) } else { Err(e) }
                })?;
            if let Some(new) = new {
                insert.execute(params![new, blob])?;
            }
        }
    }

    // ── Referencing columns (dangling refs keep their old value) ───────────
    conn.execute_batch(
        "UPDATE entities SET provenance_capture_id = COALESCE(
             (SELECT new FROM map_inbox WHERE old = entities.provenance_capture_id),
             provenance_capture_id)
         WHERE provenance_capture_id IS NOT NULL;
         UPDATE facts SET provenance_capture_id = COALESCE(
             (SELECT new FROM map_inbox WHERE old = facts.provenance_capture_id),
             provenance_capture_id)
         WHERE provenance_capture_id IS NOT NULL;
         UPDATE relations SET provenance_capture_id = COALESCE(
             (SELECT new FROM map_inbox WHERE old = relations.provenance_capture_id),
             provenance_capture_id)
         WHERE provenance_capture_id IS NOT NULL;
         UPDATE project_entries SET capture_id = COALESCE(
             (SELECT new FROM map_inbox WHERE old = project_entries.capture_id),
             capture_id);
         UPDATE entity_merge_proposals SET evidence_capture_id = COALESCE(
             (SELECT new FROM map_inbox WHERE old = entity_merge_proposals.evidence_capture_id),
             evidence_capture_id)
         WHERE evidence_capture_id IS NOT NULL;
         UPDATE entity_type_proposals SET evidence_capture_id = COALESCE(
             (SELECT new FROM map_inbox WHERE old = entity_type_proposals.evidence_capture_id),
             evidence_capture_id)
         WHERE evidence_capture_id IS NOT NULL;
         UPDATE project_attach_proposals SET capture_id = COALESCE(
             (SELECT new FROM map_inbox WHERE old = project_attach_proposals.capture_id),
             capture_id);
         UPDATE project_attach_proposals SET note_id = COALESCE(
             (SELECT new FROM map_notes WHERE old = project_attach_proposals.note_id),
             note_id)
         WHERE note_id IS NOT NULL;
         UPDATE facts SET source_inbox_id = COALESCE(
             (SELECT new FROM map_inbox WHERE CAST(old AS TEXT) = facts.source_inbox_id),
             source_inbox_id)
         WHERE source_inbox_id IS NOT NULL;",
    )?;

    // ── atomic_notes.source_ids: JSON list of capture ids → uuids ──────────
    let map_inbox: HashMap<i64, String> = {
        let mut stmt = conn.prepare("SELECT old, new FROM map_inbox")?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        rows
    };
    let notes: Vec<(String, String)> = {
        let mut stmt = conn.prepare(
            "SELECT id, source_ids FROM atomic_notes
             WHERE source_ids IS NOT NULL AND source_ids LIKE '[%'",
        )?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        rows
    };
    let mut update = conn.prepare("UPDATE atomic_notes SET source_ids = ?1 WHERE id = ?2")?;
    for (note_id, raw) in notes {
        let Ok(serde_json::Value::Array(items)) = serde_json::from_str(&raw) else {
            continue;
        };
        let mapped: Vec<String> = items
            .iter()
            .map(|v| match v.as_i64().and_then(|i| map_inbox.get(&i)) {
                Some(new) => format!("\"{new}\""),
                // Already-string ids and sentinels pass through unchanged.
                None => v.to_string(),
            })
            .collect();
        // Python json.dumps look-alike (", " separator), like py_dumps.
        update.execute(params![format!("[{}]", mapped.join(", ")), note_id])?;
    }

    // Autoincrement bookkeeping of the dropped pks (sqlite_sequence only
    // exists once an AUTOINCREMENT table has inserted at least one row).
    let has_sequence: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE name = 'sqlite_sequence'",
        [],
        |r| r.get(0),
    )?;
    if has_sequence > 0 {
        conn.execute_batch(
            "DELETE FROM sqlite_sequence WHERE name IN ('inbox', 'atomic_notes');",
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use rusqlite::{params, Connection};

    use crate::storage::register_vec_extension;
    use crate::Storage;

    /// Legacy prod shape (integer pks) → open → everything remapped to uuids.
    #[test]
    fn migrates_legacy_integer_ids_to_uuids() {
        register_vec_extension();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE inbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    client_id TEXT
                 );
                 CREATE TABLE atomic_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    content TEXT NOT NULL,
                    source_ids TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    provenance_capture_id INTEGER
                 );
                 CREATE TABLE project_entries (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    capture_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'note',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 );
                 CREATE VIRTUAL TABLE atomic_notes_vec USING vec0(embedding float[384]);
                 INSERT INTO inbox (content, client_id) VALUES ('capture app', 'cli-uuid-1');
                 INSERT INTO inbox (content) VALUES ('capture sans client_id');
                 INSERT INTO atomic_notes (content, source_ids, provenance_capture_id)
                     VALUES ('une note', '[1, 2]', 1);
                 INSERT INTO project_entries (id, project_id, capture_id, content)
                     VALUES ('pe1', 'proj1', 1, 'entrée');",
            )
            .unwrap();
            let blob: Vec<u8> = (0..384u32)
                .flat_map(|i| (if i == 0 { 1.0f32 } else { 0.0 }).to_le_bytes())
                .collect();
            conn.execute(
                "INSERT INTO atomic_notes_vec (rowid, embedding) VALUES (1, ?1)",
                params![blob],
            )
            .unwrap();
        }

        let storage = Storage::open(path.to_str().unwrap()).unwrap();
        let conn = storage.lock().unwrap();

        // client_id promoted to pk; the bare row got a fresh uuid4.
        let id1: String = conn
            .query_row("SELECT id FROM inbox WHERE content='capture app'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(id1, "cli-uuid-1");
        let id2: String = conn
            .query_row(
                "SELECT id FROM inbox WHERE content='capture sans client_id'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(id2.len(), 36);

        // Note uuid + mapped provenance + rewritten source_ids JSON.
        let (nid, prov, src): (String, String, String) = conn
            .query_row(
                "SELECT id, provenance_capture_id, source_ids FROM atomic_notes",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(nid.len(), 36);
        assert_eq!(prov, "cli-uuid-1");
        assert_eq!(src, format!("[\"cli-uuid-1\", \"{id2}\"]"));

        // Referencing table rewritten through the mapping.
        let cap: String = conn
            .query_row("SELECT capture_id FROM project_entries", [], |r| r.get(0))
            .unwrap();
        assert_eq!(cap, "cli-uuid-1");

        // Vector re-keyed by the note uuid.
        let vec_key: String = conn
            .query_row("SELECT note_id FROM atomic_notes_vec", [], |r| r.get(0))
            .unwrap();
        assert_eq!(vec_key, nid);
        drop(conn);

        // Idempotence: a second open is a no-op.
        drop(storage);
        let storage = Storage::open(path.to_str().unwrap()).unwrap();
        let conn = storage.lock().unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM inbox", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }
}
