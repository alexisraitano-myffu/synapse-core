//! Faithful port of the Python backend's idempotent `init_db()`
//! (`db/__init__.py`). Same discipline: `CREATE ... IF NOT EXISTS`, seed rows
//! with `INSERT OR IGNORE`, and best-effort `ALTER TABLE ... ADD COLUMN`
//! migrations, so the core opens an existing production database in place —
//! no migration step, no version table.
//!
//! Any schema change lands HERE first; the Python `init_db()` is now a thin
//! call into this code and must never grow DDL of its own again.

use rusqlite::Connection;

use crate::embedder::EMBEDDING_DIM;

pub(crate) fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    // note_id TEXT: vec0 supports declared text primary keys (verified on
    // sqlite-vec 0.1.9) — the vector key IS the atomic_notes uuid.
    let vec_table = format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS atomic_notes_vec \
         USING vec0(note_id TEXT PRIMARY KEY, embedding float[{EMBEDDING_DIM}])"
    );

    let creates: &[&str] = &[
        // SYN-112: uuid TEXT pks everywhere — auto-increment ids cannot give
        // rows a cross-device identity. inbox.id doubles as the capture's
        // client uuid when one is provided (idempotency moves onto the pk).
        "CREATE TABLE IF NOT EXISTS inbox (
            id           TEXT NOT NULL PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
            content      TEXT NOT NULL,
            source       TEXT NOT NULL DEFAULT 'manual',
            created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS atomic_notes (
            id         TEXT NOT NULL PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
            title      TEXT,
            content    TEXT NOT NULL,
            source_ids TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )",
        // vec_table (parallel vector index keyed by note uuid) is executed
        // separately below because of the dim interpolation.
        "CREATE TABLE IF NOT EXISTS knowledge_graph (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_a   TEXT NOT NULL,
            relation   TEXT NOT NULL,
            entity_b   TEXT NOT NULL,
            context    TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )",
        // ── Phase A+ — Entity graph ─────────────────────────────────────────
        "CREATE TABLE IF NOT EXISTS entities (
            id                TEXT PRIMARY KEY,
            type              TEXT,
            canonical_name    TEXT NOT NULL,
            aliases           TEXT DEFAULT '[]',
            attributes        TEXT DEFAULT '{}',
            mention_count     INTEGER DEFAULT 1,
            last_mentioned    DATE,
            persistence_value INTEGER DEFAULT 3,
            summary           TEXT,
            embedding         BLOB,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS facts (
            id                TEXT PRIMARY KEY,
            entity_id         TEXT REFERENCES entities(id),
            predicate         TEXT NOT NULL,
            value             TEXT NOT NULL,
            confidence        REAL DEFAULT 0.5,
            source_inbox_id   TEXT,
            persistence_value INTEGER DEFAULT 3,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_confirmed    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS relations (
            id          TEXT PRIMARY KEY,
            entity_from TEXT REFERENCES entities(id),
            predicate   TEXT NOT NULL,
            entity_to   TEXT REFERENCES entities(id),
            confidence  REAL DEFAULT 0.5,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS resources (
            id                TEXT PRIMARY KEY,
            type              TEXT,
            source            TEXT,
            title             TEXT,
            summary           TEXT,
            tags              TEXT DEFAULT '[]',
            entities_mentioned TEXT DEFAULT '[]',
            embedding         BLOB,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS pending_facts (
            id                  TEXT PRIMARY KEY,
            fact_data           TEXT NOT NULL,
            validation_strategy TEXT DEFAULT 'passive',
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS review_queue (
            id               TEXT PRIMARY KEY,
            fact_data        TEXT NOT NULL,
            suggested_entity TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS intentions (
            id         TEXT PRIMARY KEY,
            content    TEXT NOT NULL,
            ttl_hours  INTEGER DEFAULT 48,
            resolved   BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        // Append-only log of user validations (survives a rebuild, replicates).
        "CREATE TABLE IF NOT EXISTS validation_events (
            id               TEXT PRIMARY KEY,
            fact_id          TEXT,
            entity_canonical TEXT,
            predicate        TEXT,
            value            TEXT,
            confirmed        INTEGER NOT NULL,
            correction       TEXT,
            device_id        TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        // One row per Dream Cycle run (stats for the app's "last/next cycle").
        "CREATE TABLE IF NOT EXISTS cycle_runs (
            id                TEXT PRIMARY KEY,
            started_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at       TIMESTAMP,
            notes_processed   INTEGER DEFAULT 0,
            entities_total    INTEGER DEFAULT 0,
            pending_total     INTEGER DEFAULT 0,
            status            TEXT DEFAULT 'running',
            trigger           TEXT DEFAULT 'manual',
            error             TEXT
        )",
        // ── SYN-41 — Projects as aggregate entities ─────────────────────────
        "CREATE TABLE IF NOT EXISTS project_entries (
            id                TEXT PRIMARY KEY,
            project_id        TEXT NOT NULL REFERENCES entities(id),
            capture_id        TEXT NOT NULL REFERENCES inbox(id),
            content           TEXT NOT NULL,
            kind              TEXT NOT NULL DEFAULT 'note',
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS project_state_versions (
            id                TEXT PRIMARY KEY,
            project_id        TEXT NOT NULL REFERENCES entities(id),
            summary_md        TEXT NOT NULL,
            entry_count       INTEGER NOT NULL,
            trigger           TEXT NOT NULL,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS project_state (
            project_id          TEXT PRIMARY KEY REFERENCES entities(id),
            current_version_id  TEXT REFERENCES project_state_versions(id),
            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            entry_count_at_sync INTEGER DEFAULT 0
        )",
        // ── SYN-39 — Entity merge proposals (absorbed row stays, soft link) ─
        "CREATE TABLE IF NOT EXISTS entity_merge_proposals (
            id                    TEXT PRIMARY KEY,
            candidate_entity_id   TEXT NOT NULL REFERENCES entities(id),
            existing_entity_id    TEXT NOT NULL REFERENCES entities(id),
            similarity_score      REAL NOT NULL,
            similarity_reason     TEXT,
            evidence_capture_id   TEXT REFERENCES inbox(id),
            status                TEXT NOT NULL DEFAULT 'pending',
            created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at           TIMESTAMP,
            resolved_canonical_id TEXT REFERENCES entities(id)
        )",
        // ── SYN-58 — Live entity-type vocabulary + proposals ────────────────
        "CREATE TABLE IF NOT EXISTS active_entity_types (
            type        TEXT PRIMARY KEY,
            source      TEXT NOT NULL,
            added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        "CREATE TABLE IF NOT EXISTS entity_type_proposals (
            id                  TEXT PRIMARY KEY,
            proposed_type       TEXT NOT NULL,
            reason              TEXT,
            evidence_capture_id TEXT REFERENCES inbox(id),
            candidate_entity_id TEXT REFERENCES entities(id),
            status              TEXT NOT NULL DEFAULT 'pending',
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at         TIMESTAMP
        )",
        // Soft « rattacher à un projet existant ? » proposals (« À valider »).
        "CREATE TABLE IF NOT EXISTS project_attach_proposals (
            id               TEXT PRIMARY KEY,
            capture_id       TEXT NOT NULL REFERENCES inbox(id),
            note_id          TEXT REFERENCES atomic_notes(id),
            project_id       TEXT NOT NULL REFERENCES entities(id),
            content          TEXT NOT NULL,
            similarity_score REAL NOT NULL,
            status           TEXT NOT NULL DEFAULT 'pending',
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at      TIMESTAMP
        )",
        // SYN-69 — persisted map positions (projection cache, never authoritative).
        "CREATE TABLE IF NOT EXISTS node_positions (
            node_id    TEXT PRIMARY KEY,
            x          REAL NOT NULL,
            y          REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        // SYN-70 — cached cluster labels, keyed by defining-entities signature.
        "CREATE TABLE IF NOT EXISTS cluster_labels (
            signature  TEXT PRIMARY KEY,
            label      TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
    ];

    for stmt in creates {
        conn.execute(stmt, [])?;
    }
    conn.execute(&vec_table, [])?;

    // SYN-58: seed the live vocabulary with the six built-in types.
    for builtin in ["person", "place", "project", "concept", "organization", "animal"] {
        conn.execute(
            "INSERT OR IGNORE INTO active_entity_types (type, source) VALUES (?1, 'builtin')",
            [builtin],
        )?;
    }

    // Best-effort column migrations, in the same order Python applied them.
    let alters: &[&str] = &[
        "ALTER TABLE inbox ADD COLUMN processed_at TIMESTAMP",
        // Episodic-memory columns (decay = Ebbinghaus, SYN-19).
        "ALTER TABLE atomic_notes ADD COLUMN summary TEXT",
        "ALTER TABLE atomic_notes ADD COLUMN entities_mentioned TEXT DEFAULT '[]'",
        "ALTER TABLE atomic_notes ADD COLUMN memory_strength REAL DEFAULT 1.0",
        "ALTER TABLE atomic_notes ADD COLUMN last_reactivated_at TIMESTAMP",
        // SYN-21 — real resource pipeline (fetch + summary).
        "ALTER TABLE resources ADD COLUMN url        TEXT",
        "ALTER TABLE resources ADD COLUMN content    TEXT",
        "ALTER TABLE resources ADD COLUMN fetched_at TIMESTAMP",
        // Sync columns on inbox (client_id enables idempotent capture).
        "ALTER TABLE inbox ADD COLUMN client_id TEXT",
        "ALTER TABLE inbox ADD COLUMN device_id TEXT",
        "ALTER TABLE inbox ADD COLUMN captured_at TIMESTAMP",
        "ALTER TABLE inbox ADD COLUMN status TEXT DEFAULT 'queued'",
        // SYN-41 — provenance back-link to the immutable inbox row.
        "ALTER TABLE entities     ADD COLUMN provenance_capture_id TEXT REFERENCES inbox(id)",
        "ALTER TABLE facts        ADD COLUMN provenance_capture_id TEXT REFERENCES inbox(id)",
        "ALTER TABLE atomic_notes ADD COLUMN provenance_capture_id TEXT REFERENCES inbox(id)",
        "ALTER TABLE relations    ADD COLUMN provenance_capture_id TEXT REFERENCES inbox(id)",
        // SYN-44 — append vs from-scratch rebuild on project_state_versions.
        "ALTER TABLE project_state_versions ADD COLUMN kind TEXT NOT NULL DEFAULT 'append'",
        // SYN-39 — soft-link a merged entity to its absorber.
        "ALTER TABLE entities ADD COLUMN merged_into_id TEXT REFERENCES entities(id)",
        "ALTER TABLE entities ADD COLUMN merged_at TIMESTAMP",
        // SYN-58 — entity lifecycle status (active | pending | archived).
        "ALTER TABLE entities ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
        // SYN-37 + SYN-59 — fact/entity lifecycle (archived / obsoleted).
        "ALTER TABLE facts    ADD COLUMN archived_at  TIMESTAMP",
        "ALTER TABLE facts    ADD COLUMN obsoleted_at TIMESTAMP",
        "ALTER TABLE facts    ADD COLUMN obsoleted_by TEXT REFERENCES facts(id)",
        "ALTER TABLE entities ADD COLUMN archived_at  TIMESTAMP",
        // SYN-68 — entity memory_strength for the living map.
        "ALTER TABLE entities ADD COLUMN memory_strength REAL DEFAULT 1.0",
        // SYN-77 — keep the failure reason on the inbox row.
        "ALTER TABLE inbox ADD COLUMN error TEXT",
        // SYN-88 — fact category (identity | dates | work | ...).
        "ALTER TABLE facts ADD COLUMN category TEXT",
        // SYN-89 — re-résumé flag, set whenever a fact of the entity changes.
        "ALTER TABLE entities ADD COLUMN summary_stale INTEGER NOT NULL DEFAULT 0",
        // SYN-85 — note kinds (note | task | event) + user archive gesture.
        "ALTER TABLE atomic_notes ADD COLUMN kind TEXT NOT NULL DEFAULT 'note'",
        "ALTER TABLE atomic_notes ADD COLUMN event_date TIMESTAMP",
        "ALTER TABLE atomic_notes ADD COLUMN event_recurring INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE atomic_notes ADD COLUMN archived_at TIMESTAMP",
        // Low-confidence task/event notes land in « À valider ».
        "ALTER TABLE atomic_notes ADD COLUMN review_status TEXT NOT NULL DEFAULT 'confirmed'",
        // Relations join the same confidence gate.
        "ALTER TABLE relations ADD COLUMN review_status TEXT NOT NULL DEFAULT 'confirmed'",
    ];

    for ddl in alters {
        match conn.execute(ddl, []) {
            Ok(_) => {}
            // Same tolerance as Python's `except apsw.SQLError`: the column
            // already exists. Anything else (locked db, disk error) propagates.
            Err(e) if e.to_string().contains("duplicate column name") => {}
            Err(e) => return Err(e),
        }
    }

    let indexes: &[&str] = &[
        // One resource per URL (idempotent re-capture of the same link).
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_resources_url \
         ON resources(url) WHERE url IS NOT NULL",
        // Idempotency: at most one inbox row per client-generated capture id.
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_inbox_client_id \
         ON inbox(client_id) WHERE client_id IS NOT NULL",
        // Timeline access: a project's entries in reverse-chrono order.
        "CREATE INDEX IF NOT EXISTS idx_project_entries_project \
         ON project_entries(project_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_merge_proposals_status \
         ON entity_merge_proposals(status, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_type_proposals_status \
         ON entity_type_proposals(status, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_project_attach_proposals_status \
         ON project_attach_proposals(status, created_at DESC)",
    ];

    for stmt in indexes {
        conn.execute(stmt, [])?;
    }

    Ok(())
}
