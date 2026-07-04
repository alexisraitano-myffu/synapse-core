//! Storage substrate (SYN-110 / T1): the core owns the SQLite schema and all
//! vector reads/writes. Hosts keep querying non-vector columns through their
//! own SQL connections; everything vectorial goes through [`Storage`].
//!
//! Parity contract with the Python implementation it replaces
//! (`entity_search.py` + the inline vec0 SQL):
//! - on-disk format unchanged: L2-normalized little-endian float32 blobs, in
//!   the `atomic_notes_vec` vec0 table (note_id = note uuid, SYN-112) and in the
//!   `entities.embedding` / `resources.embedding` BLOB columns;
//! - notes: native sqlite-vec KNN (`MATCH ? AND k = ?`, L2 distance);
//! - entities/resources (UUID string PKs, no int rowid for vec0): exact linear
//!   scan, distance accumulated in f64 like Python floats, score
//!   `round(max(0, 1 - distance/2), 4)`, same SQL candidate filters.

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, Once};
use std::time::Duration;

use rusqlite::{params, Connection};

use crate::embedder::{CoreError, EMBEDDING_DIM};
use crate::schema;

const EMBEDDING_BYTES: usize = EMBEDDING_DIM * 4;

/// A note KNN hit; `distance` is sqlite-vec's L2 on unit vectors ([0, 2]).
#[derive(Debug, Clone, PartialEq)]
pub struct NoteHit {
    pub note_id: String,
    pub distance: f64,
}

/// An entity similarity hit; `score` = `1 - distance/2`, rounded to 4 decimals.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityHit {
    pub id: String,
    pub canonical_name: String,
    pub entity_type: Option<String>,
    pub summary: String,
    pub score: f64,
}

/// A resource similarity hit; `title` falls back to the URL like Python did.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceHit {
    pub id: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub summary: String,
    pub score: f64,
}

impl From<rusqlite::Error> for CoreError {
    fn from(e: rusqlite::Error) -> Self {
        CoreError::Storage(e.to_string())
    }
}

/// Register sqlite-vec as an auto-extension so every connection (including
/// the ones rusqlite opens internally) gets the vec0 module. Process-wide,
/// must happen before the first `Connection::open`.
pub(crate) fn register_vec_extension() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| unsafe {
        // Cast through *const (): sqlite-vec's entry point is typed against
        // its own libsqlite3-sys, rusqlite's against the bundled one.
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

pub struct Storage {
    // One connection behind a Mutex: callers are hosts doing short, single
    // operations; SQLite-level concurrency with the host's own connections is
    // handled by busy_timeout.
    conn: Mutex<Connection>,
}

impl Storage {
    /// Open (creating if needed) the database and bring the schema up to
    /// date. Safe on an existing production database: the schema init is the
    /// same idempotent DDL the Python backend applied.
    pub fn open(db_path: &str) -> Result<Self, CoreError> {
        register_vec_extension();
        if let Some(parent) = Path::new(db_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| CoreError::Storage(format!("cannot create {parent:?}: {e}")))?;
            }
        }
        let conn = Connection::open(db_path)?;
        // Hosts run their own SQL connections next to this one; wait rather
        // than fail when one of them holds the write lock.
        conn.busy_timeout(Duration::from_secs(5))?;
        // Same rationale as sql.rs: apsw-era behavior is foreign_keys OFF.
        conn.pragma_update(None, "foreign_keys", false)?;
        schema::init_schema(&conn)?;
        // SYN-112: legacy integer ids → uuid TEXT pks (no-op once done).
        crate::migrate::migrate_integer_ids(&conn)?;
        // SYN-112 (T3): sync journal + triggers. After the uuid migration so
        // the journal only ever sees TEXT pks.
        crate::sync::install(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // ── P2P sync (SYN-112 T3) — engine surface, transport comes in phase 3 ──

    pub fn sync_device_id(&self) -> Result<String, CoreError> {
        let conn = self.lock()?;
        crate::sync::device_id(&conn)
    }

    /// Changeset (protocol-v1 JSON) of everything journaled after `since`.
    pub fn sync_changes_since(&self, since: i64, limit: i64) -> Result<String, CoreError> {
        let conn = self.lock()?;
        crate::sync::changes_since(&conn, since, limit)
    }

    /// Merge a peer's changeset (LWW per column) → JSON report. The caller
    /// re-embeds the notes listed in the report's `notes_changed`.
    pub fn sync_apply(&self, changes_json: &str) -> Result<String, CoreError> {
        let conn = self.lock()?;
        crate::sync::apply_changes(&conn, changes_json)
    }

    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, Connection>, CoreError> {
        self.conn
            .lock()
            .map_err(|_| CoreError::Storage("storage mutex poisoned".into()))
    }

    // ── Notes (vec0) ─────────────────────────────────────────────────────

    pub fn upsert_note_vector(&self, note_id: &str, embedding: &[u8]) -> Result<(), CoreError> {
        check_dim(embedding)?;
        self.lock()?.execute(
            "INSERT OR REPLACE INTO atomic_notes_vec(note_id, embedding) VALUES (?1, ?2)",
            params![note_id, embedding],
        )?;
        Ok(())
    }

    pub fn delete_note_vector(&self, note_id: &str) -> Result<(), CoreError> {
        self.lock()?.execute(
            "DELETE FROM atomic_notes_vec WHERE note_id = ?1",
            params![note_id],
        )?;
        Ok(())
    }

    pub fn get_note_vector(&self, note_id: &str) -> Result<Option<Vec<u8>>, CoreError> {
        let conn = self.lock()?;
        let mut stmt =
            conn.prepare("SELECT embedding FROM atomic_notes_vec WHERE note_id = ?1")?;
        let mut rows = stmt.query(params![note_id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    /// KNN over the episodic notes, distance-ascending.
    pub fn search_notes(&self, query: &[u8], k: u32) -> Result<Vec<NoteHit>, CoreError> {
        check_dim(query)?;
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT note_id, distance FROM atomic_notes_vec \
             WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance",
        )?;
        let hits = stmt
            .query_map(params![query, k], |row| {
                Ok(NoteHit {
                    note_id: row.get(0)?,
                    distance: row.get(1)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(hits)
    }

    // ── Entities / resources (BLOB columns) ──────────────────────────────

    pub fn set_entity_embedding(&self, entity_id: &str, embedding: &[u8]) -> Result<(), CoreError> {
        check_dim(embedding)?;
        self.lock()?.execute(
            "UPDATE entities SET embedding = ?1 WHERE id = ?2",
            params![embedding, entity_id],
        )?;
        Ok(())
    }

    pub fn set_resource_embedding(
        &self,
        resource_id: &str,
        embedding: &[u8],
    ) -> Result<(), CoreError> {
        check_dim(embedding)?;
        self.lock()?.execute(
            "UPDATE resources SET embedding = ?1 WHERE id = ?2",
            params![embedding, resource_id],
        )?;
        Ok(())
    }

    /// Top-K entities most similar to `query`, score-descending.
    ///
    /// Candidate filters are identical to `entity_search.py`: vectorized,
    /// not soft-merged, status='active', not user-archived, optional exact
    /// `type`, minus `exclude_ids`; stale-dimension vectors are skipped.
    pub fn search_entities(
        &self,
        query: &[u8],
        limit: u32,
        min_score: f64,
        type_filter: Option<&str>,
        exclude_ids: &[String],
    ) -> Result<Vec<EntityHit>, CoreError> {
        let conn = self.lock()?;
        search_entities_on(&conn, query, limit, min_score, type_filter, exclude_ids)
    }

    /// Top-K resources by similarity on their embedded summary.
    pub fn search_resources(&self, query: &[u8], limit: u32) -> Result<Vec<ResourceHit>, CoreError> {
        let q = parse_query(query)?;
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, title, url, summary, embedding FROM resources \
             WHERE embedding IS NOT NULL",
        )?;
        let mut rows = stmt.query([])?;
        let mut scored: Vec<ResourceHit> = Vec::new();
        while let Some(row) = rows.next()? {
            let blob: Vec<u8> = row.get(4)?;
            let Some(score) = score_against(&q, &blob) else {
                continue;
            };
            let title: Option<String> = row.get(1)?;
            let url: Option<String> = row.get(2)?;
            scored.push(ResourceHit {
                id: row.get(0)?,
                title: title.or_else(|| url.clone()),
                url,
                summary: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                score,
            });
        }
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(limit as usize);
        Ok(scored)
    }
}

/// Same-connection variant of the entity similarity scan, callable from
/// routing code that already holds the storage connection (in-transaction).
pub(crate) fn search_entities_on(
    conn: &Connection,
    query: &[u8],
    limit: u32,
    min_score: f64,
    type_filter: Option<&str>,
    exclude_ids: &[String],
) -> Result<Vec<EntityHit>, CoreError> {
    {
        let q = parse_query(query)?;
        let exclude: HashSet<&str> = exclude_ids.iter().map(String::as_str).collect();
        let base = "SELECT id, canonical_name, type, summary, embedding FROM entities \
                    WHERE embedding IS NOT NULL AND merged_into_id IS NULL \
                    AND status = 'active' AND archived_at IS NULL";
        let mut scored: Vec<EntityHit> = Vec::new();
        let mut visit = |row: &rusqlite::Row<'_>| -> Result<(), rusqlite::Error> {
            let id: String = row.get(0)?;
            if exclude.contains(id.as_str()) {
                return Ok(());
            }
            let blob: Vec<u8> = row.get(4)?;
            let Some(score) = score_against(&q, &blob) else {
                return Ok(()); // stale dim (model changed before a re-embed)
            };
            if score < min_score {
                return Ok(());
            }
            scored.push(EntityHit {
                id,
                canonical_name: row.get(1)?,
                entity_type: row.get(2)?,
                summary: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                score,
            });
            Ok(())
        };

        match type_filter {
            Some(t) => {
                let mut stmt = conn.prepare(&format!("{base} AND type = ?1"))?;
                let mut rows = stmt.query(params![t])?;
                while let Some(row) = rows.next()? {
                    visit(row)?;
                }
            }
            None => {
                let mut stmt = conn.prepare(base)?;
                let mut rows = stmt.query([])?;
                while let Some(row) = rows.next()? {
                    visit(row)?;
                }
            }
        }

        // Stable sort, like Python's `sort(key=lambda x: -x[0])`: ties keep
        // the table scan order.
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(limit as usize);
        Ok(scored)
    }
}

fn check_dim(embedding: &[u8]) -> Result<(), CoreError> {
    if embedding.len() != EMBEDDING_BYTES {
        return Err(CoreError::Storage(format!(
            "embedding must be {EMBEDDING_BYTES} bytes ({EMBEDDING_DIM} float32), got {}",
            embedding.len()
        )));
    }
    Ok(())
}

fn parse_query(query: &[u8]) -> Result<Vec<f64>, CoreError> {
    check_dim(query)?;
    Ok(blob_to_f64(query))
}

fn blob_to_f64(blob: &[u8]) -> Vec<f64> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect()
}

/// `round(max(0, 1 - L2/2), 4)` against a stored blob, or None when the
/// stored vector has a stale dimension. f64 accumulation in storage order,
/// matching the Python float math it replaces.
fn score_against(q: &[f64], blob: &[u8]) -> Option<f64> {
    if blob.len() != q.len() * 4 {
        return None;
    }
    let v = blob_to_f64(blob);
    let dist = q
        .iter()
        .zip(v.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let score = (1.0 - dist / 2.0).max(0.0);
    Some((score * 10000.0).round() / 10000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(theta: f64) -> Vec<u8> {
        let mut v = vec![0f32; EMBEDDING_DIM];
        v[0] = theta.cos() as f32;
        v[1] = theta.sin() as f32;
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn open_temp() -> (tempfile::TempDir, Storage) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synapse.db");
        let storage = Storage::open(path.to_str().unwrap()).unwrap();
        (dir, storage)
    }

    fn insert_entity(storage: &Storage, id: &str, name: &str, etype: &str) {
        storage
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO entities (id, canonical_name, type) VALUES (?1, ?2, ?3)",
                params![id, name, etype],
            )
            .unwrap();
    }

    #[test]
    fn schema_init_is_idempotent_and_reopenable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synapse.db");
        let s1 = Storage::open(path.to_str().unwrap()).unwrap();
        drop(s1);
        // Second open replays every CREATE/ALTER on the existing file.
        let s2 = Storage::open(path.to_str().unwrap()).unwrap();
        let conn = s2.lock().unwrap();
        let n: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name = 'atomic_notes_vec'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        let types: i64 = conn
            .query_row("SELECT count(*) FROM active_entity_types", [], |r| r.get(0))
            .unwrap();
        assert_eq!(types, 6);
    }

    #[test]
    fn note_vectors_roundtrip_and_knn() {
        let (_dir, storage) = open_temp();
        let a = unit_vec(0.0);
        let b = unit_vec(0.3);
        let far = unit_vec(2.0);
        storage.upsert_note_vector("note-a", &a).unwrap();
        storage.upsert_note_vector("note-b", &b).unwrap();
        storage.upsert_note_vector("note-far", &far).unwrap();

        assert_eq!(storage.get_note_vector("note-a").unwrap().unwrap(), a);
        assert!(storage.get_note_vector("absent").unwrap().is_none());

        let hits = storage.search_notes(&a, 3).unwrap();
        assert_eq!(
            hits.iter().map(|h| h.note_id.as_str()).collect::<Vec<_>>(),
            vec!["note-a", "note-b", "note-far"]
        );
        assert!(hits[0].distance < 1e-6);
        // L2 between unit vectors at angle θ = 2·sin(θ/2).
        assert!((hits[1].distance - 2.0 * (0.3f64 / 2.0).sin()).abs() < 1e-6);

        storage.delete_note_vector("note-a").unwrap();
        assert!(storage.get_note_vector("note-a").unwrap().is_none());
        assert_eq!(storage.search_notes(&a, 3).unwrap().len(), 2);
    }

    #[test]
    fn entity_search_replicates_python_semantics() {
        let (_dir, storage) = open_temp();
        for (id, name, etype) in [
            ("e1", "Proche", "person"),
            ("e2", "Moyen", "person"),
            ("e3", "Loin", "person"),
            ("e4", "AutreType", "place"),
            ("e5", "Fusionnee", "person"),
            ("e6", "Archivee", "person"),
            ("e7", "EnAttente", "person"),
            ("e8", "SansVecteur", "person"),
            ("e9", "DimPerimee", "person"),
        ] {
            insert_entity(&storage, id, name, etype);
        }
        let q = unit_vec(0.0);
        storage.set_entity_embedding("e1", &unit_vec(0.1)).unwrap();
        storage.set_entity_embedding("e2", &unit_vec(0.8)).unwrap();
        storage.set_entity_embedding("e3", &unit_vec(2.5)).unwrap();
        storage.set_entity_embedding("e4", &unit_vec(0.05)).unwrap();
        storage.set_entity_embedding("e5", &unit_vec(0.0)).unwrap();
        storage.set_entity_embedding("e6", &unit_vec(0.0)).unwrap();
        storage.set_entity_embedding("e7", &unit_vec(0.0)).unwrap();
        {
            let conn = storage.lock().unwrap();
            conn.execute("UPDATE entities SET merged_into_id = 'e1' WHERE id = 'e5'", [])
                .unwrap();
            conn.execute("UPDATE entities SET archived_at = CURRENT_TIMESTAMP WHERE id = 'e6'", [])
                .unwrap();
            conn.execute("UPDATE entities SET status = 'pending' WHERE id = 'e7'", [])
                .unwrap();
            // Stale dim: a raw 2-float blob (model change without re-embed).
            conn.execute(
                "UPDATE entities SET embedding = ?1 WHERE id = 'e9'",
                params![vec![0u8; 8]],
            )
            .unwrap();
        }

        let hits = storage.search_entities(&q, 10, 0.0, None, &[]).unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        // e5/e6/e7 filtered by lifecycle, e8 unvectorized, e9 stale dim.
        assert_eq!(ids, vec!["e4", "e1", "e2", "e3"]);
        // Hand-checked score: d = 2·sin(0.05) → 1 - d/2, rounded to 4.
        let expected = 1.0 - (2.0 * (0.1f64 / 2.0).sin()) / 2.0;
        let expected = (expected * 10000.0).round() / 10000.0;
        assert_eq!(hits[1].score, expected);

        // type filter
        let hits = storage
            .search_entities(&q, 10, 0.0, Some("person"), &[])
            .unwrap();
        assert_eq!(
            hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            vec!["e1", "e2", "e3"]
        );

        // exclude + min_score + limit
        let hits = storage
            .search_entities(&q, 1, 0.5, None, &["e4".to_string()])
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "e1");
    }

    #[test]
    fn resource_search_scores_and_falls_back_title() {
        let (_dir, storage) = open_temp();
        {
            let conn = storage.lock().unwrap();
            conn.execute(
                "INSERT INTO resources (id, title, url, summary) VALUES \
                 ('r1', NULL, 'https://ex.am/ple', 'sans titre'), \
                 ('r2', 'Titre', NULL, NULL)",
                [],
            )
            .unwrap();
        }
        storage.set_resource_embedding("r1", &unit_vec(0.1)).unwrap();
        storage.set_resource_embedding("r2", &unit_vec(1.0)).unwrap();

        let hits = storage.search_resources(&unit_vec(0.0), 10).unwrap();
        assert_eq!(hits[0].id, "r1");
        assert_eq!(hits[0].title.as_deref(), Some("https://ex.am/ple"));
        assert_eq!(hits[1].summary, "");
    }

    #[test]
    fn rejects_wrong_dimension_writes() {
        let (_dir, storage) = open_temp();
        assert!(storage.upsert_note_vector("n1", &[0u8; 8]).is_err());
        assert!(storage.set_entity_embedding("x", &[0u8; 8]).is_err());
        assert!(storage.search_notes(&[0u8; 8], 3).is_err());
    }
}
