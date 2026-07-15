//! Local read snapshot for app replicas (SYN-132).
//!
//! A phone that embeds the core holds a fully synced copy of the database,
//! but the app's read replica historically only knew how to consume the
//! desktop backend's HTTP endpoints (`GET /changes`, `/feed`, `/projects`,
//! `/project/{id}/state`, `/pending`, the three proposal lists and
//! `/capture/{id}/generated`). This module renders the SAME JSON shapes
//! straight from the core database, so the replica can be fed with one local
//! call and the screens stay byte-compatible whichever source feeds them.
//! The SQL mirrors `api/app.py` on purpose — keep both in step.

use base64ct::{Base64, Encoding};
use rusqlite::types::ValueRef;
use rusqlite::Connection;
use serde_json::{json, Map, Value};

use crate::embedder::CoreError;

fn cell_to_json(v: ValueRef<'_>) -> Value {
    match v {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => json!(i),
        ValueRef::Real(f) => json!(f),
        ValueRef::Text(t) => Value::String(String::from_utf8_lossy(t).into_owned()),
        // JSON can't hold bytes; base64 matches what /changes ships (SYN-91).
        ValueRef::Blob(b) => Value::String(Base64::encode_string(b)),
    }
}

/// Run a SELECT and return its rows as JSON objects (the `cursor_to_dicts`
/// of the Python backend).
fn query_rows(
    conn: &Connection,
    sql: &str,
    params: &[&dyn rusqlite::ToSql],
) -> Result<Vec<Map<String, Value>>, CoreError> {
    let mut stmt = conn.prepare(sql)?;
    let columns: Vec<String> = stmt.column_names().iter().map(|c| c.to_string()).collect();
    let mut out = Vec::new();
    let mut rows = stmt.query(rusqlite::params_from_iter(params.iter()))?;
    while let Some(row) = rows.next()? {
        let mut obj = Map::with_capacity(columns.len());
        for (i, name) in columns.iter().enumerate() {
            obj.insert(name.clone(), cell_to_json(row.get_ref(i)?));
        }
        out.push(obj);
    }
    Ok(out)
}

fn first_row(
    conn: &Connection,
    sql: &str,
    params: &[&dyn rusqlite::ToSql],
) -> Result<Option<Map<String, Value>>, CoreError> {
    Ok(query_rows(conn, sql, params)?.into_iter().next())
}

/// How Python's f-string renders a JSON field in the pending question.
fn display(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "None".into(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// The `/changes` payload: full derived state, entity embeddings as base64.
fn changes(conn: &Connection) -> Result<Value, CoreError> {
    let mut entities = query_rows(conn, "SELECT * FROM entities", &[])?;
    for e in &mut entities {
        // SYN-91: the raw BLOB column is dropped and shipped as embedding_b64
        // (cell_to_json already made it a base64 string).
        let emb = e.remove("embedding").unwrap_or(Value::Null);
        e.insert("embedding_b64".into(), emb);
    }
    let relations = query_rows(
        conn,
        "SELECT * FROM relations WHERE review_status != 'pending'",
        &[],
    )?;
    let facts = query_rows(conn, "SELECT * FROM facts", &[])?;
    let notes = query_rows(conn, "SELECT * FROM atomic_notes", &[])?;
    Ok(json!({
        "entities": entities,
        "facts": facts,
        "relations": relations,
        "atomic_notes": notes,
        // Same role as the backend's datetime.now(utc).isoformat(): an opaque
        // "when was this snapshot taken" marker (never used for deltas).
        "cursor": format!("{}+00:00",
            crate::decay::resolve_now(None).format("%Y-%m-%dT%H:%M:%S%.6f")),
        "instance_id": crate::sync::device_id(conn)?,
    }))
}

/// The `/feed` payload (latest captures, legacy status fix included).
fn feed(conn: &Connection, limit: i64) -> Result<Vec<Map<String, Value>>, CoreError> {
    let mut rows = query_rows(
        conn,
        "SELECT id, client_id, content, source, created_at, captured_at, processed_at, \
         status, error FROM inbox ORDER BY created_at DESC LIMIT ?1",
        &[&limit],
    )?;
    for r in &mut rows {
        let status = match r.get("status") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => "queued".into(),
        };
        // legacy rows processed before the status column existed
        let processed = matches!(r.get("processed_at"), Some(Value::String(_)));
        let status = if status == "queued" && processed { "processed".into() } else { status };
        r.insert("status".into(), Value::String(status));
    }
    Ok(rows)
}

const PROJECTS_SQL: &str = "SELECT e.id, e.canonical_name, e.mention_count, e.persistence_value, \
        e.last_mentioned, e.summary, \
        psv.summary_md AS current_summary_md, \
        psv.kind        AS current_kind, \
        psv.created_at  AS current_synthesized_at, \
        (SELECT COUNT(*) FROM project_entries pe WHERE pe.project_id = e.id) AS entries_total \
 FROM entities e \
 LEFT JOIN project_state ps ON ps.project_id = e.id \
 LEFT JOIN project_state_versions psv ON psv.id = ps.current_version_id \
 WHERE e.type = 'project' AND e.merged_into_id IS NULL \
   AND e.status = 'active' AND e.archived_at IS NULL \
 ORDER BY COALESCE(e.last_mentioned, e.created_at) DESC";

/// The `/project/{id}/state` payload for one project.
fn project_state(conn: &Connection, id: &str, name: &Value) -> Result<Value, CoreError> {
    let state = first_row(
        conn,
        "SELECT psv.id AS version_id, psv.summary_md, psv.entry_count, \
                psv.trigger, psv.kind, psv.created_at, ps.updated_at \
         FROM project_state ps \
         JOIN project_state_versions psv ON psv.id = ps.current_version_id \
         WHERE ps.project_id = ?1",
        &[&id],
    )?;
    let entries = query_rows(
        conn,
        "SELECT id, content, kind, capture_id, created_at \
         FROM project_entries WHERE project_id = ?1 \
         ORDER BY created_at DESC LIMIT 50",
        &[&id],
    )?;
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM project_entries WHERE project_id = ?1",
        [id],
        |r| r.get(0),
    )?;
    // SYN-134 — projects carry facts now; same ACTIVE-only slice as the
    // backend endpoint (api/app.py::project_state).
    let facts = query_rows(
        conn,
        "SELECT id, predicate, value, confidence, category, \
                persistence_value, created_at, provenance_capture_id \
         FROM facts WHERE entity_id = ?1 \
         AND obsoleted_at IS NULL AND archived_at IS NULL \
         ORDER BY created_at ASC",
        &[&id],
    )?;
    Ok(json!({
        "project_id": id,
        "canonical_name": name,
        "current_state": state,
        "entries_recent": entries,
        "entries_total": total,
        "facts": facts,
    }))
}

/// The `/pending` payload: pending facts as validatable cards.
fn pending_facts(conn: &Connection) -> Result<Vec<Value>, CoreError> {
    let items = query_rows(
        conn,
        "SELECT id, fact_data, created_at FROM pending_facts ORDER BY created_at DESC",
        &[],
    )?;
    let mut out = Vec::new();
    for item in items {
        let fd: Map<String, Value> = match item.get("fact_data") {
            Some(Value::String(s)) => match serde_json::from_str(s) {
                Ok(Value::Object(m)) => m,
                _ => continue,
            },
            _ => continue,
        };
        let source_text = match fd.get("source_inbox_id") {
            None | Some(Value::Null) => Value::Null,
            Some(src) => {
                let src_id = display(Some(src));
                first_row(conn, "SELECT content FROM inbox WHERE id = ?1", &[&src_id])?
                    .and_then(|mut r| r.remove("content"))
                    .unwrap_or(Value::Null)
            }
        };
        let entity = fd.get("entity_canonical").and_then(|v| v.as_str()).unwrap_or("");
        out.push(json!({
            "id": item.get("id"),
            "entity": entity,
            "predicate": fd.get("predicate"),
            "value": fd.get("value"),
            "confidence": fd.get("confidence"),
            "question": format!("{} — {} : {} ?", entity,
                display(fd.get("predicate")), display(fd.get("value"))),
            "source_text": source_text,
            "created_at": item.get("created_at"),
        }));
    }
    Ok(out)
}

/// The `/merge-proposals?status=pending` payload (side entities + fact previews).
fn merge_proposals(conn: &Connection) -> Result<Vec<Value>, CoreError> {
    let rows = query_rows(
        conn,
        "SELECT p.id, p.candidate_entity_id, p.existing_entity_id, \
                p.similarity_score, p.similarity_reason, p.evidence_capture_id, \
                p.status, p.created_at, p.resolved_at, p.resolved_canonical_id, \
                ec.canonical_name AS candidate_name, ec.type AS candidate_type, \
                ec.mention_count   AS candidate_mention_count, \
                ee.canonical_name AS existing_name,  ee.type AS existing_type, \
                ee.mention_count   AS existing_mention_count \
         FROM entity_merge_proposals p \
         JOIN entities ec ON ec.id = p.candidate_entity_id \
         JOIN entities ee ON ee.id = p.existing_entity_id \
         WHERE p.status = 'pending' \
         ORDER BY p.created_at DESC",
        &[],
    )?;
    let mut out = Vec::new();
    for mut r in rows {
        for side in ["candidate", "existing"] {
            let eid = display(r.get(&format!("{side}_entity_id")));
            let facts = query_rows(
                conn,
                "SELECT predicate, value, confidence FROM facts \
                 WHERE entity_id = ?1 ORDER BY confidence DESC LIMIT 5",
                &[&eid],
            )?;
            r.insert(format!("{side}_facts"), json!(facts));
        }
        out.push(Value::Object(r));
    }
    Ok(out)
}

/// The `GET /atomic-notes?review_status=pending` payload (SYN-143): the
/// « À valider » task/event queue (low-confidence classifications).
fn pending_tasks(conn: &Connection) -> Result<Vec<Value>, CoreError> {
    let rows = query_rows(
        conn,
        "SELECT id, title, content, summary, entities_mentioned, memory_strength, \
                provenance_capture_id, created_at, updated_at, \
                kind, event_date, event_recurring, archived_at, review_status \
         FROM atomic_notes WHERE archived_at IS NULL AND review_status = 'pending' \
         ORDER BY created_at DESC LIMIT 50",
        &[],
    )?;
    let mut out = Vec::new();
    for mut r in rows {
        let mentioned = r
            .get("entities_mentioned")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
            .filter(Value::is_array)
            .unwrap_or_else(|| json!([]));
        r.insert("entities_mentioned".into(), mentioned);
        out.push(Value::Object(r));
    }
    Ok(out)
}

/// The `GET /relations/pending` payload (SYN-143): low-confidence relations,
/// names resolved on both ends.
fn pending_relations(conn: &Connection) -> Result<Vec<Map<String, Value>>, CoreError> {
    query_rows(
        conn,
        "SELECT r.id, r.predicate, r.confidence, r.provenance_capture_id, \
                ef.canonical_name AS entity_from_name, \
                et.canonical_name AS entity_to_name \
         FROM relations r \
         JOIN entities ef ON ef.id = r.entity_from \
         JOIN entities et ON et.id = r.entity_to \
         WHERE r.review_status = 'pending' ORDER BY r.created_at DESC",
        &[],
    )
}

/// The `GET /space` payload (SYN-139): replicated singleton + who we are and
/// who tisses. `space` stays null until the owner founds it (first cycle).
fn space(conn: &Connection) -> Result<Value, CoreError> {
    let space = first_row(
        conn,
        "SELECT space_id, name, created_at FROM space WHERE id = 'space'",
        &[],
    )?;
    let owner = first_row(
        conn,
        "SELECT device_id FROM sync_owner WHERE id = 'owner'",
        &[],
    )?;
    Ok(json!({
        "space": space.map(Value::Object).unwrap_or(Value::Null),
        "device_id": crate::sync::device_id(conn)?,
        "owner_device_id": owner
            .and_then(|mut r| r.remove("device_id"))
            .unwrap_or(Value::Null),
    }))
}

/// The `GET /devices` payload (SYN-139). `last_pull_at` is a backend notion
/// (its per-peer pull cursors, `sync_meta 'pulled_at:%'`); the mesh copy has
/// no local equivalent, so the field is omitted and the app DTO's null
/// default applies.
fn devices(conn: &Connection) -> Result<Vec<Value>, CoreError> {
    let me = crate::sync::device_id(conn)?;
    let owner_id = first_row(
        conn,
        "SELECT device_id FROM sync_owner WHERE id = 'owner'",
        &[],
    )?
    .and_then(|mut r| r.remove("device_id"))
    .and_then(|v| v.as_str().map(str::to_owned));
    let rows = query_rows(
        conn,
        "SELECT device_id, name, platform, last_seen, revoked_at \
         FROM devices ORDER BY (device_id = ?1) DESC, name",
        &[&me],
    )?;
    let mut out = Vec::new();
    for mut r in rows {
        let id = display(r.get("device_id"));
        r.insert("is_self".into(), json!(id == me));
        r.insert("is_owner".into(), json!(owner_id.as_deref() == Some(id.as_str())));
        r.insert(
            "revoked".into(),
            json!(!matches!(r.get("revoked_at"), None | Some(Value::Null))),
        );
        out.push(Value::Object(r));
    }
    Ok(out)
}

/// The whole local read snapshot, one JSON object per consumed endpoint.
pub fn read_snapshot(conn: &Connection) -> Result<Value, CoreError> {
    let projects = query_rows(conn, PROJECTS_SQL, &[])?;
    let mut states = Map::new();
    for p in &projects {
        let id = display(p.get("id"));
        let name = p.get("canonical_name").cloned().unwrap_or(Value::Null);
        states.insert(id.clone(), project_state(conn, &id, &name)?);
    }
    let type_proposals = query_rows(
        conn,
        "SELECT p.id, p.proposed_type, p.reason, p.status, p.created_at, \
                p.resolved_at, p.candidate_entity_id, p.evidence_capture_id, \
                e.canonical_name AS candidate_name, e.type AS candidate_type, \
                e.summary        AS candidate_summary, e.status AS candidate_status, \
                i.content        AS evidence_content \
         FROM entity_type_proposals p \
         LEFT JOIN entities e ON e.id = p.candidate_entity_id \
         LEFT JOIN inbox i    ON i.id = p.evidence_capture_id \
         WHERE p.status = 'pending' \
         ORDER BY p.created_at DESC",
        &[],
    )?;
    let attach_proposals = query_rows(
        conn,
        "SELECT p.id, p.capture_id, p.note_id, p.project_id, p.content, \
                p.similarity_score, p.status, p.created_at, p.resolved_at, \
                e.canonical_name AS project_name, \
                i.content        AS capture_content \
         FROM project_attach_proposals p \
         LEFT JOIN entities e ON e.id = p.project_id \
         LEFT JOIN inbox i    ON i.id = p.capture_id \
         WHERE p.status = 'pending' \
         ORDER BY p.created_at DESC",
        &[],
    )?;
    Ok(json!({
        "changes": changes(conn)?,
        "feed": feed(conn, 100)?,
        "projects": projects,
        "project_states": states,
        "pending_facts": pending_facts(conn)?,
        "merge_proposals": merge_proposals(conn)?,
        "type_proposals": type_proposals,
        "project_attach_proposals": attach_proposals,
        "space": space(conn)?,
        "devices": devices(conn)?,
        "pending_tasks": pending_tasks(conn)?,
        "pending_relations": pending_relations(conn)?,
    }))
}

/// The `/capture/{id}/generated` payload — reverse provenance for one capture
/// (SYN-92's « ce qui en est sorti » panel, served from the local core db).
pub fn generated_for_capture(conn: &Connection, capture_id: &str) -> Result<Value, CoreError> {
    let entities = query_rows(
        conn,
        "SELECT id, canonical_name, type FROM entities \
         WHERE provenance_capture_id = ?1 AND merged_into_id IS NULL \
         ORDER BY created_at",
        &[&capture_id],
    )?;
    let facts = query_rows(
        conn,
        "SELECT f.id, f.predicate, f.value, f.entity_id, f.confidence, f.category, \
                e.canonical_name AS entity_name, f.archived_at, f.obsoleted_at \
         FROM facts f LEFT JOIN entities e ON e.id = f.entity_id \
         WHERE f.provenance_capture_id = ?1 ORDER BY f.created_at",
        &[&capture_id],
    )?;
    let relations = query_rows(
        conn,
        "SELECT r.id, r.entity_from, r.predicate, r.entity_to, r.confidence, \
                ef.canonical_name AS entity_from_name, et.canonical_name AS entity_to_name \
         FROM relations r \
         LEFT JOIN entities ef ON ef.id = r.entity_from \
         LEFT JOIN entities et ON et.id = r.entity_to \
         WHERE r.provenance_capture_id = ?1 ORDER BY r.created_at",
        &[&capture_id],
    )?;
    let notes = query_rows(
        conn,
        "SELECT id, title, content, summary, kind, archived_at \
         FROM atomic_notes WHERE provenance_capture_id = ?1 ORDER BY created_at",
        &[&capture_id],
    )?;
    Ok(json!({
        "entities": entities, "facts": facts,
        "relations": relations, "notes": notes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Storage;

    fn setup() -> (tempfile::TempDir, Connection) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snap.db");
        Storage::open(path.to_str().unwrap()).unwrap(); // schema owner
        let conn = Connection::open(&path).unwrap();
        // Parity with sql::connect(): the backend's soft-link patterns rely on OFF.
        conn.pragma_update(None, "foreign_keys", false).unwrap();
        (dir, conn)
    }

    #[test]
    fn snapshot_matches_changes_shape() {
        let (_dir, conn) = setup();
        let emb: Vec<u8> = vec![0, 0, 128, 63]; // 1.0f32 LE
        conn.execute(
            "INSERT INTO entities (id, canonical_name, type, embedding, status) \
             VALUES ('e1', 'Arkose', 'organization', ?1, 'active')",
            [&emb],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO entities (id, canonical_name, type, status) \
             VALUES ('e2', 'Alexis', 'person', 'active')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO relations (id, entity_from, predicate, entity_to, review_status) \
             VALUES ('r1', 'e2', 'climbs_at', 'e1', 'confirmed'), \
                    ('r2', 'e1', 'located_in', 'e2', 'pending')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO inbox (id, content, status, processed_at) \
             VALUES ('c1', 'grimpe à Arkose', 'queued', '2026-07-14 10:00:00')",
            [],
        )
        .unwrap();

        let snap = read_snapshot(&conn).unwrap();
        let changes = &snap["changes"];
        // Embedding shipped as base64, raw column dropped.
        let arkose = changes["entities"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == "e1")
            .unwrap();
        assert_eq!(arkose["embedding_b64"], json!(Base64::encode_string(&emb)));
        assert!(arkose.get("embedding").is_none());
        let alexis = changes["entities"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == "e2")
            .unwrap();
        assert_eq!(alexis["embedding_b64"], Value::Null);
        // Pending relations are filtered like GET /changes does.
        let rels = changes["relations"].as_array().unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0]["id"], "r1");
        // Legacy feed rows with processed_at read as processed.
        assert_eq!(snap["feed"][0]["status"], "processed");
        assert!(changes["cursor"].as_str().unwrap().contains('T'));
        assert!(changes["instance_id"].is_string());
    }

    #[test]
    fn snapshot_carries_pending_tasks_and_relations() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name) VALUES ('e1', 'Alexis'), ('e2', 'Arkose')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO atomic_notes (id, content, kind, entities_mentioned) \
             VALUES ('n1', 'réserver le créneau', 'task', '[\"Arkose\"]'), \
                    ('n2', 'note sûre', 'note', NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "UPDATE atomic_notes SET review_status = 'pending' WHERE id = 'n1'",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO relations (id, entity_from, predicate, entity_to, review_status) \
             VALUES ('r1', 'e1', 'climbs_at', 'e2', 'pending'), \
                    ('r2', 'e1', 'works_at', 'e2', 'confirmed')",
            [],
        )
        .unwrap();

        let snap = read_snapshot(&conn).unwrap();
        // Only the pending task surfaces, entities_mentioned decoded to a list.
        let tasks = snap["pending_tasks"].as_array().unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0]["id"], "n1");
        assert_eq!(tasks[0]["review_status"], "pending");
        assert_eq!(tasks[0]["entities_mentioned"], json!(["Arkose"]));
        // Only the pending relation, names resolved on both ends.
        let rels = snap["pending_relations"].as_array().unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0]["id"], "r1");
        assert_eq!(rels[0]["entity_from_name"], "Alexis");
        assert_eq!(rels[0]["entity_to_name"], "Arkose");
    }

    #[test]
    fn snapshot_carries_space_and_devices() {
        let (_dir, conn) = setup();
        let me: String = conn
            .query_row("SELECT v FROM sync_meta WHERE k = 'device_id'", [], |r| r.get(0))
            .unwrap();
        conn.execute(
            "INSERT INTO space (id, space_id, name) VALUES ('space', 'sp-1', 'Mémoire')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sync_owner (id, device_id, epoch) VALUES ('owner', 'dev-b', 4)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO devices (device_id, name, platform, last_seen, revoked_at) VALUES \
             (?1, 'Pixel 9a', 'android', '2026-07-14 10:00:00', NULL), \
             ('dev-b', 'Macmini', 'macos', '2026-07-14 09:00:00', NULL), \
             ('dev-c', 'Ancien', 'macos', NULL, '2026-07-01 08:00:00')",
            [&me],
        )
        .unwrap();

        let snap = read_snapshot(&conn).unwrap();
        // /space shape: singleton + who we are + who tisses.
        assert_eq!(snap["space"]["space"]["space_id"], "sp-1");
        assert_eq!(snap["space"]["space"]["name"], "Mémoire");
        assert_eq!(snap["space"]["device_id"], json!(me));
        assert_eq!(snap["space"]["owner_device_id"], "dev-b");
        // /devices shape: self first, flags computed locally, no last_pull_at.
        let devices = snap["devices"].as_array().unwrap();
        assert_eq!(devices.len(), 3);
        assert_eq!(devices[0]["device_id"], json!(me));
        assert_eq!(devices[0]["is_self"], true);
        assert_eq!(devices[0]["is_owner"], false);
        assert_eq!(devices[0]["revoked"], false);
        let mac = devices.iter().find(|d| d["device_id"] == "dev-b").unwrap();
        assert_eq!(mac["is_owner"], true);
        assert_eq!(mac["is_self"], false);
        let old = devices.iter().find(|d| d["device_id"] == "dev-c").unwrap();
        assert_eq!(old["revoked"], true);
        assert!(devices[0].get("last_pull_at").is_none());

        // A db founded by nobody yet: space null, owner null, devices empty.
        let (_dir2, fresh) = setup();
        let snap2 = read_snapshot(&fresh).unwrap();
        assert_eq!(snap2["space"]["space"], Value::Null);
        assert_eq!(snap2["space"]["owner_device_id"], Value::Null);
        assert_eq!(snap2["devices"], json!([]));
    }

    #[test]
    fn snapshot_carries_projects_and_pending() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name, type, status) \
             VALUES ('p1', 'Escalade', 'project', 'active')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO project_entries (id, project_id, content, kind, capture_id) \
             VALUES ('pe1', 'p1', 'nouvelle voie 6b', 'capture', 'c1')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO project_state_versions (id, project_id, summary_md, entry_count, trigger, kind) \
             VALUES ('v1', 'p1', '## Escalade', 1, 'entry', 'refinement')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO project_state (project_id, current_version_id) VALUES ('p1', 'v1')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO inbox (id, content, status) VALUES ('c1', 'Nina fait de la grimpe', 'processed')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pending_facts (id, fact_data) VALUES ('pf1', \
             '{\"entity_canonical\": \"Nina\", \"predicate\": \"does_sport\", \
               \"value\": \"escalade\", \"confidence\": 0.6, \"source_inbox_id\": \"c1\"}')",
            [],
        )
        .unwrap();

        let snap = read_snapshot(&conn).unwrap();
        assert_eq!(snap["projects"][0]["id"], "p1");
        assert_eq!(snap["projects"][0]["entries_total"], 1);
        let state = &snap["project_states"]["p1"];
        assert_eq!(state["current_state"]["summary_md"], "## Escalade");
        assert_eq!(state["entries_total"], 1);
        assert_eq!(state["entries_recent"][0]["id"], "pe1");
        let pf = &snap["pending_facts"][0];
        assert_eq!(pf["question"], "Nina — does_sport : escalade ?");
        assert_eq!(pf["source_text"], "Nina fait de la grimpe");
    }

    #[test]
    fn generated_for_capture_fans_out_on_provenance() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name, type, provenance_capture_id, status) \
             VALUES ('e1', 'Arkose', 'organization', 'c1', 'active')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO facts (id, entity_id, predicate, value, provenance_capture_id) \
             VALUES ('f1', 'e1', 'is_climbing_gym', 'true', 'c1')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO atomic_notes (id, content, provenance_capture_id) \
             VALUES ('n1', 'séance à Arkose', 'c1')",
            [],
        )
        .unwrap();

        let g = generated_for_capture(&conn, "c1").unwrap();
        assert_eq!(g["entities"][0]["canonical_name"], "Arkose");
        assert_eq!(g["facts"][0]["entity_name"], "Arkose");
        assert_eq!(g["notes"][0]["id"], "n1");
        assert!(g["relations"].as_array().unwrap().is_empty());
        assert!(generated_for_capture(&conn, "absent").unwrap()["entities"]
            .as_array()
            .unwrap()
            .is_empty());
    }
}
