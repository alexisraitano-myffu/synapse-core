//! SYN-135 — local application of the mobile app's action log.
//!
//! Mirror of the desktop backend's write endpoints (`api/app.py` +
//! `dream_cycle/validation.py`) so a phone-only install applies user gestures
//! (validate a fact, archive, rename, relation CRUD, accept a merge, …) to
//! its local core db. The SQL here is a DELIBERATE mirror of the Python
//! endpoints — when one side's behaviour changes, change the other.
//!
//! Pure SQL on the caller's connection: the sync triggers (`sync.rs`) journal
//! every write, so replication to peers is automatic — no stamping needed.
//!
//! Replay semantics differ from HTTP on purpose: a missing row returns
//! `{"status": "not_found"}` (Ok) instead of an error, and an already
//! resolved proposal returns `{"status": "skipped"}` — a peer may have
//! applied/deleted the row first, and the app's action queue must never
//! wedge on a gesture that has become moot.

use rusqlite::types::Value as SqlV;
use rusqlite::{params, params_from_iter, Connection};
use serde_json::{json, Map, Value};

use crate::embedder::CoreError;
use crate::routing::{
    find_existing_entity, insert_fact, json_scalar_to_sql, new_uuid, py_dumps, query_row_map,
    query_row_maps,
};

/// User-confirmed facts are near-certain (`validation.py::CONFIRMED_CONFIDENCE`).
const CONFIRMED_CONFIDENCE: f64 = 0.95;

/// Apply one action-log entry. `payload_json` is the app's string map
/// (`Map<String, String>` — booleans/nullables are string-encoded, blank =
/// null). Returns the outcome as JSON; the caller owns the transaction.
pub fn apply_action(
    conn: &Connection,
    action_type: &str,
    payload_json: &str,
) -> Result<Value, CoreError> {
    let payload: Value = serde_json::from_str(payload_json)
        .map_err(|e| CoreError::Storage(format!("invalid action payload: {e}")))?;
    let empty = Map::new();
    let p = payload.as_object().unwrap_or(&empty);

    match action_type {
        "validate_fact" => validate_fact(conn, p),
        "archive_fact" => fact_timestamp(conn, s(p, "factId"), "archived_at=CURRENT_TIMESTAMP"),
        "unarchive_fact" => fact_timestamp(conn, s(p, "factId"), "archived_at=NULL"),
        "obsolete_fact" => fact_timestamp(conn, s(p, "factId"), "obsoleted_at=CURRENT_TIMESTAMP"),
        "restore_fact" => fact_timestamp(conn, s(p, "factId"), "obsoleted_at=NULL, obsoleted_by=NULL"),
        "edit_fact" => edit_fact(conn, p),
        "archive_entity" => entity_timestamp(conn, s(p, "entityId"), "archived_at=CURRENT_TIMESTAMP"),
        "unarchive_entity" => entity_timestamp(conn, s(p, "entityId"), "archived_at=NULL"),
        "change_type" => change_type(conn, p),
        "rename_entity" => rename_entity(conn, p),
        "create_relation" => create_relation(conn, p),
        "edit_relation" => edit_relation(conn, p),
        "delete_relation" => delete_relation(conn, p),
        "archive_note" => note_archived(conn, s(p, "noteId"), Some(iso_now())),
        "unarchive_note" => note_archived(conn, s(p, "noteId"), None),
        "reinforce_note" => reinforce_note(conn, s(p, "noteId")),
        "set_note_date" => set_note_date(conn, p),
        "promote_note" => promote_note(conn, p),
        "accept_merge" => accept_merge(conn, p),
        "reject_merge" => resolve_proposal(conn, "entity_merge_proposals", s(p, "id"), "rejected"),
        "accept_type" => accept_type(conn, p),
        "reject_type" => reject_type(conn, p),
        "accept_project_attach" => accept_project_attach(conn, p),
        "reject_project_attach" => {
            resolve_proposal(conn, "project_attach_proposals", s(p, "id"), "rejected")
        }
        "rename_space" => rename_space(conn, p),
        "rename_device" => rename_device(conn, p),
        "set_device_revoked" => set_device_revoked(conn, p),
        "confirm_note" => confirm_pending(conn, "atomic_notes", s(p, "noteId")),
        "confirm_relation" => confirm_pending(conn, "relations", s(p, "relationId")),
        "requeue_capture" => requeue_capture(conn, s(p, "captureId")),
        other => Err(CoreError::Storage(format!("unknown action type '{other}'"))),
    }
}

// ── payload helpers ──────────────────────────────────────────────────────────

fn s<'a>(p: &'a Map<String, Value>, key: &str) -> &'a str {
    p.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Blank string = null (the app encodes `x ?: ""`).
fn opt<'a>(p: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    Some(s(p, key).trim()).filter(|v| !v.is_empty())
}

fn is_true(p: &Map<String, Value>, key: &str) -> bool {
    s(p, key) == "true"
}

/// `datetime.now(timezone.utc).isoformat()` — the backend's note/fact stamps.
fn iso_now() -> String {
    format!(
        "{}+00:00",
        crate::decay::resolve_now(None).format("%Y-%m-%dT%H:%M:%S%.6f")
    )
}

fn ok(status: &str) -> Result<Value, CoreError> {
    Ok(json!({ "status": status }))
}

fn not_found() -> Result<Value, CoreError> {
    ok("not_found")
}

fn row_exists(conn: &Connection, table: &str, id: &str) -> Result<bool, CoreError> {
    let n: i64 = conn.query_row(
        &format!("SELECT COUNT(*) FROM {table} WHERE id = ?1"),
        params![id],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

/// SYN-89 — any fact write/lifecycle change invalidates the derived summary.
fn mark_fact_entity_stale(conn: &Connection, fact_id: &str) -> Result<(), CoreError> {
    conn.execute(
        "UPDATE entities SET summary_stale = 1 \
         WHERE id = (SELECT entity_id FROM facts WHERE id = ?1)",
        params![fact_id],
    )?;
    Ok(())
}

// ── facts ────────────────────────────────────────────────────────────────────

/// Port of `validation.py::record_and_apply_validation`.
fn validate_fact(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let fact_id = s(p, "id");
    let confirmed = is_true(p, "confirmed");
    let correction = opt(p, "correction");
    let device_id = opt(p, "deviceId");

    let pending = query_row_map(
        conn,
        "SELECT id, fact_data FROM pending_facts WHERE id = ?1",
        &[SqlV::from(fact_id.to_string())],
    )?;
    let Some(pending) = pending else {
        return not_found();
    };
    let mut fact_data: Value = match pending
        .get("fact_data")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str(raw).ok())
    {
        Some(v) => v,
        // Unfixable by retry — record nothing, drop the gesture (the HTTP
        // path would 404-and-stick; offline we must not wedge the queue).
        None => return ok("invalid_fact_data"),
    };

    // Append-only event — the durable record of the decision.
    conn.execute(
        "INSERT INTO validation_events \
         (id, fact_id, entity_canonical, predicate, value, confirmed, correction, device_id) \
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
        rusqlite::params![
            new_uuid(),
            fact_id,
            fact_data.get("entity_canonical").and_then(Value::as_str),
            fact_data.get("predicate").and_then(Value::as_str),
            json_scalar_to_sql(fact_data.get("value").unwrap_or(&Value::Null)),
            confirmed as i64,
            correction,
            device_id,
        ],
    )?;

    if !confirmed {
        conn.execute("DELETE FROM pending_facts WHERE id = ?1", params![fact_id])?;
        return Ok(json!({ "status": "rejected", "fact_id": fact_id }));
    }

    if let Some(correction) = correction {
        fact_data["value"] = json!(correction);
    }

    let entity_name = fact_data
        .get("entity_canonical")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    // SYN-41/112 — provenance traces back to the spawning capture; integer
    // legacy payloads keep their text form.
    let prov_id = match fact_data.get("source_inbox_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => Some(crate::routing::py_dumps(other)),
    };

    let entity_id = match find_existing_entity(conn, &entity_name, &[])? {
        Some(row) => row
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        None => {
            let id = new_uuid();
            conn.execute(
                "INSERT INTO entities (id, canonical_name, provenance_capture_id) \
                 VALUES (?1,?2,?3)",
                params![id, entity_name, prov_id],
            )?;
            id
        }
    };

    insert_fact(
        conn,
        &entity_id,
        fact_data.get("predicate").and_then(Value::as_str).unwrap_or(""),
        fact_data.get("value").cloned().unwrap_or(Value::Null),
        CONFIRMED_CONFIDENCE,
        fact_data.get("source_inbox_id").cloned().unwrap_or(Value::Null),
        fact_data
            .get("persistence_value")
            .and_then(Value::as_i64)
            .unwrap_or(3),
        prov_id,
        fact_data.get("category").cloned().unwrap_or(Value::Null),
    )?;
    conn.execute("DELETE FROM pending_facts WHERE id = ?1", params![fact_id])?;

    Ok(json!({ "status": "confirmed", "fact_id": fact_id, "entity": entity_name }))
}

/// Port of `_set_timestamp` for facts (`/fact/{id}/archive|unarchive|obsolete|
/// restore`) — plus the SYN-89 stale mark the Python helper applies to facts.
fn fact_timestamp(conn: &Connection, fact_id: &str, set_clause: &str) -> Result<Value, CoreError> {
    if !row_exists(conn, "facts", fact_id)? {
        return not_found();
    }
    conn.execute(
        &format!("UPDATE facts SET {set_clause} WHERE id = ?1"),
        params![fact_id],
    )?;
    mark_fact_entity_stale(conn, fact_id)?;
    ok("ok")
}

/// Port of `PATCH /fact/{fact_id}` — user correction is authoritative.
fn edit_fact(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let fact_id = s(p, "factId");
    let predicate = opt(p, "predicate");
    let value = opt(p, "value");
    if predicate.is_none() && value.is_none() {
        return ok("noop");
    }
    if !row_exists(conn, "facts", fact_id)? {
        return not_found();
    }
    let mut sets = vec!["confidence=1.0".to_string(), "last_confirmed=?1".to_string()];
    let mut binds: Vec<SqlV> = vec![SqlV::Text(iso_now())];
    if let Some(pred) = predicate {
        binds.push(SqlV::Text(pred.to_string()));
        sets.push(format!("predicate=?{}", binds.len()));
    }
    if let Some(val) = value {
        binds.push(SqlV::Text(val.to_string()));
        sets.push(format!("value=?{}", binds.len()));
    }
    binds.push(SqlV::Text(fact_id.to_string()));
    let sql = format!(
        "UPDATE facts SET {} WHERE id=?{}",
        sets.join(", "),
        binds.len()
    );
    conn.execute(&sql, params_from_iter(binds))?;
    mark_fact_entity_stale(conn, fact_id)?;
    ok("ok")
}

// ── entities ─────────────────────────────────────────────────────────────────

fn entity_timestamp(
    conn: &Connection,
    entity_id: &str,
    set_clause: &str,
) -> Result<Value, CoreError> {
    if !row_exists(conn, "entities", entity_id)? {
        return not_found();
    }
    conn.execute(
        &format!("UPDATE entities SET {set_clause} WHERE id = ?1"),
        params![entity_id],
    )?;
    ok("ok")
}

/// Type half of `PATCH /entity/{entity_id}`.
fn change_type(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let entity_id = s(p, "entityId");
    let Some(new_type) = opt(p, "type") else {
        return ok("noop");
    };
    if !row_exists(conn, "entities", entity_id)? {
        return not_found();
    }
    conn.execute(
        "UPDATE entities SET type = ?1 WHERE id = ?2",
        params![new_type, entity_id],
    )?;
    ok("ok")
}

/// Rename half of `PATCH /entity/{entity_id}` (SYN-82) — the old
/// canonical_name is kept as an alias so the resolver still matches it.
fn rename_entity(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let entity_id = s(p, "entityId");
    let Some(new_name) = opt(p, "name") else {
        return ok("noop");
    };
    let row = query_row_map(
        conn,
        "SELECT id, canonical_name, aliases FROM entities WHERE id = ?1",
        &[SqlV::from(entity_id.to_string())],
    )?;
    let Some(row) = row else {
        return not_found();
    };
    let old_name = row
        .get("canonical_name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if new_name == old_name {
        return ok("noop");
    }
    let mut aliases: Vec<String> = row
        .get("aliases")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default();
    if !old_name.is_empty() && !aliases.iter().any(|a| a == &old_name) {
        aliases.push(old_name);
    }
    aliases.retain(|a| a.to_lowercase() != new_name.to_lowercase());
    conn.execute(
        "UPDATE entities SET canonical_name = ?1, aliases = ?2 WHERE id = ?3",
        params![new_name, py_dumps(&json!(aliases)), entity_id],
    )?;
    ok("ok")
}

// ── relations ────────────────────────────────────────────────────────────────

/// Port of `POST /relation` (SYN-84) — user origin → confidence 1.0. The app
/// mints the id client-side so the offline replay is idempotent.
fn create_relation(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let Some(predicate) = opt(p, "predicate") else {
        return ok("noop");
    };
    for key in ["fromId", "toId"] {
        let n: i64 = conn.query_row(
            "SELECT COUNT(*) FROM entities WHERE id = ?1 AND merged_into_id IS NULL",
            params![s(p, key)],
            |r| r.get(0),
        )?;
        if n == 0 {
            return not_found();
        }
    }
    let rel_id = opt(p, "relationId")
        .map(str::to_string)
        .unwrap_or_else(new_uuid);
    conn.execute(
        "INSERT OR IGNORE INTO relations (id, entity_from, predicate, entity_to, confidence) \
         VALUES (?1,?2,?3,?4,1.0)",
        params![rel_id, s(p, "fromId"), predicate, s(p, "toId")],
    )?;
    ok("ok")
}

fn edit_relation(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let Some(predicate) = opt(p, "predicate") else {
        return ok("noop");
    };
    let changed = conn.execute(
        "UPDATE relations SET predicate = ?1, confidence = 1.0 WHERE id = ?2",
        params![predicate, s(p, "relationId")],
    )?;
    if changed == 0 {
        return not_found();
    }
    ok("ok")
}

fn delete_relation(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let changed = conn.execute(
        "DELETE FROM relations WHERE id = ?1",
        params![s(p, "relationId")],
    )?;
    if changed == 0 {
        return not_found();
    }
    ok("ok")
}

// ── atomic notes ─────────────────────────────────────────────────────────────

fn note_archived(
    conn: &Connection,
    note_id: &str,
    archived_at: Option<String>,
) -> Result<Value, CoreError> {
    let changed = conn.execute(
        "UPDATE atomic_notes SET archived_at = ?1 WHERE id = ?2",
        params![archived_at, note_id],
    )?;
    if changed == 0 {
        return not_found();
    }
    ok("ok")
}

/// `POST /atomic-note/{id}/reinforce` — 👍 « garder » on a fading note.
fn reinforce_note(conn: &Connection, note_id: &str) -> Result<Value, CoreError> {
    let now = crate::decay::resolve_now(None)
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let changed = conn.execute(
        "UPDATE atomic_notes SET last_reactivated_at = ?1, memory_strength = 1.0 WHERE id = ?2",
        params![now, note_id],
    )?;
    if changed == 0 {
        return not_found();
    }
    ok("ok")
}

/// `POST /atomic-note/{id}/date` — a dated task stays a task (SYN-23).
fn set_note_date(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let date = opt(p, "date");
    let recurring = (date.is_some() && is_true(p, "recurring")) as i64;
    let changed = conn.execute(
        "UPDATE atomic_notes SET event_date = ?1, event_recurring = ?2 WHERE id = ?3",
        params![date, recurring, s(p, "noteId")],
    )?;
    if changed == 0 {
        return not_found();
    }
    ok("ok")
}

/// Port of `POST /atomic-note/{id}/promote-to-project`. The LLM synthesis is
/// the host's post-commit job — the returned `synthesis` carries its inputs.
fn promote_note(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let note_id = s(p, "noteId");
    let note = query_row_map(
        conn,
        "SELECT id, title, content, provenance_capture_id FROM atomic_notes WHERE id = ?1",
        &[SqlV::from(note_id.to_string())],
    )?;
    let Some(note) = note else {
        return not_found();
    };
    let Some(capture_id) = note
        .get("provenance_capture_id")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty())
        .map(str::to_string)
    else {
        // « note sans capture source » — unfixable by retry, drop the gesture.
        return ok("no_source_capture");
    };
    let content = note
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let raw_name = opt(p, "name")
        .map(str::to_string)
        .or_else(|| {
            note.get("title")
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| content.clone());
    let canonical: String = raw_name.trim().chars().take(80).collect();
    if canonical.is_empty() {
        return ok("empty_name");
    }
    let synthesis =
        crate::routing::persist_project_entry(conn, &canonical, &content, &capture_id, true)?;
    conn.execute(
        "UPDATE atomic_notes SET archived_at = ?1 WHERE id = ?2",
        params![iso_now(), note_id],
    )?;
    Ok(json!({
        "status": "promoted",
        "note_id": note_id,
        "synthesis": {
            "project_id": synthesis.project_id,
            "project_name": synthesis.project_name,
            "entry_id": synthesis.entry_id,
            "entry_content": synthesis.entry_content,
            "entry_count": synthesis.entry_count,
        },
    }))
}

// ── proposals ────────────────────────────────────────────────────────────────

/// Shared reject/terminal-status path: `pending` → `status` + resolved_at.
fn resolve_proposal(
    conn: &Connection,
    table: &str,
    proposal_id: &str,
    status: &str,
) -> Result<Value, CoreError> {
    if !row_exists(conn, table, proposal_id)? {
        return not_found();
    }
    let changed = conn.execute(
        &format!(
            "UPDATE {table} SET status = ?1, resolved_at = CURRENT_TIMESTAMP \
             WHERE id = ?2 AND status = 'pending'"
        ),
        params![status, proposal_id],
    )?;
    if changed == 0 {
        return ok("skipped");
    }
    ok(status)
}

/// Port of `POST /merge-proposals/{id}/accept` + `_reroute_to_canonical`:
/// facts/relations repointed, `entities_mentioned` names swapped (SYN-42),
/// absorbed side soft-linked (`merged_into_id`, no DELETE — lineage kept).
fn accept_merge(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let proposal_id = s(p, "id");
    let canonical_id = s(p, "canonicalId");
    let proposal = query_row_map(
        conn,
        "SELECT id, candidate_entity_id, existing_entity_id, status \
         FROM entity_merge_proposals WHERE id = ?1",
        &[SqlV::from(proposal_id.to_string())],
    )?;
    let Some(proposal) = proposal else {
        return not_found();
    };
    if proposal.get("status").and_then(Value::as_str) != Some("pending") {
        return ok("skipped");
    }
    let candidate = proposal
        .get("candidate_entity_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    let existing = proposal
        .get("existing_entity_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    let absorbed_id = if canonical_id == candidate {
        existing
    } else if canonical_id == existing {
        candidate
    } else {
        return ok("skipped");
    };

    let canonical = query_row_map(
        conn,
        "SELECT id, canonical_name, aliases FROM entities WHERE id = ?1",
        &[SqlV::from(canonical_id.to_string())],
    )?;
    let absorbed = query_row_map(
        conn,
        "SELECT id, canonical_name FROM entities WHERE id = ?1",
        &[SqlV::from(absorbed_id.to_string())],
    )?;
    let (Some(canonical), Some(absorbed)) = (canonical, absorbed) else {
        return not_found();
    };
    let canonical_name = canonical
        .get("canonical_name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let absorbed_name = absorbed
        .get("canonical_name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    // Reroute the projections (`_reroute_to_canonical`).
    conn.execute(
        "UPDATE facts SET entity_id = ?1 WHERE entity_id = ?2",
        params![canonical_id, absorbed_id],
    )?;
    conn.execute(
        "UPDATE relations SET entity_from = ?1 WHERE entity_from = ?2",
        params![canonical_id, absorbed_id],
    )?;
    conn.execute(
        "UPDATE relations SET entity_to = ?1 WHERE entity_to = ?2",
        params![canonical_id, absorbed_id],
    )?;
    let notes = query_row_maps(
        conn,
        "SELECT id, entities_mentioned FROM atomic_notes WHERE entities_mentioned LIKE ?1",
        &[SqlV::from(format!("%\"{absorbed_name}\"%"))],
    )?;
    for note in notes {
        let Some(arr) = note
            .get("entities_mentioned")
            .and_then(Value::as_str)
            .and_then(|raw| serde_json::from_str::<Vec<String>>(raw).ok())
        else {
            continue;
        };
        if !arr.iter().any(|x| x == &absorbed_name) {
            continue;
        }
        let mut seen = std::collections::HashSet::new();
        let new_arr: Vec<String> = arr
            .into_iter()
            .map(|x| {
                if x == absorbed_name {
                    canonical_name.clone()
                } else {
                    x
                }
            })
            .filter(|x| seen.insert(x.clone()))
            .collect();
        conn.execute(
            "UPDATE atomic_notes SET entities_mentioned = ?1 WHERE id = ?2",
            params![
                py_dumps(&json!(new_arr)),
                note.get("id").and_then(Value::as_str)
            ],
        )?;
    }

    let mut aliases: Vec<String> = canonical
        .get("aliases")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default();
    if !aliases.iter().any(|a| a == &absorbed_name) {
        aliases.push(absorbed_name);
    }
    conn.execute(
        "UPDATE entities SET aliases = ?1 WHERE id = ?2",
        params![py_dumps(&json!(aliases)), canonical_id],
    )?;
    conn.execute(
        "UPDATE entities SET merged_into_id = ?1, merged_at = CURRENT_TIMESTAMP WHERE id = ?2",
        params![canonical_id, absorbed_id],
    )?;
    conn.execute(
        "UPDATE entity_merge_proposals SET status = 'accepted', \
         resolved_at = CURRENT_TIMESTAMP, resolved_canonical_id = ?1 WHERE id = ?2",
        params![canonical_id, proposal_id],
    )?;
    Ok(json!({
        "status": "accepted",
        "proposal_id": proposal_id,
        "canonical_id": canonical_id,
        "absorbed_id": absorbed_id,
    }))
}

/// Port of `POST /entity-type-proposals/{id}/accept` (SYN-58) — extend the
/// vocab, promote the pending candidate entity.
fn accept_type(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let proposal_id = s(p, "id");
    let proposal = query_row_map(
        conn,
        "SELECT id, status, proposed_type, candidate_entity_id \
         FROM entity_type_proposals WHERE id = ?1",
        &[SqlV::from(proposal_id.to_string())],
    )?;
    let Some(proposal) = proposal else {
        return not_found();
    };
    if proposal.get("status").and_then(Value::as_str) != Some("pending") {
        return ok("skipped");
    }
    let new_type = opt(p, "type")
        .map(str::to_string)
        .or_else(|| {
            proposal
                .get("proposed_type")
                .and_then(Value::as_str)
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
        })
        .unwrap_or_default();
    if new_type.is_empty() {
        return ok("skipped");
    }
    conn.execute(
        "INSERT OR IGNORE INTO active_entity_types (type, source) VALUES (?1, 'user')",
        params![new_type],
    )?;
    if let Some(candidate) = proposal
        .get("candidate_entity_id")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty())
    {
        conn.execute(
            "UPDATE entities SET type = ?1, status = 'active' WHERE id = ?2",
            params![new_type, candidate],
        )?;
    }
    conn.execute(
        "UPDATE entity_type_proposals SET status = 'accepted', resolved_at = CURRENT_TIMESTAMP \
         WHERE id = ?1",
        params![proposal_id],
    )?;
    Ok(json!({ "status": "accepted", "type": new_type }))
}

/// Port of `POST /entity-type-proposals/{id}/reject` — the pending candidate
/// entity is archived (drops out of default views).
fn reject_type(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let proposal_id = s(p, "id");
    let proposal = query_row_map(
        conn,
        "SELECT id, status, candidate_entity_id FROM entity_type_proposals WHERE id = ?1",
        &[SqlV::from(proposal_id.to_string())],
    )?;
    let Some(proposal) = proposal else {
        return not_found();
    };
    if proposal.get("status").and_then(Value::as_str) != Some("pending") {
        return ok("skipped");
    }
    if let Some(candidate) = proposal
        .get("candidate_entity_id")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty())
    {
        conn.execute(
            "UPDATE entities SET status = 'archived' WHERE id = ?1",
            params![candidate],
        )?;
    }
    conn.execute(
        "UPDATE entity_type_proposals SET status = 'rejected', resolved_at = CURRENT_TIMESTAMP \
         WHERE id = ?1",
        params![proposal_id],
    )?;
    ok("rejected")
}

/// Port of `POST /project-attach-proposals/{id}/accept`. The LLM synthesis is
/// the host's post-commit job — the returned `synthesis` carries its inputs.
fn accept_project_attach(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let proposal_id = s(p, "id");
    let proposal = query_row_map(
        conn,
        "SELECT p.id, p.status, p.capture_id, p.content, p.project_id, \
                e.canonical_name AS project_name \
         FROM project_attach_proposals p \
         LEFT JOIN entities e ON e.id = p.project_id \
         WHERE p.id = ?1",
        &[SqlV::from(proposal_id.to_string())],
    )?;
    let Some(proposal) = proposal else {
        return not_found();
    };
    if proposal.get("status").and_then(Value::as_str) != Some("pending") {
        return ok("skipped");
    }
    let Some(project_name) = proposal
        .get("project_name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
    else {
        // 410 on the backend — the target project entity is gone.
        return ok("project_gone");
    };
    let content = proposal
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let capture_id = proposal
        .get("capture_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let synthesis =
        crate::routing::persist_project_entry(conn, &project_name, &content, &capture_id, false)?;
    conn.execute(
        "UPDATE project_attach_proposals SET status = 'accepted', \
         resolved_at = CURRENT_TIMESTAMP WHERE id = ?1",
        params![proposal_id],
    )?;
    Ok(json!({
        "status": "accepted",
        "proposal_id": proposal_id,
        "synthesis": {
            "project_id": synthesis.project_id,
            "project_name": synthesis.project_name,
            "entry_id": synthesis.entry_id,
            "entry_content": synthesis.entry_content,
            "entry_count": synthesis.entry_count,
        },
    }))
}

// ── « À valider » : tâches + liens + retry capture (SYN-143) ─────────────────

/// Ports of `app.py::note_confirm` / `relation_confirm`: promote a
/// low-confidence row from review_status='pending' into the live view.
/// A peer may have validated it first — that replays as "skipped".
fn confirm_pending(conn: &Connection, table: &str, id: &str) -> Result<Value, CoreError> {
    let n = conn.execute(
        &format!(
            "UPDATE {table} SET review_status = 'confirmed' \
             WHERE id = ?1 AND review_status = 'pending'"
        ),
        params![id],
    )?;
    if n == 1 {
        return ok("confirmed");
    }
    if row_exists(conn, table, id)? {
        ok("skipped")
    } else {
        not_found()
    }
}

/// Port of `app.py::inbox_requeue` (SYN-77): put a failed capture back in the
/// queue for the next owned cycle.
fn requeue_capture(conn: &Connection, id: &str) -> Result<Value, CoreError> {
    let n = conn.execute(
        "UPDATE inbox SET status='queued', processed_at=NULL, error=NULL \
         WHERE id = ?1 AND status = 'failed'",
        params![id],
    )?;
    if n == 1 {
        return ok("queued");
    }
    if row_exists(conn, "inbox", id)? {
        ok("skipped")
    } else {
        not_found()
    }
}

// ── space + devices (SYN-139) ────────────────────────────────────────────────
// Mirrors of `PATCH /space` and `PATCH /device/{id}`. The HTTP guards (422 on
// blank name, 409 on self-revoke / revoking the weaver) become Ok statuses
// here: the UI blocks those gestures upfront, and a queued action must never
// wedge on a state a peer changed first.

/// Port of `app.py::space_patch`.
fn rename_space(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let Some(name) = opt(p, "name") else {
        return ok("invalid_name");
    };
    let n = conn.execute(
        "UPDATE space SET name = ?1 WHERE id = 'space'",
        params![name],
    )?;
    if n == 0 {
        return not_found();
    }
    ok("ok")
}

/// Port of the rename half of `app.py::device_patch`.
fn rename_device(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let Some(name) = opt(p, "name") else {
        return ok("invalid_name");
    };
    let n = conn.execute(
        "UPDATE devices SET name = ?1 WHERE device_id = ?2",
        params![name, s(p, "deviceId")],
    )?;
    if n == 0 {
        return not_found();
    }
    ok("ok")
}

/// Port of the revoke/restore half of `app.py::device_patch`.
fn set_device_revoked(conn: &Connection, p: &Map<String, Value>) -> Result<Value, CoreError> {
    let device_id = s(p, "deviceId");
    let revoked = is_true(p, "revoked");
    if revoked {
        if device_id == crate::sync::device_id(conn)? {
            return ok("rejected_self_revoke");
        }
        let owner: Option<String> = conn
            .query_row(
                "SELECT device_id FROM sync_owner WHERE id = 'owner'",
                [],
                |r| r.get(0),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        if owner.as_deref() == Some(device_id) {
            return ok("rejected_weaver");
        }
    }
    let n = conn.execute(
        if revoked {
            "UPDATE devices SET revoked_at = CURRENT_TIMESTAMP WHERE device_id = ?1"
        } else {
            "UPDATE devices SET revoked_at = NULL WHERE device_id = ?1"
        },
        params![device_id],
    )?;
    if n == 0 {
        return not_found();
    }
    ok("ok")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Storage;

    fn setup() -> (tempfile::TempDir, Connection) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("actions.db");
        Storage::open(path.to_str().unwrap()).unwrap(); // schema owner
        let conn = Connection::open(&path).unwrap();
        conn.pragma_update(None, "foreign_keys", false).unwrap();
        (dir, conn)
    }

    fn apply(conn: &Connection, action: &str, payload: Value) -> Value {
        apply_action(conn, action, &payload.to_string()).unwrap()
    }

    fn one<T: rusqlite::types::FromSql>(conn: &Connection, sql: &str, id: &str) -> T {
        conn.query_row(sql, params![id], |r| r.get(0)).unwrap()
    }

    #[test]
    fn pending_validation_and_requeue_actions() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name) VALUES ('e1', 'Alexis'), ('e2', 'Arkose')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO atomic_notes (id, content, kind) VALUES ('n1', 'réserver le créneau', 'task')",
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
             VALUES ('r1', 'e1', 'climbs_at', 'e2', 'pending')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO inbox (id, content, status, error) VALUES ('c1', 'x', 'failed', 'boom')",
            [],
        )
        .unwrap();

        // Confirm promotes pending → confirmed ; replays are skipped, ghosts not_found.
        assert_eq!(apply(&conn, "confirm_note", json!({"noteId": "n1"}))["status"], "confirmed");
        let rs: String = one(&conn, "SELECT review_status FROM atomic_notes WHERE id = ?1", "n1");
        assert_eq!(rs, "confirmed");
        assert_eq!(apply(&conn, "confirm_note", json!({"noteId": "n1"}))["status"], "skipped");
        assert_eq!(apply(&conn, "confirm_note", json!({"noteId": "ghost"}))["status"], "not_found");
        assert_eq!(
            apply(&conn, "confirm_relation", json!({"relationId": "r1"}))["status"],
            "confirmed"
        );
        let rs: String = one(&conn, "SELECT review_status FROM relations WHERE id = ?1", "r1");
        assert_eq!(rs, "confirmed");

        // Requeue resets a failed capture ; only failed rows are eligible.
        assert_eq!(apply(&conn, "requeue_capture", json!({"captureId": "c1"}))["status"], "queued");
        let (st, err): (String, Option<String>) = conn
            .query_row("SELECT status, error FROM inbox WHERE id = 'c1'", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(st, "queued");
        assert!(err.is_none());
        assert_eq!(apply(&conn, "requeue_capture", json!({"captureId": "c1"}))["status"], "skipped");
        assert_eq!(
            apply(&conn, "requeue_capture", json!({"captureId": "ghost"}))["status"],
            "not_found"
        );
    }

    #[test]
    fn space_and_device_actions_apply_the_backend_guards() {
        let (_dir, conn) = setup();
        let me: String = conn
            .query_row("SELECT v FROM sync_meta WHERE k = 'device_id'", [], |r| r.get(0))
            .unwrap();

        // No space founded yet → not_found, the queue moves on.
        let r = apply(&conn, "rename_space", json!({"name": "Notre mémoire"}));
        assert_eq!(r["status"], "not_found");
        conn.execute(
            "INSERT INTO space (id, space_id, name) VALUES ('space', 'sp-1', 'Ma mémoire')",
            [],
        )
        .unwrap();
        assert_eq!(apply(&conn, "rename_space", json!({"name": ""}))["status"], "invalid_name");
        assert_eq!(
            apply(&conn, "rename_space", json!({"name": "Notre mémoire"}))["status"],
            "ok"
        );
        let name: String = conn
            .query_row("SELECT name FROM space WHERE id = 'space'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(name, "Notre mémoire");

        conn.execute(
            "INSERT INTO devices (device_id, name, platform) VALUES \
             (?1, 'Pixel 9a', 'android'), ('dev-mac', 'Macmini', 'darwin')",
            params![me],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sync_owner (id, device_id, epoch) VALUES ('owner', ?1, 1)",
            params![me],
        )
        .unwrap();

        assert_eq!(
            apply(&conn, "rename_device", json!({"deviceId": "dev-mac", "name": "Mac du salon"}))
                ["status"],
            "ok"
        );
        let name: String = one(
            &conn,
            "SELECT name FROM devices WHERE device_id = ?1",
            "dev-mac",
        );
        assert_eq!(name, "Mac du salon");
        assert_eq!(
            apply(&conn, "rename_device", json!({"deviceId": "ghost", "name": "X"}))["status"],
            "not_found"
        );

        // A device cannot revoke itself, and the weaver must hand over first.
        let r = apply(
            &conn,
            "set_device_revoked",
            json!({"deviceId": me, "revoked": "true"}),
        );
        assert_eq!(r["status"], "rejected_self_revoke");
        conn.execute("UPDATE sync_owner SET device_id = 'dev-mac' WHERE id = 'owner'", [])
            .unwrap();
        let r = apply(
            &conn,
            "set_device_revoked",
            json!({"deviceId": "dev-mac", "revoked": "true"}),
        );
        assert_eq!(r["status"], "rejected_weaver");
        conn.execute("UPDATE sync_owner SET device_id = ?1 WHERE id = 'owner'", params![me])
            .unwrap();
        assert_eq!(
            apply(&conn, "set_device_revoked", json!({"deviceId": "dev-mac", "revoked": "true"}))
                ["status"],
            "ok"
        );
        let revoked: Option<String> = one(
            &conn,
            "SELECT revoked_at FROM devices WHERE device_id = ?1",
            "dev-mac",
        );
        assert!(revoked.is_some());
        // Restore is unguarded, like the backend.
        assert_eq!(
            apply(&conn, "set_device_revoked", json!({"deviceId": "dev-mac", "revoked": "false"}))
                ["status"],
            "ok"
        );
        let revoked: Option<String> = one(
            &conn,
            "SELECT revoked_at FROM devices WHERE device_id = ?1",
            "dev-mac",
        );
        assert!(revoked.is_none());
    }

    #[test]
    fn validate_fact_confirm_and_reject() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name) VALUES ('e1', 'Cici Huang')",
            [],
        )
        .unwrap();
        conn.execute(
            "UPDATE entities SET aliases = '[\"Cici\"]' WHERE id = 'e1'",
            [],
        )
        .unwrap();
        for id in ["p1", "p2"] {
            conn.execute(
                "INSERT INTO pending_facts (id, fact_data) VALUES (?1, ?2)",
                params![
                    id,
                    r#"{"entity_canonical": "Cici", "predicate": "works_at", "value": "Acme", "source_inbox_id": "c1", "persistence_value": 4, "category": "work"}"#
                ],
            )
            .unwrap();
        }

        // Confirm with a correction — alias-aware resolution, no duplicate shell.
        let r = apply(
            &conn,
            "validate_fact",
            json!({"id": "p1", "confirmed": "true", "correction": "Globex", "deviceId": "dev-a"}),
        );
        assert_eq!(r["status"], "confirmed");
        let entities: i64 = conn
            .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
            .unwrap();
        assert_eq!(entities, 1);
        let (value, confidence): (String, f64) = conn
            .query_row(
                "SELECT value, confidence FROM facts WHERE entity_id = 'e1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(value, "Globex");
        assert_eq!(confidence, CONFIRMED_CONFIDENCE);
        let stale: i64 = one(&conn, "SELECT summary_stale FROM entities WHERE id = ?1", "e1");
        assert_eq!(stale, 1);

        // Reject — event recorded, pending gone, no fact.
        let r = apply(&conn, "validate_fact", json!({"id": "p2", "confirmed": "false"}));
        assert_eq!(r["status"], "rejected");
        let pendings: i64 = conn
            .query_row("SELECT COUNT(*) FROM pending_facts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(pendings, 0);
        let events: i64 = conn
            .query_row("SELECT COUNT(*) FROM validation_events", [], |r| r.get(0))
            .unwrap();
        assert_eq!(events, 2);

        // Idempotent replay of a moot gesture.
        let r = apply(&conn, "validate_fact", json!({"id": "p1", "confirmed": "true"}));
        assert_eq!(r["status"], "not_found");
    }

    #[test]
    fn fact_lifecycle_and_edit_mark_summary_stale() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name) VALUES ('e1', 'Léa')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO facts (id, entity_id, predicate, value, confidence) \
             VALUES ('f1', 'e1', 'lives_in', 'Lyon', 0.9)",
            [],
        )
        .unwrap();

        assert_eq!(apply(&conn, "archive_fact", json!({"factId": "f1"}))["status"], "ok");
        let archived: Option<String> =
            one(&conn, "SELECT archived_at FROM facts WHERE id = ?1", "f1");
        assert!(archived.is_some());
        apply(&conn, "unarchive_fact", json!({"factId": "f1"}));
        let archived: Option<String> =
            one(&conn, "SELECT archived_at FROM facts WHERE id = ?1", "f1");
        assert!(archived.is_none());

        apply(
            &conn,
            "edit_fact",
            json!({"factId": "f1", "predicate": "lives_in", "value": "Paris"}),
        );
        let (value, confidence): (String, f64) = conn
            .query_row("SELECT value, confidence FROM facts WHERE id = 'f1'", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(value, "Paris");
        assert_eq!(confidence, 1.0);
        let stale: i64 = one(&conn, "SELECT summary_stale FROM entities WHERE id = ?1", "e1");
        assert_eq!(stale, 1);

        assert_eq!(
            apply(&conn, "obsolete_fact", json!({"factId": "missing"}))["status"],
            "not_found"
        );
    }

    #[test]
    fn rename_keeps_old_name_as_alias() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name) VALUES ('e1', 'Cici')",
            [],
        )
        .unwrap();
        apply(&conn, "rename_entity", json!({"entityId": "e1", "name": "Cici Huang"}));
        let (name, aliases): (String, String) = conn
            .query_row(
                "SELECT canonical_name, aliases FROM entities WHERE id = 'e1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(name, "Cici Huang");
        assert_eq!(aliases, r#"["Cici"]"#);
    }

    #[test]
    fn relations_crud_and_note_gestures() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name) VALUES ('e1', 'A'), ('e2', 'B')",
            [],
        )
        .unwrap();
        apply(
            &conn,
            "create_relation",
            json!({"relationId": "r1", "fromId": "e1", "predicate": "knows", "toId": "e2"}),
        );
        let (conf, status): (f64, String) = conn
            .query_row(
                "SELECT confidence, review_status FROM relations WHERE id = 'r1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(conf, 1.0);
        assert_eq!(status, "confirmed");
        apply(&conn, "edit_relation", json!({"relationId": "r1", "predicate": "cousin_of"}));
        apply(&conn, "delete_relation", json!({"relationId": "r1"}));
        let relations: i64 = conn
            .query_row("SELECT COUNT(*) FROM relations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(relations, 0);

        conn.execute(
            "INSERT INTO atomic_notes (id, content, kind) VALUES ('n1', 'faire le CV', 'task')",
            [],
        )
        .unwrap();
        apply(
            &conn,
            "set_note_date",
            json!({"noteId": "n1", "date": "2026-08-01", "recurring": "false"}),
        );
        let date: Option<String> =
            one(&conn, "SELECT event_date FROM atomic_notes WHERE id = ?1", "n1");
        assert_eq!(date.as_deref(), Some("2026-08-01"));
        apply(&conn, "reinforce_note", json!({"noteId": "n1"}));
        let strength: f64 =
            one(&conn, "SELECT memory_strength FROM atomic_notes WHERE id = ?1", "n1");
        assert_eq!(strength, 1.0);
        apply(&conn, "archive_note", json!({"noteId": "n1"}));
        let archived: Option<String> =
            one(&conn, "SELECT archived_at FROM atomic_notes WHERE id = ?1", "n1");
        assert!(archived.unwrap().contains("+00:00"));
    }

    #[test]
    fn accept_merge_reroutes_everything() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name) VALUES ('keep', 'Cici Huang'), ('gone', 'Cici')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO facts (id, entity_id, predicate, value) VALUES ('f1', 'gone', 'works_at', 'Acme')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO relations (id, entity_from, predicate, entity_to) VALUES ('r1', 'gone', 'knows', 'keep')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO atomic_notes (id, content, entities_mentioned) \
             VALUES ('n1', 'vu Cici', '[\"Cici\", \"Cici Huang\"]')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO entity_merge_proposals \
             (id, candidate_entity_id, existing_entity_id, similarity_score, similarity_reason) \
             VALUES ('m1', 'gone', 'keep', 0.92, 'substring')",
            [],
        )
        .unwrap();

        let r = apply(&conn, "accept_merge", json!({"id": "m1", "canonicalId": "keep"}));
        assert_eq!(r["status"], "accepted");
        assert_eq!(r["absorbed_id"], "gone");
        let fact_entity: String = one(&conn, "SELECT entity_id FROM facts WHERE id = ?1", "f1");
        assert_eq!(fact_entity, "keep");
        let rel_from: String = one(&conn, "SELECT entity_from FROM relations WHERE id = ?1", "r1");
        assert_eq!(rel_from, "keep");
        let mentioned: String =
            one(&conn, "SELECT entities_mentioned FROM atomic_notes WHERE id = ?1", "n1");
        assert_eq!(mentioned, r#"["Cici Huang"]"#); // swapped + deduped, order kept
        let merged_into: Option<String> =
            one(&conn, "SELECT merged_into_id FROM entities WHERE id = ?1", "gone");
        assert_eq!(merged_into.as_deref(), Some("keep"));
        let aliases: String = one(&conn, "SELECT aliases FROM entities WHERE id = ?1", "keep");
        assert_eq!(aliases, r#"["Cici"]"#);

        // Terminal: replaying skips.
        let r = apply(&conn, "accept_merge", json!({"id": "m1", "canonicalId": "keep"}));
        assert_eq!(r["status"], "skipped");
    }

    #[test]
    fn type_proposals_extend_vocab_and_promote() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO entities (id, canonical_name, status) VALUES ('e1', 'Tarte tatin', 'pending')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO entity_type_proposals (id, proposed_type, candidate_entity_id) \
             VALUES ('t1', 'recipe', 'e1')",
            [],
        )
        .unwrap();
        let r = apply(&conn, "accept_type", json!({"id": "t1", "type": ""}));
        assert_eq!(r["status"], "accepted");
        assert_eq!(r["type"], "recipe");
        let vocab: i64 = one(
            &conn,
            "SELECT COUNT(*) FROM active_entity_types WHERE type = ?1",
            "recipe",
        );
        assert_eq!(vocab, 1);
        let (etype, estatus): (String, String) = conn
            .query_row("SELECT type, status FROM entities WHERE id = 'e1'", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(etype, "recipe");
        assert_eq!(estatus, "active");
    }

    #[test]
    fn promote_note_creates_project_and_archives() {
        let (_dir, conn) = setup();
        conn.execute(
            "INSERT INTO inbox (id, content) VALUES ('c1', 'refaire la terrasse')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO atomic_notes (id, title, content, provenance_capture_id) \
             VALUES ('n1', 'Terrasse', 'refaire la terrasse', 'c1')",
            [],
        )
        .unwrap();
        let r = apply(&conn, "promote_note", json!({"noteId": "n1", "name": ""}));
        assert_eq!(r["status"], "promoted");
        assert_eq!(r["synthesis"]["project_name"], "Terrasse");
        let projects: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM entities WHERE type = 'project' AND canonical_name = 'Terrasse'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(projects, 1);
        let archived: Option<String> =
            one(&conn, "SELECT archived_at FROM atomic_notes WHERE id = ?1", "n1");
        assert!(archived.is_some());
    }
}
