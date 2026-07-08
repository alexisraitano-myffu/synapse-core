//! SYN-23 — weekly digest (T5 port of `dream_cycle/digest.py`).
//!
//! One durable note per ISO week condensing the past week AND the week ahead:
//! retrospective (new entities/facts/notes + "tendances"), forward-looking
//! (dated events & tasks within 7 days, incl. recurring birthdays — SYN-97 —
//! and open undated tasks). Idempotent per week: re-running overwrites.
//!
//! Split mirrors the host call sites:
//! - [`gather_week`] is pure SQL on the CALLER's connection (offline-testable,
//!   exposed via `SqlConnection::gather_week`);
//! - [`Brain::summarize_digest`] renders French markdown via the classifier's
//!   HTTP path — prompt is DATA (`prompts/digest.md`), timeless rule inside;
//! - [`Brain::write_digest_note`] persists note + vector on the Brain's own
//!   connection — call it OUTSIDE any host transaction.

use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime};
use rusqlite::types::Value as SqlV;
use rusqlite::{params, Connection};
use serde_json::{json, Value};

use crate::embedder::CoreError;
use crate::llm::{load_prompt, post_messages, response_text, LlmConfig};
use crate::routing::{new_uuid, query_row_maps, Brain};

// Bounds so a busy week doesn't blow up the prompt — the digest is a summary,
// not an exhaustive log.
const MAX_ENTITIES: i64 = 25;
const MAX_FACTS: i64 = 25;
const MAX_NOTES: i64 = 25;
const MAX_TRENDS: i64 = 8;
const MAX_TASKS: i64 = 20;

// SYN-97 — birthday facts surfaced under « à venir » (recurring yearly,
// month-day match). Subset of SINGLE_VALUED_PREDICATES denoting a birth date.
const BIRTHDAY_PREDICATES: [&str; 4] =
    ["has_birthday", "birthday", "born_on", "date_of_birth"];

const SQLITE_FMT: &str = "%Y-%m-%d %H:%M:%S";
const ISO_FMT: &str = "%Y-%m-%d";

/// Collect the retrospective (past `days`) and forward-looking (next `days`)
/// material for the digest. No LLM call — safe to unit-test offline.
pub(crate) fn gather_week(
    conn: &Connection,
    now: NaiveDateTime,
    days: i64,
) -> Result<Value, CoreError> {
    let since = (now - Duration::days(days)).format(SQLITE_FMT).to_string();
    let today = now.date();
    let horizon = today + Duration::days(days);

    let new_entities = query_row_maps(
        conn,
        "SELECT canonical_name, type FROM entities \
         WHERE created_at >= ?1 AND (status IS NULL OR status = 'active') \
         AND merged_into_id IS NULL \
         ORDER BY created_at DESC LIMIT ?2",
        &[SqlV::from(since.clone()), SqlV::from(MAX_ENTITIES)],
    )?;

    let new_facts = query_row_maps(
        conn,
        "SELECT e.canonical_name AS entity, f.predicate, f.value \
         FROM facts f JOIN entities e ON e.id = f.entity_id \
         WHERE f.created_at >= ?1 AND f.archived_at IS NULL AND f.obsoleted_at IS NULL \
         ORDER BY f.created_at DESC LIMIT ?2",
        &[SqlV::from(since.clone()), SqlV::from(MAX_FACTS)],
    )?;

    let validated_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM validation_events WHERE confirmed = 1 AND created_at >= ?1",
        params![since],
        |r| r.get(0),
    )?;

    let new_notes = query_row_maps(
        conn,
        "SELECT title, content, kind FROM atomic_notes \
         WHERE created_at >= ?1 AND archived_at IS NULL \
         AND kind IN ('note', 'task', 'event') AND review_status != 'pending' \
         ORDER BY created_at DESC LIMIT ?2",
        &[SqlV::from(since.clone()), SqlV::from(MAX_NOTES)],
    )?;

    // Tendances: entities reactivated (mentioned) over the window, busiest first.
    let window_start = (today - Duration::days(days)).format(ISO_FMT).to_string();
    let trends = query_row_maps(
        conn,
        "SELECT canonical_name, type, mention_count FROM entities \
         WHERE last_mentioned >= ?1 AND (status IS NULL OR status = 'active') \
         AND merged_into_id IS NULL \
         ORDER BY mention_count DESC, last_mentioned DESC LIMIT ?2",
        &[SqlV::from(window_start), SqlV::from(MAX_TRENDS)],
    )?;

    let captures: i64 = conn.query_row(
        "SELECT COUNT(*) FROM inbox WHERE created_at >= ?1",
        params![since],
        |r| r.get(0),
    )?;

    // Forward-looking — dated events AND dated tasks (SYN-23) not archived,
    // within the horizon. Recurring (birthdays) compared on month-day;
    // one-shots on the absolute date. Filtered in code so the year-boundary
    // case stays correct.
    let dated_raw = query_row_maps(
        conn,
        "SELECT title, content, kind, event_date, event_recurring FROM atomic_notes \
         WHERE kind IN ('event', 'task') AND archived_at IS NULL AND event_date IS NOT NULL \
         AND review_status != 'pending'",
        &[],
    )?;
    let mut upcoming: Vec<Value> = Vec::new();
    for ev in &dated_raw {
        let recurring = ev
            .get("event_recurring")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            != 0;
        let date_str = ev.get("event_date").and_then(Value::as_str).unwrap_or("");
        let Some(occ) = next_occurrence(date_str, recurring, today) else {
            continue;
        };
        if occ < today || occ > horizon {
            continue;
        }
        upcoming.push(json!({
            "title": ev.get("title").cloned().unwrap_or(Value::Null),
            "content": ev.get("content").cloned().unwrap_or(Value::Null),
            "kind": ev.get("kind").cloned().unwrap_or(Value::Null),
            "date": occ.format(ISO_FMT).to_string(),
            "recurring": recurring,
        }));
    }

    // SYN-97 — birthdays live as `has_birthday` facts, not (only) as event
    // notes. Treat them as recurring (yearly month-day) and dedup against any
    // event note that already names the same person on the same day (the
    // cycle emits BOTH for a birthday).
    let birthday_raw = query_row_maps(
        conn,
        "SELECT e.canonical_name AS entity, f.value AS value \
         FROM facts f JOIN entities e ON e.id = f.entity_id \
         WHERE f.predicate IN (?1, ?2, ?3, ?4) \
         AND f.archived_at IS NULL AND f.obsoleted_at IS NULL \
         AND (e.status IS NULL OR e.status = 'active') AND e.merged_into_id IS NULL",
        &BIRTHDAY_PREDICATES.map(|p| SqlV::from(p.to_string())),
    )?;
    for b in &birthday_raw {
        let value = b.get("value").and_then(Value::as_str).unwrap_or("");
        let Some(occ) = next_occurrence(value, true, today) else {
            continue;
        };
        if occ < today || occ > horizon {
            continue;
        }
        let iso = occ.format(ISO_FMT).to_string();
        let name = b
            .get("entity")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let needle = name.to_lowercase();
        let already = !name.is_empty()
            && upcoming.iter().any(|ev| {
                ev["date"].as_str() == Some(iso.as_str())
                    && format!(
                        "{} {}",
                        ev["title"].as_str().unwrap_or(""),
                        ev["content"].as_str().unwrap_or("")
                    )
                    .to_lowercase()
                    .contains(&needle)
            });
        if already {
            continue;
        }
        upcoming.push(json!({
            "title": if name.is_empty() {
                "Anniversaire".to_string()
            } else {
                format!("Anniversaire de {name}")
            },
            "content": Value::Null,
            "kind": "birthday",
            "date": iso,
            "recurring": true,
        }));
    }
    upcoming.sort_by(|a, b| a["date"].as_str().cmp(&b["date"].as_str()));

    // Open tasks WITHOUT a date (dated ones already surface under « à venir »).
    let open_tasks = query_row_maps(
        conn,
        "SELECT title, content FROM atomic_notes \
         WHERE kind = 'task' AND archived_at IS NULL AND event_date IS NULL \
         AND review_status != 'pending' \
         ORDER BY created_at DESC LIMIT ?1",
        &[SqlV::from(MAX_TASKS)],
    )?;

    let week_start = today - Duration::days(today.weekday().num_days_from_monday() as i64);
    Ok(json!({
        "week_start": week_start.format(ISO_FMT).to_string(),
        "generated_at": today.format(ISO_FMT).to_string(),
        "days": days,
        "counts": {
            "captures": captures,
            "new_entities": new_entities.len(),
            "new_facts": new_facts.len(),
            "validated_facts": validated_count,
            "new_notes": new_notes.len(),
        },
        "new_entities": new_entities,
        "new_facts": new_facts,
        "new_notes": new_notes,
        "trends": trends,
        "upcoming_events": upcoming,
        "open_tasks": open_tasks,
    }))
}

/// Resolve an event's next concrete date. One-shots return their absolute
/// date; recurring ones this year's (or next year's) matching month-day.
/// 29 Feb on a non-leap year resolves to 1 Mar (Python's ValueError branch).
pub(crate) fn next_occurrence(
    event_date: &str,
    recurring: bool,
    today: NaiveDate,
) -> Option<NaiveDate> {
    let head: String = event_date.trim().chars().take(10).collect();
    let d = NaiveDate::parse_from_str(&head, ISO_FMT).ok()?;
    if !recurring {
        return Some(d);
    }
    let this_year = NaiveDate::from_ymd_opt(today.year(), d.month(), d.day())
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(today.year(), 3, 1).unwrap());
    if this_year >= today {
        Some(this_year)
    } else {
        NaiveDate::from_ymd_opt(today.year() + 1, this_year.month(), this_year.day())
            .or_else(|| NaiveDate::from_ymd_opt(today.year() + 1, 3, 1))
    }
}

/// String-typed variant for the bindings (hosts pass/get ISO dates).
pub fn next_occurrence_str(event_date: &str, recurring: bool, today: &str) -> Option<String> {
    let today = NaiveDate::parse_from_str(today.trim(), ISO_FMT).ok()?;
    next_occurrence(event_date, recurring, today).map(|d| d.format(ISO_FMT).to_string())
}

impl Brain {
    /// Render the gathered week into French markdown via the LLM. Prompt =
    /// `prompts/digest.md`; the payload is the week JSON pretty-printed like
    /// Python's `json.dumps(indent=2)` (preserve_order keeps the key order).
    pub fn summarize_digest(&self, week: &Value, config: &LlmConfig) -> Result<String, CoreError> {
        let system = load_prompt(&config.prompts_dir, "digest.md")?;
        let payload = serde_json::to_string_pretty(week)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let week_start = week["week_start"].as_str().unwrap_or("");
        let params_json = json!({
            "model": config.model,
            "max_tokens": 1400,
            "system": system,
            "messages": [{"role": "user", "content": format!(
                "Matière de la semaine (semaine du {week_start}) :\n\n{payload}")}],
        });
        let body = post_messages(config, &params_json)?;
        let text = response_text(&body);
        if text.is_empty() {
            return Err(CoreError::LlmContent("digest: réponse Haiku vide".into()));
        }
        Ok(text)
    }

    /// Store the digest as an atomic_note (kind='digest'), replacing any
    /// existing digest for the same ISO week, then (re)write its vector so
    /// search surfaces it. Runs on the Brain's own connection — call OUTSIDE
    /// any host transaction. Returns the note id.
    pub fn write_digest_note(&self, week: &Value, markdown: &str) -> Result<String, CoreError> {
        let week_start = week["week_start"].as_str().unwrap_or("");
        let title = format!("Digest — semaine du {week_start}");
        let mut names: Vec<&str> = week["new_entities"]
            .as_array()
            .map(|a| a.iter().filter_map(|e| e["canonical_name"].as_str()).collect())
            .unwrap_or_default();
        if let Some(trends) = week["trends"].as_array() {
            for t in trends {
                if let Some(n) = t["canonical_name"].as_str() {
                    if !names.contains(&n) {
                        names.push(n);
                    }
                }
            }
        }
        let counts = &week["counts"];
        let summary = format!(
            "Digest hebdo : {} captures, {} entités, {} faits.",
            counts["captures"].as_i64().unwrap_or(0),
            counts["new_entities"].as_i64().unwrap_or(0),
            counts["new_facts"].as_i64().unwrap_or(0),
        );

        let note_id = new_uuid();
        let stale: Vec<String>;
        {
            let conn = self.storage.lock()?;
            stale = {
                let mut stmt = conn
                    .prepare("SELECT id FROM atomic_notes WHERE kind = 'digest' AND title = ?1")?;
                let rows = stmt.query_map(params![title], |r| r.get::<_, String>(0))?;
                rows.collect::<Result<_, _>>()?
            };
            let tx = conn.unchecked_transaction()?;
            for nid in &stale {
                tx.execute("DELETE FROM atomic_notes WHERE id = ?1", params![nid])?;
            }
            tx.execute(
                "INSERT INTO atomic_notes \
                 (id, title, content, summary, entities_mentioned, memory_strength, kind) \
                 VALUES (?1,?2,?3,?4,?5,?6,'digest')",
                params![
                    note_id,
                    title,
                    markdown,
                    summary,
                    serde_json::to_string(&names).unwrap_or_else(|_| "[]".into()),
                    1.0f64
                ],
            )?;
            tx.commit()?;
        }
        // Vector rows after the note commit (own connection): stale ones out,
        // the new digest in — best-effort, like Python.
        for nid in &stale {
            let _ = self.storage.delete_note_vector(nid);
        }
        if let Some(vec) = self.embed(&format!("{title}\n{markdown}")) {
            let _ = self.storage.upsert_note_vector(&note_id, &vec);
        }
        Ok(note_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, ISO_FMT).unwrap()
    }

    #[test]
    fn next_occurrence_like_python() {
        let today = d("2026-06-17");
        // Already passed this year → next year; upcoming stays this year.
        assert_eq!(next_occurrence("1990-01-10", true, today), Some(d("2027-01-10")));
        assert_eq!(next_occurrence("1990-12-25", true, today), Some(d("2026-12-25")));
        // One-shot returns its absolute date untouched.
        assert_eq!(next_occurrence("2026-06-20", false, today), Some(d("2026-06-20")));
        assert_eq!(next_occurrence("garbage", true, today), None);
        // 29 Feb on a non-leap year → 1 Mar.
        assert_eq!(
            next_occurrence("2024-02-29", true, d("2025-01-15")),
            Some(d("2025-03-01"))
        );
    }

    #[test]
    fn gather_week_buckets_and_dedups_birthdays() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("digest.db");
        let _storage = crate::Storage::open(path.to_str().unwrap()).unwrap();
        let c = crate::connect(path.to_str().unwrap()).unwrap();
        for sql in [
            // Recent entity vs one outside the retrospective window.
            "INSERT INTO entities (id, type, canonical_name, mention_count, persistence_value, \
             last_mentioned, created_at, status) \
             VALUES ('cici','person','Cici',4,3,'2026-06-16','2026-06-15 10:00:00','active')",
            "INSERT INTO entities (id, type, canonical_name, mention_count, persistence_value, \
             last_mentioned, created_at, status) \
             VALUES ('old','person','Ancien',2,3,'2026-05-10','2026-05-01 10:00:00','active')",
            // Birthday event note in range + a far event out of range.
            "INSERT INTO atomic_notes (id, title, content, kind, event_date, event_recurring, \
             created_at) VALUES ('n1','Anniversaire de Cici','fête','event','1990-06-19',1,\
             '2026-06-15 10:00:00')",
            "INSERT INTO atomic_notes (id, title, content, kind, event_date, created_at) \
             VALUES ('n2','Conf lointaine','x','event','2026-07-30','2026-06-15 10:00:00')",
            // has_birthday fact for the SAME person/day — must be deduped.
            "INSERT INTO facts (id, entity_id, predicate, value, confidence, created_at) \
             VALUES ('f1','cici','has_birthday','1990-06-19',0.9,'2026-06-15 10:00:00')",
            // Open undated task.
            "INSERT INTO atomic_notes (id, title, content, kind, created_at) \
             VALUES ('n3','Refondre le design','todo','task','2026-06-15 10:00:00')",
        ] {
            c.execute(sql, &[]).unwrap();
        }

        let week = c.gather_week(Some("2026-06-17 12:00:00"), 7).unwrap();
        assert_eq!(week["week_start"], "2026-06-15");
        assert_eq!(week["counts"]["new_entities"], 1); // 'Ancien' filtered by window
        assert_eq!(week["counts"]["new_facts"], 1);
        let names: Vec<&str> = week["new_entities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["canonical_name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["Cici"]);
        assert!(week["trends"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["canonical_name"] == "Cici"));
        // One upcoming entry only: the event note (fact deduped, far conf excluded).
        let ups = week["upcoming_events"].as_array().unwrap();
        assert_eq!(ups.len(), 1, "{ups:?}");
        assert_eq!(ups[0]["kind"], "event");
        assert_eq!(ups[0]["date"], "2026-06-19");
        assert_eq!(week["open_tasks"].as_array().unwrap().len(), 1);
    }
}
