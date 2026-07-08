//! SYN-19/68 — Ebbinghaus graceful forgetting (T5 port of `dream_cycle/decay.py`).
//!
//! `memory_strength = exp(-Δdays / τ)` where Δdays is the time since the row's
//! reactivation anchor (`atomic_notes.last_reactivated_at`, falling back to
//! `created_at`; `entities.last_mentioned` for SYN-68). The score is
//! **recomputed from elapsed time**, never decremented in place, so it is
//! independent of how often the decay job runs. Reactivation moves the anchor
//! toward now: full reset for a fresh mention (factor 1.0), partial for a
//! search hit (0 < factor < 1).
//!
//! All functions take the CALLER's connection so an open host transaction
//! wraps the writes (same rationale as `SqlConnection::insert_fact`).

use std::collections::HashSet;

use chrono::{Duration, NaiveDate, NaiveDateTime};
use rusqlite::{params, Connection};

use crate::embedder::CoreError;

const SQLITE_FMT: &str = "%Y-%m-%d %H:%M:%S";

/// τ default: `SYNAPSE_DECAY_TAU_DAYS` env, else 30 days (same as Python).
fn tau_default() -> f64 {
    std::env::var("SYNAPSE_DECAY_TAU_DAYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30.0)
}

fn now_naive_utc() -> NaiveDateTime {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    chrono::DateTime::from_timestamp(secs, 0)
        .map(|dt| dt.naive_utc())
        .unwrap_or_default()
}

/// Parse an optional now override ('YYYY-MM-DD HH:MM:SS', naive UTC) — hosts
/// inject a fixed clock in tests; None = system now.
pub(crate) fn resolve_now(now_sql: Option<&str>) -> NaiveDateTime {
    now_sql
        .and_then(|s| NaiveDateTime::parse_from_str(s.trim(), SQLITE_FMT).ok())
        .unwrap_or_else(now_naive_utc)
}

/// Tolerant SQLite timestamp parse — mirror of Python `decay._parse`: ISO 'T'
/// normalized, timezone/fraction suffixes dropped, bare DATE accepted
/// (`entities.last_mentioned`), anything else falls back to `now`.
fn parse_ts(ts: &str, now: NaiveDateTime) -> NaiveDateTime {
    let s = ts.trim().replace('T', " ");
    let s = s
        .split('+')
        .next()
        .unwrap_or("")
        .split('.')
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    NaiveDateTime::parse_from_str(&s, SQLITE_FMT)
        .ok()
        .or_else(|| {
            NaiveDate::parse_from_str(&s, "%Y-%m-%d")
                .ok()
                .and_then(|d| d.and_hms_opt(0, 0, 0))
        })
        .unwrap_or(now)
}

/// Anchor = first non-empty of (reactivated, created), like Python's
/// `r["last_reactivated_at"] or r["created_at"]`.
fn anchor(reactivated: Option<&str>, created: Option<&str>, now: NaiveDateTime) -> NaiveDateTime {
    let eff = reactivated
        .filter(|s| !s.trim().is_empty())
        .or_else(|| created.filter(|s| !s.trim().is_empty()));
    match eff {
        Some(ts) => parse_ts(ts, now),
        None => now,
    }
}

fn strength(base: NaiveDateTime, now: NaiveDateTime, tau: f64) -> f64 {
    let delta_days = ((now - base).num_seconds() as f64 / 86400.0).max(0.0);
    (-delta_days / tau).exp()
}

/// Recompute `memory_strength` for every atomic_note. Returns the count.
pub(crate) fn apply_decay(
    conn: &Connection,
    tau_days: Option<f64>,
    now: NaiveDateTime,
) -> Result<i64, CoreError> {
    let tau = tau_days.unwrap_or_else(tau_default);
    let rows: Vec<(String, Option<String>, Option<String>)> = {
        let mut stmt =
            conn.prepare("SELECT id, created_at, last_reactivated_at FROM atomic_notes")?;
        let mapped = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        mapped.collect::<Result<_, _>>()?
    };
    let n = rows.len() as i64;
    for (id, created, reactivated) in rows {
        let base = anchor(reactivated.as_deref(), created.as_deref(), now);
        conn.execute(
            "UPDATE atomic_notes SET memory_strength = ?1 WHERE id = ?2",
            params![strength(base, now, tau), id],
        )?;
    }
    Ok(n)
}

/// SYN-68: same law for entities, anchored on `last_mentioned`.
pub(crate) fn apply_entity_decay(
    conn: &Connection,
    tau_days: Option<f64>,
    now: NaiveDateTime,
) -> Result<i64, CoreError> {
    let tau = tau_days.unwrap_or_else(tau_default);
    let rows: Vec<(String, Option<String>, Option<String>)> = {
        let mut stmt = conn.prepare("SELECT id, created_at, last_mentioned FROM entities")?;
        let mapped = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        mapped.collect::<Result<_, _>>()?
    };
    let n = rows.len() as i64;
    for (id, created, mentioned) in rows {
        let base = anchor(mentioned.as_deref(), created.as_deref(), now);
        conn.execute(
            "UPDATE entities SET memory_strength = ?1 WHERE id = ?2",
            params![strength(base, now, tau), id],
        )?;
    }
    Ok(n)
}

/// Move `last_reactivated_at` toward now. factor ≥ 1.0 → full reset (mention);
/// 0 < factor < 1 → partial "light" bump (search hit). Returns the count touched.
pub(crate) fn reactivate_notes(
    conn: &Connection,
    note_ids: &[String],
    factor: f64,
    now: NaiveDateTime,
) -> Result<i64, CoreError> {
    let mut touched = 0i64;
    for nid in note_ids {
        let row: Option<(Option<String>, Option<String>)> = {
            let mut stmt = conn.prepare(
                "SELECT created_at, last_reactivated_at FROM atomic_notes WHERE id = ?1",
            )?;
            let first = stmt
                .query_map(params![nid], |r| Ok((r.get(0)?, r.get(1)?)))?
                .next()
                .transpose()?;
            first
        };
        let Some((created, reactivated)) = row else {
            continue;
        };
        let base = anchor(reactivated.as_deref(), created.as_deref(), now);
        let new = if factor >= 1.0 {
            now
        } else {
            base + Duration::seconds(((now - base).num_seconds() as f64 * factor) as i64)
        };
        conn.execute(
            "UPDATE atomic_notes SET last_reactivated_at = ?1 WHERE id = ?2",
            params![new.format(SQLITE_FMT).to_string(), nid],
        )?;
        touched += 1;
    }
    Ok(touched)
}

/// Strong reactivation of every note whose `entities_mentioned` JSON names one
/// of `entity_names` (the routing path calls this on every fresh mention).
pub(crate) fn reactivate_notes_for_entities(
    conn: &Connection,
    entity_names: &[String],
    now: NaiveDateTime,
) -> Result<i64, CoreError> {
    let mut ids: HashSet<String> = HashSet::new();
    for name in entity_names {
        if name.is_empty() {
            continue;
        }
        let pattern = format!("%\"{name}\"%");
        let mut stmt =
            conn.prepare("SELECT id FROM atomic_notes WHERE entities_mentioned LIKE ?1")?;
        let rows = stmt
            .query_map(params![pattern], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        ids.extend(rows);
    }
    let ids: Vec<String> = ids.into_iter().collect();
    reactivate_notes(conn, &ids, 1.0, now)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(s, SQLITE_FMT).unwrap()
    }

    #[test]
    fn parse_is_tolerant() {
        let now = ts("2026-05-31 12:00:00");
        assert_eq!(parse_ts("2026-05-01 08:30:00", now), ts("2026-05-01 08:30:00"));
        assert_eq!(parse_ts("2026-05-01T08:30:00+00:00", now), ts("2026-05-01 08:30:00"));
        assert_eq!(parse_ts("2026-05-01 08:30:00.123456", now), ts("2026-05-01 08:30:00"));
        assert_eq!(parse_ts("2026-05-01", now), ts("2026-05-01 00:00:00"));
        assert_eq!(parse_ts("garbage", now), now);
    }

    #[test]
    fn strength_matches_ebbinghaus() {
        let now = ts("2026-05-31 12:00:00");
        let month_ago = ts("2026-05-01 12:00:00");
        let s = strength(month_ago, now, 30.0);
        assert!((s - (-1.0f64).exp()).abs() < 1e-9, "30d @ τ=30 => e^-1, got {s}");
        assert!((strength(now, now, 30.0) - 1.0).abs() < 1e-12);
        // future anchor clamps to 0 elapsed
        assert!((strength(ts("2026-06-30 12:00:00"), now, 30.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn partial_bump_moves_anchor_halfway() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE atomic_notes (id TEXT PRIMARY KEY, created_at TEXT, \
             last_reactivated_at TEXT, memory_strength REAL, entities_mentioned TEXT)",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO atomic_notes (id, created_at) VALUES ('n1', '2026-05-01 12:00:00')",
            [],
        )
        .unwrap();
        let now = ts("2026-05-31 12:00:00");
        reactivate_notes(&conn, &["n1".into()], 0.5, now).unwrap();
        let moved: String = conn
            .query_row("SELECT last_reactivated_at FROM atomic_notes WHERE id='n1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(moved, "2026-05-16 12:00:00");
    }
}
