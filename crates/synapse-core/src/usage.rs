//! SYN-160 — what every LLM call actually consumed.
//!
//! One row per call, never aggregated at write time: a total can always be
//! recomputed from rows, the reverse is not true. The rows replicate like any
//! other table (see `sync::synced_tables`), keyed by uuid — a row seen twice
//! is the SAME row and upserts onto itself, so a two-device space sums each
//! device's real spending without double-counting.
//!
//! Note on what this table is NOT: it is not a mirror of the provider's
//! billing. Two devices routing the same capture during a no-sync window make
//! two real API calls and are really billed twice; both rows are kept on
//! purpose. `SYN-133` collapses the derived twins, not the money spent.

use rusqlite::{params, Connection};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::embedder::CoreError;
use crate::llm::{LlmConfig, LlmProvider};
use crate::sync;

/// Tokens billed by one call, in the provider-agnostic Anthropic shape.
///
/// The four buckets are priced differently (cache writes cost more than plain
/// input, cache reads cost a tenth), so collapsing them at write time would
/// destroy the only information that makes a price computable later.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LlmUsage {
    pub input: i64,
    pub output: i64,
    pub cache_write: i64,
    pub cache_read: i64,
}

impl LlmUsage {
    /// Read the `usage` block off a response body. Absent fields read 0: an
    /// OpenAI-compatible endpoint reports no cache buckets, and the on-device
    /// path reports nothing at all.
    pub(crate) fn from_body(body: &Value) -> Self {
        let u = &body["usage"];
        let n = |k: &str| u[k].as_i64().unwrap_or(0);
        Self {
            input: n("input_tokens"),
            output: n("output_tokens"),
            cache_write: n("cache_creation_input_tokens"),
            cache_read: n("cache_read_input_tokens"),
        }
    }

    /// Sum across the retries of one logical call — a retried call is billed
    /// twice, so the row must carry both attempts.
    pub(crate) fn add(self, other: Self) -> Self {
        Self {
            input: self.input + other.input,
            output: self.output + other.output,
            cache_write: self.cache_write + other.cache_write,
            cache_read: self.cache_read + other.cache_read,
        }
    }

    pub fn total(&self) -> i64 {
        self.input + self.output + self.cache_write + self.cache_read
    }
}

/// Which pass of the cycle spent the tokens.
///
/// Every call site names its own: the expensive passes are NOT the classifier
/// (the digest and the project syntheses carry far longer prompts and outputs),
/// so a measurement that only covered `classify` would understate the real
/// cost — which is the whole point of the ticket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Classify,
    Resummarize,
    ProjectSummary,
    Digest,
    Resource,
}

impl Op {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Op::Classify => "classify",
            Op::Resummarize => "resummarize",
            Op::ProjectSummary => "project_summary",
            Op::Digest => "digest",
            Op::Resource => "resource",
        }
    }
}

fn provider_str(p: LlmProvider) -> &'static str {
    match p {
        LlmProvider::Anthropic => "anthropic",
        LlmProvider::OpenAiCompatible => "openai",
        LlmProvider::Local => "local",
    }
}

/// Persist one call. Best-effort by design: a bookkeeping failure must never
/// abort a cycle that has already spent the money and produced its result.
pub(crate) fn record(conn: &Connection, config: &LlmConfig, op: Op, usage: LlmUsage) {
    // The on-device path bills nothing and reports nothing; a zero row would
    // only add sync traffic. A metered call with a zero usage block (a
    // provider that omits it) is still worth a row: it says a call happened.
    if config.provider == LlmProvider::Local {
        return;
    }
    let device = sync::device_id(conn).unwrap_or_default();
    let res = conn.execute(
        "INSERT INTO llm_usage \
           (id, occurred_at, day, device_id, provider, model, operation, \
            input_tokens, output_tokens, cache_write_tokens, cache_read_tokens) \
         VALUES (?1, CURRENT_TIMESTAMP, date('now'), ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            Uuid::new_v4().to_string(),
            device,
            provider_str(config.provider),
            config.model,
            op.as_str(),
            usage.input,
            usage.output,
            usage.cache_write,
            usage.cache_read,
        ],
    );
    if let Err(e) = res {
        // Same policy as the rest of the core's telemetry: log, never fail the
        // caller. Losing one accounting row is cheaper than losing a cycle.
        eprintln!("[usage] could not record {} call: {e}", op.as_str());
    }
}

/// Consumption for `month` (`YYYY-MM`), split by operation, plus the whole
/// history's earliest day so the host can tell whether a projection is worth
/// showing at all.
///
/// The core deliberately returns TOKENS ONLY, never a price. Converting to
/// money needs the tariff of the day, who holds the key, and whether the user
/// is the one paying — host concerns, and getting them wrong would print a
/// figure worse than printing none.
pub fn usage_summary(conn: &Connection, month: &str) -> Result<Value, CoreError> {
    let like = format!("{month}%");
    let mut stmt = conn.prepare(
        "SELECT operation, model, \
                SUM(input_tokens), SUM(output_tokens), \
                SUM(cache_write_tokens), SUM(cache_read_tokens), COUNT(*) \
         FROM llm_usage WHERE day LIKE ?1 \
         GROUP BY operation, model ORDER BY operation, model",
    )?;
    let rows = stmt
        .query_map(params![like], |r| {
            Ok(json!({
                "operation": r.get::<_, String>(0)?,
                "model": r.get::<_, String>(1)?,
                "input_tokens": r.get::<_, i64>(2)?,
                "output_tokens": r.get::<_, i64>(3)?,
                "cache_write_tokens": r.get::<_, i64>(4)?,
                "cache_read_tokens": r.get::<_, i64>(5)?,
                "calls": r.get::<_, i64>(6)?,
            }))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    // Captures actually processed this month — the unit a user thinks in
    // ("I wrote 40 notes"), which tokens alone never convey.
    let captures: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM inbox WHERE processed_at IS NOT NULL \
             AND substr(processed_at, 1, 7) = ?1",
            params![month],
            |r| r.get(0),
        )
        .unwrap_or(0);

    // Days with at least one call, this month: a projection over 2 days of
    // history is noise, and the host needs to know before drawing one.
    let days_with_data: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT day) FROM llm_usage WHERE day LIKE ?1",
            params![like],
            |r| r.get(0),
        )
        .unwrap_or(0);

    let first_day: Option<String> = conn
        .query_row("SELECT MIN(day) FROM llm_usage", [], |r| r.get(0))
        .unwrap_or(None);

    Ok(json!({
        "month": month,
        "by_operation": rows,
        "captures_processed": captures,
        "days_with_data": days_with_data,
        "first_day": first_day,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::init_schema;

    fn db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn
    }

    fn cfg(provider: LlmProvider) -> LlmConfig {
        LlmConfig {
            model: "claude-haiku-4-5-20251001".into(),
            api_key: String::new(),
            provider,
            base_url: None,
            fuel_token: None,
            prompts_dir: String::new(),
            today: "2026-07-28".into(),
            local: None,
        }
    }

    #[test]
    fn reads_the_four_token_buckets() {
        let body = json!({"usage": {
            "input_tokens": 333, "output_tokens": 404,
            "cache_creation_input_tokens": 4929, "cache_read_input_tokens": 0,
        }});
        let u = LlmUsage::from_body(&body);
        assert_eq!((u.input, u.output, u.cache_write, u.cache_read), (333, 404, 4929, 0));
        assert_eq!(u.total(), 5666);
    }

    #[test]
    fn missing_usage_block_reads_zero_not_error() {
        let u = LlmUsage::from_body(&json!({"content": [{"text": "hi"}]}));
        assert_eq!(u, LlmUsage::default());
    }

    #[test]
    fn retries_are_summed_because_both_attempts_are_billed() {
        let a = LlmUsage { input: 10, output: 5, cache_write: 0, cache_read: 100 };
        let b = LlmUsage { input: 10, output: 7, cache_write: 0, cache_read: 100 };
        assert_eq!(a.add(b).total(), 232);
    }

    #[test]
    fn records_one_row_per_call_and_groups_by_operation() {
        let conn = db();
        let c = cfg(LlmProvider::Anthropic);
        record(&conn, &c, Op::Classify, LlmUsage { input: 333, output: 404, cache_write: 4929, cache_read: 0 });
        record(&conn, &c, Op::Classify, LlmUsage { input: 300, output: 400, cache_write: 0, cache_read: 4929 });
        record(&conn, &c, Op::Digest, LlmUsage { input: 8000, output: 1200, cache_write: 0, cache_read: 0 });

        let month: String = conn.query_row("SELECT substr(date('now'), 1, 7)", [], |r| r.get(0)).unwrap();
        let s = usage_summary(&conn, &month).unwrap();
        let rows = s["by_operation"].as_array().unwrap();
        assert_eq!(rows.len(), 2, "une ligne par (opération, modèle)");
        let classify = rows.iter().find(|r| r["operation"] == "classify").unwrap();
        assert_eq!(classify["calls"], 2);
        assert_eq!(classify["input_tokens"], 633);
        assert_eq!(classify["cache_read_tokens"], 4929);
        let digest = rows.iter().find(|r| r["operation"] == "digest").unwrap();
        assert_eq!(digest["output_tokens"], 1200);
    }

    #[test]
    fn on_device_calls_are_not_recorded() {
        let conn = db();
        record(&conn, &cfg(LlmProvider::Local), Op::Classify, LlmUsage::default());
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM llm_usage", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 0, "le on-device ne facture rien, pas de ligne");
    }

    #[test]
    fn summary_of_an_untouched_month_is_empty_not_missing() {
        let conn = db();
        let s = usage_summary(&conn, "2020-01").unwrap();
        assert_eq!(s["by_operation"].as_array().unwrap().len(), 0);
        assert_eq!(s["captures_processed"], 0);
        assert_eq!(s["days_with_data"], 0);
    }
}
