//! Classifier orchestration (SYN-111 / T2): prompt build + Anthropic HTTP +
//! tolerant parse — the LLM I/O half of the Dream Cycle brain.
//!
//! The prompt is DATA (SYN-96 invariant): `prompts/classifier.md`, versioned
//! in this repo, read at runtime from `prompts_dir`, `{today}` substituted.
//! Editing it never requires a recompilation, and the apps bundle the same
//! files as assets.
//!
//! Two entry points, mirroring the Python split:
//! - [`Brain::build_classify_params`] returns the full `messages.create`
//!   params as JSON — the host's Batch API path submits them via its SDK;
//! - [`Brain::classify`] performs the synchronous HTTP call (direct
//!   Anthropic or the fuel proxy) and parses the response.
//!
//! Error taxonomy matters to the host: an HTTP/network failure maps to
//! `CoreError::LlmHttp` (the run aborts, entries stay queued — the
//! `anthropic.APIError` policy), a truncation/JSON failure maps to
//! `CoreError::LlmContent` (that one entry is marked failed).

use serde_json::{json, Map, Value};

use crate::embedder::CoreError;
use crate::routing::Brain;

/// Everything the host resolves about "how to call the model": key handling
/// (incl. the fuel-proxy seam) stays host policy; the core just executes.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub model: String,
    pub api_key: String,
    /// None → https://api.anthropic.com ; the fuel proxy passes its origin.
    pub base_url: Option<String>,
    /// Set for `syn-fuel-` tokens: sent as `x-synapse-token`, with the
    /// placeholder api key the proxy ignores.
    pub fuel_token: Option<String>,
    pub prompts_dir: String,
    /// Injected into the prompt (`{today}`) — Python used module-load date.
    pub today: String,
}

const MAX_TOKENS: u32 = 4096;
const ANTHROPIC_VERSION: &str = "2023-06-01";
const FALLBACK_TYPES: &[&str] =
    &["person", "place", "project", "concept", "organization", "animal"];

impl Brain {
    /// Port of `_classify_params`: stable rules (cached) + optional working
    /// memory (cached) + live vocab / projects / owner blocks (uncached).
    pub fn build_classify_params(
        &self,
        content: &str,
        day_context: Option<&str>,
        config: &LlmConfig,
    ) -> Result<Value, CoreError> {
        let prompt_path = std::path::Path::new(&config.prompts_dir).join("classifier.md");
        let template = std::fs::read_to_string(&prompt_path).map_err(|e| {
            CoreError::Storage(format!("cannot read {}: {e}", prompt_path.display()))
        })?;
        let classifier = template.replace("{today}", &config.today);

        let mut system_blocks = vec![json!({
            "type": "text",
            "text": classifier,
            "cache_control": {"type": "ephemeral"},
        })];
        if let Some(day_context) = day_context {
            system_blocks.push(json!({
                "type": "text",
                "text": day_context,
                "cache_control": {"type": "ephemeral"},
            }));
        }

        let conn = self.storage.lock()?;
        system_blocks.push(json!({"type": "text", "text": active_types_block(&conn)?}));
        system_blocks.push(json!({"type": "text", "text": active_projects_block(&conn)?}));
        if let Some(owner) = owner_block(&conn)? {
            system_blocks.push(json!({"type": "text", "text": owner}));
        }
        drop(conn);

        Ok(json!({
            "model": config.model,
            "max_tokens": MAX_TOKENS,
            "system": system_blocks,
            "messages": [{"role": "user", "content": content}],
        }))
    }

    /// Port of `step1_classify`: build params, POST /v1/messages, parse.
    pub fn classify(
        &self,
        content: &str,
        day_context: Option<&str>,
        config: &LlmConfig,
    ) -> Result<Value, CoreError> {
        let params = self.build_classify_params(content, day_context, config)?;
        let body = post_messages(config, &params)?;
        let text = body["content"][0]["text"].as_str().unwrap_or("");
        let stop_reason = body["stop_reason"].as_str();
        parse_classify_text(text, content.chars().count(), stop_reason)
    }
}

/// POST /v1/messages with the config's key/fuel-proxy routing — the single
/// HTTP path shared by classify and the T5 summary calls. Returns the parsed
/// response body; HTTP/network failures are `LlmHttp` (abort-the-run policy).
pub(crate) fn post_messages(config: &LlmConfig, params: &Value) -> Result<Value, CoreError> {
    let base = config
        .base_url
        .as_deref()
        .unwrap_or("https://api.anthropic.com")
        .trim_end_matches('/');
    let url = format!("{base}/v1/messages");

    let mut request = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(600))
        .set("content-type", "application/json")
        .set("anthropic-version", ANTHROPIC_VERSION);
    request = match &config.fuel_token {
        Some(token) => request
            .set("x-api-key", "placeholder-real-key-lives-on-the-proxy")
            .set("x-synapse-token", token),
        None => request.set("x-api-key", &config.api_key),
    };

    let response = request.send_string(&params.to_string()).map_err(|e| match e {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            CoreError::LlmHttp(format!("HTTP {code}: {}", &body[..body.len().min(500)]))
        }
        other => CoreError::LlmHttp(other.to_string()),
    })?;
    response
        .into_json()
        .map_err(|e| CoreError::LlmHttp(format!("invalid response body: {e}")))
}

/// `content[0].text` of a /v1/messages response, stripped — the plain-text
/// consumers (summaries). Empty text is the caller's problem.
pub(crate) fn response_text(body: &Value) -> String {
    body["content"][0]["text"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Read a prompt file from `prompts_dir`, dropping the trailing newline so
/// the text stays byte-identical to the historical Python constants.
pub(crate) fn load_prompt(prompts_dir: &str, file: &str) -> Result<String, CoreError> {
    let path = std::path::Path::new(prompts_dir).join(file);
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| CoreError::Storage(format!("cannot read {}: {e}", path.display())))?;
    Ok(raw.strip_suffix('\n').unwrap_or(&raw).to_string())
}

/// Port of `_parse_classify_text` (max_tokens guard + fence strip + parse).
pub fn parse_classify_text(
    text: &str,
    content_len: usize,
    stop_reason: Option<&str>,
) -> Result<Value, CoreError> {
    if stop_reason == Some("max_tokens") {
        return Err(CoreError::LlmContent(format!(
            "classification tronquée (max_tokens) — capture trop longue/dense ({content_len} chars)"
        )));
    }
    let mut raw = text.trim().to_string();
    if raw.starts_with("```") {
        // Same two-step strip as Python: drop the fence line, drop the tail.
        raw = raw
            .split_once('\n')
            .map(|(_, rest)| rest)
            .unwrap_or(&raw)
            .rsplit_once("```")
            .map(|(head, _)| head)
            .unwrap_or(&raw)
            .trim()
            .to_string();
    }
    serde_json::from_str(&raw)
        .map_err(|e| CoreError::LlmContent(format!("classification JSON invalide: {e}")))
}

// ── context blocks (ports of _load_*_block) ─────────────────────────────

fn active_types_block(conn: &rusqlite::Connection) -> Result<String, CoreError> {
    let mut stmt = conn.prepare("SELECT type FROM active_entity_types ORDER BY source, type")?;
    let mut types = stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    if types.is_empty() {
        types = FALLBACK_TYPES.iter().map(|s| s.to_string()).collect();
    }
    Ok(format!(
        "[TYPES D'ENTITÉ ACTIFS — choisis EXACTEMENT l'un d'eux pour `type`]\n{}\n\
         Aucun ne convient ? → type=\"concept\" + type_proposal \
         {{\"value\": \"<type_snake>\", \"reason\": \"...\"}}.",
        types.join(", ")
    ))
}

fn active_projects_block(conn: &rusqlite::Connection) -> Result<String, CoreError> {
    let mut stmt = conn.prepare(
        "SELECT canonical_name, summary, aliases FROM entities \
         WHERE type='project' AND merged_into_id IS NULL \
         ORDER BY mention_count DESC, last_mentioned DESC LIMIT 50",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    if rows.is_empty() {
        return Ok("[PROJETS EXISTANTS]\n(aucun pour l'instant — toute mention de \
                   'nouveau projet : X' doit créer l'entité)"
            .to_string());
    }
    let mut lines =
        vec!["[PROJETS EXISTANTS — utilise leur canonical_name exact pour le rattachement]"
            .to_string()];
    for (name, summary, aliases) in rows {
        let aliases: Vec<String> = aliases
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();
        let alias_str = if aliases.is_empty() {
            String::new()
        } else {
            format!(" (alias: {})", aliases.join(", "))
        };
        let summary: String = summary
            .unwrap_or_default()
            .trim()
            .replace('\n', " ")
            .chars()
            .take(120)
            .collect();
        let tail = if summary.is_empty() {
            String::new()
        } else {
            format!(" — {summary}")
        };
        lines.push(format!("- {name}{alias_str}{tail}"));
    }
    Ok(lines.join("\n"))
}

fn owner_block(conn: &rusqlite::Connection) -> Result<Option<String>, CoreError> {
    let Some(oid) = owner_entity_id() else {
        return Ok(None);
    };
    let mut stmt = conn.prepare(
        "SELECT canonical_name, aliases FROM entities WHERE id = ?1 AND merged_into_id IS NULL",
    )?;
    let mut rows = stmt.query(rusqlite::params![oid])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let name: String = row.get(0)?;
    let aliases: Vec<String> = row
        .get::<_, Option<String>>(1)?
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    let alias_str = if aliases.is_empty() {
        String::new()
    } else {
        format!(" (alias : {})", aliases.join(", "))
    };
    Ok(Some(format!(
        "[AUTEUR — l'utilisateur de ce second cerveau]\n\
         L'auteur des captures est « {name} »{alias_str}. Toute référence à la PREMIÈRE \
         PERSONNE (je, j', me, m', moi, mon, ma, mes, le mien/la mienne…) le désigne. \
         Utilise EXACTEMENT le canonical_name « {name} » comme entité pour les faits et \
         relations le concernant — ex : « Romain est mon frère » → relation \
         (Romain, is_sibling_of, {name}) ; « j'habite à Lyon » → fact (lives_in, Lyon) sur \
         « {name} ». Ne crée JAMAIS d'entité générique « auteur », « Auteur », « User » ou « moi »."
    )))
}

/// Port of `config_store.get_owner_entity_id`: `$SYNAPSE_HOME/config.json`.
fn owner_entity_id() -> Option<String> {
    let base = std::env::var("SYNAPSE_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            std::path::Path::new(&home).join(".synapse")
        });
    let raw = std::fs::read_to_string(base.join("config.json")).ok()?;
    let config: Map<String, Value> = serde_json::from_str(&raw).ok()?;
    config
        .get("owner_entity_id")
        .and_then(Value::as_str)
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_strips_fences_and_guards_truncation() {
        let v = parse_classify_text("```json\n{\"a\": 1}\n```", 10, Some("end_turn")).unwrap();
        assert_eq!(v["a"], 1);
        let v = parse_classify_text("  {\"a\": 2} ", 10, None).unwrap();
        assert_eq!(v["a"], 2);
        assert!(matches!(
            parse_classify_text("{}", 10, Some("max_tokens")),
            Err(CoreError::LlmContent(_))
        ));
        assert!(matches!(
            parse_classify_text("pas du json", 10, None),
            Err(CoreError::LlmContent(_))
        ));
    }
}
