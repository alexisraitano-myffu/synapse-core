//! Classifier orchestration (SYN-111 / T2): prompt build + Anthropic HTTP +
//! tolerant parse — the LLM I/O half of the Dream Cycle brain.
//!
//! The prompt is DATA (SYN-96 invariant): versioned in this repo, read at
//! runtime from `prompts_dir`, `{today}` substituted. Editing it never requires
//! a recompilation, and the apps bundle the same files as assets.
//!
//! Since SYN-171 there are TWO prompts, not one: `classifier-note.md` (routing
//! + prose) and `classifier-graph.md` (entities, facts, relations, projects).
//! `classifier.md` is the superseded single-call prompt — kept as the documented
//! fallback and as the size reference the tests compare against, no longer read
//! by this module. One measured consequence to keep in mind before growing or
//! shrinking a half: Haiku caches a system prefix only above 4096 tokens, and
//! both halves sit below it while the single call sat above.
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
use crate::usage::LlmUsage;

/// Which wire dialect the host's chosen model speaks (SYN-126, open provider
/// seam). The core stays model-agnostic: it builds ONE Anthropic-shaped request
/// and reads ONE Anthropic-shaped response; a non-Anthropic provider is
/// translated at the boundary in [`post_messages`], so every call site
/// (classify, summaries, digest, resources) is unaware of the provider.
///
/// - `Anthropic` — Claude's Messages API (and the `syn-fuel-` proxy).
/// - `OpenAiCompatible` — the `/v1/chat/completions` dialect spoken by OpenAI,
///   Ollama, vLLM, LM Studio, OpenRouter… anyone the user brings their own
///   key/endpoint for.
///
/// - `Local` — an on-device runtime (LiteRT/Gemma, SYN-155). NOT a wire format:
///   the host owns the runtime and the core calls INTO it through the
///   [`LocalLlm`] backend on [`LlmConfig::local`], bypassing HTTP entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlmProvider {
    #[default]
    Anthropic,
    OpenAiCompatible,
    Local,
}

impl LlmProvider {
    /// Host-facing parse (a config/UI string → enum). Unknown or `None` falls
    /// back to `Anthropic`, so an unset field keeps today's behaviour.
    pub fn parse(s: Option<&str>) -> Self {
        match s.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("openai") | Some("openai-compatible") | Some("openai_compatible") => {
                Self::OpenAiCompatible
            }
            Some("local") | Some("gemma-local") | Some("on-device") | Some("on_device") => {
                Self::Local
            }
            _ => Self::Anthropic,
        }
    }
}

/// A host-supplied on-device LLM (SYN-155): the mobile app owns the runtime
/// (LiteRT-LM / Gemma) and the core calls INTO it instead of over HTTP. This is
/// the foreign-backend extension point the provider seam (SYN-150) reserved.
///
/// The core hands the already-built, `{today}`-substituted prompt (system + the
/// user turn); the host applies the model's chat template and returns the raw
/// generated text — the same string an HTTP body's `content[0].text` would
/// carry (JSON for classify, prose for summaries). Called on the core's own
/// (worker) thread, synchronously.
pub trait LocalLlm: Send + Sync {
    fn generate(&self, system: String, user: String, max_tokens: u32)
        -> Result<String, CoreError>;
}

/// Everything the host resolves about "how to call the model": key handling
/// (incl. the fuel-proxy seam) stays host policy; the core just executes.
#[derive(Clone)]
pub struct LlmConfig {
    pub model: String,
    pub api_key: String,
    /// Which wire dialect `model` speaks. `Anthropic` by default (`LlmProvider`
    /// derives `Default`) — an existing host that never sets it is unchanged.
    pub provider: LlmProvider,
    /// None → the provider's default origin (Anthropic: api.anthropic.com,
    /// OpenAI-compatible: api.openai.com). Ollama/vLLM/etc. pass their own.
    /// The fuel proxy passes its origin here too (Anthropic only).
    pub base_url: Option<String>,
    /// Set for `syn-fuel-` tokens: sent as `x-synapse-token`, with the
    /// placeholder api key the proxy ignores. Anthropic-only (ignored elsewhere).
    pub fuel_token: Option<String>,
    pub prompts_dir: String,
    /// Injected into the prompt (`{today}`) — Python used module-load date.
    pub today: String,
    /// The on-device backend, required when `provider == Local` (SYN-155). Kept
    /// out of `Debug` (a foreign callback has no meaningful repr).
    pub local: Option<std::sync::Arc<dyn LocalLlm>>,
}

impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConfig")
            .field("model", &self.model)
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("prompts_dir", &self.prompts_dir)
            .field("today", &self.today)
            .field("local", &self.local.as_ref().map(|_| "<on-device backend>"))
            .finish_non_exhaustive()
    }
}

const MAX_TOKENS: u32 = 4096;
const ANTHROPIC_VERSION: &str = "2023-06-01";
const FALLBACK_TYPES: &[&str] =
    &["person", "place", "project", "concept", "organization", "animal"];

/// SYN-171 — which half of the classifier a call is running.
///
/// The capture used to go through ONE prompt holding every rule at once. Measured
/// 2026-08-21 on 59 cases: compacting that prompt cost Gemma E2B a THIRD of its
/// notes (34 → 22) while leaving Haiku untouched, and a complete 2×2 (prompt ×
/// constrained schema) cleared the schema — it was the prompt alone. The rules
/// the compaction dropped were repetitions of one invariant ("extracting facts
/// never absorbs the note"), useless to a large model and load-bearing for a
/// small one.
///
/// Splitting removes the need to repeat it: the graph call has no `atomic_note`
/// field at all, so the invariant stops being a sentence to hammer and becomes
/// STRUCTURAL. The two calls are deliberately INDEPENDENT — the extractor never
/// sees the routing — because chaining them would restore the coupling we are
/// removing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ClassifyHalf {
    /// What the capture leaves behind: note, kind, owner, dates, ephemerality.
    Note,
    /// What it says about the world: entities, facts, relations, project entries.
    Graph,
}

impl ClassifyHalf {
    fn prompt_file(self) -> &'static str {
        match self {
            ClassifyHalf::Note => "classifier-note.md",
            ClassifyHalf::Graph => "classifier-graph.md",
        }
    }
}

impl Brain {
    /// Port of `_classify_params`: stable rules (cached) + optional working
    /// memory (cached) + live vocab / projects / owner blocks (uncached).
    pub fn build_classify_params(
        &self,
        content: &str,
        day_context: Option<&str>,
        config: &LlmConfig,
        half: ClassifyHalf,
    ) -> Result<Value, CoreError> {
        let prompt_path = std::path::Path::new(&config.prompts_dir).join(half.prompt_file());
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
        // The live vocabulary and the project list only steer fields the GRAPH
        // half emits (`entities[].type`, `project_entries[].project_canonical`).
        // Sending them to the note half would spend input on blocks whose target
        // fields do not exist in its schema — and the whole point of the split is
        // that a half only ever reads what it can write.
        if half == ClassifyHalf::Graph {
            system_blocks.push(json!({"type": "text", "text": active_types_block(&conn)?}));
            system_blocks.push(json!({"type": "text", "text": active_projects_block(&conn)?}));
        }
        // Who the author is matters to BOTH: the note half needs it to tell a
        // reported action from the author's own (SYN-182 `atomic_note_owner`),
        // the graph half to resolve "me"/"my" onto the right entity.
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

    /// One half: build params, POST /v1/messages, parse. Billing is recorded per
    /// call — the split doubles the number of calls, and a books entry that
    /// counted them as one would understate what the cycle actually costs.
    fn classify_half(
        &self,
        content: &str,
        day_context: Option<&str>,
        config: &LlmConfig,
        half: ClassifyHalf,
    ) -> Result<Value, CoreError> {
        let params = self.build_classify_params(content, day_context, config, half)?;
        let body = post_messages(config, &params)?;
        // SYN-160 — recorded before parsing: the call is billed whether or not
        // its JSON turns out to be usable, so a parse failure must not make the
        // spending disappear from the books.
        {
            let conn = self.storage.lock()?;
            crate::usage::record(&conn, config, crate::usage::Op::Classify, LlmUsage::from_body(&body));
        }
        let text = body["content"][0]["text"].as_str().unwrap_or("");
        let stop_reason = body["stop_reason"].as_str();
        parse_classify_text(text, content.chars().count(), stop_reason)
    }

    /// Port of `step1_classify`, in TWO independent calls since SYN-171.
    ///
    /// The note half runs first so a failure costs one call instead of two, but
    /// nothing is threaded from it into the graph half: independence IS the
    /// mechanism. `route_capture` consumes a single merged object, so the split
    /// stops at this boundary and no caller downstream has to know about it.
    pub fn classify(
        &self,
        content: &str,
        day_context: Option<&str>,
        config: &LlmConfig,
    ) -> Result<Value, CoreError> {
        let note = self.classify_half(content, day_context, config, ClassifyHalf::Note)?;
        let graph = self.classify_half(content, day_context, config, ClassifyHalf::Graph)?;
        Ok(merge_halves(note, graph))
    }
}

/// Fuse the two halves back into the shape `route_capture` has always consumed.
///
/// Each half owns its fields outright — no key is written by both, so there is
/// nothing to arbitrate and the merge cannot silently prefer one call's opinion
/// over the other's. `language` is the one exception: both halves detect it, and
/// the note half wins because it is the half that WRITES prose in that language.
///
/// The three collections are forced to arrays even when the graph half omits
/// them: `route_capture` reads them with `arr()`, and a missing key would read
/// as "nothing to extract" — indistinguishable from a half that failed to answer.
pub fn merge_halves(note: Value, graph: Value) -> Value {
    let mut merged = serde_json::Map::new();
    if let Value::Object(g) = graph {
        for (k, v) in g {
            merged.insert(k, v);
        }
    }
    if let Value::Object(n) = note {
        for (k, v) in n {
            merged.insert(k, v);
        }
    }
    for key in ["entities", "relations", "project_entries"] {
        if !merged.get(key).map(Value::is_array).unwrap_or(false) {
            merged.insert(key.into(), json!([]));
        }
    }
    Value::Object(merged)
}

/// The single LLM chokepoint shared by classify and the T5 summary calls
/// (SYN-126): dispatch on the provider, but ALWAYS return an Anthropic-shaped
/// body (`content[0].text` + `stop_reason`) so every caller and the response
/// helpers ([`response_text`], [`unusable`], [`parse_classify_text`]) are
/// provider-agnostic. HTTP/network failures are `LlmHttp` (abort-the-run policy).
pub(crate) fn post_messages(config: &LlmConfig, params: &Value) -> Result<Value, CoreError> {
    // SYN-171 — the cycle is a classifier, not a writer: sampling buys nothing
    // and costs reproducibility. Left unset, the same capture took different
    // routing branches between two runs, which made every parity measurement
    // argue with itself. A caller that wants sampling can still pass its own.
    let owned;
    let params = match params.get("temperature") {
        Some(_) => params,
        None => {
            let mut p = params.clone();
            if let Value::Object(m) = &mut p {
                m.insert("temperature".into(), json!(0));
            }
            owned = p;
            &owned
        }
    };
    match config.provider {
        LlmProvider::Anthropic => post_anthropic(config, params),
        LlmProvider::OpenAiCompatible => post_openai(config, params),
        LlmProvider::Local => post_local(config, params),
    }
}

/// On-device path (SYN-155): flatten the request to system+user text, hand it to
/// the host's [`LocalLlm`] backend, wrap the reply in the Anthropic shape. No
/// HTTP. A truncation is the host's concern — we report `end_turn` (the guard
/// keys off `max_tokens`, which an on-device backend never emits).
fn post_local(config: &LlmConfig, params: &Value) -> Result<Value, CoreError> {
    let backend = config.local.as_ref().ok_or_else(|| {
        CoreError::LlmHttp("provider=local but no on-device backend was supplied".to_string())
    })?;
    let system = params.get("system").map(flatten_text).unwrap_or_default();
    let user = params
        .get("messages")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|m| flatten_text(m.get("content").unwrap_or(&Value::Null)))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default();
    let max_tokens = params
        .get("max_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(MAX_TOKENS as u64) as u32;
    let text = backend.generate(system, user, max_tokens)?;
    Ok(json!({
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }))
}

/// Map a `ureq` transport/status error to `LlmHttp` (shared by both providers).
fn http_error(e: ureq::Error) -> CoreError {
    match e {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            CoreError::LlmHttp(format!("HTTP {code}: {}", &body[..body.len().min(500)]))
        }
        other => CoreError::LlmHttp(other.to_string()),
    }
}

/// POST /v1/messages with the config's key/fuel-proxy routing. The request is
/// already Anthropic-shaped and the response needs no translation.
fn post_anthropic(config: &LlmConfig, params: &Value) -> Result<Value, CoreError> {
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

    let response = request
        .send_string(&params.to_string())
        .map_err(http_error)?;
    response
        .into_json()
        .map_err(|e| CoreError::LlmHttp(format!("invalid response body: {e}")))
}

/// POST /v1/chat/completions (OpenAI, Ollama, vLLM, LM Studio, OpenRouter…):
/// translate the Anthropic-shaped `params` to a chat request, then normalise the
/// response back to the Anthropic shape. `cache_control` is dropped and
/// `finish_reason` is mapped to `stop_reason` — the two normalisations SYN-126
/// flagged (a foreign provider chokes on `cache_control` and never emits
/// Anthropic's `stop_reason`, so both must be synthesised at this boundary).
fn post_openai(config: &LlmConfig, params: &Value) -> Result<Value, CoreError> {
    let base = config
        .base_url
        .as_deref()
        .unwrap_or("https://api.openai.com")
        .trim_end_matches('/');
    let url = format!("{base}/v1/chat/completions");

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(600))
        .set("content-type", "application/json")
        .set("authorization", &format!("Bearer {}", config.api_key))
        .send_string(&anthropic_to_openai(config, params).to_string())
        .map_err(http_error)?;
    let body: Value = response
        .into_json()
        .map_err(|e| CoreError::LlmHttp(format!("invalid response body: {e}")))?;
    Ok(openai_to_anthropic(&body))
}

/// Concatenate the `text` of an Anthropic content value — a bare string, or an
/// array of blocks — into one string, dropping `cache_control` and any non-text
/// block. The join keeps blocks readable when a system had several.
fn flatten_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

/// Anthropic `messages.create` params → OpenAI `chat/completions` request. The
/// `system` blocks become a leading `system` message; each user/assistant
/// message's content is flattened to text. `cache_control` never survives (we
/// read only `.text`).
fn anthropic_to_openai(config: &LlmConfig, params: &Value) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = params.get("system") {
        let text = flatten_text(system);
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }
    if let Some(arr) = params.get("messages").and_then(Value::as_array) {
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = flatten_text(m.get("content").unwrap_or(&Value::Null));
            messages.push(json!({"role": role, "content": content}));
        }
    }
    let mut out = json!({
        "model": config.model,
        "max_tokens": params.get("max_tokens").cloned().unwrap_or(json!(MAX_TOKENS)),
        "messages": messages,
    });
    if let (Some(t), Value::Object(m)) = (params.get("temperature"), &mut out) {
        m.insert("temperature".into(), t.clone());
    }
    out
}

/// OpenAI `chat/completions` response → Anthropic-shaped body. `finish_reason`
/// `length` is Anthropic's `max_tokens` (truncation — the guard/parse both key
/// off it); everything else collapses to `end_turn`. A body with no usable
/// choice yields empty text, which the `unusable`/retry path already handles.
fn openai_to_anthropic(body: &Value) -> Value {
    let choice = &body["choices"][0];
    let text = choice["message"]["content"].as_str().unwrap_or("");
    let stop_reason = match choice["finish_reason"].as_str() {
        Some("length") => "max_tokens",
        _ => "end_turn",
    };
    json!({
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
    })
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

/// Extra attempts granted to a generative call that came back unusable.
const EMPTY_RETRIES: usize = 1;

/// True when a generative response can't be used as-is: no text at all, or
/// truncated at `max_tokens` (a summary cut mid-sentence is not a summary).
///
/// SYN-124 — measured root cause: reasoning-capable models sometimes spend the
/// whole budget on a thinking block and return `stop_reason = max_tokens` with
/// an EMPTY body. On Gemma E4B re-summarising an entity, ~1000 chars of
/// thinking consumed all 300 tokens; the same prompt succeeds on a retry when
/// the model doesn't think. Retrying on emptiness alone caught that only by
/// accident, and never caught a truncation that left partial text behind.
fn unusable(body: &Value, text: &str) -> bool {
    text.is_empty() || body["stop_reason"].as_str() == Some("max_tokens")
}

/// `post_messages` + `response_text`, retried while the response is unusable
/// (see [`unusable`]). Callers keep the same contract as `response_text` — an
/// empty string is still possible once the retries are spent, and stays the
/// caller's problem.
/// SYN-160 — the tuple is deliberate. Returning the usage forces every call
/// site to say what it spent and under which operation; a signature that let
/// callers ignore it would drift back to "we only measure classify", which is
/// exactly the blind spot the ticket set out to remove. A retried call bills
/// twice, so both attempts are summed into the returned usage.
pub(crate) fn post_messages_text(
    config: &LlmConfig,
    params: &Value,
) -> Result<(String, LlmUsage), CoreError> {
    let mut body = post_messages(config, params)?;
    let mut text = response_text(&body);
    let mut usage = LlmUsage::from_body(&body);
    for _ in 0..EMPTY_RETRIES {
        if !unusable(&body, &text) {
            break;
        }
        body = post_messages(config, params)?;
        text = response_text(&body);
        usage = usage.add(LlmUsage::from_body(&body));
    }
    Ok((text, usage))
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
        "[ACTIVE ENTITY TYPES — pick EXACTLY one of these for `type`]\n{}\n\
         None fits? → type=\"concept\" + type_proposal \
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
        return Ok("[EXISTING PROJECTS]\n(none yet — any mention of 'new project: X' \
                   must create the entity)"
            .to_string());
    }
    let mut lines =
        vec!["[EXISTING PROJECTS — use their exact canonical_name to attach]"
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
        format!(" (aliases: {})", aliases.join(", "))
    };
    Ok(Some(format!(
        "[AUTHOR — the user of this second brain]\n\
         The author of the captures is \"{name}\"{alias_str}. Any FIRST-PERSON reference \
         designates them — EN: I, me, my, mine, myself; FR: je, j', me, m', moi, mon, ma, mes, \
         le mien/la mienne. Use EXACTLY the canonical_name \"{name}\" as the entity for facts \
         and relations about them — e.g. \"Romain is my brother\" (FR: « Romain est mon frère ») \
         → relation (Romain, is_sibling_of, {name}); \"I live in Lyon\" (FR: « j'habite à Lyon ») \
         → fact (lives_in, Lyon) on \"{name}\". NEVER create a generic entity \"author\", \
         \"Author\", \"User\" or \"me\"."
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

    /// Minimal one-shot /v1/messages stub: serves `bodies` in order, one per
    /// connection, and reports how many requests it actually received. Keeps
    /// the retry test dependency-free (no HTTP mocking crate).
    /// Sends the FULL raw request text of each served connection back over the
    /// channel, so tests can both count requests (`rx.iter().count()`) and
    /// assert on the outgoing body (cache_control strip / message shape).
    fn stub_server(bodies: Vec<&'static str>) -> (String, std::sync::mpsc::Receiver<String>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for (i, mut stream) in listener.incoming().flatten().enumerate() {
                // Read the FULL request (headers + body) before answering:
                // closing a socket with unread input sends an RST, which the
                // client reports as "connection reset" instead of our response.
                let mut req: Vec<u8> = Vec::new();
                let mut buf = [0u8; 1024];
                loop {
                    let Ok(n) = stream.read(&mut buf) else { break };
                    if n == 0 {
                        break;
                    }
                    req.extend_from_slice(&buf[..n]);
                    let Some(head_end) = req
                        .windows(4)
                        .position(|w| w == b"\r\n\r\n")
                        .map(|p| p + 4)
                    else {
                        continue;
                    };
                    let head = String::from_utf8_lossy(&req[..head_end]).to_lowercase();
                    let want: usize = head
                        .split("content-length:")
                        .nth(1)
                        .and_then(|s| s.split("\r\n").next())
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);
                    if req.len() - head_end >= want {
                        break;
                    }
                }
                let payload = bodies.get(i).copied().unwrap_or("{}");
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                let _ = tx.send(String::from_utf8_lossy(&req).to_string());
                if i + 1 >= bodies.len() {
                    break;
                }
            }
        });
        (base, rx)
    }

    fn cfg(base: String) -> LlmConfig {
        cfg_provider(base, LlmProvider::Anthropic)
    }

    fn cfg_provider(base: String, provider: LlmProvider) -> LlmConfig {
        LlmConfig {
            model: "test".into(),
            api_key: "k".into(),
            provider,
            base_url: Some(base),
            fuel_token: None,
            prompts_dir: String::new(),
            today: "2026-07-20".into(),
            local: None,
        }
    }

    const EMPTY: &str = r#"{"content":[{"type":"text","text":"   "}]}"#;
    const FILLED: &str = r#"{"content":[{"type":"text","text":"une fiche"}]}"#;
    /// The measured Gemma failure: budget spent thinking, body empty, cut at max_tokens.
    const THOUGHT_AWAY: &str =
        r#"{"stop_reason":"max_tokens","content":[{"type":"text","text":""}]}"#;
    /// Truncated but non-empty — a summary cut mid-sentence.
    const CUT: &str =
        r#"{"stop_reason":"max_tokens","content":[{"type":"text","text":"une fiche coup"}]}"#;

    #[test]
    fn thinking_that_ate_the_budget_is_retried() {
        let (base, rx) = stub_server(vec![THOUGHT_AWAY, FILLED]);
        let (text, _) = post_messages_text(&cfg(base), &json!({})).unwrap();
        assert_eq!(text, "une fiche");
        assert_eq!(rx.iter().count(), 2);
    }

    #[test]
    fn truncated_but_non_empty_is_also_retried() {
        // The old empty-only guard let this through: partial text, silently stored.
        let (base, rx) = stub_server(vec![CUT, FILLED]);
        let (text, _) = post_messages_text(&cfg(base), &json!({})).unwrap();
        assert_eq!(text, "une fiche", "a summary cut mid-sentence is not a summary");
        assert_eq!(rx.iter().count(), 2);
    }

    #[test]
    fn empty_generation_is_retried_once() {
        let (base, rx) = stub_server(vec![EMPTY, FILLED]);
        let (text, _) = post_messages_text(&cfg(base), &json!({})).unwrap();
        assert_eq!(text, "une fiche", "the retry's text must win");
        assert_eq!(rx.iter().count(), 2, "exactly one retry");
    }

    #[test]
    fn non_empty_generation_is_not_retried() {
        let (base, rx) = stub_server(vec![FILLED]);
        let (text, _) = post_messages_text(&cfg(base), &json!({})).unwrap();
        assert_eq!(text, "une fiche");
        assert_eq!(rx.iter().count(), 1, "no retry when the first call is useful");
    }

    #[test]
    fn still_empty_after_retry_stays_empty_for_the_caller() {
        // Contract: retries are bounded — callers still handle the empty case
        // (summaries keeps summary_stale=1, resources falls back to a snippet).
        let (base, rx) = stub_server(vec![EMPTY, EMPTY]);
        let (text, _) = post_messages_text(&cfg(base), &json!({})).unwrap();
        assert!(text.is_empty());
        assert_eq!(rx.iter().count(), 2, "bounded at EMPTY_RETRIES, no loop");
    }

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

    // ── SYN-150: OpenAI-compatible provider normalisation ────────────────
    const OA_FILLED: &str =
        r#"{"choices":[{"message":{"content":"une fiche"},"finish_reason":"stop"}]}"#;
    /// finish_reason=length → the OpenAI spelling of Anthropic's max_tokens.
    const OA_CUT: &str =
        r#"{"choices":[{"message":{"content":"une fiche coup"},"finish_reason":"length"}]}"#;

    #[test]
    fn provider_parse_defaults_to_anthropic() {
        assert_eq!(LlmProvider::parse(None), LlmProvider::Anthropic);
        assert_eq!(LlmProvider::parse(Some("gibberish")), LlmProvider::Anthropic);
        assert_eq!(LlmProvider::parse(Some("anthropic")), LlmProvider::Anthropic);
        assert_eq!(
            LlmProvider::parse(Some(" OpenAI ")),
            LlmProvider::OpenAiCompatible
        );
        assert_eq!(
            LlmProvider::parse(Some("openai-compatible")),
            LlmProvider::OpenAiCompatible
        );
    }

    #[test]
    fn openai_response_is_normalised_to_anthropic_text() {
        let (base, _rx) = stub_server(vec![OA_FILLED]);
        let cfg = cfg_provider(base, LlmProvider::OpenAiCompatible);
        // A classify-style body still parses: the normalised shape carries
        // content[0].text + stop_reason, exactly what the caller reads.
        let body = post_messages(&cfg, &json!({})).unwrap();
        assert_eq!(body["content"][0]["text"], "une fiche");
        assert_eq!(body["stop_reason"], "end_turn");
    }

    #[test]
    fn openai_length_maps_to_truncation_and_is_retried() {
        // finish_reason=length must normalise to stop_reason=max_tokens so the
        // existing truncation guard fires — here it drives the retry.
        let (base, rx) = stub_server(vec![OA_CUT, OA_FILLED]);
        let cfg = cfg_provider(base, LlmProvider::OpenAiCompatible);
        let (text, _) = post_messages_text(&cfg, &json!({})).unwrap();
        assert_eq!(text, "une fiche", "the untruncated retry must win");
        assert_eq!(rx.iter().count(), 2);
    }

    // ── SYN-155: on-device (Local) backend ───────────────────────────────
    /// Records what the core flattened, echoes a canned reply — stands in for
    /// the Kotlin LiteRT backend without a device.
    struct StubLocal {
        seen_system: std::sync::Mutex<String>,
        seen_user: std::sync::Mutex<String>,
        reply: String,
    }
    impl LocalLlm for StubLocal {
        fn generate(&self, system: String, user: String, _max: u32) -> Result<String, CoreError> {
            *self.seen_system.lock().unwrap() = system;
            *self.seen_user.lock().unwrap() = user;
            Ok(self.reply.clone())
        }
    }

    #[test]
    fn local_backend_receives_flattened_prompt_and_reply_is_wrapped() {
        let stub = std::sync::Arc::new(StubLocal {
            seen_system: std::sync::Mutex::new(String::new()),
            seen_user: std::sync::Mutex::new(String::new()),
            reply: "{\"language\":\"fr\"}".to_string(),
        });
        let mut cfg = cfg_provider(String::new(), LlmProvider::Local);
        cfg.local = Some(stub.clone());
        let params = json!({
            "max_tokens": 512,
            "system": [{"type": "text", "text": "tu es un classifieur",
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "note du jour"}],
        });
        let body = post_messages(&cfg, &params).unwrap();
        // The reply is wrapped so response helpers stay provider-agnostic.
        assert_eq!(body["content"][0]["text"], "{\"language\":\"fr\"}");
        assert_eq!(body["stop_reason"], "end_turn");
        // The backend saw the flattened prompt, cache_control dropped.
        assert_eq!(*stub.seen_system.lock().unwrap(), "tu es un classifieur");
        assert_eq!(*stub.seen_user.lock().unwrap(), "note du jour");
    }

    #[test]
    fn local_provider_without_backend_errors() {
        let cfg = cfg_provider(String::new(), LlmProvider::Local); // local = None
        assert!(matches!(
            post_messages(&cfg, &json!({})),
            Err(CoreError::LlmHttp(_))
        ));
    }

    #[test]
    fn openai_request_drops_cache_control_and_flattens_system() {
        let (base, rx) = stub_server(vec![OA_FILLED]);
        let cfg = cfg_provider(base, LlmProvider::OpenAiCompatible);
        let params = json!({
            "model": "ignored-cfg-wins",
            "max_tokens": 123,
            "system": [{"type": "text", "text": "tu es un classifieur",
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "note du jour"}],
        });
        post_messages(&cfg, &params).unwrap();
        let req = rx.recv().unwrap();
        let sent = &req[req.find("\r\n\r\n").unwrap() + 4..];
        let sent: Value = serde_json::from_str(sent).unwrap();
        // No Anthropic-only keys reach an OpenAI endpoint.
        assert!(!req.contains("cache_control"), "cache_control must be stripped");
        assert!(sent.get("system").is_none(), "system folds into messages");
        // The config's model wins; system becomes the first message.
        assert_eq!(sent["model"], "test");
        assert_eq!(sent["max_tokens"], 123);
        assert_eq!(sent["messages"][0]["role"], "system");
        assert_eq!(sent["messages"][0]["content"], "tu es un classifieur");
        assert_eq!(sent["messages"][1]["role"], "user");
        assert_eq!(sent["messages"][1]["content"], "note du jour");
    }
}
