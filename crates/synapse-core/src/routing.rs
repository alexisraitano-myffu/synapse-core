//! Deterministic Dream Cycle routing (SYN-111 / T2).
//!
//! Faithful port of the Python brain's per-capture pipeline
//! (`dream_cycle/cycle.py::_process_entry` and everything it fans out to:
//! `step2_resolve`, `compute_confidence`, `step4_route`,
//! `step5_validate_pending`, `facts_store.insert_fact`, intentions, atomic
//! notes, project entries) — LLM I/O excluded. The seam is: classified JSON
//! in, database writes out. Sub-LLM work (project synthesis, resummary,
//! resource fetch) is returned to the host as a work list, never performed
//! here.
//!
//! Parity discipline (golden-tested against the frozen Python reference):
//! - SQL casefolding stays SQL (`LOWER(...)` in the same statements); Python
//!   `str.lower()` sites use Rust `to_lowercase()`;
//! - float order of operations matches `compute_confidence` exactly;
//! - Python truthiness is reproduced where the code branched on it
//!   (`classified.get("entities")` is false for `[]`);
//! - `json.dumps(..., ensure_ascii=False)` byte layout only matters where
//!   the string feeds the embedder (`entity_embedding_text`) — reproduced by
//!   `py_dumps`; stored JSON is compared content-wise by the golden
//!   normalizer, so serde's compact form is fine there.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rusqlite::types::Value as SqlV;
use rusqlite::{params, params_from_iter, Connection};
use serde_json::{json, Map, Value};

use crate::embedder::{CoreError, Embedder};
use crate::storage::{search_entities_on, Storage};

// Same tunables and defaults as cycle.py; the env overrides keep working
// because host and core share the process environment.
const MIN_ENTITY_PERSISTENCE: f64 = 2.0;
const MERGE_EMBEDDING_THRESHOLD_DEFAULT: f64 = 0.85;
const PROJECT_ATTACH_THRESHOLD_DEFAULT: f64 = 0.30;
const PROJECT_ATTACH_MARGIN_DEFAULT: f64 = 0.03;
const REVIEW_CONFIDENCE_THRESHOLD_DEFAULT: f64 = 0.7;

const SINGLE_VALUED_PREDICATES: &[&str] = &[
    "works_at", "current_workplace", "employer",
    "lives_in", "current_city", "lives", "address",
    "has_birthday", "birthday", "born_on", "date_of_birth",
    "phone", "phone_number", "email",
    "age", "job_title", "current_role", "role",
];

const DATE_PREDICATE_KEYWORDS: &[&str] =
    &["birthday", "birth", "date", "born", "anniversary", "anniversaire"];

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Wall-clock inputs, provided by the host so the core stays deterministic
/// and testable (the Python code read the clock in place).
#[derive(Debug, Clone)]
pub struct RouteContext {
    /// ISO timestamp for `inbox.processed_at` (Python's `now` parameter).
    pub now: String,
    /// ISO date for `entities.last_mentioned` (Python: `now(utc).date()`).
    pub today: String,
    /// ISO timestamp 48h ago — expired-intentions cutoff. Kept as an opaque
    /// string: Python compared `created_at < isoformat(...)` textually.
    pub intentions_cutoff: String,
    /// `%Y-%m-%d %H:%M:%S` timestamp for note reactivation writes.
    pub now_sql: String,
}

/// One project entry persisted this capture — the host runs the live
/// synthesis (Haiku) for each, exactly like `_persist_project_entry` did
/// when given a client.
#[derive(Debug, Clone)]
pub struct ProjectSynthesis {
    pub project_id: String,
    pub entry_id: String,
    pub project_name: String,
    pub entry_content: String,
    pub entry_count: i64,
}

#[derive(Debug, Default)]
pub struct RouteReport {
    pub entity_ids: Vec<String>,
    /// Flattened facts (with entity_canonical + source_inbox_id), the input
    /// step5 accumulates across the run.
    pub new_facts: Vec<Value>,
    pub created_note_id: Option<String>,
    pub project_syntheses: Vec<ProjectSynthesis>,
    pub fast_exit: bool,
}

/// The routing brain: storage + (optionally) the embedder that powers the
/// merge fallback, project-attach proposals and note vectorization. Without
/// an embedder those paths degrade exactly like Python's `except Exception`
/// around `embed_text` (skip silently / leave the note unvectorized).
pub struct Brain {
    pub storage: Storage,
    embedder: Option<Arc<Embedder>>,
}

impl Brain {
    pub fn open(db_path: &str, model_dir: Option<&str>) -> Result<Self, CoreError> {
        let embedder = match model_dir {
            Some(dir) => Some(Arc::new(Embedder::new(dir)?)),
            None => None,
        };
        Self::open_shared(db_path, embedder)
    }

    /// Open sharing an already-loaded embedder (the model weighs ~235 MB and
    /// takes seconds to load; hosts opening several Brains — e.g. a test
    /// suite with one database per test — must not pay it per instance).
    pub fn open_shared(
        db_path: &str,
        embedder: Option<Arc<Embedder>>,
    ) -> Result<Self, CoreError> {
        let storage = Storage::open(db_path)?;
        Ok(Self { storage, embedder })
    }

    pub(crate) fn embed(&self, text: &str) -> Option<Vec<u8>> {
        let vec = self.embedder.as_ref()?.embed(text).ok()?;
        Some(vec.iter().flat_map(|x| x.to_le_bytes()).collect())
    }

    /// One serialized vector per ~128-token window (SYN-118): the storage
    /// keeps them all and search takes the best window per note.
    pub(crate) fn embed_chunks(&self, text: &str) -> Option<Vec<Vec<u8>>> {
        let chunks = self.embedder.as_ref()?.embed_chunks(text).ok()?;
        Some(
            chunks
                .into_iter()
                .map(|v| v.iter().flat_map(|x| x.to_le_bytes()).collect())
                .collect(),
        )
    }

    /// Chunk vectors concatenated into ONE blob (SYN-118) — the layout of the
    /// `entities`/`resources` embedding columns; scorers take the best frame.
    pub(crate) fn embed_frames(&self, text: &str) -> Option<Vec<u8>> {
        Some(self.embed_chunks(text)?.concat())
    }

    /// Embed arbitrary text with the Brain's already-loaded embedder — the
    /// host-side re-embed path after a sync apply (mirror of the backend's
    /// `embed_text`), without paying a second model load.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        match &self.embedder {
            Some(e) => e.embed(text),
            None => Err(CoreError::Embedding(
                "brain opened without a model_dir".into(),
            )),
        }
    }

    /// Chunked variant (SYN-118) for the same re-embed path: one vector per
    /// ~128-token window, so a mobile host stores the same per-chunk rows as
    /// the desktop backend after a sync apply.
    pub fn embed_text_chunks(&self, text: &str) -> Result<Vec<Vec<f32>>, CoreError> {
        match &self.embedder {
            Some(e) => e.embed_chunks(text),
            None => Err(CoreError::Embedding(
                "brain opened without a model_dir".into(),
            )),
        }
    }

    /// Port of `_process_entry` minus classification/resources/LLM calls.
    /// `entry` = `{id, content}`; `classified` = the classifier JSON.
    /// Marks the inbox row processed. The caller handles errors by marking
    /// the entry failed (content-error policy stays host-side).
    pub fn route_capture(
        &self,
        entry: &Value,
        classified: &Value,
        ctx: &RouteContext,
    ) -> Result<RouteReport, CoreError> {
        // uuid string post-SYN-112; legacy integer ids (golden corpus,
        // pre-migration callers) are accepted as their text form.
        let capture_id: String = match entry.get("id") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            _ => return Err(CoreError::Storage("entry.id missing".into())),
        };
        let capture_id = capture_id.as_str();
        let content = entry.get("content").and_then(Value::as_str).unwrap_or("");

        let mut report = RouteReport::default();

        let is_ephemeral = truthy(classified.get("is_ephemeral"))
            || classified.get("input_type").and_then(Value::as_str) == Some("ephemeral");
        let note_kind = classified
            .get("atomic_note_kind")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("note")
            .to_string();
        let atomic = classified
            .get("atomic_note")
            .and_then(Value::as_str)
            .unwrap_or("");
        let durable_note =
            !atomic.trim().is_empty() && (note_kind == "task" || note_kind == "event");

        let conn = self.storage.lock()?;

        // Pure ephemeral fast exit (no entities, no project, no durable note).
        if is_ephemeral
            && !(truthy(classified.get("entities"))
                || truthy(classified.get("project_entries"))
                || durable_note)
        {
            conn.execute_batch("BEGIN")?;
            let r = (|| -> Result<(), CoreError> {
                self.propose_project_attach_if_similar(&conn, capture_id, content, None)?;
                handle_intentions(&conn, classified, ctx)?;
                mark(&conn, capture_id, &ctx.now, "processed")?;
                Ok(())
            })();
            finish_txn(&conn, r)?;
            report.fast_exit = true;
            return Ok(report);
        }

        // Resolve (step 2) outside the transaction, like Python.
        let resolved = if truthy(classified.get("entities")) {
            Some(self.resolve(&conn, classified, ctx))
        } else {
            None
        };
        if let Some(resolved) = &resolved {
            for ent in resolved {
                for fact in &ent.facts {
                    let mut nf = fact.clone();
                    if let Value::Object(m) = &mut nf {
                        m.insert("entity_canonical".into(),
                                 ent.data.get("canonical_name").cloned().unwrap_or(Value::Null));
                        m.insert("source_inbox_id".into(), json!(capture_id));
                    }
                    report.new_facts.push(nf);
                }
            }
        }

        conn.execute_batch("BEGIN")?;
        let mut pending_note_vec: Option<(String, String)> = None;
        let r = (|| -> Result<(), CoreError> {
            if let Some(resolved) = &resolved {
                report.entity_ids =
                    self.step4_route(&conn, classified, resolved, capture_id, durable_note, ctx)?;
            }

            // Atomic note (SYN-56/58/85 gates).
            let mut created_note_id: Option<String> = None;
            if !atomic.trim().is_empty() && (!is_ephemeral || durable_note) {
                let mut mentioned: Vec<String> = arr(classified.get("entities"))
                    .iter()
                    .filter_map(|e| e.get("canonical_name").and_then(Value::as_str))
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                for pe in arr(classified.get("project_entries")) {
                    if let Some(pc) = pe.get("project_canonical").and_then(Value::as_str) {
                        if !pc.is_empty() && !mentioned.iter().any(|m| m == pc) {
                            mentioned.push(pc.to_string());
                        }
                    }
                }
                let conf = py_float(classified.get("classification_confidence")).unwrap_or(1.0);
                let threshold = env_f64(
                    "SYNAPSE_REVIEW_CONFIDENCE_THRESHOLD",
                    REVIEW_CONFIDENCE_THRESHOLD_DEFAULT,
                );
                let review_status = if (note_kind == "task" || note_kind == "event")
                    && conf < threshold
                {
                    "pending"
                } else {
                    "confirmed"
                };
                let summary = classified
                    .get("summary")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                // SYN-119 — the classifier detects the capture language server-side.
                let language = classified
                    .get("language")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty());
                let note_id = persist_atomic_note(
                    &conn,
                    atomic.trim(),
                    summary,
                    &mentioned,
                    capture_id,
                    &note_kind,
                    classified
                        .get("event_date")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty()),
                    truthy(classified.get("event_recurring")),
                    review_status,
                    language,
                )?;
                created_note_id = Some(note_id.clone());
                let title: String = if summary.is_empty() { atomic.trim() } else { summary }
                    .chars()
                    .take(60)
                    .collect();
                pending_note_vec = Some((note_id, format!("{title}\n{}", atomic.trim())));
            }
            report.created_note_id = created_note_id.clone();

            // Project entries — N per capture, dedup by lowercased canonical.
            let mut seen_projects: HashSet<String> = HashSet::new();
            for pe in arr(classified.get("project_entries")) {
                let Some(pc) = pe.get("project_canonical").and_then(Value::as_str) else {
                    continue;
                };
                let key = pc.trim().to_lowercase();
                if key.is_empty() || seen_projects.contains(&key) {
                    continue;
                }
                seen_projects.insert(key);
                let entry_content = pe
                    .get("content")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .unwrap_or(content)
                    .trim()
                    .to_string();
                let synthesis = persist_project_entry(
                    &conn,
                    pc.trim(),
                    &entry_content,
                    capture_id,
                    truthy(pe.get("is_new")),
                )?;
                report.project_syntheses.push(synthesis);
            }

            // Soft project-attach proposal for unrouted actionable captures.
            if seen_projects.is_empty() && (note_kind == "task" || is_ephemeral) {
                let attach_content = if !atomic.trim().is_empty() { atomic.trim() } else { content };
                self.propose_project_attach_if_similar(
                    &conn,
                    capture_id,
                    attach_content.trim(),
                    created_note_id.as_deref(),
                )?;
            }

            handle_intentions(&conn, classified, ctx)?;

            // SYN-19: a new mention reactivates the notes referencing it.
            let mentioned: Vec<String> = arr(classified.get("entities"))
                .iter()
                .filter_map(|e| e.get("canonical_name").and_then(Value::as_str))
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            crate::decay::reactivate_notes_for_entities(
                &conn,
                &mentioned,
                crate::decay::resolve_now(Some(&ctx.now_sql)),
            )?;

            mark(&conn, capture_id, &ctx.now, "processed")?;
            Ok(())
        })();
        finish_txn(&conn, r)?;
        drop(conn);

        // Post-commit, best-effort — mirrors the deferred vec flush.
        if let Some((note_id, text)) = pending_note_vec {
            if let Some(chunks) = self.embed_chunks(&text) {
                let _ = self.storage.upsert_note_vectors(&note_id, &chunks);
            }
        }

        Ok(report)
    }

    /// Report → JSON for the FFI/PyO3 boundary.
    pub fn report_to_json(report: &RouteReport) -> Value {
        json!({
            "entity_ids": report.entity_ids,
            "new_facts": report.new_facts,
            "created_note_id": report.created_note_id,
            "fast_exit": report.fast_exit,
            "project_syntheses": report.project_syntheses.iter().map(|s| json!({
                "project_id": s.project_id,
                "entry_id": s.entry_id,
                "project_name": s.project_name,
                "entry_content": s.entry_content,
                "entry_count": s.entry_count,
            })).collect::<Vec<_>>(),
        })
    }

    /// Host-facing `insert_fact` (validation endpoints, reclassify) — same
    /// dedup-reinforce + SYN-37 supersede as the routing path.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_user_fact(
        &self,
        entity_id: &str,
        predicate: &str,
        value: Value,
        confidence: f64,
        source_inbox_id: Value,
        persistence_value: i64,
        provenance_capture_id: Option<String>,
        category: Value,
    ) -> Result<String, CoreError> {
        let conn = self.storage.lock()?;
        insert_fact(
            &conn, entity_id, predicate, value, confidence, source_inbox_id,
            persistence_value, provenance_capture_id, category,
        )
    }

    /// Host-facing `_find_existing_entity` (alias-aware) → entity id.
    pub fn find_entity(
        &self,
        canonical_name: &str,
        aliases: &[String],
    ) -> Result<Option<String>, CoreError> {
        let conn = self.storage.lock()?;
        Ok(find_existing_entity(&conn, canonical_name, aliases)?
            .and_then(|row| row.get("id").and_then(Value::as_str).map(String::from)))
    }

    /// Port of `step5_validate_pending`: corroborated pending facts promote.
    pub fn validate_pending(&self, new_facts: &[Value]) -> Result<i64, CoreError> {
        let conn = self.storage.lock()?;
        let pending: Vec<(String, String)> = {
            let mut stmt = conn.prepare("SELECT id, fact_data FROM pending_facts")?;
            let rows = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };

        let mut promoted = 0i64;
        for (pending_id, raw) in pending {
            let Ok(pf) = serde_json::from_str::<Value>(&raw) else {
                continue;
            };
            let corroborator = new_facts.iter().find(|nf| {
                nf.get("predicate") == pf.get("predicate")
                    && nf
                        .get("entity_canonical")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_lowercase()
                        == pf
                            .get("entity_canonical")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_lowercase()
                    && py_str(nf.get("source_inbox_id")) != py_str(pf.get("source_inbox_id"))
            });
            let Some(corroborator) = corroborator else {
                continue;
            };

            let new_conf = compute_confidence(
                persistence_value(&pf),
                corroborator
                    .get("evidence_strength")
                    .and_then(Value::as_str)
                    .unwrap_or("explicit"),
                true,
                2,
            );
            if new_conf <= 0.85 {
                continue;
            }

            let entity_name = pf
                .get("entity_canonical")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let row = find_existing_entity(&conn, entity_name, &[])?;
            // Post-SYN-112 payloads carry uuid strings; a pre-migration
            // number is kept verbatim (same dangling-ref policy as migrate).
            let prov_id: Option<String> = match pf.get("source_inbox_id") {
                Some(Value::Number(n)) => Some(n.to_string()),
                Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            };
            let entity_id = match row {
                Some(r) => r.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
                None => {
                    let id = new_uuid();
                    conn.execute(
                        "INSERT INTO entities (id, canonical_name, provenance_capture_id) \
                         VALUES (?1, ?2, ?3)",
                        params![id, entity_name, prov_id],
                    )?;
                    id
                }
            };
            insert_fact(
                &conn,
                &entity_id,
                pf.get("predicate").and_then(Value::as_str).unwrap_or(""),
                pf.get("value").cloned().unwrap_or(Value::Null),
                new_conf,
                pf.get("source_inbox_id").cloned().unwrap_or(Value::Null),
                persistence_value(&pf),
                prov_id,
                pf.get("category").cloned().unwrap_or(Value::Null),
            )?;
            conn.execute("DELETE FROM pending_facts WHERE id = ?1", params![pending_id])?;
            promoted += 1;
        }
        Ok(promoted)
    }

    // ── step 2 — resolve ────────────────────────────────────────────────

    fn resolve(&self, conn: &Connection, classified: &Value, ctx: &RouteContext) -> Vec<Resolved> {
        let mut out = Vec::new();
        for entity_data in arr(classified.get("entities")) {
            let aliases: Vec<String> = arr(entity_data.get("aliases"))
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect();
            let canonical = entity_data
                .get("canonical_name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let existing = find_existing_entity(conn, canonical, &aliases).unwrap_or(None);

            let mut facts = Vec::new();
            for fact in arr(entity_data.get("facts")) {
                let mut fact = fact.clone();
                let predicate = fact.get("predicate").and_then(Value::as_str).unwrap_or("");
                if DATE_PREDICATE_KEYWORDS.iter().any(|kw| predicate.contains(kw)) {
                    if let Some(v) = fact.get("value").and_then(Value::as_str) {
                        let resolved = resolve_date(v, &ctx.today);
                        if let Value::Object(m) = &mut fact {
                            m.insert("value".into(), Value::String(resolved));
                        }
                    }
                }
                facts.push(fact);
            }
            out.push(Resolved {
                data: entity_data.clone(),
                existing,
                facts,
            });
        }
        out
    }

    // ── step 4 — route ──────────────────────────────────────────────────

    fn step4_route(
        &self,
        conn: &Connection,
        classified: &Value,
        resolved: &[Resolved],
        source_inbox_id: &str,
        anchors_durable_note: bool,
        ctx: &RouteContext,
    ) -> Result<Vec<String>, CoreError> {
        let mut entity_ids: Vec<String> = Vec::new();

        let mut relation_names: HashSet<String> = HashSet::new();
        let mut relation_targets_by_from: HashMap<String, HashSet<String>> = HashMap::new();
        for rel in arr(classified.get("relations")) {
            for key in ["from", "to"] {
                if let Some(name) = rel.get(key).and_then(Value::as_str) {
                    if !name.is_empty() {
                        relation_names.insert(name.trim().to_lowercase());
                    }
                }
            }
            let rfrom = rel.get("from").and_then(Value::as_str).unwrap_or("").trim().to_lowercase();
            let rto = rel.get("to").and_then(Value::as_str).unwrap_or("").trim().to_lowercase();
            if !rfrom.is_empty() && !rto.is_empty() {
                relation_targets_by_from.entry(rfrom).or_default().insert(rto);
            }
        }

        let active_types: HashSet<String> = {
            let mut stmt = conn.prepare("SELECT type FROM active_entity_types")?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            rows.into_iter().collect()
        };
        let project_canonicals: HashSet<String> = arr(classified.get("project_entries"))
            .iter()
            .map(|pe| {
                pe.get("project_canonical")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_lowercase()
            })
            .collect();

        for res in resolved {
            let mut entity_data = res.data.clone();
            let canonical = entity_data
                .get("canonical_name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            if canonical.is_empty() {
                continue;
            }
            let existing = &res.existing;
            let mention_count = existing
                .as_ref()
                .map(|e| e.get("mention_count").and_then(Value::as_i64).unwrap_or(1) + 1)
                .unwrap_or(1);

            // SYN-58 type guards — new entities only.
            let mut type_proposal: Option<(String, Option<String>)> = None;
            let mut entity_status = "active";
            if existing.is_none() {
                let etype = entity_data
                    .get("type")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("concept")
                    .trim()
                    .to_string();
                if etype == "project" && !project_canonicals.contains(&canonical.to_lowercase()) {
                    if let Value::Object(m) = &mut entity_data {
                        m.insert("type".into(), Value::String("concept".into()));
                    }
                }
                if let Some(tp) = entity_data.get("type_proposal").filter(|v| v.is_object()) {
                    let proposed = tp
                        .get("value")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !proposed.is_empty() && !active_types.contains(&proposed) {
                        let reason = tp.get("reason").and_then(Value::as_str).map(String::from);
                        type_proposal = Some((proposed, reason));
                        entity_status = "pending";
                    }
                }
            }

            // Fact scoring + anti-redite dedup.
            let empty: HashSet<String> = HashSet::new();
            let rel_targets = relation_targets_by_from
                .get(&canonical.to_lowercase())
                .unwrap_or(&empty);
            let mut scored: Vec<(Value, f64)> = Vec::new();
            for fact in &res.facts {
                let value_lower = fact
                    .get("value")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_lowercase();
                if rel_targets.contains(&value_lower) {
                    continue; // covered by a relation edge (anti-redite)
                }
                let confidence = compute_confidence(
                    persistence_value(fact),
                    fact.get("evidence_strength")
                        .and_then(Value::as_str)
                        .unwrap_or("explicit"),
                    existing.is_some(),
                    mention_count,
                );
                scored.push((fact.clone(), confidence));
            }

            // Entity creation, decoupled from fact confidence.
            let has_facts = !res.facts.is_empty();
            let max_persistence = if has_facts {
                entity_persistence(&res.facts)
            } else {
                0.0
            };
            let should_create = existing.is_some()
                || relation_names.contains(&canonical.to_lowercase())
                || max_persistence >= MIN_ENTITY_PERSISTENCE
                || anchors_durable_note;

            let mut entity_id: Option<String> = None;
            if should_create {
                let id = upsert_entity(
                    conn,
                    &entity_data,
                    existing.as_ref(),
                    &res.facts,
                    source_inbox_id,
                    entity_status,
                    &ctx.today,
                )?;
                if !entity_ids.contains(&id) {
                    entity_ids.push(id.clone());
                }
                if existing.is_none() {
                    if let Some((proposed, reason)) = &type_proposal {
                        conn.execute(
                            "INSERT INTO entity_type_proposals \
                             (id, proposed_type, reason, evidence_capture_id, candidate_entity_id) \
                             VALUES (?1,?2,?3,?4,?5)",
                            params![new_uuid(), proposed, reason, source_inbox_id, id],
                        )?;
                    }
                    // Python `entity_data.get("type", "concept")`: missing →
                    // "concept", an explicit null stays None (SQL `type = NULL`
                    // matches nothing; the embedding fallback then searches all
                    // types because type_filter=None).
                    let merge_type = match entity_data.get("type") {
                        None => Some("concept"),
                        Some(Value::Null) => None,
                        Some(v) => v.as_str(),
                    };
                    self.propose_merge_if_similar(
                        conn,
                        &id,
                        &canonical,
                        merge_type,
                        source_inbox_id,
                    )?;
                }
                entity_id = Some(id);
            }

            for (fact, confidence) in &scored {
                if *confidence > 0.85 {
                    if let Some(eid) = &entity_id {
                        insert_fact(
                            conn,
                            eid,
                            fact.get("predicate").and_then(Value::as_str).unwrap_or(""),
                            fact.get("value").cloned().unwrap_or(Value::Null),
                            *confidence,
                            Value::String(source_inbox_id.to_string()),
                            persistence_value(fact),
                            Some(source_inbox_id.to_string()),
                            fact.get("category").cloned().unwrap_or(Value::Null),
                        )?;
                    }
                } else {
                    let fact_data = json!({
                        "entity_canonical": entity_data.get("canonical_name"),
                        "predicate": fact.get("predicate"),
                        "value": fact.get("value"),
                        "persistence_value": fact.get("persistence_value").cloned()
                            .unwrap_or(json!(3)),
                        "evidence_strength": fact.get("evidence_strength").cloned()
                            .unwrap_or(json!("explicit")),
                        "category": fact.get("category"),
                        "confidence": confidence,
                        "source_inbox_id": source_inbox_id,
                    });
                    if *confidence >= 0.5 {
                        conn.execute(
                            "INSERT INTO pending_facts (id, fact_data, validation_strategy) \
                             VALUES (?1,?2,?3)",
                            params![new_uuid(), py_dumps_ascii(&fact_data), "passive"],
                        )?;
                    } else {
                        conn.execute(
                            "INSERT INTO review_queue (id, fact_data, suggested_entity) \
                             VALUES (?1,?2,?3)",
                            params![
                                new_uuid(),
                                py_dumps_ascii(&fact_data),
                                entity_data.get("canonical_name").and_then(Value::as_str)
                            ],
                        )?;
                    }
                }
            }
        }

        // Relations — both endpoints must already exist; confidence-gated.
        let rel_threshold = env_f64(
            "SYNAPSE_REVIEW_CONFIDENCE_THRESHOLD",
            REVIEW_CONFIDENCE_THRESHOLD_DEFAULT,
        );
        for rel in arr(classified.get("relations")) {
            let from_name = rel.get("from").and_then(Value::as_str).unwrap_or("");
            let predicate = rel.get("predicate").and_then(Value::as_str).unwrap_or("");
            let to_name = rel.get("to").and_then(Value::as_str).unwrap_or("");
            if from_name.is_empty() || predicate.is_empty() || to_name.is_empty() {
                continue;
            }
            let rel_conf = py_float(rel.get("confidence")).unwrap_or(1.0);
            let review_status = if rel_conf < rel_threshold { "pending" } else { "confirmed" };
            let lookup = |name: &str| -> Result<Option<String>, CoreError> {
                let mut stmt = conn.prepare(
                    "SELECT id FROM entities WHERE LOWER(canonical_name) = LOWER(?1)",
                )?;
                let mut rows = stmt.query(params![name])?;
                Ok(match rows.next()? {
                    Some(row) => Some(row.get(0)?),
                    None => None,
                })
            };
            if let (Some(from_id), Some(to_id)) = (lookup(from_name)?, lookup(to_name)?) {
                conn.execute(
                    "INSERT INTO relations \
                     (id, entity_from, predicate, entity_to, confidence, review_status, \
                      provenance_capture_id) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    params![new_uuid(), from_id, predicate, to_id, rel_conf, review_status,
                            source_inbox_id],
                )?;
            }
        }

        Ok(entity_ids)
    }

    // ── merge + attach proposals ────────────────────────────────────────

    fn propose_merge_if_similar(
        &self,
        conn: &Connection,
        new_id: &str,
        new_name: &str,
        new_type: Option<&str>,
        capture_id: &str,
    ) -> Result<(), CoreError> {
        if new_name.is_empty() {
            return Ok(());
        }
        let needle = new_name.to_lowercase().trim().to_string();
        let needle_tokens: HashSet<&str> = needle.split_whitespace().collect();
        let candidates: Vec<(String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT id, canonical_name FROM entities \
                 WHERE id != ?1 AND type = ?2 AND merged_into_id IS NULL",
            )?;
            let type_param: SqlV = match new_type {
                Some(t) => SqlV::Text(t.to_string()),
                None => SqlV::Null,
            };
            let rows = stmt
                .query_map(params![new_id, type_param], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default()))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        for (cid, cname) in &candidates {
            let ex_lower = cname.trim().to_lowercase();
            if ex_lower == needle {
                continue;
            }
            if !(ex_lower.contains(&needle) || needle.contains(&ex_lower)) {
                continue;
            }
            let ex_tokens: HashSet<&str> = ex_lower.split_whitespace().collect();
            if needle_tokens.is_disjoint(&ex_tokens) {
                continue;
            }
            if record_merge_proposal(conn, new_id, cid, 0.9, "name_substring", capture_id)? {
                return Ok(());
            }
        }

        // SYN-61 embedding fallback.
        let threshold = env_f64(
            "SYNAPSE_MERGE_EMBEDDING_THRESHOLD",
            MERGE_EMBEDDING_THRESHOLD_DEFAULT,
        );
        let entity = query_row_map(conn, "SELECT * FROM entities WHERE id = ?1", &[SqlV::from(new_id.to_string())])?;
        let Some(entity) = entity else { return Ok(()) };
        let Some(vec) = self.embed(&entity_embedding_text(&entity)) else {
            return Ok(()); // embed failure → skipped, like Python
        };
        let matches = search_entities_on(conn, &vec, 5, threshold, new_type,
                                         &[new_id.to_string()])?;
        for m in &matches {
            let reason = format!("embedding_{:.2}", m.score);
            if record_merge_proposal(conn, new_id, &m.id, m.score, &reason, capture_id)? {
                return Ok(());
            }
        }
        Ok(())
    }

    fn propose_project_attach_if_similar(
        &self,
        conn: &Connection,
        capture_id: &str,
        content: &str,
        note_id: Option<&str>,
    ) -> Result<bool, CoreError> {
        if content.trim().is_empty() {
            return Ok(false);
        }
        let already: i64 = conn.query_row(
            "SELECT COUNT(*) FROM project_entries WHERE capture_id = ?1",
            params![capture_id],
            |r| r.get(0),
        )?;
        if already > 0 {
            return Ok(false);
        }
        let threshold = env_f64("SYNAPSE_PROJECT_ATTACH_THRESHOLD", PROJECT_ATTACH_THRESHOLD_DEFAULT);
        let margin = env_f64("SYNAPSE_PROJECT_ATTACH_MARGIN", PROJECT_ATTACH_MARGIN_DEFAULT);
        let Some(vec) = self.embed(content) else {
            return Ok(false);
        };
        let matches = search_entities_on(conn, &vec, 2, 0.0, Some("project"), &[])?;
        if matches.is_empty() || matches[0].score < threshold {
            return Ok(false);
        }
        if matches.len() > 1 && (matches[0].score - matches[1].score) < margin {
            return Ok(false);
        }
        let m = &matches[0];
        let dup: i64 = conn.query_row(
            "SELECT COUNT(*) FROM project_attach_proposals \
             WHERE capture_id = ?1 AND project_id = ?2 AND status = 'pending'",
            params![capture_id, m.id],
            |r| r.get(0),
        )?;
        if dup > 0 {
            return Ok(false);
        }
        conn.execute(
            "INSERT INTO project_attach_proposals \
             (id, capture_id, note_id, project_id, content, similarity_score) \
             VALUES (?1,?2,?3,?4,?5,?6)",
            params![new_uuid(), capture_id, note_id, m.id, content.trim(), m.score],
        )?;
        Ok(true)
    }
}

struct Resolved {
    data: Value,
    existing: Option<Map<String, Value>>,
    facts: Vec<Value>,
}

// ── shared helpers (ports of the module-level Python functions) ─────────

pub(crate) fn new_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn finish_txn(conn: &Connection, r: Result<(), CoreError>) -> Result<(), CoreError> {
    match r {
        Ok(()) => {
            conn.execute_batch("COMMIT")?;
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// Python truthiness over JSON values (None/False/0/""/[]/{} are false).
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

fn arr(v: Option<&Value>) -> &[Value] {
    v.and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

/// Python `float(x)` over a JSON value: number, numeric string or bool;
/// anything else (incl. missing/null) is the caller's fallback.
fn py_float(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse().ok(),
        Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Python `str(x)` for the source-id comparison in step5 (None → "None").
fn py_str(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "None".into(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => if *b { "True".into() } else { "False".into() },
        Some(other) => other.to_string(),
    }
}

fn persistence_value(fact: &Value) -> i64 {
    fact.get("persistence_value").and_then(Value::as_i64).unwrap_or(3)
}

/// `_entity_persistence`: strongest persistence among the facts, 3 if none.
fn entity_persistence(facts: &[Value]) -> f64 {
    if facts.is_empty() {
        return 3.0;
    }
    facts
        .iter()
        .map(|f| persistence_value(f) as f64)
        .fold(f64::NEG_INFINITY, f64::max)
}

pub(crate) fn compute_confidence(
    persistence: i64,
    evidence_strength: &str,
    existing: bool,
    mention_count: i64,
) -> f64 {
    let base = match evidence_strength {
        "hedged" => 0.65,
        "implicit" => 0.40,
        _ => 0.92, // explicit + unknown values fall back to explicit
    };
    let mut bonus = 0.0_f64;
    if existing {
        bonus += 0.05;
    }
    bonus += (mention_count as f64 * 0.02).min(0.05);
    bonus += match persistence {
        5 => 0.2,
        4 => 0.15,
        3 => 0.05,
        2 => 0.0,
        1 => -0.1,
        _ => 0.0,
    };
    let mut score = base + bonus;
    if evidence_strength == "hedged" {
        score = score.min(0.84);
    }
    score.clamp(0.0, 1.0)
}

/// Generic row → JSON map (blobs become Null — never consumed by routing).
pub(crate) fn query_row_map(
    conn: &Connection,
    sql: &str,
    params: &[SqlV],
) -> Result<Option<Map<String, Value>>, CoreError> {
    let mut stmt = conn.prepare(sql)?;
    let columns: Vec<String> = stmt.column_names().iter().map(|c| c.to_string()).collect();
    let mut rows = stmt.query(params_from_iter(params.iter().cloned()))?;
    match rows.next()? {
        Some(row) => {
            let mut map = Map::new();
            for (i, col) in columns.iter().enumerate() {
                let v = match row.get_ref(i)? {
                    rusqlite::types::ValueRef::Null => Value::Null,
                    rusqlite::types::ValueRef::Integer(n) => json!(n),
                    rusqlite::types::ValueRef::Real(f) => json!(f),
                    rusqlite::types::ValueRef::Text(t) => {
                        Value::String(String::from_utf8_lossy(t).into_owned())
                    }
                    rusqlite::types::ValueRef::Blob(_) => Value::Null,
                };
                map.insert(col.clone(), v);
            }
            Ok(Some(map))
        }
        None => Ok(None),
    }
}

pub(crate) fn query_row_maps(
    conn: &Connection,
    sql: &str,
    params: &[SqlV],
) -> Result<Vec<Map<String, Value>>, CoreError> {
    let mut stmt = conn.prepare(sql)?;
    let columns: Vec<String> = stmt.column_names().iter().map(|c| c.to_string()).collect();
    let mut rows = stmt.query(params_from_iter(params.iter().cloned()))?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let mut map = Map::new();
        for (i, col) in columns.iter().enumerate() {
            let v = match row.get_ref(i)? {
                rusqlite::types::ValueRef::Null => Value::Null,
                rusqlite::types::ValueRef::Integer(n) => json!(n),
                rusqlite::types::ValueRef::Real(f) => json!(f),
                rusqlite::types::ValueRef::Text(t) => {
                    Value::String(String::from_utf8_lossy(t).into_owned())
                }
                rusqlite::types::ValueRef::Blob(_) => Value::Null,
            };
            map.insert(col.clone(), v);
        }
        out.push(map);
    }
    Ok(out)
}

/// Port of `_find_existing_entity`: primary SQL-cased lookup, then the
/// Python-cased alias scan (first DB-row match wins).
fn find_existing_entity(
    conn: &Connection,
    canonical_name: &str,
    aliases: &[String],
) -> Result<Option<Map<String, Value>>, CoreError> {
    if let Some(row) = query_row_map(
        conn,
        "SELECT * FROM entities WHERE LOWER(canonical_name) = LOWER(?1) \
         AND merged_into_id IS NULL",
        &[SqlV::from(canonical_name.to_string())],
    )? {
        return Ok(Some(row));
    }

    let mut search_names: HashSet<String> = HashSet::new();
    search_names.insert(canonical_name.to_lowercase());
    for a in aliases {
        search_names.insert(a.to_lowercase());
    }
    for entity in query_row_maps(conn, "SELECT * FROM entities WHERE merged_into_id IS NULL", &[])? {
        let entity_aliases: Vec<String> = entity
            .get("aliases")
            .and_then(Value::as_str)
            .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
            .unwrap_or_default();
        let mut existing_names: HashSet<String> = HashSet::new();
        existing_names.insert(
            entity
                .get("canonical_name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_lowercase(),
        );
        for a in &entity_aliases {
            existing_names.insert(a.to_lowercase());
        }
        if !search_names.is_disjoint(&existing_names) {
            return Ok(Some(entity));
        }
    }
    Ok(None)
}

/// Port of `_upsert_entity` (aliases union, attributes merge new-wins,
/// mention bump, MAX persistence; INSERT carries provenance + status).
fn upsert_entity(
    conn: &Connection,
    entity_data: &Value,
    existing: Option<&Map<String, Value>>,
    facts: &[Value],
    capture_id: &str,
    status: &str,
    today: &str,
) -> Result<String, CoreError> {
    let summary = entity_data.get("summary").and_then(Value::as_str);
    let attributes = entity_data
        .get("attributes")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let persistence = if facts.is_empty() {
        3.0
    } else {
        entity_persistence(facts)
    };

    if let Some(existing) = existing {
        let entity_id = existing
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let existing_aliases: Vec<String> = existing
            .get("aliases")
            .and_then(Value::as_str)
            .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
            .unwrap_or_default();
        let mut merged: Vec<String> = existing_aliases;
        for a in arr(entity_data.get("aliases")).iter().filter_map(Value::as_str) {
            if !merged.iter().any(|m| m == a) {
                merged.push(a.to_string());
            }
        }
        let mut merged_attrs: Map<String, Value> = existing
            .get("attributes")
            .and_then(Value::as_str)
            .and_then(|s| serde_json::from_str::<Map<String, Value>>(s).ok())
            .unwrap_or_default();
        for (k, v) in attributes {
            merged_attrs.insert(k, v); // new keys win
        }
        let new_summary = summary
            .map(String::from)
            .or_else(|| existing.get("summary").and_then(Value::as_str).map(String::from));
        conn.execute(
            "UPDATE entities SET aliases=?1, attributes=?2, summary=?3, \
             mention_count=mention_count+1, last_mentioned=?4, \
             persistence_value=MAX(persistence_value, ?5) WHERE id=?6",
            params![
                serde_json::to_string(&merged).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&Value::Object(merged_attrs)).unwrap_or_else(|_| "{}".into()),
                new_summary,
                today,
                persistence,
                entity_id,
            ],
        )?;
        Ok(entity_id)
    } else {
        let entity_id = new_uuid();
        let aliases: Vec<String> = arr(entity_data.get("aliases"))
            .iter()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect();
        conn.execute(
            "INSERT INTO entities \
             (id, type, canonical_name, aliases, attributes, summary, last_mentioned, \
              persistence_value, provenance_capture_id, status) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                entity_id,
                match entity_data.get("type") {
                    None => SqlV::Text("concept".into()),
                    Some(Value::Null) => SqlV::Null,
                    Some(v) => SqlV::Text(v.as_str().unwrap_or("concept").into()),
                },
                entity_data.get("canonical_name").and_then(Value::as_str).unwrap_or(""),
                serde_json::to_string(&aliases).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&Value::Object(attributes)).unwrap_or_else(|_| "{}".into()),
                summary,
                today,
                persistence,
                capture_id,
                status,
            ],
        )?;
        Ok(entity_id)
    }
}

fn record_merge_proposal(
    conn: &Connection,
    new_id: &str,
    existing_id: &str,
    score: f64,
    reason: &str,
    capture_id: &str,
) -> Result<bool, CoreError> {
    let exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM entity_merge_proposals \
         WHERE (candidate_entity_id=?1 AND existing_entity_id=?2) \
            OR (candidate_entity_id=?2 AND existing_entity_id=?1)",
        params![new_id, existing_id],
        |r| r.get(0),
    )?;
    if exists > 0 {
        return Ok(false);
    }
    conn.execute(
        "INSERT INTO entity_merge_proposals \
         (id, candidate_entity_id, existing_entity_id, similarity_score, \
          similarity_reason, evidence_capture_id) VALUES (?1,?2,?3,?4,?5,?6)",
        params![new_uuid(), new_id, existing_id, score, reason, capture_id],
    )?;
    Ok(true)
}

/// Port of `facts_store.insert_fact` (dedup-reinforce + SYN-37 supersede).
/// pub(crate): the SQL gateway re-exposes it on the HOST's connection
/// (`SqlConnection::insert_fact`) so user-action endpoints keep their open
/// transaction (T5 — the Python copy is gone).
#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_fact(
    conn: &Connection,
    entity_id: &str,
    predicate: &str,
    value: Value,
    confidence: f64,
    source_inbox_id: Value,
    persistence_value: i64,
    provenance_capture_id: Option<String>,
    category: Value,
) -> Result<String, CoreError> {
    let fact_id = new_uuid();
    let value_sql = json_scalar_to_sql(&value);
    let dup = {
        let mut stmt = conn.prepare(
            "SELECT id, confidence FROM facts \
             WHERE entity_id = ?1 AND LOWER(TRIM(predicate)) = LOWER(TRIM(?2)) \
             AND LOWER(TRIM(value)) = LOWER(TRIM(?3)) \
             AND obsoleted_at IS NULL AND archived_at IS NULL LIMIT 1",
        )?;
        let mut rows = stmt.query(params![entity_id, predicate, value_sql])?;
        match rows.next()? {
            Some(row) => Some((row.get::<_, String>(0)?, row.get::<_, Option<f64>>(1)?)),
            None => None,
        }
    };
    if let Some((dup_id, dup_conf)) = dup {
        conn.execute(
            "UPDATE facts SET confidence = ?1, last_confirmed = CURRENT_TIMESTAMP WHERE id = ?2",
            params![confidence.max(dup_conf.unwrap_or(0.0)), dup_id],
        )?;
        return Ok(dup_id);
    }
    if SINGLE_VALUED_PREDICATES.contains(&predicate.trim().to_lowercase().as_str()) {
        let existing: Vec<(String, Option<f64>)> = {
            let mut stmt = conn.prepare(
                "SELECT id, confidence FROM facts \
                 WHERE entity_id = ?1 AND predicate = ?2 \
                 AND obsoleted_at IS NULL AND archived_at IS NULL",
            )?;
            let rows = stmt
                .query_map(params![entity_id, predicate], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        for (ex_id, ex_conf) in existing {
            if confidence >= ex_conf.unwrap_or(0.0) {
                conn.execute(
                    "UPDATE facts SET obsoleted_at = CURRENT_TIMESTAMP, obsoleted_by = ?1 \
                     WHERE id = ?2",
                    params![fact_id, ex_id],
                )?;
            }
        }
    }
    conn.execute(
        "INSERT INTO facts \
         (id, entity_id, predicate, value, confidence, source_inbox_id, \
          persistence_value, provenance_capture_id, category) \
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
        params![
            fact_id,
            entity_id,
            predicate,
            value_sql,
            confidence,
            json_scalar_to_sql(&source_inbox_id),
            persistence_value,
            provenance_capture_id,
            json_scalar_to_sql(&category),
        ],
    )?;
    conn.execute(
        "UPDATE entities SET summary_stale = 1 WHERE id = ?1",
        params![entity_id],
    )?;
    Ok(fact_id)
}

/// Bind a JSON scalar like Python bound the native value (str/int/float/
/// bool/None); structures fall back to their compact JSON text.
fn json_scalar_to_sql(v: &Value) -> SqlV {
    match v {
        Value::Null => SqlV::Null,
        Value::Bool(b) => SqlV::Integer(*b as i64),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                SqlV::Integer(i)
            } else {
                SqlV::Real(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(s) => SqlV::Text(s.clone()),
        other => SqlV::Text(other.to_string()),
    }
}

fn mark(conn: &Connection, entry_id: &str, now: &str, status: &str) -> Result<(), CoreError> {
    conn.execute(
        "UPDATE inbox SET processed_at=?1, status=?2, error=NULL WHERE id=?3",
        params![now, status, entry_id],
    )?;
    Ok(())
}

/// Port of `handle_intentions` (expired purge + optional insert).
fn handle_intentions(
    conn: &Connection,
    classified: &Value,
    ctx: &RouteContext,
) -> Result<(), CoreError> {
    conn.execute(
        "DELETE FROM intentions WHERE created_at < ?1 AND resolved = 0",
        params![ctx.intentions_cutoff],
    )?;
    let is_ephemeral = truthy(classified.get("is_ephemeral"))
        || classified.get("input_type").and_then(Value::as_str) == Some("ephemeral");
    if is_ephemeral {
        let source = classified
            .get("ephemeral_content")
            .filter(|v| truthy(Some(v)))
            .cloned()
            .unwrap_or_else(|| classified.get("summary").cloned().unwrap_or(json!("")));
        let content = intention_text(&source);
        if !content.is_empty() {
            conn.execute(
                "INSERT INTO intentions (id, content, ttl_hours) VALUES (?1,?2,?3)",
                params![new_uuid(), content, 48],
            )?;
        }
    }
    Ok(())
}

/// Port of `_intention_text` (dict/list coercion into TEXT).
fn intention_text(value: &Value) -> String {
    let mut v = value.clone();
    if let Value::Object(m) = &v {
        v = m
            .get("content")
            .filter(|x| truthy(Some(x)))
            .or_else(|| m.get("text").filter(|x| truthy(Some(x))))
            .or_else(|| m.get("description").filter(|x| truthy(Some(x))))
            .or_else(|| m.get("items").filter(|x| truthy(Some(x))))
            .cloned()
            .unwrap_or_else(|| Value::String(py_dumps(&v)));
    }
    if let Value::Array(items) = &v {
        let joined: Vec<String> = items
            .iter()
            .filter(|x| truthy(Some(x)))
            .map(py_scalar_str)
            .collect();
        v = Value::String(joined.join(" · "));
    }
    match &v {
        Value::Null => String::new(),
        Value::String(s) => s.trim().to_string(),
        other => py_scalar_str(other).trim().to_string(),
    }
}

/// Python `str()` of a JSON scalar (dicts/lists via py_dumps-ish repr are
/// not needed on the corpus; keep JSON text for them).
fn py_scalar_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
fn persist_atomic_note(
    conn: &Connection,
    content: &str,
    summary: &str,
    entities_mentioned: &[String],
    capture_id: &str,
    kind: &str,
    event_date: Option<&str>,
    event_recurring: bool,
    review_status: &str,
    language: Option<&str>,
) -> Result<String, CoreError> {
    let kind = if ["note", "task", "event"].contains(&kind) { kind } else { "note" };
    let review_status = if ["confirmed", "pending"].contains(&review_status) {
        review_status
    } else {
        "confirmed"
    };
    let title: String = if summary.is_empty() { content } else { summary }
        .chars()
        .take(60)
        .collect();
    let durable = kind == "event" || kind == "task";
    let note_id = new_uuid();
    conn.execute(
        "INSERT INTO atomic_notes \
         (id, title, content, summary, entities_mentioned, memory_strength, \
          provenance_capture_id, kind, event_date, event_recurring, review_status, language) \
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
        params![
            note_id,
            title,
            content,
            summary,
            serde_json::to_string(entities_mentioned).unwrap_or_else(|_| "[]".into()),
            1.0,
            capture_id,
            kind,
            if durable { event_date } else { None },
            (durable && event_recurring) as i64,
            review_status,
            language,
        ],
    )?;
    Ok(note_id)
}

pub(crate) fn persist_project_entry(
    conn: &Connection,
    canonical: &str,
    content: &str,
    capture_id: &str,
    is_new_project: bool,
) -> Result<ProjectSynthesis, CoreError> {
    let existing: Option<String> = {
        let mut stmt = conn.prepare(
            "SELECT id FROM entities WHERE type='project' AND LOWER(canonical_name) = LOWER(?1)",
        )?;
        let mut rows = stmt.query(params![canonical])?;
        match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        }
    };
    let project_id = match existing {
        Some(id) => {
            conn.execute(
                "UPDATE entities SET mention_count = mention_count + 1, \
                 last_mentioned = DATE('now') WHERE id = ?1",
                params![id],
            )?;
            id
        }
        None => {
            let id = new_uuid();
            conn.execute(
                "INSERT INTO entities \
                 (id, type, canonical_name, mention_count, last_mentioned, persistence_value, \
                  summary, provenance_capture_id) \
                 VALUES (?1, 'project', ?2, 1, DATE('now'), 3, ?3, ?4)",
                params![
                    id,
                    canonical,
                    if is_new_project {
                        Some("Projet créé automatiquement par le Dream Cycle.")
                    } else {
                        None
                    },
                    capture_id
                ],
            )?;
            id
        }
    };
    let entry_id = new_uuid();
    conn.execute(
        "INSERT INTO project_entries (id, project_id, capture_id, content, kind) \
         VALUES (?1, ?2, ?3, ?4, 'note')",
        params![entry_id, project_id, capture_id, content],
    )?;
    let entry_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM project_entries WHERE project_id = ?1",
        params![project_id],
        |r| r.get(0),
    )?;
    Ok(ProjectSynthesis {
        project_id,
        entry_id,
        project_name: canonical.to_string(),
        entry_content: content.to_string(),
        entry_count,
    })
}

/// Port of `entity_embedding_text` — the exact text fastembed/the core
/// embeds for an entity; `py_dumps` keeps Python's JSON byte layout so the
/// vectors stay comparable.
pub(crate) fn entity_embedding_text(entity: &Map<String, Value>) -> String {
    let aliases: Vec<String> = entity
        .get("aliases")
        .and_then(Value::as_str)
        .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
        .unwrap_or_default();
    let attributes: Value = entity
        .get("attributes")
        .and_then(Value::as_str)
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_else(|| json!({}));
    format!(
        "Nom: {}\nType: {}\nAliases: {}\nAttributs: {}\nRésumé: {}",
        entity.get("canonical_name").and_then(Value::as_str).unwrap_or(""),
        entity.get("type").and_then(Value::as_str).unwrap_or(""),
        aliases.join(", "),
        py_dumps(&attributes),
        entity.get("summary").and_then(Value::as_str).unwrap_or(""),
    )
}

/// `json.dumps(v, ensure_ascii=False)` — Python's default separators
/// (", ", ": ") and insertion order (serde_json preserve_order).
fn py_dumps(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let inner: Vec<String> = m
                .iter()
                .map(|(k, val)| format!("{}: {}", serde_json::to_string(k).unwrap(), py_dumps(val)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Array(a) => {
            let inner: Vec<String> = a.iter().map(py_dumps).collect();
            format!("[{}]", inner.join(", "))
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// `json.dumps(v)` with ensure_ascii=True (pending/review fact_data uses the
/// Python default). Non-ASCII chars are \uXXXX-escaped like CPython.
fn py_dumps_ascii(v: &Value) -> String {
    let raw = py_dumps(v);
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        if (c as u32) < 0x80 {
            out.push(c);
        } else {
            let mut buf = [0u16; 2];
            for unit in c.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// Minimal deterministic stand-in for `dateparser.parse(...).date()`,
/// covering the value shapes the classifier actually produces (it is told
/// to resolve dates itself): ISO dates pass through, a bare year resolves
/// like dateparser does (current month, PREFER_DAY_OF_MONTH=first), and
/// the few English/French relative phrases seen in the wild. Anything else
/// returns unchanged — same as a dateparser miss.
fn resolve_date(value: &str, today: &str) -> String {
    let v = value.trim();
    // Already ISO (date or datetime prefix).
    let bytes = v.as_bytes();
    let is_iso = v.len() >= 10
        && bytes[0..4].iter().all(u8::is_ascii_digit)
        && bytes[4] == b'-'
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[7] == b'-'
        && bytes[8..10].iter().all(u8::is_ascii_digit);
    if is_iso {
        return v[0..10].to_string();
    }
    // Bare year → year-<current month>-01 (dateparser fills missing month
    // from the current date and the day from PREFER_DAY_OF_MONTH=first).
    if v.len() == 4 && v.chars().all(|c| c.is_ascii_digit()) {
        return format!("{v}-{}-01", &today[5..7]);
    }
    // Partial ISO month-day ("07-04") → current year prepended (dateparser
    // read these MDY and filled the year from the current date).
    if v.len() == 5
        && bytes[0..2].iter().all(u8::is_ascii_digit)
        && bytes[2] == b'-'
        && bytes[3..5].iter().all(u8::is_ascii_digit)
    {
        return format!("{}-{v}", &today[0..4]);
    }
    let lower = v.to_lowercase();
    if let Some(days) = match lower.as_str() {
        "today" | "aujourd'hui" => Some(0),
        "tomorrow" | "demain" => Some(1),
        "next week" | "la semaine prochaine" => Some(7),
        _ => None,
    } {
        return add_days_iso(today, days);
    }
    value.to_string()
}

/// Day arithmetic on an ISO `YYYY-MM-DD` string (Gregorian, no deps).
fn add_days_iso(date: &str, days: i64) -> String {
    fn leap(y: i64) -> bool {
        (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
    }
    fn month_len(y: i64, m: i64) -> i64 {
        match m {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            _ => if leap(y) { 29 } else { 28 },
        }
    }
    let (mut y, mut m, mut d) = (
        date[0..4].parse::<i64>().unwrap_or(1970),
        date[5..7].parse::<i64>().unwrap_or(1),
        date[8..10].parse::<i64>().unwrap_or(1),
    );
    d += days;
    while d > month_len(y, m) {
        d -= month_len(y, m);
        m += 1;
        if m > 12 {
            m = 1;
            y += 1;
        }
    }
    while d < 1 {
        m -= 1;
        if m < 1 {
            m = 12;
            y -= 1;
        }
        d += month_len(y, m);
    }
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_matches_python_formula() {
        // explicit, new entity, mention 1, persistence 3: 0.92 + 0.02 + 0.05
        assert_eq!(compute_confidence(3, "explicit", false, 1), 0.99);
        // hedged clamps under the facts threshold whatever the bonuses
        assert_eq!(compute_confidence(5, "hedged", true, 9), 0.84);
        // implicit low
        assert!((compute_confidence(1, "implicit", false, 1) - 0.32).abs() < 1e-12);
        // cap at 1.0
        assert_eq!(compute_confidence(5, "explicit", true, 9), 1.0);
    }

    #[test]
    fn date_resolution_covers_recorded_shapes() {
        assert_eq!(resolve_date("2026-06-16", "2026-07-04"), "2026-06-16");
        assert_eq!(resolve_date("1993", "2026-07-04"), "1993-07-01");
        assert_eq!(resolve_date("07-04", "2026-07-04"), "2026-07-04");
        assert_eq!(resolve_date("next week", "2026-07-04"), "2026-07-11");
        assert_eq!(resolve_date("bientôt", "2026-07-04"), "bientôt");
        assert_eq!(add_days_iso("2026-12-28", 7), "2027-01-04");
    }

    #[test]
    fn py_dumps_matches_python_layout() {
        let v: Value = serde_json::from_str(r#"{"b": 1, "a": ["x", 2], "c": "é"}"#).unwrap();
        assert_eq!(py_dumps(&v), r#"{"b": 1, "a": ["x", 2], "c": "é"}"#);
        assert_eq!(py_dumps_ascii(&v), r#"{"b": 1, "a": ["x", 2], "c": "\u00e9"}"#);
    }

    #[test]
    fn truthiness_is_python_truthiness() {
        assert!(!truthy(Some(&json!([]))));
        assert!(!truthy(Some(&json!(""))));
        assert!(!truthy(Some(&json!(0))));
        assert!(!truthy(Some(&json!(null))));
        assert!(truthy(Some(&json!([1]))));
        assert!(truthy(Some(&json!(false))) == false);
    }

    // SYN-119 — the language the classifier detected must flow end-to-end:
    // route_capture reads `classified["language"]` and persists it on the note.
    // A note-only capture needs neither the embedder nor the network.
    #[test]
    fn route_capture_persists_note_language() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("s.db");
        let brain = Brain::open(db.to_str().unwrap(), None).unwrap();
        {
            let conn = brain.storage.lock().unwrap();
            conn.execute(
                "INSERT INTO inbox (id, content) VALUES ('c1', ?1)",
                params!["Je me demande si je devrais arrêter le café"],
            )
            .unwrap();
        }
        let entry = json!({"id": "c1", "content": "Je me demande si je devrais arrêter le café"});
        let classified = json!({
            "language": "fr",
            "input_type": "episodic",
            "atomic_note": "Je me demande si je devrais arrêter le café",
            "atomic_note_kind": "note",
            "is_ephemeral": false,
            "summary": "réflexion sur le café",
            "entities": [],
            "relations": [],
            "project_entries": [],
            "classification_confidence": 1.0
        });
        let ctx = RouteContext {
            now: "2026-07-13T12:00:00".into(),
            today: "2026-07-13".into(),
            intentions_cutoff: "2026-07-11T12:00:00".into(),
            now_sql: "2026-07-13 12:00:00".into(),
        };
        brain.route_capture(&entry, &classified, &ctx).unwrap();
        let conn = brain.storage.lock().unwrap();
        let lang: Option<String> = conn
            .query_row(
                "SELECT language FROM atomic_notes WHERE provenance_capture_id = 'c1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(lang.as_deref(), Some("fr"));
    }
}
