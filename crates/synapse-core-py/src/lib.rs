use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes};
use synapse_core::SqlValue;

fn core_err(e: synapse_core::CoreError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn to_sql_value(ob: &Bound<'_, PyAny>) -> PyResult<SqlValue> {
    if ob.is_none() {
        return Ok(SqlValue::Null);
    }
    if let Ok(b) = ob.cast::<PyBool>() {
        return Ok(SqlValue::Integer(b.is_true() as i64));
    }
    if let Ok(b) = ob.cast::<PyBytes>() {
        return Ok(SqlValue::Blob(b.as_bytes().to_vec()));
    }
    if let Ok(i) = ob.extract::<i64>() {
        return Ok(SqlValue::Integer(i));
    }
    if let Ok(f) = ob.extract::<f64>() {
        return Ok(SqlValue::Real(f));
    }
    if let Ok(s) = ob.extract::<String>() {
        return Ok(SqlValue::Text(s));
    }
    Err(PyTypeError::new_err(format!(
        "unsupported SQL parameter type: {}",
        ob.get_type().name()?
    )))
}

fn from_sql_value<'py>(py: Python<'py>, v: SqlValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        SqlValue::Null => py.None().into_bound(py),
        SqlValue::Integer(i) => i.into_pyobject(py)?.into_any(),
        SqlValue::Real(f) => f.into_pyobject(py)?.into_any(),
        SqlValue::Text(s) => s.into_pyobject(py)?.into_any(),
        SqlValue::Blob(b) => PyBytes::new(py, &b).into_any(),
    })
}

/// Text embedder backed by the shared Rust core.
///
/// Python usage:
///     from synapse_core import Embedder
///     e = Embedder("/path/to/model-dir")
///     vec = e.embed("some text")   # list[float], 384-d, L2-normalized
#[pyclass]
struct Embedder {
    inner: std::sync::Arc<synapse_core::Embedder>,
}

#[pymethods]
impl Embedder {
    #[new]
    fn new(py: Python<'_>, model_dir: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| synapse_core::Embedder::new(model_dir))
            .map_err(core_err)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    fn embed(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        // ONNX inference can take tens of ms: release the GIL while it runs.
        py.detach(|| self.inner.embed(text)).map_err(core_err)
    }

    /// One vector per ~128-token window of the text (SYN-118); a short text
    /// yields a single vector identical to `embed`.
    fn embed_chunks(&self, py: Python<'_>, text: &str) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| self.inner.embed_chunks(text)).map_err(core_err)
    }
}

/// Storage substrate backed by the shared Rust core (SYN-110 / T1).
///
/// Owns the SQLite schema (created/migrated on open) and every vector
/// read/write. Blobs are the sqlite-vec serialized float32 format, exactly
/// what `embed_text` produces. Search results come back as plain tuples;
/// the Python callers wrap them in their historical dict shapes.
///
/// Python usage:
///     from synapse_core import Storage
///     s = Storage(str(DB_PATH))            # opens + init/migrate schema
///     s.upsert_note_vector(note_id, vec)
///     hits = s.search_entities(vec, limit=5, min_score=0.85)
#[pyclass]
struct Storage {
    inner: synapse_core::Storage,
}

#[pymethods]
impl Storage {
    #[new]
    fn new(py: Python<'_>, db_path: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| synapse_core::Storage::open(db_path))
            .map_err(core_err)?;
        Ok(Self { inner })
    }

    fn upsert_note_vector(&self, py: Python<'_>, note_id: &str, embedding: &[u8]) -> PyResult<()> {
        py.detach(|| self.inner.upsert_note_vector(note_id, embedding))
            .map_err(core_err)
    }

    /// Chunked upsert (SYN-118): one blob per ~128-token window, chunk 0
    /// keyed by the note uuid (back-compat), then `uuid#k`.
    fn upsert_note_vectors(
        &self,
        py: Python<'_>,
        note_id: &str,
        embeddings: Vec<Vec<u8>>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.upsert_note_vectors(note_id, &embeddings))
            .map_err(core_err)
    }

    fn delete_note_vector(&self, py: Python<'_>, note_id: &str) -> PyResult<()> {
        py.detach(|| self.inner.delete_note_vector(note_id))
            .map_err(core_err)
    }

    fn get_note_vector<'py>(
        &self,
        py: Python<'py>,
        note_id: &str,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let vec = py
            .detach(|| self.inner.get_note_vector(note_id))
            .map_err(core_err)?;
        Ok(vec.map(|v| PyBytes::new(py, &v)))
    }

    /// KNN over episodic notes → [(note_id, l2_distance)], distance-ascending.
    fn search_notes(&self, py: Python<'_>, query: &[u8], k: u32) -> PyResult<Vec<(String, f64)>> {
        let hits = py
            .detach(|| self.inner.search_notes(query, k))
            .map_err(core_err)?;
        Ok(hits.into_iter().map(|h| (h.note_id, h.distance)).collect())
    }

    fn set_entity_embedding(
        &self,
        py: Python<'_>,
        entity_id: &str,
        embedding: &[u8],
    ) -> PyResult<()> {
        py.detach(|| self.inner.set_entity_embedding(entity_id, embedding))
            .map_err(core_err)
    }

    fn set_resource_embedding(
        &self,
        py: Python<'_>,
        resource_id: &str,
        embedding: &[u8],
    ) -> PyResult<()> {
        py.detach(|| self.inner.set_resource_embedding(resource_id, embedding))
            .map_err(core_err)
    }

    /// → [(id, canonical_name, type, summary, score)], score-descending.
    #[pyo3(signature = (query, limit=10, min_score=0.0, type_filter=None, exclude_ids=None))]
    fn search_entities(
        &self,
        py: Python<'_>,
        query: &[u8],
        limit: u32,
        min_score: f64,
        type_filter: Option<&str>,
        exclude_ids: Option<Vec<String>>,
    ) -> PyResult<Vec<(String, String, Option<String>, String, f64)>> {
        let exclude = exclude_ids.unwrap_or_default();
        let hits = py
            .detach(|| {
                self.inner
                    .search_entities(query, limit, min_score, type_filter, &exclude)
            })
            .map_err(core_err)?;
        Ok(hits
            .into_iter()
            .map(|h| (h.id, h.canonical_name, h.entity_type, h.summary, h.score))
            .collect())
    }

    // ── P2P sync (SYN-112 T3): engine surface for the phase-3 transport ──

    fn sync_device_id(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.sync_device_id()).map_err(core_err)
    }

    /// Changeset (protocol-v1 JSON string) of everything journaled after
    /// `since`; paginate with the returned `next` cursor while `has_more`.
    #[pyo3(signature = (since, limit=10000))]
    fn sync_changes_since(&self, py: Python<'_>, since: i64, limit: i64) -> PyResult<String> {
        py.detach(|| self.inner.sync_changes_since(since, limit))
            .map_err(core_err)
    }

    /// Merge a peer's changeset (per-column LWW) → JSON report. The caller
    /// re-embeds the notes listed under `notes_changed`.
    fn sync_apply(&self, py: Python<'_>, changes_json: &str) -> PyResult<String> {
        py.detach(|| self.inner.sync_apply(changes_json))
            .map_err(core_err)
    }

    /// SYN-133 — post-pull twin dedup (collapse on the smallest uuid,
    /// tombstones journaled, doomed notes' vectors swept) → JSON report.
    fn dedup_after_pull(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.dedup_after_pull()).map_err(core_err)
    }

    /// → [(id, title, url, summary, score)], score-descending.
    #[pyo3(signature = (query, limit=10))]
    fn search_resources(
        &self,
        py: Python<'_>,
        query: &[u8],
        limit: u32,
    ) -> PyResult<Vec<(String, Option<String>, Option<String>, String, f64)>> {
        let hits = py
            .detach(|| self.inner.search_resources(query, limit))
            .map_err(core_err)?;
        Ok(hits
            .into_iter()
            .map(|h| (h.id, h.title, h.url, h.summary, h.score))
            .collect())
    }
}

/// One SQL connection to the core's bundled SQLite (the ONLY SQLite in the
/// process — mixing two SQLite libraries on one file corrupts it, see the
/// core's `sql.rs`). The Python `db.Connection` adapter wraps this with the
/// cursor/transaction surface the backend historically used.
#[pyclass]
struct SqlConnection {
    inner: Option<synapse_core::SqlConnection>,
}

impl SqlConnection {
    fn get(&self) -> PyResult<&synapse_core::SqlConnection> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("connection is closed"))
    }
}

#[pymethods]
impl SqlConnection {
    /// Returns `(columns | None, rows)`; rows are tuples of native values.
    #[pyo3(signature = (sql, params=None))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        sql: &str,
        params: Option<Vec<Bound<'py, PyAny>>>,
    ) -> PyResult<(Option<Vec<String>>, Vec<Vec<Bound<'py, PyAny>>>)> {
        let values: Vec<SqlValue> = params
            .unwrap_or_default()
            .iter()
            .map(to_sql_value)
            .collect::<PyResult<_>>()?;
        let conn = self.get()?;
        let result = py
            .detach(|| conn.execute(sql, &values))
            .map_err(core_err)?;
        let rows = result
            .rows
            .into_iter()
            .map(|row| row.into_iter().map(|v| from_sql_value(py, v)).collect())
            .collect::<PyResult<_>>()?;
        Ok((result.columns, rows))
    }

    fn last_insert_rowid(&self) -> PyResult<i64> {
        self.get()?.last_insert_rowid().map_err(core_err)
    }

    /// Shared fact write (SYN-37 supersede + dedup) executed on THIS
    /// connection, so the caller's open `with conn:` transaction wraps it
    /// (the `Brain` variant runs on its own connection — SQLITE_BUSY inside
    /// a host transaction). JSON scalars like `Brain.insert_fact`.
    #[pyo3(signature = (entity_id, predicate, value_json, confidence,
                        source_inbox_id_json="null", persistence_value=3,
                        provenance_capture_id=None, category_json="null"))]
    #[allow(clippy::too_many_arguments)]
    fn insert_fact(
        &self,
        py: Python<'_>,
        entity_id: &str,
        predicate: &str,
        value_json: &str,
        confidence: f64,
        source_inbox_id_json: &str,
        persistence_value: i64,
        provenance_capture_id: Option<String>,
        category_json: &str,
    ) -> PyResult<String> {
        let value: serde_json::Value = serde_json::from_str(value_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let source: serde_json::Value = serde_json::from_str(source_inbox_id_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let category: serde_json::Value = serde_json::from_str(category_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let conn = self.get()?;
        py.detach(|| {
            conn.insert_fact(
                entity_id,
                predicate,
                value,
                confidence,
                source,
                persistence_value,
                provenance_capture_id,
                category,
            )
        })
        .map_err(core_err)
    }

    /// SYN-19 decay pass over atomic_notes; `now` = optional fixed clock
    /// 'YYYY-MM-DD HH:MM:SS' (tests inject it), None = system now.
    #[pyo3(signature = (tau_days=None, now=None))]
    fn apply_decay(
        &self,
        py: Python<'_>,
        tau_days: Option<f64>,
        now: Option<String>,
    ) -> PyResult<i64> {
        let conn = self.get()?;
        py.detach(|| conn.apply_decay(tau_days, now.as_deref()))
            .map_err(core_err)
    }

    /// SYN-68 decay pass over entities (anchor `last_mentioned`).
    #[pyo3(signature = (tau_days=None, now=None))]
    fn apply_entity_decay(
        &self,
        py: Python<'_>,
        tau_days: Option<f64>,
        now: Option<String>,
    ) -> PyResult<i64> {
        let conn = self.get()?;
        py.detach(|| conn.apply_entity_decay(tau_days, now.as_deref()))
            .map_err(core_err)
    }

    /// Move notes' reactivation anchor toward now (1.0 = full mention reset,
    /// <1 = light search bump). Returns the count touched.
    #[pyo3(signature = (note_ids, factor=1.0, now=None))]
    fn reactivate_notes(
        &self,
        py: Python<'_>,
        note_ids: Vec<String>,
        factor: f64,
        now: Option<String>,
    ) -> PyResult<i64> {
        let conn = self.get()?;
        py.detach(|| conn.reactivate_notes(&note_ids, factor, now.as_deref()))
            .map_err(core_err)
    }

    /// Full reactivation of every note mentioning one of `entity_names`.
    #[pyo3(signature = (entity_names, now=None))]
    fn reactivate_notes_for_entities(
        &self,
        py: Python<'_>,
        entity_names: Vec<String>,
        now: Option<String>,
    ) -> PyResult<i64> {
        let conn = self.get()?;
        py.detach(|| conn.reactivate_notes_for_entities(&entity_names, now.as_deref()))
            .map_err(core_err)
    }

    /// SYN-23 — the digest's structured week as JSON (pure SQL on THIS
    /// connection, offline). `now` = optional fixed clock 'YYYY-MM-DD HH:MM:SS'.
    #[pyo3(signature = (now=None, days=7))]
    fn gather_week(&self, py: Python<'_>, now: Option<String>, days: i64) -> PyResult<String> {
        let conn = self.get()?;
        let week = py
            .detach(|| conn.gather_week(now.as_deref(), days))
            .map_err(core_err)?;
        Ok(week.to_string())
    }

    /// Host-facing project-entry write on THIS connection (the caller's open
    /// transaction wraps it). Returns
    /// {project_id, entry_id, project_name, entry_content, entry_count};
    /// the LLM synthesis is Brain.synthesize_project, run after commit.
    #[pyo3(signature = (canonical, content, capture_id, is_new_project=false))]
    fn add_project_entry(
        &self,
        py: Python<'_>,
        canonical: &str,
        content: &str,
        capture_id: &str,
        is_new_project: bool,
    ) -> PyResult<String> {
        let conn = self.get()?;
        let s = py
            .detach(|| conn.add_project_entry(canonical, content, capture_id, is_new_project))
            .map_err(core_err)?;
        Ok(serde_json::json!({
            "project_id": s.project_id,
            "entry_id": s.entry_id,
            "project_name": s.project_name,
            "entry_content": s.entry_content,
            "entry_count": s.entry_count,
        })
        .to_string())
    }

    /// Close the underlying SQLite connection (further calls raise).
    fn close(&mut self) {
        self.inner = None;
    }
}

/// Open a SQL connection. No schema init — that's `Storage`'s job.
#[pyfunction]
fn connect(py: Python<'_>, db_path: &str) -> PyResult<SqlConnection> {
    let inner = py
        .detach(|| synapse_core::connect(db_path))
        .map_err(core_err)?;
    Ok(SqlConnection { inner: Some(inner) })
}

fn brain_err(e: synapse_core::CoreError) -> PyErr {
    use pyo3::exceptions::{PyConnectionError, PyValueError};
    match e {
        // Host policy: HTTP/network aborts the run (like anthropic.APIError),
        // a content error only fails the one entry.
        synapse_core::CoreError::LlmHttp(msg) => PyConnectionError::new_err(msg),
        synapse_core::CoreError::LlmContent(msg) => PyValueError::new_err(msg),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// The Dream Cycle brain (SYN-111): deterministic routing + classifier
/// orchestration. JSON strings across the boundary — the classified dict is
/// polymorphic and the report is consumed as a dict anyway.
#[pyclass]
struct Brain {
    inner: synapse_core::Brain,
}

#[pymethods]
impl Brain {
    /// `embedder`: share one loaded model across Brains (per-test databases
    /// must not reload 235 MB each). `model_dir` loads a private one.
    #[new]
    #[pyo3(signature = (db_path, model_dir=None, embedder=None))]
    fn new(
        py: Python<'_>,
        db_path: &str,
        model_dir: Option<&str>,
        embedder: Option<PyRef<'_, Embedder>>,
    ) -> PyResult<Self> {
        let shared = embedder.map(|e| e.inner.clone());
        let inner = py
            .detach(|| match shared {
                Some(e) => synapse_core::Brain::open_shared(db_path, Some(e)),
                None => synapse_core::Brain::open(db_path, model_dir),
            })
            .map_err(brain_err)?;
        Ok(Self { inner })
    }

    /// Route one capture. Returns the RouteReport as JSON:
    /// {entity_ids, new_facts, created_note_id, fast_exit, project_syntheses}.
    #[pyo3(signature = (entry_json, classified_json, now, today, intentions_cutoff, now_sql))]
    fn route_capture(
        &self,
        py: Python<'_>,
        entry_json: &str,
        classified_json: &str,
        now: &str,
        today: &str,
        intentions_cutoff: &str,
        now_sql: &str,
    ) -> PyResult<String> {
        let entry: serde_json::Value =
            serde_json::from_str(entry_json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let classified: serde_json::Value = serde_json::from_str(classified_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let ctx = synapse_core::RouteContext {
            now: now.to_string(),
            today: today.to_string(),
            intentions_cutoff: intentions_cutoff.to_string(),
            now_sql: now_sql.to_string(),
        };
        let report = py
            .detach(|| self.inner.route_capture(&entry, &classified, &ctx))
            .map_err(brain_err)?;
        Ok(synapse_core::Brain::report_to_json(&report).to_string())
    }

    /// step5 — behavioral validation over the run's accumulated new facts.
    fn validate_pending(&self, py: Python<'_>, new_facts_json: &str) -> PyResult<i64> {
        let new_facts: Vec<serde_json::Value> = serde_json::from_str(new_facts_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        py.detach(|| self.inner.validate_pending(&new_facts))
            .map_err(brain_err)
    }

    /// `messages.create` params for one capture (Batch API path) → JSON.
    #[pyo3(signature = (content, day_context, model, prompts_dir, today))]
    fn build_classify_params(
        &self,
        py: Python<'_>,
        content: &str,
        day_context: Option<&str>,
        model: &str,
        prompts_dir: &str,
        today: &str,
    ) -> PyResult<String> {
        let config = synapse_core::LlmConfig {
            model: model.to_string(),
            api_key: String::new(),
            base_url: None,
            fuel_token: None,
            prompts_dir: prompts_dir.to_string(),
            today: today.to_string(),
        };
        let params = py
            .detach(|| self.inner.build_classify_params(content, day_context, &config))
            .map_err(brain_err)?;
        Ok(params.to_string())
    }

    /// Synchronous classification (build + HTTP Anthropic + parse) → JSON.
    /// Raises ConnectionError on HTTP/network failure (abort-the-run policy)
    /// and ValueError on truncated/invalid content (fail-one-entry policy).
    #[pyo3(signature = (content, day_context, model, api_key, prompts_dir, today,
                        base_url=None, fuel_token=None))]
    #[allow(clippy::too_many_arguments)]
    fn classify(
        &self,
        py: Python<'_>,
        content: &str,
        day_context: Option<&str>,
        model: &str,
        api_key: &str,
        prompts_dir: &str,
        today: &str,
        base_url: Option<&str>,
        fuel_token: Option<&str>,
    ) -> PyResult<String> {
        let config = synapse_core::LlmConfig {
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.map(String::from),
            fuel_token: fuel_token.map(String::from),
            prompts_dir: prompts_dir.to_string(),
            today: today.to_string(),
        };
        let classified = py
            .detach(|| self.inner.classify(content, day_context, &config))
            .map_err(brain_err)?;
        Ok(classified.to_string())
    }

    /// SYN-89 re-summary pass (T5): entities touched by the run + stale ones,
    /// summaries rebuilt from active facts/relations via the LLM. Returns the
    /// regenerated entity ids as a JSON array. HTTP failure stops the pass
    /// silently (stale flags survive) — mirror of the Python `break`.
    #[pyo3(signature = (touched_ids, model, api_key, prompts_dir, today,
                        base_url=None, fuel_token=None))]
    #[allow(clippy::too_many_arguments)]
    fn resummarize(
        &self,
        py: Python<'_>,
        touched_ids: Vec<String>,
        model: &str,
        api_key: &str,
        prompts_dir: &str,
        today: &str,
        base_url: Option<&str>,
        fuel_token: Option<&str>,
    ) -> PyResult<String> {
        let config = synapse_core::LlmConfig {
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.map(String::from),
            fuel_token: fuel_token.map(String::from),
            prompts_dir: prompts_dir.to_string(),
            today: today.to_string(),
        };
        let ids = py
            .detach(|| self.inner.resummarize(&touched_ids, &config))
            .map_err(brain_err)?;
        Ok(serde_json::Value::from(ids).to_string())
    }

    /// SYN-43/44 living project synthesis (T5): append + threshold-triggered
    /// refinement. Returns the new summary_md or None (failures never block).
    #[pyo3(signature = (project_id, project_name, new_entry_content, new_entry_count,
                        model, api_key, prompts_dir, today, base_url=None, fuel_token=None))]
    #[allow(clippy::too_many_arguments)]
    fn synthesize_project(
        &self,
        py: Python<'_>,
        project_id: &str,
        project_name: &str,
        new_entry_content: &str,
        new_entry_count: i64,
        model: &str,
        api_key: &str,
        prompts_dir: &str,
        today: &str,
        base_url: Option<&str>,
        fuel_token: Option<&str>,
    ) -> PyResult<Option<String>> {
        let config = synapse_core::LlmConfig {
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.map(String::from),
            fuel_token: fuel_token.map(String::from),
            prompts_dir: prompts_dir.to_string(),
            today: today.to_string(),
        };
        py.detach(|| {
            self.inner.synthesize_project(
                project_id,
                project_name,
                new_entry_content,
                new_entry_count,
                &config,
            )
        })
        .map_err(brain_err)
    }

    /// Host-facing project-entry write (manual API endpoints): find/create
    /// the project, INSERT the entry. Returns
    /// {project_id, entry_id, project_name, entry_content, entry_count}.
    #[pyo3(signature = (canonical, content, capture_id, is_new_project=false))]
    fn add_project_entry(
        &self,
        py: Python<'_>,
        canonical: &str,
        content: &str,
        capture_id: &str,
        is_new_project: bool,
    ) -> PyResult<String> {
        let s = py
            .detach(|| {
                self.inner
                    .add_project_entry(canonical, content, capture_id, is_new_project)
            })
            .map_err(brain_err)?;
        Ok(serde_json::json!({
            "project_id": s.project_id,
            "entry_id": s.entry_id,
            "project_name": s.project_name,
            "entry_content": s.entry_content,
            "entry_count": s.entry_count,
        })
        .to_string())
    }

    /// Port of `step6_vectorize` (T5): embed each entity's composite text and
    /// store the vector; per-entity failures skip. Returns the count.
    fn vectorize_entities(&self, py: Python<'_>, entity_ids: Vec<String>) -> PyResult<i64> {
        py.detach(|| self.inner.vectorize_entities(&entity_ids))
            .map_err(brain_err)
    }

    /// Shared fact write (SYN-37 supersede + dedup) for the validation /
    /// reclassify endpoints. `value`/`source_inbox_id`/`category` are JSON
    /// scalars (bound like Python bound the native values).
    #[pyo3(signature = (entity_id, predicate, value_json, confidence,
                        source_inbox_id_json="null", persistence_value=3,
                        provenance_capture_id=None, category_json="null"))]
    #[allow(clippy::too_many_arguments)]
    fn insert_fact(
        &self,
        py: Python<'_>,
        entity_id: &str,
        predicate: &str,
        value_json: &str,
        confidence: f64,
        source_inbox_id_json: &str,
        persistence_value: i64,
        provenance_capture_id: Option<String>,
        category_json: &str,
    ) -> PyResult<String> {
        let value: serde_json::Value = serde_json::from_str(value_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let source: serde_json::Value = serde_json::from_str(source_inbox_id_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let category: serde_json::Value = serde_json::from_str(category_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        py.detach(|| {
            self.inner.insert_user_fact(
                entity_id,
                predicate,
                value,
                confidence,
                source,
                persistence_value,
                provenance_capture_id,
                category,
            )
        })
        .map_err(brain_err)
    }

    /// SYN-23 — render the gathered week (JSON str) into the digest markdown
    /// (prompt = data `digest.md`, LLM via the core HTTP path). Raises
    /// ConnectionError on HTTP failure, ValueError on empty content.
    #[pyo3(signature = (week_json, model, api_key, prompts_dir, today,
                        base_url=None, fuel_token=None))]
    #[allow(clippy::too_many_arguments)]
    fn summarize_digest(
        &self,
        py: Python<'_>,
        week_json: &str,
        model: &str,
        api_key: &str,
        prompts_dir: &str,
        today: &str,
        base_url: Option<&str>,
        fuel_token: Option<&str>,
    ) -> PyResult<String> {
        let week: serde_json::Value = serde_json::from_str(week_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let config = synapse_core::LlmConfig {
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.map(String::from),
            fuel_token: fuel_token.map(String::from),
            prompts_dir: prompts_dir.to_string(),
            today: today.to_string(),
        };
        py.detach(|| self.inner.summarize_digest(&week, &config))
            .map_err(brain_err)
    }

    /// SYN-23 — store the digest note (idempotent per ISO week) + its vector,
    /// on the Brain's OWN connection: call outside host transactions. Returns
    /// the note id.
    fn write_digest_note(
        &self,
        py: Python<'_>,
        week_json: &str,
        markdown: &str,
    ) -> PyResult<String> {
        let week: serde_json::Value = serde_json::from_str(week_json)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        py.detach(|| self.inner.write_digest_note(&week, markdown))
            .map_err(brain_err)
    }

    /// SYN-21 — fetch → extract → summarise → store one URL (idempotent on
    /// the URL, Brain's OWN connection: call outside host transactions).
    /// `model=None` → no LLM (snippet-fallback summary), like client=None.
    #[pyo3(signature = (url, capture_id=None, model=None, api_key=None, prompts_dir=None,
                        today=None, base_url=None, fuel_token=None))]
    #[allow(clippy::too_many_arguments)]
    fn process_resource(
        &self,
        py: Python<'_>,
        url: &str,
        capture_id: Option<&str>,
        model: Option<&str>,
        api_key: Option<&str>,
        prompts_dir: Option<&str>,
        today: Option<&str>,
        base_url: Option<&str>,
        fuel_token: Option<&str>,
    ) -> PyResult<Option<String>> {
        let config = llm_config_opt(model, api_key, prompts_dir, today, base_url, fuel_token);
        py.detach(|| self.inner.process_resource(url, capture_id, config.as_ref()))
            .map_err(brain_err)
    }

    /// SYN-21 — process every URL found in a capture (each independent, one
    /// failure never blocks the others). Returns the stored resource ids as a
    /// JSON array.
    #[pyo3(signature = (content, capture_id=None, model=None, api_key=None, prompts_dir=None,
                        today=None, base_url=None, fuel_token=None))]
    #[allow(clippy::too_many_arguments)]
    fn process_capture_resources(
        &self,
        py: Python<'_>,
        content: &str,
        capture_id: Option<&str>,
        model: Option<&str>,
        api_key: Option<&str>,
        prompts_dir: Option<&str>,
        today: Option<&str>,
        base_url: Option<&str>,
        fuel_token: Option<&str>,
    ) -> PyResult<String> {
        let config = llm_config_opt(model, api_key, prompts_dir, today, base_url, fuel_token);
        let ids = py
            .detach(|| self.inner.process_capture_resources(content, capture_id, config.as_ref()))
            .map_err(brain_err)?;
        Ok(serde_json::Value::from(ids).to_string())
    }

    /// Alias-aware entity resolution → entity id or None.
    #[pyo3(signature = (canonical_name, aliases=None))]
    fn find_entity(
        &self,
        py: Python<'_>,
        canonical_name: &str,
        aliases: Option<Vec<String>>,
    ) -> PyResult<Option<String>> {
        let aliases = aliases.unwrap_or_default();
        py.detach(|| self.inner.find_entity(canonical_name, &aliases))
            .map_err(brain_err)
    }
}

/// Port of `_parse_classify_text` — shared by the host's Batch API path.
#[pyfunction]
#[pyo3(signature = (text, content_len, stop_reason=None))]
fn parse_classify_text(text: &str, content_len: usize, stop_reason: Option<&str>) -> PyResult<String> {
    synapse_core::parse_classify_text(text, content_len, stop_reason)
        .map(|v| v.to_string())
        .map_err(brain_err)
}

/// SYN-23 — next concrete date of a (possibly recurring) event, ISO strings;
/// None when `event_date` doesn't parse (Python returned None there too).
#[pyfunction]
fn next_occurrence(event_date: &str, recurring: bool, today: &str) -> Option<String> {
    synapse_core::next_occurrence_str(event_date, recurring, today)
}

/// Build an LlmConfig only when the host resolved a model (client=None parity).
fn llm_config_opt(
    model: Option<&str>,
    api_key: Option<&str>,
    prompts_dir: Option<&str>,
    today: Option<&str>,
    base_url: Option<&str>,
    fuel_token: Option<&str>,
) -> Option<synapse_core::LlmConfig> {
    model.map(|model| synapse_core::LlmConfig {
        model: model.to_string(),
        api_key: api_key.unwrap_or_default().to_string(),
        base_url: base_url.map(String::from),
        fuel_token: fuel_token.map(String::from),
        prompts_dir: prompts_dir.unwrap_or_default().to_string(),
        today: today.unwrap_or_default().to_string(),
    })
}

/// SYN-21 — all http(s) URLs in a text, de-duplicated, order-preserving.
#[pyfunction]
fn extract_urls(text: &str) -> Vec<String> {
    synapse_core::extract_urls(text)
}

/// SYN-21 — title + visible text of an HTML document, as JSON {title, text}.
#[pyfunction]
fn extract_page(html: &str) -> String {
    let page = synapse_core::extract_page(html);
    serde_json::json!({"title": page.title, "text": page.text}).to_string()
}

/// SYN-21 — GET a URL and extract {title, text} (JSON); None on any failure.
#[pyfunction]
#[pyo3(signature = (url, timeout=10.0))]
fn fetch_and_extract(py: Python<'_>, url: &str, timeout: f64) -> Option<String> {
    let timeout = std::time::Duration::from_secs_f64(timeout.max(0.0));
    py.detach(|| synapse_core::fetch_and_extract(url, timeout))
        .map(|p| serde_json::json!({"title": p.title, "text": p.text}).to_string())
}

/// Pairing offerer session (SYN-128): the device that SHOWS the QR keeps this
/// between showing the offer and receiving the scanner's returned key.
#[pyclass]
struct PairingSession {
    inner: synapse_core::PairingSession,
    offer_pub: [u8; 32],
}

#[pymethods]
impl PairingSession {
    /// Start a pairing. `addrs` = how the joiner can reach us (LAN URLs).
    /// Returns (session, qr_string). Render `qr_string` as a QR code.
    #[staticmethod]
    fn offer(addrs: Vec<String>) -> PyResult<(Self, String)> {
        let (inner, offer) = synapse_core::PairingSession::offer(addrs).map_err(core_err)?;
        let qr = offer.encode();
        let offer_pub = inner.offer_public();
        Ok((Self { inner, offer_pub }, qr))
    }

    /// The offerer's ephemeral public key (bytes), for AAD in seal.
    fn offer_pub<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.offer_pub)
    }

    /// Complete with the scanner's returned public key → channel key (bytes).
    fn channel_key<'py>(&self, py: Python<'py>, accept_pub: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let ap: [u8; 32] = accept_pub
            .try_into()
            .map_err(|_| PyRuntimeError::new_err("accept_pub must be 32 bytes"))?;
        Ok(PyBytes::new(py, &self.inner.channel_key(&ap)))
    }
}

/// Scanner side (SYN-128): decode the QR, return (accept_pub, channel_key) as
/// bytes. Send accept_pub back to the offerer over the transport.
#[pyfunction]
fn pairing_accept<'py>(
    py: Python<'py>,
    qr: &str,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let offer = synapse_core::PairingOffer::decode(qr).map_err(core_err)?;
    let (accept_pub, key) = synapse_core::pairing_accept(&offer).map_err(core_err)?;
    Ok((PyBytes::new(py, &accept_pub), PyBytes::new(py, &key)))
}

/// The reachability hints embedded in a QR offer (so a joiner knows where to
/// call back) — decoded without completing the exchange.
#[pyfunction]
fn pairing_offer_addrs(qr: &str) -> PyResult<Vec<String>> {
    Ok(synapse_core::PairingOffer::decode(qr)
        .map_err(core_err)?
        .addrs)
}

/// AEAD-seal a payload under the channel key (SYN-128/137). Returns base64.
/// `offer_pub`/`accept_pub` are the two handshake messages of the channel —
/// X25519 keys (32 B, QR) or SPAKE2 messages (33 B, code) — bound as AAD.
#[pyfunction]
fn pairing_seal(
    channel_key: &[u8],
    offer_pub: &[u8],
    accept_pub: &[u8],
    plaintext: &[u8],
) -> PyResult<String> {
    let ck = channel_key32(channel_key)?;
    synapse_core::pairing_seal(&ck, offer_pub, accept_pub, plaintext).map_err(core_err)
}

/// Open what `pairing_seal` produced (SYN-128/137) → the plaintext bytes.
#[pyfunction]
fn pairing_open<'py>(
    py: Python<'py>,
    channel_key: &[u8],
    offer_pub: &[u8],
    accept_pub: &[u8],
    sealed_b64: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let ck = channel_key32(channel_key)?;
    let out =
        synapse_core::pairing_open(&ck, offer_pub, accept_pub, sealed_b64).map_err(core_err)?;
    Ok(PyBytes::new(py, &out))
}

fn channel_key32(channel_key: &[u8]) -> PyResult<[u8; 32]> {
    channel_key
        .try_into()
        .map_err(|_| PyRuntimeError::new_err("channel_key must be 32 bytes"))
}

/// SYN-137 — one side of the PAKE on the 6-digit code (symmetric: member and
/// joiner run the same role). Keep it between sending our message and
/// receiving the peer's; `finish` is one-shot. Never log the code, the
/// messages or the key.
#[pyclass]
struct CodePairing {
    inner: Option<synapse_core::CodePairing>,
    msg: Vec<u8>,
}

#[pymethods]
impl CodePairing {
    #[new]
    fn new(code: &str) -> Self {
        let (inner, msg) = synapse_core::CodePairing::start(code);
        Self {
            inner: Some(inner),
            msg,
        }
    }

    /// Our handshake message (bytes) to send to the peer.
    fn msg<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.msg)
    }

    /// Complete with the peer's message → the 32-byte channel key. A wrong
    /// code still succeeds but yields a different key — confirm with
    /// `pairing_code_confirm_mac`/`_verify` before trusting the channel.
    fn finish<'py>(&mut self, py: Python<'py>, peer_msg: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("code pairing already finished"))?;
        let key = inner.finish(peer_msg).map_err(core_err)?;
        Ok(PyBytes::new(py, &key))
    }
}

/// SYN-137 joiner side: key-confirmation MAC over the transcript.
#[pyfunction]
fn pairing_code_confirm_mac<'py>(
    py: Python<'py>,
    channel_key: &[u8],
    member_msg: &[u8],
    joiner_msg: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let ck = channel_key32(channel_key)?;
    Ok(PyBytes::new(
        py,
        &synapse_core::code_confirm_mac(&ck, member_msg, joiner_msg),
    ))
}

/// SYN-137 member side: constant-time verify; a mismatch burns one attempt.
#[pyfunction]
fn pairing_code_confirm_verify(
    channel_key: &[u8],
    member_msg: &[u8],
    joiner_msg: &[u8],
    mac: &[u8],
) -> PyResult<bool> {
    let ck = channel_key32(channel_key)?;
    Ok(synapse_core::code_confirm_verify(
        &ck, member_msg, joiner_msg, mac,
    ))
}

#[pymodule(name = "synapse_core")]
fn synapse_core_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Embedder>()?;
    m.add_class::<Storage>()?;
    m.add_class::<SqlConnection>()?;
    m.add_class::<Brain>()?;
    m.add_class::<PairingSession>()?;
    m.add_class::<CodePairing>()?;
    m.add_function(wrap_pyfunction!(pairing_accept, m)?)?;
    m.add_function(wrap_pyfunction!(pairing_offer_addrs, m)?)?;
    m.add_function(wrap_pyfunction!(pairing_seal, m)?)?;
    m.add_function(wrap_pyfunction!(pairing_open, m)?)?;
    m.add_function(wrap_pyfunction!(pairing_code_confirm_mac, m)?)?;
    m.add_function(wrap_pyfunction!(pairing_code_confirm_verify, m)?)?;
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add_function(wrap_pyfunction!(parse_classify_text, m)?)?;
    m.add_function(wrap_pyfunction!(next_occurrence, m)?)?;
    m.add_function(wrap_pyfunction!(extract_urls, m)?)?;
    m.add_function(wrap_pyfunction!(extract_page, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_and_extract, m)?)?;
    m.add("EMBEDDING_DIM", synapse_core::EMBEDDING_DIM)?;
    Ok(())
}
