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

#[pymodule(name = "synapse_core")]
fn synapse_core_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Embedder>()?;
    m.add_class::<Storage>()?;
    m.add_class::<SqlConnection>()?;
    m.add_class::<Brain>()?;
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add_function(wrap_pyfunction!(parse_classify_text, m)?)?;
    m.add("EMBEDDING_DIM", synapse_core::EMBEDDING_DIM)?;
    Ok(())
}
