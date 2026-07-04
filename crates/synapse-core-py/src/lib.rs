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
    inner: synapse_core::Embedder,
}

#[pymethods]
impl Embedder {
    #[new]
    fn new(model_dir: &str) -> PyResult<Self> {
        let inner = synapse_core::Embedder::new(model_dir).map_err(core_err)?;
        Ok(Self { inner })
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

    fn upsert_note_vector(&self, py: Python<'_>, note_id: i64, embedding: &[u8]) -> PyResult<()> {
        py.detach(|| self.inner.upsert_note_vector(note_id, embedding))
            .map_err(core_err)
    }

    fn delete_note_vector(&self, py: Python<'_>, note_id: i64) -> PyResult<()> {
        py.detach(|| self.inner.delete_note_vector(note_id))
            .map_err(core_err)
    }

    fn get_note_vector<'py>(
        &self,
        py: Python<'py>,
        note_id: i64,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let vec = py
            .detach(|| self.inner.get_note_vector(note_id))
            .map_err(core_err)?;
        Ok(vec.map(|v| PyBytes::new(py, &v)))
    }

    /// KNN over episodic notes → [(note_id, l2_distance)], distance-ascending.
    fn search_notes(&self, py: Python<'_>, query: &[u8], k: u32) -> PyResult<Vec<(i64, f64)>> {
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

#[pymodule(name = "synapse_core")]
fn synapse_core_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Embedder>()?;
    m.add_class::<Storage>()?;
    m.add_class::<SqlConnection>()?;
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add("EMBEDDING_DIM", synapse_core::EMBEDDING_DIM)?;
    Ok(())
}
