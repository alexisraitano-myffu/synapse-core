use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
        let inner = synapse_core::Embedder::new(model_dir)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn embed(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        // ONNX inference can take tens of ms: release the GIL while it runs.
        let result = py.detach(|| self.inner.embed(text));
        result.map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pymodule(name = "synapse_core")]
fn synapse_core_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Embedder>()?;
    m.add("EMBEDDING_DIM", synapse_core::EMBEDDING_DIM)?;
    Ok(())
}
