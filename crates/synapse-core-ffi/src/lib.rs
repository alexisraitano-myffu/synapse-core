//! UniFFI surface of synapse-core, consumed by Kotlin (Android) and Swift (iOS).

use std::sync::Arc;

uniffi::setup_scaffolding!();

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum CoreError {
    #[error("model load failed: {msg}")]
    ModelLoad { msg: String },
    #[error("embedding failed: {msg}")]
    Embedding { msg: String },
}

impl From<synapse_core::CoreError> for CoreError {
    fn from(e: synapse_core::CoreError) -> Self {
        match e {
            synapse_core::CoreError::ModelLoad(msg) => CoreError::ModelLoad { msg },
            synapse_core::CoreError::Embedding(msg) => CoreError::Embedding { msg },
        }
    }
}

#[uniffi::export]
pub fn embedding_dim() -> u32 {
    synapse_core::EMBEDDING_DIM as u32
}

/// Text embedder. `model_dir` must contain the bundled ONNX + tokenizer files
/// (same files as the desktop backend, shipped as app assets).
#[derive(uniffi::Object)]
pub struct Embedder {
    inner: synapse_core::Embedder,
}

#[uniffi::export]
impl Embedder {
    #[uniffi::constructor]
    pub fn new(model_dir: String) -> Result<Arc<Self>, CoreError> {
        let inner = synapse_core::Embedder::new(&model_dir)?;
        Ok(Arc::new(Self { inner }))
    }

    /// L2-normalized 384-d vector, bit-parity with the desktop core.
    pub fn embed(&self, text: String) -> Result<Vec<f32>, CoreError> {
        Ok(self.inner.embed(&text)?)
    }
}
