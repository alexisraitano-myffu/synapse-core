use std::path::Path;
use std::sync::Mutex;

use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};

/// Must match `EMBEDDING_DIM` in the Python backend's `config.py`.
pub const EMBEDDING_DIM: usize = 384;

/// Truncation fallback when the tokenizer config carries no usable value.
const DEFAULT_MAX_LENGTH: usize = 512;

/// ONNX file names probed inside the model directory, in order. The first is
/// the exact file the Python backend uses (qdrant HF repo layout).
const ONNX_CANDIDATES: &[&str] = &["model_optimized.onnx", "model.onnx", "onnx/model.onnx"];

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("embedding failed: {0}")]
    Embedding(String),
}

/// Text embedder backed by a user-provided ONNX model directory.
///
/// The directory must contain the model file (`model_optimized.onnx` or
/// `model.onnx`) plus `tokenizer.json`, `config.json`,
/// `special_tokens_map.json` and `tokenizer_config.json`.
pub struct Embedder {
    // fastembed's embed() needs &mut; the Mutex also gives us the Send + Sync
    // bound UniFFI requires for interface objects.
    model: Mutex<TextEmbedding>,
}

impl Embedder {
    pub fn new(model_dir: &str) -> Result<Self, CoreError> {
        let dir = Path::new(model_dir);

        let onnx_path = ONNX_CANDIDATES
            .iter()
            .map(|c| dir.join(c))
            .find(|p| p.is_file())
            .ok_or_else(|| {
                CoreError::ModelLoad(format!(
                    "no ONNX model file found in {model_dir} (tried {ONNX_CANDIDATES:?})"
                ))
            })?;

        let read = |name: &str| -> Result<Vec<u8>, CoreError> {
            std::fs::read(dir.join(name))
                .map_err(|e| CoreError::ModelLoad(format!("cannot read {name}: {e}")))
        };

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };

        let onnx_file = std::fs::read(&onnx_path)
            .map_err(|e| CoreError::ModelLoad(format!("cannot read {onnx_path:?}: {e}")))?;

        // Parity with Python fastembed (preprocessor_utils.py): the effective
        // truncation is min(model_max_length, max_length) over whichever keys
        // tokenizer_config.json defines. The qdrant export of our model says
        // max_length=128 / model_max_length=512, so truncation happens at 128
        // tokens; hardcoding 512 here would silently diverge on long texts.
        let max_length = effective_max_length(&tokenizer_files.tokenizer_config_file)
            .unwrap_or(DEFAULT_MAX_LENGTH);

        // Mean pooling: what the Python fastembed applies to this model (the
        // backend silences the very warning that announced this default).
        let model_def =
            UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

        let options = InitOptionsUserDefined::default().with_max_length(max_length);

        let model = TextEmbedding::try_new_from_user_defined(model_def, options)
            .map_err(|e| CoreError::ModelLoad(e.to_string()))?;

        Ok(Self {
            model: Mutex::new(model),
        })
    }

    /// Embed a text into an L2-normalized `EMBEDDING_DIM` vector.
    ///
    /// Parity contract with the Python backend's `embed_text`: mean pooling
    /// then L2 normalization, so downstream `score = 1 - distance/2` over
    /// sqlite-vec L2 distances stays valid.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| CoreError::Embedding("embedder mutex poisoned".into()))?;

        let mut vec = model
            .embed(vec![text], None)
            .map_err(|e| CoreError::Embedding(e.to_string()))?
            .pop()
            .ok_or_else(|| CoreError::Embedding("model returned no embedding".into()))?;

        if vec.len() != EMBEDDING_DIM {
            return Err(CoreError::Embedding(format!(
                "model returned {} dims, expected {EMBEDDING_DIM}",
                vec.len()
            )));
        }

        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }

        Ok(vec)
    }
}

/// min(model_max_length, max_length) over the keys present, like Python
/// fastembed's tokenizer setup. None if neither key holds a usable number.
fn effective_max_length(tokenizer_config: &[u8]) -> Option<usize> {
    let config: serde_json::Value = serde_json::from_slice(tokenizer_config).ok()?;
    let read = |key: &str| config.get(key).and_then(|v| v.as_u64()).map(|v| v as usize);
    match (read("model_max_length"), read("max_length")) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> Option<String> {
        std::env::var("SYNAPSE_MODEL_DIR").ok().filter(|d| {
            ONNX_CANDIDATES
                .iter()
                .any(|c| Path::new(d).join(c).is_file())
        })
    }

    #[test]
    fn embeds_l2_normalized_384_dims() {
        let Some(dir) = model_dir() else {
            eprintln!("SYNAPSE_MODEL_DIR not set or empty; skipping");
            return;
        };
        let embedder = Embedder::new(&dir).unwrap();
        let v = embedder.embed("Alexis travaille sur le projet Synapse.").unwrap();
        assert_eq!(v.len(), EMBEDDING_DIM);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm was {norm}");
    }

    #[test]
    fn same_text_same_vector() {
        let Some(dir) = model_dir() else {
            eprintln!("SYNAPSE_MODEL_DIR not set or empty; skipping");
            return;
        };
        let embedder = Embedder::new(&dir).unwrap();
        let a = embedder.embed("bonjour le monde").unwrap();
        let b = embedder.embed("bonjour le monde").unwrap();
        assert_eq!(a, b);
    }
}
