use std::path::Path;
use std::sync::Mutex;

use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};

/// Must match `EMBEDDING_DIM` in the Python backend's `config.py`.
pub const EMBEDDING_DIM: usize = 384;

/// Truncation fallback when the tokenizer config carries no usable value.
const DEFAULT_MAX_LENGTH: usize = 512;

// ── Chunked embedding (SYN-118) ──────────────────────────────────────────
// The qdrant export truncates at 128 tokens, and that is the RIGHT granule:
// this is a sentence model — mean-pooling a 460-token digest into one vector
// dilutes every section (measured: tail queries 0.31→0.22 when embedding the
// full text at 512). Long texts are therefore embedded as one vector PER
// ~128-token window and the search takes the best-scoring window.

/// Tokens shared between consecutive windows so a sentence cut at a boundary
/// still lives whole in one of them.
const CHUNK_OVERLAP_TOKENS: usize = 24;

/// Upper bound on windows per text (~16 × 124 tokens ≈ 2000 tokens covered;
/// beyond that the tail is dropped, like the old 128 truncation but 16× later).
/// Storage sweeps this many candidate chunk keys on delete — keep in sync.
pub const MAX_CHUNKS: usize = 16;

/// ONNX file names probed inside the model directory, in order. The first is
/// the exact file the Python backend uses (qdrant HF repo layout).
const ONNX_CANDIDATES: &[&str] = &["model_optimized.onnx", "model.onnx", "onnx/model.onnx"];

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("embedding failed: {0}")]
    Embedding(String),
    #[error("storage error: {0}")]
    Storage(String),
    // The two LLM failure classes have DIFFERENT host policies: an HTTP/
    // network error aborts the whole run (entries stay queued for a retry),
    // a content error only fails the one entry.
    #[error("llm http error: {0}")]
    LlmHttp(String),
    #[error("llm content error: {0}")]
    LlmContent(String),
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
    // Same tokenizer.json as the model, used ONLY to window long texts for
    // embed_chunks (truncation disabled: it must see the real length).
    chunker: tokenizers::Tokenizer,
    /// Effective per-window token budget (the model's truncation cap).
    max_tokens: usize,
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

        let mut chunker = tokenizers::Tokenizer::from_bytes(&tokenizer_files.tokenizer_file)
            .map_err(|e| CoreError::ModelLoad(format!("chunker tokenizer: {e}")))?;
        chunker.with_truncation(None).map_err(|e| {
            CoreError::ModelLoad(format!("chunker truncation off: {e}"))
        })?;

        // Mean pooling: what the Python fastembed applies to this model (the
        // backend silences the very warning that announced this default).
        let model_def =
            UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

        // Single intra-op thread: onnxruntime's default pool sizing relies on
        // cpuinfo, which fails on emulated/unknown CPUs (hangs observed on the
        // Android emulator), and one thread is the sane default for embedding
        // one short text at a time on mobile anyway.
        let options = InitOptionsUserDefined::default()
            .with_max_length(max_length)
            .with_intra_threads(1);

        let model = TextEmbedding::try_new_from_user_defined(model_def, options)
            .map_err(|e| CoreError::ModelLoad(e.to_string()))?;

        Ok(Self {
            model: Mutex::new(model),
            chunker,
            max_tokens: max_length,
        })
    }

    /// Embed a text as one L2-normalized vector per ~`max_tokens` window
    /// (SYN-118). A text that fits in one window returns a single vector,
    /// embedded from the ORIGINAL string (bit-identical to `embed`); longer
    /// texts are windowed on token ids with `CHUNK_OVERLAP_TOKENS` overlap,
    /// each window decoded back to text and embedded. At most `MAX_CHUNKS`
    /// windows: a pathological text is covered ~16× further than before.
    pub fn embed_chunks(&self, text: &str) -> Result<Vec<Vec<f32>>, CoreError> {
        let enc = self
            .chunker
            .encode(text, false)
            .map_err(|e| CoreError::Embedding(format!("chunker encode: {e}")))?;
        let ids = enc.get_ids();
        // Leave room for the special tokens the real encode adds per window.
        let budget = self.max_tokens.saturating_sub(4).max(16);
        if ids.len() <= budget {
            return Ok(vec![self.embed(text)?]);
        }

        let overlap = CHUNK_OVERLAP_TOKENS.min(budget / 2);
        let stride = budget - overlap;
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < ids.len() && chunks.len() < MAX_CHUNKS {
            let end = (start + budget).min(ids.len());
            let piece = self
                .chunker
                .decode(&ids[start..end], true)
                .map_err(|e| CoreError::Embedding(format!("chunker decode: {e}")))?;
            if !piece.trim().is_empty() {
                chunks.push(self.embed(&piece)?);
            }
            if end == ids.len() {
                break;
            }
            start += stride;
        }
        if chunks.is_empty() {
            chunks.push(self.embed(text)?);
        }
        Ok(chunks)
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

    // SYN-118 — chunked embedding of long texts.
    #[test]
    fn embed_chunks_windows_long_text_and_matches_embed_for_short() {
        let Some(dir) = model_dir() else {
            eprintln!("SYNAPSE_MODEL_DIR not set or empty; skipping");
            return;
        };
        let embedder = Embedder::new(&dir).unwrap();

        // Short text: exactly one chunk, bit-identical to embed().
        let short = "Alexis travaille sur le projet Synapse.";
        let chunks = embedder.embed_chunks(short).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], embedder.embed(short).unwrap());

        // Long text (~50 distinct sentences ≫ 128 tokens): several windows,
        // all normalized, and the windows are genuinely different vectors.
        let long = (0..50)
            .map(|i| format!("La phrase numéro {i} parle du sujet {} en détail.", i * 7))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = embedder.embed_chunks(&long).unwrap();
        assert!(chunks.len() > 1, "expected windows, got {}", chunks.len());
        assert!(chunks.len() <= MAX_CHUNKS);
        for c in &chunks {
            assert_eq!(c.len(), EMBEDDING_DIM);
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4);
        }
        assert_ne!(chunks[0], chunks[chunks.len() - 1]);
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
