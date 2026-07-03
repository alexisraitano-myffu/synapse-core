//! synapse-core: the single compiled brain shared by the Synapse desktop host
//! (via PyO3) and the mobile apps (via UniFFI).
//!
//! T0 scope: local text embeddings with strict parity against the Python
//! backend (`embeddings.py`). The model files are DATA, not code: the host
//! passes a directory containing the exact same ONNX + tokenizer files the
//! Python fastembed uses (qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q),
//! so vectors stay compatible with the existing database.

mod embedder;

pub use embedder::{CoreError, Embedder, EMBEDDING_DIM};
