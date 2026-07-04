//! synapse-core: the single compiled brain shared by the Synapse desktop host
//! (via PyO3) and the mobile apps (via UniFFI).
//!
//! Current scope:
//! - T0: local text embeddings with strict parity against the Python backend
//!   (`embeddings.py`). The model files are DATA, not code: the host passes a
//!   directory containing the exact same ONNX + tokenizer files the Python
//!   fastembed uses (qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q), so
//!   vectors stay compatible with the existing database.
//! - T1: the storage substrate. The core owns the SQLite schema (rusqlite +
//!   statically linked sqlite-vec) and every vector read/write; hosts keep
//!   their own SQL access for non-vector columns.

mod embedder;
mod llm;
mod routing;
mod schema;
mod sql;
mod storage;

pub use embedder::{CoreError, Embedder, EMBEDDING_DIM};
pub use llm::{parse_classify_text, LlmConfig};
pub use routing::{Brain, ProjectSynthesis, RouteContext, RouteReport};
pub use sql::{connect, SqlConnection, SqlResult, SqlValue};
pub use storage::{EntityHit, NoteHit, ResourceHit, Storage};
