//! UniFFI surface of synapse-core, consumed by Kotlin (Android) and Swift (iOS).

use std::sync::Arc;

uniffi::setup_scaffolding!();

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum CoreError {
    #[error("model load failed: {msg}")]
    ModelLoad { msg: String },
    #[error("embedding failed: {msg}")]
    Embedding { msg: String },
    #[error("storage error: {msg}")]
    Storage { msg: String },
    #[error("llm http error: {msg}")]
    LlmHttp { msg: String },
    #[error("llm content error: {msg}")]
    LlmContent { msg: String },
}

impl From<synapse_core::CoreError> for CoreError {
    fn from(e: synapse_core::CoreError) -> Self {
        match e {
            synapse_core::CoreError::ModelLoad(msg) => CoreError::ModelLoad { msg },
            synapse_core::CoreError::Embedding(msg) => CoreError::Embedding { msg },
            synapse_core::CoreError::Storage(msg) => CoreError::Storage { msg },
            synapse_core::CoreError::LlmHttp(msg) => CoreError::LlmHttp { msg },
            synapse_core::CoreError::LlmContent(msg) => CoreError::LlmContent { msg },
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

    /// One vector per ~128-token window (SYN-118); short text = one vector.
    pub fn embed_chunks(&self, text: String) -> Result<Vec<Vec<f32>>, CoreError> {
        Ok(self.inner.embed_chunks(&text)?)
    }
}

/// A note KNN hit; `distance` is sqlite-vec's L2 on unit vectors ([0, 2]).
#[derive(uniffi::Record)]
pub struct NoteHit {
    pub note_id: String,
    pub distance: f64,
}

/// An entity similarity hit; `score` = `1 - distance/2`, rounded to 4 decimals.
#[derive(uniffi::Record)]
pub struct EntityHit {
    pub id: String,
    pub canonical_name: String,
    pub entity_type: Option<String>,
    pub summary: String,
    pub score: f64,
}

#[derive(uniffi::Record)]
pub struct ResourceHit {
    pub id: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub summary: String,
    pub score: f64,
}

/// Storage substrate (SYN-110 / T1): schema ownership + vector reads/writes.
/// Embedding blobs are the sqlite-vec serialized float32 format (what
/// `Embedder.embed` yields once packed little-endian).
#[derive(uniffi::Object)]
pub struct Storage {
    inner: synapse_core::Storage,
}

#[uniffi::export]
impl Storage {
    /// Open (creating if needed) the database and init/migrate the schema.
    #[uniffi::constructor]
    pub fn open(db_path: String) -> Result<Arc<Self>, CoreError> {
        let inner = synapse_core::Storage::open(&db_path)?;
        Ok(Arc::new(Self { inner }))
    }

    pub fn upsert_note_vector(&self, note_id: String, embedding: Vec<u8>) -> Result<(), CoreError> {
        Ok(self.inner.upsert_note_vector(&note_id, &embedding)?)
    }

    /// Chunked upsert (SYN-118): chunk 0 keyed by the note uuid, then `uuid#k`.
    pub fn upsert_note_vectors(
        &self,
        note_id: String,
        embeddings: Vec<Vec<u8>>,
    ) -> Result<(), CoreError> {
        Ok(self.inner.upsert_note_vectors(&note_id, &embeddings)?)
    }

    pub fn delete_note_vector(&self, note_id: String) -> Result<(), CoreError> {
        Ok(self.inner.delete_note_vector(&note_id)?)
    }

    pub fn get_note_vector(&self, note_id: String) -> Result<Option<Vec<u8>>, CoreError> {
        Ok(self.inner.get_note_vector(&note_id)?)
    }

    pub fn search_notes(&self, query: Vec<u8>, k: u32) -> Result<Vec<NoteHit>, CoreError> {
        Ok(self
            .inner
            .search_notes(&query, k)?
            .into_iter()
            .map(|h| NoteHit {
                note_id: h.note_id,
                distance: h.distance,
            })
            .collect())
    }

    pub fn set_entity_embedding(
        &self,
        entity_id: String,
        embedding: Vec<u8>,
    ) -> Result<(), CoreError> {
        Ok(self.inner.set_entity_embedding(&entity_id, &embedding)?)
    }

    pub fn set_resource_embedding(
        &self,
        resource_id: String,
        embedding: Vec<u8>,
    ) -> Result<(), CoreError> {
        Ok(self.inner.set_resource_embedding(&resource_id, &embedding)?)
    }

    pub fn search_entities(
        &self,
        query: Vec<u8>,
        limit: u32,
        min_score: f64,
        type_filter: Option<String>,
        exclude_ids: Vec<String>,
    ) -> Result<Vec<EntityHit>, CoreError> {
        Ok(self
            .inner
            .search_entities(&query, limit, min_score, type_filter.as_deref(), &exclude_ids)?
            .into_iter()
            .map(|h| EntityHit {
                id: h.id,
                canonical_name: h.canonical_name,
                entity_type: h.entity_type,
                summary: h.summary,
                score: h.score,
            })
            .collect())
    }

    pub fn search_resources(
        &self,
        query: Vec<u8>,
        limit: u32,
    ) -> Result<Vec<ResourceHit>, CoreError> {
        Ok(self
            .inner
            .search_resources(&query, limit)?
            .into_iter()
            .map(|h| ResourceHit {
                id: h.id,
                title: h.title,
                url: h.url,
                summary: h.summary,
                score: h.score,
            })
            .collect())
    }

    // ── P2P sync (SYN-112/SYN-113): engine surface for the mobile transport ──

    pub fn sync_device_id(&self) -> Result<String, CoreError> {
        Ok(self.inner.sync_device_id()?)
    }

    /// Changeset (protocol-v1 JSON string) of everything journaled after
    /// `since`; paginate with the returned `next` cursor while `has_more`.
    pub fn sync_changes_since(&self, since: i64, limit: i64) -> Result<String, CoreError> {
        Ok(self.inner.sync_changes_since(since, limit)?)
    }

    /// Merge a peer's changeset (per-column LWW) → JSON report. The caller
    /// re-embeds the notes listed under `notes_changed`.
    pub fn sync_apply(&self, changes_json: String) -> Result<String, CoreError> {
        Ok(self.inner.sync_apply(&changes_json)?)
    }
}

/// One SQLite value crossing the FFI boundary, in either direction.
#[derive(uniffi::Enum)]
pub enum SqlValue {
    Null,
    Integer { value: i64 },
    Real { value: f64 },
    Text { value: String },
    Blob { value: Vec<u8> },
}

impl From<SqlValue> for synapse_core::SqlValue {
    fn from(v: SqlValue) -> Self {
        match v {
            SqlValue::Null => Self::Null,
            SqlValue::Integer { value } => Self::Integer(value),
            SqlValue::Real { value } => Self::Real(value),
            SqlValue::Text { value } => Self::Text(value),
            SqlValue::Blob { value } => Self::Blob(value),
        }
    }
}

impl From<synapse_core::SqlValue> for SqlValue {
    fn from(v: synapse_core::SqlValue) -> Self {
        match v {
            synapse_core::SqlValue::Null => Self::Null,
            synapse_core::SqlValue::Integer(value) => Self::Integer { value },
            synapse_core::SqlValue::Real(value) => Self::Real { value },
            synapse_core::SqlValue::Text(value) => Self::Text { value },
            synapse_core::SqlValue::Blob(value) => Self::Blob { value },
        }
    }
}

/// Result of an `execute`: `columns` is None for statements returning no
/// result set (INSERT/UPDATE/DDL).
#[derive(uniffi::Record)]
pub struct SqlResult {
    pub columns: Option<Vec<String>>,
    pub rows: Vec<Vec<SqlValue>>,
}

/// Generic SQL access to the core-owned database — the ONLY SQLite in the
/// process (mixing two SQLite libraries on one file corrupts it).
#[derive(uniffi::Object)]
pub struct SqlConnection {
    inner: synapse_core::SqlConnection,
}

#[uniffi::export]
impl SqlConnection {
    /// Open a SQL connection. No schema init — that's `Storage`'s job.
    #[uniffi::constructor]
    pub fn open(db_path: String) -> Result<Arc<Self>, CoreError> {
        let inner = synapse_core::connect(&db_path)?;
        Ok(Arc::new(Self { inner }))
    }

    pub fn execute(&self, sql: String, params: Vec<SqlValue>) -> Result<SqlResult, CoreError> {
        let values: Vec<synapse_core::SqlValue> = params.into_iter().map(Into::into).collect();
        let result = self.inner.execute(&sql, &values)?;
        Ok(SqlResult {
            columns: result.columns,
            rows: result
                .rows
                .into_iter()
                .map(|row| row.into_iter().map(Into::into).collect())
                .collect(),
        })
    }

    pub fn last_insert_rowid(&self) -> Result<i64, CoreError> {
        Ok(self.inner.last_insert_rowid()?)
    }
}

/// The Dream Cycle brain (SYN-111): deterministic routing + classifier
/// orchestration. JSON strings across the boundary, same shapes as PyO3.
#[derive(uniffi::Object)]
pub struct Brain {
    inner: synapse_core::Brain,
}

#[uniffi::export]
impl Brain {
    #[uniffi::constructor]
    pub fn open(db_path: String, model_dir: Option<String>) -> Result<Arc<Self>, CoreError> {
        let inner = synapse_core::Brain::open(&db_path, model_dir.as_deref())?;
        Ok(Arc::new(Self { inner }))
    }

    /// Route one capture; returns the RouteReport as JSON.
    #[allow(clippy::too_many_arguments)]
    pub fn route_capture(
        &self,
        entry_json: String,
        classified_json: String,
        now: String,
        today: String,
        intentions_cutoff: String,
        now_sql: String,
    ) -> Result<String, CoreError> {
        let entry: serde_json::Value = serde_json::from_str(&entry_json)
            .map_err(|e| CoreError::Storage { msg: e.to_string() })?;
        let classified: serde_json::Value = serde_json::from_str(&classified_json)
            .map_err(|e| CoreError::Storage { msg: e.to_string() })?;
        let ctx = synapse_core::RouteContext {
            now,
            today,
            intentions_cutoff,
            now_sql,
        };
        let report = self.inner.route_capture(&entry, &classified, &ctx)?;
        Ok(synapse_core::Brain::report_to_json(&report).to_string())
    }

    /// Embed with the Brain's already-loaded model (no second Embedder):
    /// the re-embed path after a sync apply on mobile hosts.
    pub fn embed(&self, text: String) -> Result<Vec<f32>, CoreError> {
        Ok(self.inner.embed_text(&text)?)
    }

    pub fn validate_pending(&self, new_facts_json: String) -> Result<i64, CoreError> {
        let new_facts: Vec<serde_json::Value> = serde_json::from_str(&new_facts_json)
            .map_err(|e| CoreError::Storage { msg: e.to_string() })?;
        Ok(self.inner.validate_pending(&new_facts)?)
    }

    /// Synchronous classification (prompt build + HTTP + parse) → JSON.
    #[allow(clippy::too_many_arguments)]
    pub fn classify(
        &self,
        content: String,
        day_context: Option<String>,
        model: String,
        api_key: String,
        prompts_dir: String,
        today: String,
        base_url: Option<String>,
        fuel_token: Option<String>,
    ) -> Result<String, CoreError> {
        let config = synapse_core::LlmConfig {
            model,
            api_key,
            base_url,
            fuel_token,
            prompts_dir,
            today,
        };
        Ok(self
            .inner
            .classify(&content, day_context.as_deref(), &config)?
            .to_string())
    }
}
