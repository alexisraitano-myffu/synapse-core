# synapse-core

The single compiled brain of [Synapse](https://github.com/alexisraitano-myffu/synapse): embeddings, storage, routing, decay, summaries, LLM orchestration and P2P sync, written once in Rust and consumed everywhere.

- **Desktop host** (macOS/Windows FastAPI backend): via the PyO3 binding (`crates/synapse-core-py`).
- **Mobile apps** (Android/iOS): via the UniFFI binding (`crates/synapse-core-ffi`).

One implementation, zero logic divergence between platforms.

## What's in the core

Everything the "Dream Cycle" does lives here as Rust modules in `crates/synapse-core/src/`:

| Module | Responsibility |
| -- | -- |
| `embedder.rs` | Local ONNX embeddings (fastembed, `paraphrase-multilingual-MiniLM-L12-v2`, 384-d, L2-normalized) |
| `storage.rs` · `schema.rs` · `sql.rs` · `migrate.rs` | SQLite substrate (`rusqlite` + `sqlite-vec`), schema, the SQL gateway the host writes through, migrations |
| `routing.rs` | The pipeline: classify → resolve/coreference → confidence-score → route (fact / note / relation / ephemeral) → fact⇄relation dedup → `review_status` gating |
| `llm.rs` | Claude Haiku orchestration — build prompt, blocking HTTP (`ureq`, rustls), parse JSON |
| `decay.rs` | Graceful forgetting: `memory_strength = exp(-Δdays/τ)` |
| `summaries.rs` · `digest.rs` | Derived entity summaries (regenerated from active facts/relations) and the weekly digest |
| `resources.rs` | URL fetch + summarize into searchable resources |
| `sync.rs` | The P2P sync engine (see below) |

## Prompts are data

Prompts are **not compiled in** — they live as versioned files under [`prompts/`](prompts/) (`classifier.md`, `digest.md`, `project-summary.md`, … plus a `manifest.json`) and are read at runtime. They can be edited without recompiling and are synchronized between devices as ordinary data, so every platform runs identical prompts.

## P2P sync (SYN-112)

A **homemade** sync engine, not a third-party CRDT. cr-sqlite was dormant and rejected the schema, and Automerge is the wrong model for ~20 relational tables — and because an owner-lock means a single device runs the Dream Cycle, all derived tables are effectively single-writer, so the engine stays deliberately small: a `sync_log` change journal (a version map, not an event log), a hybrid logical clock computed in pure SQL, per-column **last-writer-wins** merge, and tombstones, over a versioned protocol. Any writer — the core, the Python host through the `sql.rs` gateway, even a `sqlite3` CLI — journals correctly with zero registration.

## Crates

| Crate | Role |
| -- | -- |
| `crates/synapse-core` | Pure Rust library (the brain) |
| `crates/synapse-core-py` | PyO3 binding, built as a Python wheel with maturin |
| `crates/synapse-core-ffi` | UniFFI binding for Kotlin and Swift |

## Model files are data

The embedding model (ONNX + tokenizer files) is never compiled in. Hosts pass a directory containing the model files at runtime; apps bundle them as assets. This keeps vectors byte-compatible across platforms and satisfies App Store rule 2.5.2 (no downloaded code).

## Build

```bash
cargo build                        # desktop (onnxruntime downloaded at build time)
cargo test                         # SYNAPSE_MODEL_DIR=<model dir> to run embedding tests

# Python wheel (desktop host)
cd crates/synapse-core-py && maturin build --release

# Android (the app ships libonnxruntime.so and ort loads it dynamically)
cargo ndk -t arm64-v8a -t x86_64 build -p synapse-core-ffi --no-default-features --features ort-dynamic --release
```

## License

Apache-2.0
