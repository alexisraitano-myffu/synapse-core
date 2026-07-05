# synapse-core

The single compiled brain of [Synapse](https://github.com/alexisraitano-myffu/synapse): embeddings, storage, routing and decay logic, written once in Rust and consumed everywhere.

- **Desktop host** (macOS/Windows FastAPI backend): via the PyO3 binding (`crates/synapse-core-py`).
- **Mobile apps** (Android/iOS): via the UniFFI binding (`crates/synapse-core-ffi`).

One implementation, zero logic divergence between platforms.

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
