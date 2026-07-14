#!/usr/bin/env bash
# Build the Android core artifacts consumed by the app:
#   bindings/android/jniLibs/arm64-v8a/libsynapse_core_ffi.so  (ort-dynamic)
#   bindings/android/kotlin/uniffi/synapse_core_ffi/synapse_core_ffi.kt
#
# The app supplies libonnxruntime.so via the onnxruntime-android AAR; the core
# is built with --features ort-dynamic so ort dlopens it by soname at runtime.
# Never set ORT_DYLIB_PATH in an Android app (extractNativeLibs=false: the .so
# does not exist on disk; a dangling path deadlocks ort rc.12).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-bindings/android}"
TARGETS=(arm64-v8a)

for t in "${TARGETS[@]}"; do
    cargo ndk -t "$t" build -p synapse-core-ffi \
        --no-default-features --features ort-dynamic --release
done

# The Kotlin binding is generated from a HOST build of the same crate: the
# UniFFI surface is feature-independent, so host metadata == Android metadata.
cargo build -p synapse-core-ffi
cargo run -p synapse-core-ffi --bin uniffi-bindgen -- generate \
    --library target/debug/libsynapse_core_ffi.dylib \
    --language kotlin --out-dir "$OUT/kotlin"

for t in "${TARGETS[@]}"; do
    case "$t" in
        arm64-v8a) rust_target=aarch64-linux-android ;;
        x86_64) rust_target=x86_64-linux-android ;;
        *) echo "unknown target $t" >&2; exit 1 ;;
    esac
    mkdir -p "$OUT/jniLibs/$t"
    cp "target/$rust_target/release/libsynapse_core_ffi.so" "$OUT/jniLibs/$t/"
done

echo "OK — artifacts in $OUT:"
find "$OUT" -type f -exec ls -lh {} \;
