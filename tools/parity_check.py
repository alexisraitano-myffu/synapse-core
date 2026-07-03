"""
Vector parity harness: Rust core (synapse_core, PyO3) vs Python fastembed.

Both sides must load the SAME model files (qdrant paraphrase-multilingual-
MiniLM-L12-v2-onnx-Q) and produce the same L2-normalized 384-d vector for the
same text, within epsilon. This is the T0 acceptance gate (SYN-109).

Usage (from the synapse backend venv, which has fastembed):
    python tools/parity_check.py --model-dir ~/.synapse/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q

Requires the synapse_core wheel to be installed in the same venv
(maturin develop / pip install the built wheel).
"""

import argparse
import math
import sys

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_DIM = 384

# Thresholds: same ONNX graph + same tokenizer on both sides, only the
# onnxruntime build differs, so disagreement should be float-noise level.
MAX_ABS_DIFF = 1e-3
MIN_COSINE = 0.99999

CORPUS = [
    "Alexis travaille sur le projet Synapse.",
    "bonjour le monde",
    "Réunion avec Vincent demain à 14h pour parler du backend.",
    "The quick brown fox jumps over the lazy dog.",
    "Métro, boulot, dodo : la vie à Lyon en été, c'est génial !",
    "L'entité « Marie-Hélène » habite à Saint-Étienne depuis 2019.",
    "emoji test 🚀🧠✨ and mixed ASCII",
    "a",
    "  des espaces   partout   ",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "El zorro marrón rápido salta sobre el perro perezoso.",
    # Long text: exercises the 512-token truncation path on both sides.
    ("Synapse est un système de mémoire sémantique personnelle local-first. " * 60),
]


def l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else list(vec)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))  # both are L2-normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    from fastembed import TextEmbedding
    import synapse_core

    print(f"python ref : fastembed TextEmbedding({MODEL_NAME!r})")
    py_model = TextEmbedding(MODEL_NAME)
    print(f"rust core  : synapse_core.Embedder({args.model_dir!r})")
    rs_model = synapse_core.Embedder(args.model_dir)

    failures = 0
    worst_diff, worst_cos = 0.0, 1.0
    for text in CORPUS:
        # Reference = embeddings.py semantics: fastembed output + manual L2 norm.
        py_vec = l2_normalize(list(next(py_model.embed([text]))))
        rs_vec = rs_model.embed(text)

        if len(py_vec) != EXPECTED_DIM or len(rs_vec) != EXPECTED_DIM:
            print(f"FAIL dim mismatch ({len(py_vec)} vs {len(rs_vec)}): {text[:50]!r}")
            failures += 1
            continue

        max_diff = max(abs(p - r) for p, r in zip(py_vec, rs_vec))
        cos = cosine(py_vec, rs_vec)
        worst_diff = max(worst_diff, max_diff)
        worst_cos = min(worst_cos, cos)

        ok = max_diff <= MAX_ABS_DIFF and cos >= MIN_COSINE
        status = "ok  " if ok else "FAIL"
        print(f"{status} cos={cos:.7f} max|Δ|={max_diff:.2e}  {text[:60]!r}")
        if not ok:
            failures += 1

    print()
    print(f"worst: cos={worst_cos:.7f}, max|Δ|={worst_diff:.2e} over {len(CORPUS)} texts")
    if failures:
        print(f"PARITY FAILED on {failures}/{len(CORPUS)} texts")
        sys.exit(1)
    print("PARITY OK")


if __name__ == "__main__":
    main()
