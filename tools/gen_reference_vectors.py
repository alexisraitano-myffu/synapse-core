"""
Generate reference vectors for the on-device parity test (T0, SYN-109).

Reference = the Python backend's embedding semantics (fastembed + manual L2
norm), i.e. what the production database vectors look like today. The Android
instrumented test embeds the same texts through the Rust core and asserts
cosine parity against this file.

Usage (backend venv):
    python tools/gen_reference_vectors.py > reference_vectors.json
"""

import json
import math
import sys

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TEXTS = [
    "Alexis travaille sur le projet Synapse.",
    "bonjour le monde",
    "Réunion avec Vincent demain à 14h pour parler du backend.",
    "The quick brown fox jumps over the lazy dog.",
    "emoji test 🚀🧠✨ and mixed ASCII",
    ("Synapse est un système de mémoire sémantique personnelle local-first. " * 60),
]


def main():
    from fastembed import TextEmbedding

    model = TextEmbedding(MODEL_NAME)
    out = []
    for text in TEXTS:
        vec = list(next(model.embed([text])))
        norm = math.sqrt(sum(x * x for x in vec))
        vec = [x / norm for x in vec] if norm > 0 else vec
        out.append({"text": text, "vector": vec})
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
