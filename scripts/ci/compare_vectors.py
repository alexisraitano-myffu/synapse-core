#!/usr/bin/env python3
"""Compare two embed_smoke outputs (host vs iOS): per-sentence cosine.

Exit 0 iff every cosine >= THRESHOLD. Stdlib only.
"""
import json
import math
import sys

THRESHOLD = 0.99999  # T0 parity was ~0.9999996 (Mac) / 0.9999999 (Pixel)


def main() -> int:
    with open(sys.argv[1]) as f:
        a = json.load(f)
    with open(sys.argv[2]) as f:
        b = json.load(f)
    if a.keys() != b.keys():
        print(f"sentence sets differ: {set(a) ^ set(b)}")
        return 1
    worst = 1.0
    for sentence, va in a.items():
        vb = b[sentence]
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        cos = dot / (na * nb)
        worst = min(worst, cos)
        print(f"cos={cos:.9f}  {sentence[:60]!r}")
    print(f"worst cosine: {worst:.9f} (threshold {THRESHOLD})")
    return 0 if worst >= THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
