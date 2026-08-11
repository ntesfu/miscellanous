#!/usr/bin/env python
"""Fuse giant-SSv2 (1408) + IV2-B14 (768) clip features -> 2176-d (v4 fusion stream).

Per the v4 diagram: L2-normalize each stream independently, then concatenate.
Streams are clip-aligned (IV2 extraction reused the giant *_starts.npy). If the two
lengths differ by a small margin, truncate to the common length.
"""
import argparse, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def l2norm(a, eps=1e-6):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--giant", default=os.path.join(ROOT, "data_v2", "features"))
    ap.add_argument("--iv2", default=os.path.join(ROOT, "fusion", "data", "features_iv2"))
    ap.add_argument("--out", default=os.path.join(ROOT, "fusion", "data", "features"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    recs = [f[:-4] for f in os.listdir(args.giant)
            if f.endswith(".npy") and not f.endswith("_starts.npy")]
    done = 0
    for rec in sorted(recs):
        gp = os.path.join(args.giant, rec + ".npy")
        ip = os.path.join(args.iv2, rec + ".npy")
        if not os.path.exists(ip):
            print(f"  skip {rec}: no IV2 feature"); continue
        g = np.load(gp); v = np.load(ip)
        T = min(len(g), len(v))
        if len(g) != len(v):
            print(f"  {rec}: length mismatch giant={len(g)} iv2={len(v)} -> {T}")
        fused = np.concatenate([l2norm(g[:T]), l2norm(v[:T])], axis=1).astype(np.float32)
        fused = fused.T  # -> [D, T] = [2176, T], MS-TCN/DiffAct channels-first convention
        np.save(os.path.join(args.out, rec + ".npy"), fused)
        done += 1
        print(f"  fused {rec}: [{fused.shape[0]}, {fused.shape[1]}]  (D,T)")
    print(f"== fused {done} recs -> {args.out} (shape [2176, T]) ==")


if __name__ == "__main__":
    main()
