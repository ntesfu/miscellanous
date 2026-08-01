#!/usr/bin/env python
"""Assemble a DiffAct dataset dir from fused features + clip-aligned step labels.

DiffAct asserts feature_T == label_length, but our per-frame groundTruth has N frames
while fused features have T = #clips (< N). We build clip-aligned labels: clip i (start s)
gets the per-frame STEP label at the clip centre (s+8, the middle of its 16-frame window).

Outputs -> extern/DiffAct/datasets/IndustReal-Fusion/
    features/<rec>.npy       (symlink to fusion/data/features/<rec>.npy, [T,2176])
    groundTruth/<rec>.txt    (T clip-aligned STEP class names)
    mapping.txt              (copied from data/mapping.txt)
    splits/{train,test}.split1.bundle
"""
import argparse, os, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default=os.path.join(ROOT, "fusion", "data", "features"))
    ap.add_argument("--starts", default=os.path.join(ROOT, "data_v2", "features"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "data", "groundTruth"),
                    help="per-frame step groundTruth (use groundTruth_type for the type head)")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "extern", "DiffAct", "datasets", "IndustReal-Fusion"))
    ap.add_argument("--center", type=int, default=8, help="frame offset within the 16-frame clip")
    args = ap.parse_args()

    fdir = os.path.join(args.out, "features")
    gdir = os.path.join(args.out, "groundTruth")
    sdir = os.path.join(args.out, "splits")
    for d in (fdir, gdir, sdir):
        os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(args.data, "mapping.txt"), os.path.join(args.out, "mapping.txt"))
    for b in ("train.split1.bundle", "test.split1.bundle"):
        shutil.copy(os.path.join(args.data, "splits", b), os.path.join(sdir, b))

    recs = sorted(f[:-4] for f in os.listdir(args.fused) if f.endswith(".npy"))
    n_ok = 0
    for rec in recs:
        feat = np.load(os.path.join(args.fused, rec + ".npy"))
        T = feat.shape[1]  # fused features are [D, T]
        starts = np.load(os.path.join(args.starts, rec + "_starts.npy"))
        per_frame = np.loadtxt(os.path.join(args.gt, rec + ".txt"), dtype=str)
        N = len(per_frame)
        # clip-aligned labels: centre frame of each 16-frame clip
        centres = np.clip(starts[:T] + args.center, 0, N - 1)
        labels = per_frame[centres]
        # symlink feature (avoid duplicating GBs)
        link = os.path.join(fdir, rec + ".npy")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(os.path.join(args.fused, rec + ".npy")), link)
        with open(os.path.join(gdir, rec + ".txt"), "w") as f:
            f.write("\n".join(labels) + "\n")
        assert len(labels) == T, f"{rec}: {len(labels)} != {T}"
        n_ok += 1
    print(f"== prepared DiffAct dataset '{os.path.basename(args.out)}': {n_ok} recs ==")
    print(f"   feature dim: {feat.shape[0]}  (expect 2176)")
    print(f"   -> {args.out}")


if __name__ == "__main__":
    main()
