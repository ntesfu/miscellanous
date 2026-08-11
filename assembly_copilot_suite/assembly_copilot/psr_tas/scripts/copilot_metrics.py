#!/usr/bin/env python3
"""Full metric suite for a trained DiffAct head on the copilot dataset.

The stock TAS suite (Acc, Edit, F1@10/25/50) is largely blind on this data: only
6 distinct segment orderings exist across all 40 recordings and every test video's
ordering also appears in training, so Edit and the low-IoU F1s sit at ~99 no matter
what the model does. This adds the measurements that still move:

  F1@75 / F1@90   same metric, stricter IoU -- restores dynamic range for free
  completion-time  |predicted - true| step-completion time, in SECONDS. The labels
                   are natively completion events; densifying them into per-frame
                   classes was our doing. This measures the original signal, and is
                   what a copilot actually promises ("flags the step within 2 s").
  per-class acc    the mean hides the rare classes -- they are the weak ones
  segment count    over/under-segmentation, which a saturated Edit conceals

    python scripts/copilot_metrics.py --pred extern/DiffAct/result/Copilot-Fusion-S1/prediction \
                                      --gt   extern/DiffAct/datasets/Copilot-Fusion/groundTruth
"""
import argparse, os
import numpy as np


def segments(labels):
    """[(label, start, end)] inclusive, from a per-timestep label list."""
    out, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append((labels[s], s, i - 1)); s = i
    return out


def f1_at(gt, pred, thr):
    """Standard TAS segmental F1: greedy IoU match, each GT segment usable once."""
    g, p = segments(gt), segments(pred)
    used = [False] * len(g)
    tp = 0
    for lab, ps, pe in p:
        best, bi = 0.0, -1
        for i, (gl, gs, ge) in enumerate(g):
            if gl != lab or used[i]:
                continue
            inter = min(pe, ge) - max(ps, gs) + 1
            if inter <= 0:
                continue
            iou = inter / (max(pe, ge) - min(ps, gs) + 1)
            if iou > best:
                best, bi = iou, i
        if bi >= 0 and best >= thr:
            used[bi] = True; tp += 1
    fp, fn = len(p) - tp, len(g) - tp
    return 2 * tp / (2 * tp + fp + fn) * 100 if tp else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--stride", type=int, default=6, help="feature stride in source frames")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    sec = args.stride / args.fps                     # seconds per feature timestep

    thrs = [.10, .25, .50, .75, .90]
    F = {t: [] for t in thrs}
    accs, errs, exact, ncls = [], [], 0, {}
    files = sorted(f for f in os.listdir(args.pred) if f.endswith(".txt"))

    for fn in files:
        # DiffAct writes: line 0 is a header, line 1 is space-separated labels
        pred = open(os.path.join(args.pred, fn)).read().split("\n")[1].split()
        gt = [l.strip() for l in open(os.path.join(args.gt, fn)) if l.strip()]
        n = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]

        accs.append(100 * np.mean([a == b for a, b in zip(gt, pred)]))
        for t in thrs:
            F[t].append(f1_at(gt, pred, t))
        g, p = segments(gt), segments(pred)
        exact += (len(g) == len(p))
        for gl, _, ge in g:                          # completion = segment END
            cand = [pe for (pl, _, pe) in p if pl == gl]
            if cand:
                errs.append(abs(min(cand, key=lambda x: abs(x - ge)) - ge) * sec)
        for a, b in zip(gt, pred):
            d = ncls.setdefault(a, [0, 0]); d[1] += 1; d[0] += (a == b)

    e = np.array(errs)
    print(f"== {len(files)} test recordings ==\n")
    print(f"frame accuracy      {np.mean(accs):6.2f}   (per-video sd {np.std(accs):.2f})")
    print("\nsegmental F1 by IoU threshold")
    for t in thrs:
        tag = "  <- saturated, do not tune on" if t <= .25 else ""
        print(f"   F1@{int(t*100):<3d}          {np.mean(F[t]):6.2f}{tag}")
    print(f"\nstep-completion timing  (n={len(e)})")
    print(f"   median            {np.median(e):6.2f} s")
    print(f"   p90               {np.percentile(e, 90):6.2f} s")
    print(f"   max               {e.max():6.2f} s")
    for tol in (1, 2, 5):
        print(f"   within +-{tol}s        {100*(e <= tol).mean():6.1f} %")
    print(f"\nsegment count exact   {exact}/{len(files)} videos")
    print("\nper-class frame accuracy (ascending support -- the rare ones are the weak ones)")
    tot = sum(v[1] for v in ncls.values())
    for k, v in sorted(ncls.items(), key=lambda x: x[1][1]):
        print(f"   {k:24s} {100*v[0]/v[1]:6.2f} %   support {100*v[1]/tot:5.2f} %")
    print(f"\nn={len(files)} test videos: one video is ~{100/len(files):.0f}% of every number "
          f"above. Differences of a few points are noise.")
    print("All figures are OFFLINE (the model sees the whole video). Live/causal "
          "accuracy will be lower and is not measured here.")


if __name__ == "__main__":
    main()
