#!/usr/bin/env python
"""Causal debounce: the streaming replacement for DiffAct's `purge` postprocess.

The raw causal head is accurate per-frame (92.25%) but flickers: 3.9x more segments
than ground truth, 34% of them under 0.6 s. Offline this is cleaned up by diffusion
smoothing plus postprocess={purge:3}, both of which need the finished sequence.

The streaming equivalent is hysteresis: hold the committed class until a challenger
wins K consecutive ticks. Strictly causal -- it consults only the last K frames -- and
it costs exactly K ticks of extra latency, which is the honest trade being made.

    python scripts/debounce.py --sweep          find K
    python scripts/debounce.py --k 7 --write    apply and write predictions
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = ("/media/lm-ciss/LM_4TB/egocentric_videos/ego_psr_repro/industReal/"
      "psr_tas/extern/DiffAct/datasets/Copilot-Fusion")
SEC = 6 / 30.0


def apply_debounce(pred, k):
    """Commit to a new class only after it wins k consecutive frames."""
    if k <= 1:
        return list(pred)
    out = list(pred)
    committed = pred[0]
    run_val, run_len = pred[0], 0
    for i, p in enumerate(pred):
        if p == run_val:
            run_len += 1
        else:
            run_val, run_len = p, 1
        if run_val != committed and run_len >= k:
            committed = run_val
        out[i] = committed
    return out


def segs(L):
    o, s = [], 0
    for i in range(1, len(L) + 1):
        if i == len(L) or L[i] != L[s]:
            o.append((L[s], s, i - 1)); s = i
    return o


def f1_at(gt, pred, thr):
    g, p = segs(gt), segs(pred)
    used = [False] * len(g); tp = 0
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


def score(k, preds, gts):
    acc, f50, f75, nseg, exact, errs = [], [], [], 0, 0, []
    for pr, gt in zip(preds, gts):
        d = apply_debounce(pr, k)
        acc.append(100 * np.mean([a == b for a, b in zip(gt, d)]))
        f50.append(f1_at(gt, d, .50)); f75.append(f1_at(gt, d, .75))
        g, p = segs(gt), segs(d)
        nseg += len(p); exact += (len(g) == len(p))
        for gl, _, ge in g:
            cand = [pe for (pl, _, pe) in p if pl == gl]
            if cand:
                errs.append(abs(min(cand, key=lambda x: abs(x - ge)) - ge) * SEC)
    e = np.array(errs)
    return dict(k=k, acc=np.mean(acc), f50=np.mean(f50), f75=np.mean(f75),
                nseg=nseg, exact=exact, med=np.median(e),
                p90=np.percentile(e, 90), w2=100 * (e <= 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(HERE, "..", "runs", "causal_step", "prediction"))
    ap.add_argument("--gt", default=os.path.join(DS, "groundTruth"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    files = sorted(os.listdir(args.pred))
    preds, gts = [], []
    for fn in files:
        p = open(os.path.join(args.pred, fn)).read().split("\n")[1].split()
        g = [l.strip() for l in open(os.path.join(args.gt, fn)) if l.strip()]
        n = min(len(p), len(g)); preds.append(p[:n]); gts.append(g[:n])

    if args.sweep:
        print(f"{'K':>3} {'lag':>6} {'Acc':>7} {'F1@50':>7} {'F1@75':>7} "
              f"{'segs':>6} {'exact':>6} {'med':>6} {'p90':>6} {'±2s':>6}")
        print(f"{'':>3} {'(s)':>6} {'':>7} {'':>7} {'':>7} {'/110':>6} {'/10':>6} "
              f"{'(s)':>6} {'(s)':>6} {'(%)':>6}")
        for k in (1, 3, 5, 7, 10, 15, 20, 30, 45):
            r = score(k, preds, gts)
            print(f"{k:3d} {k*SEC:6.1f} {r['acc']:7.2f} {r['f50']:7.2f} {r['f75']:7.2f} "
                  f"{r['nseg']:6d} {r['exact']:6d} {r['med']:6.2f} {r['p90']:6.2f} {r['w2']:6.1f}")
        return

    r = score(args.k, preds, gts)
    print(f"K={args.k} (+{args.k*SEC:.1f}s lag): Acc {r['acc']:.2f}  F1@50 {r['f50']:.2f}  "
          f"F1@75 {r['f75']:.2f}  segs {r['nseg']}/110  exact {r['exact']}/10  "
          f"median {r['med']:.2f}s  within±2s {r['w2']:.1f}%")
    if args.write:
        out = args.pred.rstrip("/") + f"_k{args.k}"
        os.makedirs(out, exist_ok=True)
        for fn, pr in zip(files, preds):
            with open(os.path.join(out, fn), "w") as f:
                f.write("### Frame level recognition: ###\n")
                f.write(" ".join(apply_debounce(pr, args.k)) + "\n")
        print(f"wrote debounced predictions -> {out}")


if __name__ == "__main__":
    main()
