#!/usr/bin/env python
"""Sweep causal decoding configs and report the full comparable metric suite.

Every config here is strictly causal: the state emitted for timestep t uses only
observations up to t (+ an explicit fixed lag, charged in the latency column).
Scored with the same definitions as psr_tas/scripts/copilot_metrics.py so the numbers
sit directly beside the offline head's.
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from causal_decode import learn_transitions, decode              # noqa: E402
from debounce import apply_debounce, segs, f1_at                 # noqa: E402

DS = ("/media/lm-ciss/LM_4TB/egocentric_videos/ego_psr_repro/industReal/"
      "psr_tas/extern/DiffAct/datasets/Copilot-Fusion")
SEC = 6 / 30.0


def score(preds, gts):
    acc, f50, f75, f90, nseg, exact, errs = [], [], [], [], 0, 0, []
    for pr, gt in zip(preds, gts):
        acc.append(100 * np.mean([a == b for a, b in zip(gt, pr)]))
        f50.append(f1_at(gt, pr, .50)); f75.append(f1_at(gt, pr, .75))
        f90.append(f1_at(gt, pr, .90))
        g, p = segs(gt), segs(pr)
        nseg += len(p); exact += (len(g) == len(p))
        for gl, _, ge in g:
            cand = [pe for (pl, _, pe) in p if pl == gl]
            if cand:
                errs.append(abs(min(cand, key=lambda x: abs(x - ge)) - ge) * SEC)
    e = np.array(errs)
    return (np.mean(acc), np.mean(f50), np.mean(f75), np.mean(f90), nseg, exact,
            np.median(e), np.percentile(e, 90), 100 * (e <= 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DS)
    ap.add_argument("--logits", default=os.path.join(HERE, "..", "runs", "causal_step", "logits"))
    args = ap.parse_args()

    classes = [l.split(maxsplit=1)[1].strip()
               for l in open(os.path.join(args.dataset, "mapping.txt"))]
    tr_gt = [[l.strip() for l in open(os.path.join(args.dataset, "groundTruth", r[:-4] + ".txt")) if l.strip()]
             for r in open(os.path.join(args.dataset, "splits", "train.split1.bundle")).read().split()]
    logA = learn_transitions(tr_gt, classes)

    recs = [r[:-4] for r in open(os.path.join(args.dataset, "splits", "test.split1.bundle")).read().split()]
    LP, GT = [], []
    for r in recs:
        lp = np.load(os.path.join(args.logits, r + ".npy"))
        gt = [l.strip() for l in open(os.path.join(args.dataset, "groundTruth", r + ".txt")) if l.strip()]
        n = min(len(lp), len(gt)); LP.append(lp[:n]); GT.append(gt[:n])

    hdr = (f"{'config':<28}{'lag':>6}{'Acc':>8}{'F1@50':>8}{'F1@75':>8}{'F1@90':>8}"
           f"{'segs':>7}{'exact':>7}{'med':>7}{'p90':>7}{'±2s':>7}")
    print(hdr); print("-" * len(hdr))

    def row(name, lag_s, preds):
        a, f50, f75, f90, ns, ex, med, p90, w2 = score(preds, GT)
        print(f"{name:<28}{lag_s:>6.1f}{a:>8.2f}{f50:>8.2f}{f75:>8.2f}{f90:>8.2f}"
              f"{ns:>7d}{ex:>7d}{med:>7.2f}{p90:>7.2f}{w2:>7.1f}")

    row("raw argmax", 0.0, [[classes[i] for i in lp.argmax(1)] for lp in LP])
    for k in (3, 5):
        row(f"debounce K={k}", k * SEC,
            [apply_debounce([classes[i] for i in lp.argmax(1)], k) for lp in LP])
    for sb in (0.0, 2.0, 4.0, 6.0, 8.0):
        row(f"viterbi lag=0 bias={sb}", 0.0,
            [[classes[i] for i in decode(lp, logA, 0, sb)] for lp in LP])
    for lag in (5, 10, 25):
        for sb in (4.0, 6.0):
            row(f"viterbi lag={lag} bias={sb}", lag * SEC,
                [[classes[i] for i in decode(lp, logA, lag, sb)] for lp in LP])
    print("\noffline DiffAct reference    n/a   96.24   97.51   89.78   72.62    "
          "~124      8   0.60   2.24   87.2   (NOT causal)")


if __name__ == "__main__":
    main()
