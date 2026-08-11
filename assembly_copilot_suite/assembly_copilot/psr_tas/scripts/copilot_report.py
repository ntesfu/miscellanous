#!/usr/bin/env python3
"""Summarise a DiffAct training log: the eval curve, where it plateaus, final epoch.

Reports the plateau ("knee") and the final-epoch score SEPARATELY and on purpose.
DiffAct evaluates on the TEST set every log_freq epochs, so picking the best epoch
off that curve is selecting on test. With no val split (30/10) the defensible
headline is the fixed final epoch; the knee is reported only as a compute-budget
finding -- "N epochs would have done" -- not as a score to quote.

    python scripts/copilot_report.py logs/copilot_train_step.log [--metric Acc]
"""
import argparse, collections, re, sys

ROW = re.compile(r"Epoch (\d+) - (\S+?)-Test-(\S+) ([\d.]+)")


def parse(path):
    """-> {decode_head: {epoch: {metric: value}}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(path) as f:
        for line in f:
            m = ROW.search(line)
            if m:
                ep, head, metric, val = int(m[1]), m[2], m[3], float(m[4])
                out[head][ep][metric] = val
    return out


def knee(epochs, vals, tol=0.5):
    """First epoch within `tol` points of the best value ever reached.

    Deliberately not argmax: on a noisy curve argmax is a single lucky eval, while
    "first time it got this good and stayed" is what tells you the budget you
    actually needed.
    """
    best = max(vals)
    for e, v in zip(epochs, vals):
        if v >= best - tol:
            return e, v, best
    return epochs[-1], vals[-1], best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--metric", default="Acc",
                    help="metric to locate the plateau on. Default Acc: on this "
                         "dataset Edit/F1@10/F1@25 saturate (only 6 distinct segment "
                         "orderings exist, all seen in training), so frame accuracy "
                         "is the only one with real headroom.")
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    data = parse(args.log)
    if not data:
        sys.exit(f"no eval rows found in {args.log}")

    for head, per_epoch in data.items():
        eps = sorted(per_epoch)
        metrics = sorted({m for d in per_epoch.values() for m in d})
        print(f"\n=== {head} ===")
        print("  epoch  " + "".join(f"{m:>10}" for m in metrics))
        for e in eps:
            print(f"  {e:5d}  " + "".join(f"{per_epoch[e].get(m, float('nan')):10.2f}"
                                          for m in metrics))
        if args.metric in metrics and len(eps) > 1:
            vals = [per_epoch[e].get(args.metric, float("nan")) for e in eps]
            ke, kv, best = knee(eps, vals, args.tol)
            fe = eps[-1]
            print(f"\n  plateau on {args.metric}: epoch {ke} ({kv:.2f}), "
                  f"within {args.tol} of best {best:.2f}")
            print(f"  FINAL epoch {fe}: " +
                  "  ".join(f"{m} {per_epoch[fe].get(m, float('nan')):.2f}" for m in metrics))
            if ke < fe:
                print(f"  -> {fe - ke} of {fe} epochs bought nothing measurable; "
                      f"budget ~{ke} next time.")
            print("  NOTE: quote the FINAL-epoch row. The plateau is a budget hint -- "
                  "this curve is measured on test, so selecting on it leaks.")


if __name__ == "__main__":
    main()
