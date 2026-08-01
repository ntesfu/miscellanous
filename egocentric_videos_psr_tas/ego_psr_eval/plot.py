#!/usr/bin/env python
"""Render charts from results.json (metrics) and gpu_usage.csv (live GPU) -> results/charts/."""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1f2733", "#5b6b7b", "#dde3ea"
BLUE, GREEN, AMBER, RED, PURP = "#3b78b5", "#3d8a4d", "#c8722a", "#b03a3a", "#7e57c2"
BEST = "#2e7d3e"


def style(ax, title, sub=None, ylabel=None):
    if sub:
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=20)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")
    else:
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def ok(results, group=None, kind=None):
    out = []
    for name, r in results.items():
        if r.get("status") != "ok":
            continue
        if group and r["group"] != group:
            continue
        if kind and r["kind"] != kind:
            continue
        out.append((name, r))
    return out


def short(label):
    return label.replace(" + ", "+").replace("  ", " ").strip()


def chart_offline_f1(results, outdir):
    rows = ok(results, group="offline_step")
    if not rows:
        return None
    rows.sort(key=lambda kv: kv[1]["metrics"]["best"]["F1@50"])
    labels = [short(r["label"]) for _, r in rows]
    vals = [r["metrics"]["best"]["F1@50"] for _, r in rows]
    top = max(range(len(vals)), key=lambda i: vals[i])
    colors = [BEST if i == top else BLUE for i in range(len(vals))]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 2))
    bars = ax.barh(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9, color=INK)
    style(ax, "Offline step segmentation — F1@50 (best decode)",
          "IndustReal test · higher is better · green = best", "F1@50")
    ax.set_xlim(0, max(vals) * 1.12)
    p = os.path.join(outdir, "offline_f1.png")
    fig.tight_layout(); fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    return p


def chart_offline_metrics(results, outdir):
    rows = ok(results, group="offline_step")
    if not rows:
        return None
    rows.sort(key=lambda kv: kv[1]["metrics"]["best"]["F1@50"], reverse=True)
    mets = ["Acc", "Edit", "F1@10", "F1@25", "F1@50"]
    labels = [short(r["label"]) for _, r in rows]
    import numpy as np
    x = np.arange(len(mets)); w = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = [BEST, BLUE, AMBER, PURP, "#4a9d9d", "#9d4a7a"]
    for i, (_, r) in enumerate(rows):
        b = r["metrics"]["best"]
        ax.bar(x + i * w, [b[m] for m in mets], w, label=labels[i],
               color=palette[i % len(palette)])
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(mets)
    style(ax, "Offline step segmentation — all metrics", "IndustReal test · higher is better")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    p = os.path.join(outdir, "offline_metrics.png")
    fig.tight_layout(); fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    return p


def chart_correctness(results, outdir):
    rows = ok(results, kind="type")
    if not rows:
        return None
    import numpy as np
    labels = [short(r["label"]) for _, r in rows]
    inc = [r["metrics"].get("incorrect_recall", 0.0) for _, r in rows]
    rem = [r["metrics"].get("remove_recall", 0.0) for _, r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, inc, w, label="incorrect-install recall", color=RED)
    ax.bar(x + w / 2, rem, w, label="remove recall", color=GREEN)
    for xi, v in zip(x - w / 2, inc):
        ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    for xi, v in zip(x + w / 2, rem):
        ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    style(ax, "Correctness / fault detection — per-class recall",
          "the type head · incorrect-install is the hard, data-limited class", "recall (%)")
    ax.legend(fontsize=9, frameon=False)
    p = os.path.join(outdir, "correctness.png")
    fig.tight_layout(); fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    return p


def chart_streaming(results, outdir):
    rows = ok(results, group="streaming")
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (_, r), col in zip(rows, [GREEN, RED, BLUE, PURP]):
        lg = sorted(r["metrics"]["lags"], key=lambda d: d["latency_s"])
        ax.plot([d["latency_s"] for d in lg], [d["F1@50"] for d in lg],
                "o-", color=col, label=short(r["label"]))
        for d in lg:
            ax.annotate(f"L={d['L']}", (d["latency_s"], d["F1@50"]),
                        textcoords="offset points", xytext=(4, 5), fontsize=7, color=MUTED)
    style(ax, "Real-time streaming — F1@50 vs latency",
          "causal · latency dial L · up-and-left is better", "F1@50")
    ax.set_xlabel("latency (s)", fontsize=10, color=MUTED)
    ax.legend(fontsize=9, frameon=False)
    p = os.path.join(outdir, "streaming.png")
    fig.tight_layout(); fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    return p


def chart_gpu(gpu_csv, outdir):
    if not gpu_csv or not os.path.exists(gpu_csv):
        return None
    series = {}  # gpu -> (t, util, mem_used, mem_total)
    any_active = False
    with open(gpu_csv) as fh:
        for row in csv.DictReader(fh):
            g = row.get("gpu") or "-"
            try:
                t = float(row["elapsed_s"])
            except (ValueError, KeyError):
                continue
            u = row.get("util_pct") or ""
            mu = row.get("mem_used_mb") or ""
            mt = row.get("mem_total_mb") or ""
            util = float(u) if u not in ("", None) else None
            memu = float(mu) if mu not in ("", None) else None
            memt = float(mt) if mt not in ("", None) else None
            if util is not None:
                any_active = True
            series.setdefault(g, []).append((t, util, memu, memt))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    if not any_active:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No GPU activity captured\n(the evals run on CPU — run on a machine with an NVIDIA GPU to see usage)",
                    ha="center", va="center", fontsize=11, color=MUTED, transform=ax.transAxes)
    else:
        for g, pts in sorted(series.items()):
            t = [p[0] for p in pts]
            ax1.plot(t, [p[1] if p[1] is not None else float("nan") for p in pts], label=f"GPU {g}")
            ax2.plot(t, [p[2] if p[2] is not None else float("nan") for p in pts], label=f"GPU {g}")
        ax1.legend(fontsize=8, frameon=False, ncol=4)
    style(ax1, "Live GPU utilization", "sampled during evaluation", "util (%)")
    style(ax2, "Live GPU memory", None, "mem used (MB)")
    ax2.set_xlabel("elapsed (s)", fontsize=10, color=MUTED)
    p = os.path.join(outdir, "gpu_usage.png")
    fig.tight_layout(); fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    payload = json.load(open(args.results))
    results = payload.get("results", {})
    made = []
    for fn in (chart_offline_f1, chart_offline_metrics, chart_correctness, chart_streaming):
        try:
            p = fn(results, args.outdir)
            if p:
                made.append(p)
        except Exception as ex:
            print(f"  [plot] {fn.__name__} failed: {ex!r}")
    try:
        p = chart_gpu(args.gpu, args.outdir)
        if p:
            made.append(p)
    except Exception as ex:
        print(f"  [plot] chart_gpu failed: {ex!r}")
    print("charts written:")
    for p in made:
        print("   ", p)


if __name__ == "__main__":
    main()
