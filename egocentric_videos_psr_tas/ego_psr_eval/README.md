# ego_psr_eval

**One command to evaluate every trained architecture in the ego-centric Procedure
Step Recognition project** — offline segmentation, correctness/fault heads, real-time
streaming, and MECCANO — with a live GPU-usage monitor and auto-generated charts.

```bash
./run.sh                 # evaluate ALL architectures, chart everything
```

That single command: activates the project env, verifies every required artifact is
present, samples GPU usage live while it runs, executes each architecture's **own**
eval script (so numbers match the project exactly), collects everything into
`results/results.json`, and renders charts into `results/charts/`.

## Usage

```bash
./run.sh                              # all architectures
./run.sh --arch v4_fusion_diffact     # one architecture
./run.sh --arch offline_step          # a whole group
./run.sh --arch v2_ssv2,v4_fusion_diffact,v3_gru   # a comma list
./run.sh --list                       # list architectures + groups, then exit
./run.sh --check                      # preflight only: are all artifacts present?
./run.sh --no-gpu-monitor             # skip the live GPU sampler
./run.sh --gpu-interval 1             # GPU sampling interval in seconds (default 3)
```

The same selection works directly on the Python driver:
`psr_env/bin/python evaluate.py --arch <all|group|name[,name...]>`.

## What it evaluates

| group | architectures |
|---|---|
| `offline_step` | v1 Huge+ASFormer · v2 SSv2+ASFormer · Fusion-B14+ASFormer · Fusion-L14+ASFormer · v2 SSv2+DiffAct · **v4 Fusion+DiffAct** |
| `offline_type` | correctness heads: v1 · v2 SSv2 · Fusion-B14 · Fusion-L14 (incorrect / remove recall) |
| `streaming` | v3 causal ViT-B+GRU · v3 TeSTra (F1@50 vs latency, per lag L) |
| `meccano` | MECCANO Fusion+ASFormer (step + type) |

Metrics: **Acc, Edit, F1@10/25/50** for step segmentation (raw + a Viterbi penalty
sweep, best reported); **per-class precision/recall** for the type heads
(the key **incorrect-install** and **remove** recall); **per-lag F1@50 + latency**
for streaming.

## Outputs

```
results/
  results.json          # all metrics, machine-readable
  gpu_usage.csv         # live GPU samples (elapsed_s, util%, mem, ...)
  logs/<arch>.log       # full raw stdout of every eval (nothing is lost)
  logs/gpu_monitor.log
  charts/
    offline_f1.png      # F1@50 across offline architectures (best highlighted)
    offline_metrics.png # Acc/Edit/F1@10/25/50 grouped bars
    correctness.png     # incorrect-install vs remove recall
    streaming.png       # F1@50 vs latency (GRU vs TeSTra)
    gpu_usage.png       # live GPU utilization + memory timeline
```

## Live GPU monitor

A background sampler (`gpu_monitor.py`) records GPU utilization and memory every few
seconds into `gpu_usage.csv` and charts it. It uses **`nvidia-smi`** (primary) with a
**`rocm-smi`** fallback. Note: the ASFormer/streaming/DiffAct evals here run on **CPU**,
so the GPU chart reflects whatever the host's NVIDIA GPU is doing during the run; on a
GPU-less host it is a flat "no activity" line.

Control it with `--gpu-interval <sec>` (default 3) or disable it with `--no-gpu-monitor`.

## How it works / design notes

- **Reuses the project's own eval scripts** (`scripts/eval_step.py`, `scripts/eval_type.py`,
  `rt/scripts/eval_online.py`) via subprocess and parses their output, so the numbers are
  identical to the project's — the harness never re-implements a metric.
- **DiffAct** has no eval-only entrypoint, so its metrics are recomputed on CPU from its
  **cached predictions** via DiffAct's own `func_eval` (no GPU retrain). Falls back to the
  saved `test_results_*.npy` if predictions are absent.
- **Nothing in the project is modified** — this repo only reads the existing models,
  features, and predictions under `../industReal/psr_tas` (and `../MECCANO`).

## Scope

This evaluates **already-trained** architectures against **already-extracted** features
(all present on disk). Feature re-extraction from raw video and model training need GPUs
and gated backbone weights and are intentionally **out of scope** — the existing project
pipelines (`01_extract_features*.py`, `slurm/*.sbatch`) handle those.

## Requirements

The project conda env `psr_env` (torch + numpy + pyyaml + matplotlib), created by the
main project. `run.sh` activates it automatically; if it's missing, create it per the main
project's setup. No other dependencies.
