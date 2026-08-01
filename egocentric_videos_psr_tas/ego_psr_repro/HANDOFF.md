# Ego-centric PSR — Project Handoff

> Written 2026-07-22 to transfer full context to a new Claude instance (or human), e.g. when
> setting the project up on another machine. Everything lives under
> `/vast/users/fahad.khan/ketan/ego_centric/` on the CAMD cluster (login `172.27.112.247`,
> user `fahad.khan`), AMD **MI210 / ROCm**, **SLURM** (partition `faculty`, qos `gtqos`).
> **Hard rule: write only inside `/vast/users/fahad.khan/ketan`.**

---

## 1. What this project is

**Procedure Step Recognition (PSR)** as Temporal Action Segmentation: input an egocentric
assembly video → output a timeline `from t0 to t1 -> <part> (correct | incorrect | remove)`.
Two-stage recipe: **frozen video backbone → per-clip features → trained segmentation head →
per-frame STEP + TYPE labels → merge into segments**. Only the head is trained.

Two datasets: **IndustReal** (84 recs, 36 train / 16 val / 32 test; 11 step classes = 10 parts +
background; 4 type classes) and **MECCANO** (20 recs, 11/2/7; 18 step classes) — the 2nd is a
generalization test.

## 2. Directory map

```
ego_centric/
  industReal/
    psr_tas/            # THE main project (all scripts, configs, extern repos, weights, env)
      scripts/          # 00_build_labels, 01_extract_features(+_v2), 02_train_asformer, 03_predict_segments, eval_step, eval_type, viterbi
      configs/          # default.yaml (Huge), default_ssv2.yaml (giant SSv2)
      fusion/           # InternVideo2 fusion: scripts/{extract_iv2,fuse}.py, configs/{fusion,fusion_l14}.yaml, weights/, slurm/
      rt/               # real-time streaming: scripts/{extract_causal,train_causal,eval_online}, configs/rt.yaml, models/, slurm/
      extern/           # cloned model repos: ASFormer, DiffAct, VideoMAEv2, InternVideo2 (single_modality), MiniROAD, TeSTra
      slurm/            # extract.sbatch, extract_v2.sbatch, train.sbatch, train_v2.sbatch, diffact_v2.sbatch, diffact_v4.sbatch, ...
      weights/          # VideoMAEv2-Huge/, vit_g_ssv2_ft.pth, vit_b_k710_dl_from_giant.pth
      data/ data_v2/    # extracted features + groundTruth (1280-d / 1408-d)
      models/           # trained ASFormer heads: step*, type* (see §4)
      psr_env/          # conda env (ROCm torch 2.4.1 + transformers 4.42.4 + easydict + tensorboard + matplotlib)
    dataset/            # 51 GB IndustReal raw (train/val/test/<rec>/{rgb,json})  [SIDE-LOADED]
  MECCANO/
    dataset/, PSR-annotations/, pipeline/{extract_fusion.py, train_meccano.sbatch, meccano.yaml}, data/
  ego_psr_eval/         # one-command EVAL harness for every architecture (git repo)
  ego_psr_repro/        # one-command REPRODUCTION DAG: download->extract->finetune->eval (git repo)  <-- YOU ARE HERE
  assets/               # comparison decks + diagrams (arch_comparison.pptx, all_arch_comparison.pptx, architecture_v4.png, metrics_comparison.png)
```

Transfer bundle of the two repos: `/vast/users/fahad.khan/ketan/ego_psr_repo_bundle.tgz` (344 KB).

## 3. The architectures & results (IndustReal test, unless noted)

**Offline step segmentation** (higher = better):

| Config | Acc | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| v1  Huge-K710 + ASFormer (baseline) | 73.3 | 68.8 | 72.9 | 68.6 | 56.3 |
| v1  + Viterbi | 73.4 | 74.2 | 78.6 | 73.9 | 62.2 |
| v2  SSv2-giant + ASFormer + Viterbi | 74.0 | 77.3 | 80.0 | 76.5 | 66.5 |
| v2  SSv2-giant + DiffAct | 74.4 | 79.6 | 81.1 | 77.6 | 68.9 |
| Fusion (giant+IV2-B14) + ASFormer + Viterbi | 75.7 | 77.6 | 79.0 | 75.8 | 68.2 |
| Fusion (giant+IV2-L14) + ASFormer + Viterbi | ~76 | — | — | — | 67.1 |
| **★ v4  Fusion + DiffAct  (NEW BEST)** | **74.9** | **79.5** | **83.1** | **79.1** | **70.1** |

**Correctness / fault detection** (type head, incorrect-install recall):

| Config | Incorrect recall | Remove recall |
|---|---|---|
| any VideoMAE-only + ASFormer | 0.0 % | ~68 % |
| **Fusion  giant + InternVideo2-B14** | **10.1 %** (prec 24.4) | 69.4 % |
| Fusion  giant + InternVideo2-L14 | 0.0 % | — |

**Real-time streaming** (causal): GRU **L=16 → Edit 55.3 / F1@50 37.4 @ 2.67 s** (best); TeSTra L=16 → 44.5 / 24.8 (worse).
**MECCANO** (generalization): Fusion+ASFormer+Viterbi → Acc 55 / Edit 64.5 / **F1@50 38.0**; type recall 0 %.

**Key findings:** (1) v4 = fusion FEATURES (from the fusion backbone) + DiffAct HEAD — the two wins
were orthogonal; F1@50 70.1 is the first result >70. (2) backbone drives segmentation quality
(SSv2 > Huge). (3) only the fusion appearance stream detects faults (0→10.1 %). (4) bigger encoder
(L14) and TeSTra both regressed → **correctness is DATA-limited, not architecture-limited** (MECCANO
0 %, L14 0 %); the lever is more/sharper error labels, not capacity. (5) the framework generalizes
unchanged to a 2nd dataset.

## 4. Trained artifacts on disk (what "finetune" produces)

- ASFormer heads: `psr_tas/models/{step,type}` (Huge, 120 ep), `{step_ssv2,type_ssv2}` (60 ep),
  `{step_fusion,type_fusion}` (60 ep), `{step_fusionl14,type_fusionl14}` (60 ep),
  `{step_meccano,type_meccano}` (60 ep). File: `epoch-<N>.model`.
- DiffAct: `psr_tas/extern/DiffAct/result/IndustReal-S1/` (v2) and `IndustReal-Fusion-S1/` (v4) —
  `epoch-*.model`, `latest.pt`, `prediction/*.txt`, `test_results_decoder-agg_epoch*.npy`.
- Streaming: `psr_tas/rt/models/{step,type,step_testra,type_testra}/model.pt`.
- Features: `data/features` (1280), `data_v2/features` (1408), `fusion/data/features` (2176),
  `fusion/data/features_iv2` (768), `fusion/data_l14/features` (2176), `rt/data/features` (768),
  `MECCANO/data/features` (2176). All aligned; `*_starts.npy` records clip positions.

## 5. `ego_psr_eval` — the evaluation harness

One command evaluates every architecture (reuses each one's own eval script; DiffAct is re-scored on
CPU from its cached predictions — it has **no eval-only entrypoint**). Live GPU monitor
(**nvidia-smi primary**, rocm-smi fallback) + charts.

```bash
cd ego_psr_eval
./run.sh                              # all architectures + charts
./run.sh --arch v4_fusion_diffact     # one
./run.sh --arch offline_step          # a group (offline_step|offline_type|streaming|meccano)
./run.sh --list | --check | --no-gpu-monitor | --gpu-interval N
```
Outputs → `results/{results.json, gpu_usage.csv, logs/, charts/}` (gitignored). All evals are
**CPU / login-node safe**. Verified numbers match §3 (v4 70.1, fusion type 10.1 %, MECCANO 38.0, GRU 37.4).

## 6. `ego_psr_repro` — the reproduction DAG (this repo)

Chains the project's existing SLURM scripts + CPU steps with correct `--dependency` edges;
shared stages (labels, giant features, fusion) run once; **idempotent** (skips DONE stages).
**Dry-run is the default — it submits nothing.**

```bash
./repro.sh                          # DRY-RUN the whole pipeline (all 9 archs)
./repro.sh --arch v4_fusion_diffact # dry-run one chain
./repro.sh --stages extract,train   # restrict to stage kinds: download,labels,extract,fuse,train,eval
./repro.sh --provision [--fetch]    # check / download datasets + weights
./repro.sh --status                 # built vs to-do
./repro.sh --list
./repro.sh --submit                 # ACTUALLY submit the SLURM DAG (GPUs!)
```
Files: `repro.sh` (entry), `orchestrate.py` (DAG + dry-run/submit), `provision.py` (assets),
`status.py`. The eval stage calls `../ego_psr_eval/run.sh`.

### Pipeline per architecture (exact commands, all sbatch submitted from `psr_tas/`)

Shared spine: `labels` → `extract_v2` (giant SSv2 **S1**) → `extract_iv2_b14` → `fuse_b14` (**S2**).

| stage | command | gpu | produces |
|---|---|---|---|
| labels | `python scripts/00_build_labels.py --dataset ../dataset --out data` | no | `data/{mapping.txt,splits,groundTruth}` |
| extract_v1 | `sbatch slurm/extract.sbatch` (array 0-7) | ✓ | `data/features` [1280] |
| train_v1 | `sbatch slurm/train.sbatch` (120 ep) | ✓ | `models/{step,type}` |
| extract_v2 (S1) | `sbatch slurm/extract_v2.sbatch` (array 0-7) | ✓ | `data_v2/features` [1408] |
| train_v2 | `sbatch slurm/train_v2.sbatch` (60 ep) | ✓ | `models/{step_ssv2,type_ssv2}` |
| diffact_v2 | `sbatch slurm/diffact_v2.sbatch` | ✓ | `extern/DiffAct/result/IndustReal-S1` |
| extract_iv2_b14 | `sbatch fusion/slurm/extract_iv2.sbatch` (array 0-7) | ✓ | `fusion/data/features_iv2` [768] |
| fuse_b14 (S2) | `python fusion/scripts/fuse.py`  (**CPU, no sbatch**) | no | `fusion/data/features` [2176] |
| train_fusion_b14 | `sbatch fusion/slurm/train_fusion.sbatch` (60 ep) | ✓ | `models/{step_fusion,type_fusion}` |
| diffact_v4 | `sbatch slurm/diffact_v4.sbatch` | ✓ | `extern/DiffAct/result/IndustReal-Fusion-S1` |
| extract_iv2_l14 | `sbatch fusion/slurm/extract_iv2_l14.sbatch` (array 0-7) | ✓ | `fusion/data/features_iv2_l14` |
| fuse_l14 | `python fusion/scripts/fuse.py --iv2_name features_iv2_l14 --out_name data_l14` (**CPU**) | no | `fusion/data_l14/features` |
| train_fusion_l14 | `sbatch fusion/slurm/train_fusion_l14.sbatch` (60 ep) | ✓ | `models/{step_fusionl14,type_fusionl14}` |
| rt_extract | `sbatch rt/slurm/extract_causal.sbatch` (array 0-7) | ✓ | `rt/data/features` [768] |
| train_rt_gru | `sbatch rt/slurm/train_rt.sbatch` | ✓ | `rt/models/{step,type}/model.pt` |
| train_rt_testra | `sbatch rt/slurm/train_testra.sbatch` | ✓ | `rt/models/{step_testra,type_testra}/model.pt` |
| mecc_extract | `sbatch pipeline/extract_fusion.sbatch` (from `MECCANO/`, array 0-7) | ✓ | `MECCANO/data/{features,labels,splits}` |
| train_meccano | `sbatch pipeline/train_meccano.sbatch` (from `MECCANO/`, 60 ep) | ✓ | `models/{step_meccano,type_meccano}` |
| eval_* | `bash ego_psr_eval/run.sh --arch <names> --no-gpu-monitor` | no | metrics |

Extract array jobs dominate GPU cost (~8 concurrent GPUs each, 4–8 h caps). Train/eval are light.

## 7. Provisioning — datasets + weights (`./repro.sh --provision`)

| Asset | On-disk path (rel to psr_tas unless noted) | Auto? | How / why blocked |
|---|---|---|---|
| VideoMAEv2-Huge (1280) | `weights/VideoMAEv2-Huge/` | ✅ | `huggingface-cli download OpenGVLab/VideoMAEv2-Huge --local-dir weights/VideoMAEv2-Huge` |
| ViT-B distilled (768, RT) | `weights/vit_b_k710_dl_from_giant.pth` | ✅ | `hf_hub_download('OpenGVLab/VideoMAE2','distill/vit_b_k710_dl_from_giant.pth')` |
| MECCANO videos + PSR annotations | `MECCANO/dataset/`, `MECCANO/PSR-annotations/` | ✅ | HF `ketanmore/MECCANO` + `git clone TimSchoonbeek/PSR-annotations` |
| **IndustReal raw (51 GB)** | `industReal/dataset/` | ❌ | 4tu.nl (DOI 10.4121/c.6104020) proxy-blocked → **side-load** |
| **giant SSv2 (1408)** | `weights/vit_g_ssv2_ft.pth` | ❌ | Google-form gated (`vit_g_hybrid_pt_1200e_ssv2_ft`) → **side-load** |
| **InternVideo2 B14 / L14 (768)** | `fusion/weights/iv2_{b14,l14}_k710.bin` | ❌ | license-gated → **side-load** |

`--fetch` downloads the ✅ ones and prints exact side-load instructions for the ❌ ones; it never
fakes a download and now **exits nonzero if a required side-load asset is missing** (fail-fast gate).

## 8. Setting up on ANOTHER machine (e.g. NVIDIA box)

The two repos use **relative paths**, so keep the layout: `ego_centric/{industReal/psr_tas,
industReal/dataset, MECCANO, ego_psr_eval, ego_psr_repro}`.

1. **Transfer code** (not the 100 GB of data/features/weights/env):
   `rsync -av --exclude 'dataset/' --exclude 'data*/features' --exclude 'weights/' --exclude 'fusion/weights/' --exclude 'psr_env/' --exclude '*/result/' --exclude '__pycache__/' <src>/ego_centric/ <dst>/ego_centric/`
2. **Create env** at `psr_tas/psr_env` (the repos expect that name) with **CUDA** torch (not ROCm):
   `conda create -p ./psr_env python=3.10 -y`; `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`;
   `pip install numpy pyyaml opencv-python-headless timm transformers easydict tensorboard matplotlib huggingface_hub scipy tqdm einops`
3. **Fix machine-specific paths** (the one real gotcha):
   - Conda path `/vast/users/fahad.khan/miniconda3/.../conda.sh` → your conda, in `ego_psr_repro/{repro.sh,orchestrate.py}`, `ego_psr_eval/run.sh`, and the `source ...` lines inside the `slurm/*.sbatch`.
   - **SLURM headers** in `psr_tas/slurm/*`, `fusion/slurm/*`, `rt/slurm/*`, `MECCANO/pipeline/*` and `CPU_SB` in `orchestrate.py`: `--partition=faculty --qos=gtqos --gres=gpu:mi210:1 --exclude=auh7-1b-gpu-199` → your partition + `--gres=gpu:1`. **No SLURM?** skip `--submit`; run the python commands the dry-run prints directly.
4. **Provision:** `./repro.sh --provision --fetch`, then side-load the 3 gated assets to the paths printed.
5. **Verify → run:** `./repro.sh --provision` (exit 0) → `--status` → `./repro.sh` (dry-run) → `--submit`.
6. **Eval:** `cd ../ego_psr_eval && ./run.sh`.

**Eval-only shortcut** (no retrain): transfer the small trained checkpoints (`models/*`,
`rt/models/*`, `extern/DiffAct/result/*`) + the feature dirs, then `ego_psr_eval/run.sh`. Avoids the
51 GB dataset + gated backbones entirely.

## 9. Gotchas (hard-won — will bite a fresh setup)

- **DiffAct has no eval-only entrypoint** — `main.py` always trains. Get its metrics by re-scoring the
  cached `prediction/*.txt` with `utils.func_eval`, or read `test_results_*_epoch1000.npy`. Do NOT
  just re-run `main.py` (resumes from `latest.pt`@1000, `num_epochs=1200`, `log_freq=200` → trains
  1001–1199, hits no eval epoch, wastes GPU).
- **Eval epoch is 60, not the script default 120** for v2 / fusion / L14 / MECCANO (only v1 is 120).
- **`eval_type.py` has no `--data` flag** — it reads `cfg["paths"]["data"]`, so pass the matching `--config`.
- **MECCANO config path**: from `psr_tas/scripts`, config is `../../../MECCANO/pipeline/meccano.yaml`,
  data `../../MECCANO/data` (data joins onto ROOT=psr_tas, config resolves vs cwd). The repro
  orchestrator resolves it absolutely (needs two `..` from psr_tas).
- **`fuse.py` and the L14/MECCANO evals have no committed sbatch** — the orchestrator supplies them
  (CPU steps wrapped as short cpu SLURM jobs under `--submit`).
- **Backbone loaders**: giant SSv2 uses the VideoMAEv2 **repo builder** `vit_giant_patch14_224`
  (NOT the HF snapshot); timm 1.x `create_model` injects `pretrained_cfg` → call the registered
  builder directly. InternVideo2 built with `use_flash_attn/fused=False` for ROCm.
- **Node `auh7-1b-gpu-199` is chronically slow** on CAMD — excluded in sbatch (`--exclude=`); drop this
  on another cluster.
- **GPU monitor** = nvidia-smi primary (this project targets NVIDIA); on a CPU-only login node the GPU
  chart is a flat "no activity" line — that's correct, not a bug.

## 10. State at handoff & open threads

- **All architectures are already trained and evaluated on CAMD** (results in §3, verified via the
  harness). Nothing pending to train.
- **Transfer to the NVIDIA box (`192.168.20.148`) is blocked from the cluster** — the login node has
  no route to that LAN. Bridge via a relay machine (your Mac reaches both) or HuggingFace. Bundle
  ready at `/vast/users/fahad.khan/ketan/ego_psr_repo_bundle.tgz`. Neither repo is pushed to a git
  remote yet.
- **Not built (future work, from the ideation pass):** the ASD **assembly-state track** in
  `OD_labels.json` (per-frame 11-bit part-presence, 26,925 dense frames, currently unused) is the
  highest-leverage next asset — it enables OD-state boundary relabeling (top segmentation win),
  filling the empty Viterbi `forbidden` mask (free decode gain), and a Graph-Consistency Auditor
  (skip/mis-order detection the type head can't do). `error_state` is never assigned (0 frames) →
  label-free correctness catches omission/order only. **REMOVE** (~159 events) is NOT data-limited and
  should be promoted to a first-class track; **incorrect-install** (~29 events) is capped → needs
  event-level / LOOCV eval to report honestly. An agentic layer (Graph Auditor → label sharpener →
  active-learning flywheel) was proposed; see `assets/` diagrams.
- **Comparison decks**: `assets/{all_arch_comparison.pptx, arch_comparison.pptx, ego_arch_comparison.pptx}`,
  diagram `architecture_v4.png`, table `metrics_comparison.png`.
