# Egocentric PSR — Procedure Step Recognition on IndustReal

Given an egocentric assembly video (HoloLens footage of a person building a
gearbox), predict a timeline of **what procedural step is happening and
whether it was done correctly**:

```
from t0 to t1  ->  <part>  (correct-install | incorrect-install | remove)
from t1 to t2  ->  <part>  ...
```

This is framed as **Temporal Action Segmentation (TAS)**: a two-stage
recipe where a *frozen* video backbone produces per-clip features, and only a
lightweight segmentation *head* is trained on top.

```
video clips -> frozen backbone -> per-clip features -> trained head -> per-frame
  (STEP, TYPE) labels -> decode/merge into segments
```

- **STEP** = which of 10 gearbox parts is being worked on (+ background) — 11 classes.
- **TYPE** = correctness of that step — 4 classes (none / correct / incorrect / remove).

**Best result on IndustReal test** (fusion features + DiffAct head):
Accuracy **74.9** / Edit **79.5** / F1@10 **83.1** / F1@25 **79.1** / F1@50 **70.1**.
A causal/real-time variant reaches Edit 55.3 / F1@50 37.4 at 2.67s latency.
The same framework was validated on a second dataset, MECCANO (F1@50 38.0),
confirming it generalizes.

## Dataset

[IndustReal](https://doi.org/10.4121/c.6104020) — egocentric HoloLens video of
people assembling/disassembling a gearbox. 84 recordings (36 train / 16 val /
32 test). Ground truth comes from IndustReal's official `procedure_info.json`
(33 fine actions = 11 part-states x {correct-install, incorrect-install,
remove}); state_idx 0 ("base") is never a procedure step, giving 10 real parts
+ background = the 11 STEP classes.

The raw video itself is **not** in this repo (see "What you need to add"
below) — only the derived, per-frame text labels (`psr_tas/data/`) are
included, since those are small and are this project's own artifact.

## Architecture

Two families of segmentation head are supported over the same features:

- **ASFormer** — a transformer-based action-segmentation model
  (`extern/ASFormer/model.py`).
- **DiffAct** — a diffusion-based action-segmentation model, the stronger of
  the two (`extern/DiffAct/model.py`).

Two backbones, usable alone or fused:

- **VideoMAEv2-giant, SSv2-finetuned** (`extern/VideoMAEv2/models/modeling_finetune.py`,
  `vit_giant_patch14_224`) — 16-frame clips, 1408-d pooled feature.
- **InternVideo2-B14, K710-finetuned** (`extern/InternVideo/InternVideo2/single_modality/models/internvideo2.py`,
  `internvideo2_base_patch14_224`) — 8-frame clips (same 16-frame window,
  uniformly subsampled), 768-d pooled feature.
- **Fusion** = concatenate the two aligned per-clip feature streams -> 2176-d
  (`psr_tas/fusion/scripts/fuse.py`). This is the best-performing input.

Both backbones are used **frozen** purely as feature extractors; nothing
inside `extern/VideoMAEv2` or `extern/InternVideo` is trained. Only the
ASFormer/DiffAct head is trained, on top of the frozen features.

## Repository structure

```
psr_tas/                        the main project
  scripts/
    00_build_labels.py          IndustReal procedure_info.json -> per-frame
                                 STEP/TYPE ground truth (mapping.txt,
                                 groundTruth/, groundTruth_type/, splits/)
    01_extract_v2.py            extract VideoMAEv2-giant (SSv2) clip features
                                 -> data_v2/features/<rec>.npy [T,1408]
    03_prepare_diffact.py       assemble a DiffAct-format dataset dir from
                                 fused features + clip-aligned STEP labels
    run_full_pipeline.sh        giant extract -> IV2-B14 extract -> fuse ->
                                 prepare DiffAct dataset, all recordings
    run_all.sh                  full pipeline + DiffAct training in one shot
    run_extract_chain.sh        chains IV2-B14 extraction after giant extraction

  fusion/scripts/
    extract_iv2.py              extract InternVideo2-B14 (K710) clip features
                                 -> fusion/data/features_iv2/<rec>.npy [T,768]
    fuse.py                     concat giant(1408) + IV2-B14(768) aligned by
                                 clip start -> fusion/data/features/<rec>.npy [T,2176]

  data/                         ground-truth labels (from 00_build_labels.py):
    mapping.txt, mapping_type.txt     class_id -> name (STEP / TYPE)
    groundTruth/<rec>.txt             per-frame STEP class name
    groundTruth_type/<rec>.txt        per-frame TYPE class name
    splits/{train,test}.split1.bundle recording lists per split

  live_monitor.py               local stdlib-only HTTP dashboard; visualizes
                                 the fusion+DiffAct architecture with a live
                                 training overlay (reads logs/ + nvidia-smi)

  extern/                       vendored third-party model code (trimmed to
                                 the source actually imported; no checkpoints)
    ASFormer/                   ASFormer head (model.py, main.py, eval.py)
    DiffAct/                    DiffAct head (model.py, dataset.py, main.py,
                                 configs/IndustReal-Fusion-S1.json)
    VideoMAEv2/                 VideoMAEv2 backbone builder
    InternVideo/InternVideo2/single_modality/models/
                                 InternVideo2-B14 backbone (only the 3 files
                                 extract_iv2.py imports)
    IndustReal/                 official dataset repo: PSR/procedure_info.json
                                 (taxonomy), PSR/psr_baseline.py, AR/, ASD/

ego_psr_eval/                   one-command harness that evaluates every
                                 trained architecture and collects results.json
  evaluate.py                   architecture registry + eval driver (reuses
                                 each architecture's own eval script)
  run.sh                        entry point: ./run.sh [--arch NAME|GROUP]
  gpu_monitor.py, plot.py       live GPU usage + result charts

ego_psr_repro/                  reproduction DAG: chains the SLURM/CPU stages
                                 with correct dependencies, idempotent
  orchestrate.py                the DAG (dry-run by default; --submit to run)
  provision.py                  checks/downloads datasets + weights
  repro.sh                      entry point: ./repro.sh [--status|--submit|...]
  HANDOFF.md                    full original project write-up: every
                                 architecture's exact results table, every
                                 pipeline stage's exact command, and the
                                 hard-won gotchas (see below) — read this
                                 for anything not covered here
```

## What you need to add before running anything

This repo has the code; it does not have the data or the pretrained weights
(too large / license-gated to redistribute):

| Asset | Where it goes | How to get it |
|---|---|---|
| IndustReal raw video (~50GB) | `psr_tas/../dataset/` | 4TU.nl, DOI `10.4121/c.6104020` |
| VideoMAEv2-giant SSv2-ft checkpoint | `psr_tas/weights/vit_g_ssv2_ft.pth` | Google-form gated (`vit_g_hybrid_pt_1200e_ssv2_ft`) |
| InternVideo2-B14 K710-ft checkpoint | `psr_tas/fusion/weights/iv2_b14_k710.bin` | license-gated, OpenGVLab/InternVideo2 |
| VideoMAEv2-Huge (v1 pipeline only) | `psr_tas/weights/VideoMAEv2-Huge/` | `huggingface-cli download OpenGVLab/VideoMAEv2-Huge` |

Verify checksums for any gated checkpoint against the publisher's listed hash
before loading it. Full provisioning details: `ego_psr_repro/HANDOFF.md` §7.

## Environment

```bash
conda create -p ./psr_tas/psr_env python=3.10 -y
conda activate ./psr_tas/psr_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pyyaml opencv-python-headless timm transformers easydict \
            tensorboard matplotlib huggingface_hub scipy tqdm einops
```

## Training pipeline (fusion + DiffAct — the best-performing configuration)

All commands run from `psr_tas/`.

```bash
# 1. Build per-frame STEP/TYPE labels from IndustReal's procedure_info.json
python scripts/00_build_labels.py --dataset ../dataset --out data

# 2. Extract VideoMAEv2-giant (SSv2) features, all splits -> data_v2/features [1408-d]
python scripts/01_extract_v2.py --splits train,test,val --stride 2 --batch 8

# 3. Extract InternVideo2-B14 features, clip-aligned to step 2 -> fusion/data/features_iv2 [768-d]
python fusion/scripts/extract_iv2.py --splits train,test,val --batch 32

# 4. Fuse the two streams -> fusion/data/features [2176-d] (CPU, no GPU needed)
python fusion/scripts/fuse.py

# 5. Prepare a DiffAct-format dataset dir (clip-aligned labels) from the fused features
python scripts/03_prepare_diffact.py

# 6. Train DiffAct (1200 epochs, evaluates on TEST every 200 epochs)
cd extern/DiffAct
python main.py --config configs/IndustReal-Fusion-S1.json --device 0
```

Or run steps 2-6 unattended and resumable with `bash scripts/run_all.sh`
(logs to `logs/run_all.log`; each stage skips already-completed work).

**ASFormer** (either backbone alone, or the fusion features) trains the same
way via `extern/ASFormer/main.py` — see `ego_psr_repro/HANDOFF.md` §6 for the
exact per-architecture command table (9 architectures: v1 Huge+ASFormer,
v2 SSv2+ASFormer, fusion B14/L14+ASFormer, v2/fusion+DiffAct, real-time
GRU/TeSTra, MECCANO).

## Evaluation

```bash
cd ego_psr_eval
./run.sh                          # every trained architecture -> results.json + charts
./run.sh --arch v4_fusion_diffact # just the best config
./run.sh --list                   # see all registered architectures
```

`evaluate.py` re-runs each architecture's *own* eval script (`eval_step.py`,
`eval_type.py`, `rt/eval_online.py`) so numbers match what those scripts
produce natively. DiffAct has no eval-only entrypoint, so its metrics are
recomputed on CPU from its cached `prediction/*.txt` via its own
`utils.func_eval` — never re-run DiffAct's `main.py` just to get numbers (see
Gotchas).

## Results

Offline step segmentation, IndustReal test:

| Config | Acc | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| v1 Huge-K710 + ASFormer + Viterbi | 73.4 | 74.2 | 78.6 | 73.9 | 62.2 |
| v2 SSv2-giant + ASFormer + Viterbi | 74.0 | 77.3 | 80.0 | 76.5 | 66.5 |
| v2 SSv2-giant + DiffAct | 74.4 | 79.6 | 81.1 | 77.6 | 68.9 |
| Fusion (giant+IV2-B14) + ASFormer + Viterbi | 75.7 | 77.6 | 79.0 | 75.8 | 68.2 |
| **v4 Fusion + DiffAct (best)** | **74.9** | **79.5** | **83.1** | **79.1** | **70.1** |

Real-time streaming (causal): GRU L=16 -> Edit 55.3 / F1@50 37.4 @ 2.67s latency.
MECCANO (generalization test): Fusion+ASFormer+Viterbi -> Acc 55 / Edit 64.5 / F1@50 38.0.

Correctness/fault detection (TYPE head, incorrect-install recall): only the
fusion (giant+InternVideo2-B14) stream detects any faults at all (10.1%
recall / 24.4% precision) — all VideoMAE-only and IV2-L14 configs score 0%.
This is a data-scarcity effect (very few real incorrect-install events in
IndustReal), not an architecture limitation.

## Gotchas (from the original project's hard-won notes)

- **DiffAct has no eval-only entrypoint** — `main.py` always trains. Rerunning
  it just resumes from `latest.pt`; get metrics from the cached
  `prediction/*.txt` instead (see `ego_psr_eval/evaluate.py`).
- **Eval epoch is 60, not 120**, for every config except v1.
- **`extern/InternVideo`** here is trimmed to only the 3 files
  `extract_iv2.py` actually imports (`models/internvideo2.py`,
  `pos_embed.py`, `flash_attention_class.py`) out of the full InternVideo
  monorepo — from `github.com/OpenGVLab/InternVideo` (Apache-2.0); go there
  for anything beyond feature extraction (training/finetuning the backbone
  itself, other InternVideo versions, etc).
- **Backbone loader**: the giant SSv2 backbone uses VideoMAEv2's own builder
  `vit_giant_patch14_224` — not the HuggingFace snapshot — because timm 1.x's
  `create_model` path injects an incompatible `pretrained_cfg`.
- Full machine-setup notes, path fixes, and SLURM-to-single-GPU adaptation:
  `ego_psr_repro/HANDOFF.md` §8-9.

## Credits

Built on top of:
[ASFormer](https://github.com/ChinaYi/ASFormer),
[DiffAct](https://github.com/Finspire13/DiffAct),
[VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2),
[InternVideo2](https://github.com/OpenGVLab/InternVideo), and the
[IndustReal](https://github.com/TimSchoonbeek/IndustReal) dataset + baseline
code. Each vendored directory under `extern/` keeps its own LICENSE/README
where included.
