# ego_psr_repro

**Reproduce every PSR architecture end-to-end** — *provision (download) → extract features →
finetune heads → evaluate* — as one dependency-ordered SLURM DAG.

It chains the project's **existing** scripts (label build, feature extractors, ASFormer/DiffAct/
GRU trainers, and the `ego_psr_eval` harness) with correct `--dependency` edges, running each
**shared** stage (labels, giant features, fusion) exactly once. It is **idempotent**: any stage
whose outputs already exist is marked `DONE` and skipped, so a re-run only fills gaps.

> **Dry-run is the default. It prints the whole plan and submits nothing.**
> Launching the real GPU DAG requires the explicit `--submit` flag.

## Usage

```bash
./repro.sh                          # DRY-RUN the whole pipeline (all 9 architectures)
./repro.sh --arch v4_fusion_diffact # dry-run one architecture's full chain
./repro.sh --arch v2_ssv2,fusion_b14
./repro.sh --stages extract,train   # restrict to stage kinds (download,labels,extract,fuse,train,eval)
./repro.sh --provision              # check datasets + weights are present
./repro.sh --provision --fetch      # download the auto-downloadable assets (HF/git)
./repro.sh --status                 # which stages are built vs to-do
./repro.sh --list                   # list architectures + stages
./repro.sh --submit                 # ACTUALLY submit the SLURM DAG (allocates GPUs!)
```

Architectures: `v1_huge · v2_ssv2 · fusion_b14 · fusion_l14 · v2_diffact · v4_fusion_diffact ·
v3_gru · v3_testra · meccano`.

## The pipeline (per architecture)

```
provision ─▶ labels ─▶ extract (frozen backbone, GPU array 0-7) ─▶ [fuse, CPU] ─▶ finetune (GPU) ─▶ evaluate (CPU)
```

Shared spine (built once, reused): `labels` → `extract_v2` (giant SSv2 1408-d, **S1**) →
`extract_iv2_b14` → `fuse_b14` (2176-d, **S2**). Fusion, both DiffActs, and MECCANO all hang off it.
`./repro.sh --list` shows every stage; `--status` shows what's already done.

## Provisioning — what auto-downloads vs. side-load

`./repro.sh --provision` prints this live. Summary:

| Asset | Auto? | How |
|---|---|---|
| VideoMAEv2-Huge (1280) | ✅ | `huggingface-cli download OpenGVLab/VideoMAEv2-Huge` |
| VideoMAEv2 ViT-B distilled (768, RT) | ✅ | `hf_hub_download OpenGVLab/VideoMAE2 distill/...` |
| MECCANO videos + PSR annotations | ✅ | HF mirror (`ketanmore/MECCANO`) + GitHub |
| **IndustReal raw dataset (51 GB)** | ❌ | 4tu.nl is proxy-blocked → **side-load** to `industReal/dataset/` |
| **VideoMAEv2 ViT-giant SSv2 (1408)** | ❌ | Google-form gated → **side-load** to `weights/vit_g_ssv2_ft.pth` |
| **InternVideo2 B14 / L14 (768)** | ❌ | license-gated → **side-load** to `fusion/weights/` |

`--fetch` downloads only the reachable ones and prints exact side-load instructions for the rest —
it never fabricates a download. (Everything is already on disk today; provisioning is for a fresh box.)

## Rough GPU cost (from-scratch, MI210 / any NVIDIA)

Feature-extraction array jobs dominate (8 concurrent GPUs each): `extract_v1/v2` ~8 h, the three
InternVideo2/causal extracts ~4 h, MECCANO ~6 h. Training/eval jobs are light (tiny ASFormer/GRU
heads over ≤84 feature files; DiffAct ~48 min). Everything is idempotent, so re-runs are cheap.

## What it does *not* do

- It does not re-implement any stage — it drives the existing `psr_tas` / `MECCANO` scripts and the
  `ego_psr_eval` harness, read-only over their code.
- It cannot fetch the proxy-blocked / license-gated assets above (physical constraint, not a bug).
- `fuse` (the 2 CPU concat steps) and the fusion-L14 / MECCANO evals have no committed sbatch in the
  project; the orchestrator supplies them (CPU steps are wrapped as short cpu SLURM jobs under `--submit`).

## Layout

```
repro.sh        # one entry point (env + dispatch); dry-run by default
orchestrate.py  # the stage DAG, dependency resolution, dry-run + --submit
provision.py    # asset presence check + fetch of the downloadable ones
status.py       # built-vs-todo per architecture
```

Requires the project conda env `psr_env` (activated automatically). The eval stage reuses
`../ego_psr_eval`.
