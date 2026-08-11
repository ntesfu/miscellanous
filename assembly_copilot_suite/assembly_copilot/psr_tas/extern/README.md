# `extern/` — third-party dependencies (not committed)

The training pipeline builds on five upstream repositories. They are **not
vendored** in this repo — clone them here at the pinned commits below, then
apply the small first-party patch and copy in the experiment configs.

## 1. Clone at the pinned commits

```bash
cd psr_tas/extern

git clone https://github.com/Finspire13/DiffAct.git         && git -C DiffAct     checkout a223e0c27a6f02adfb321a518cb0f7dd0d0896b6
git clone https://github.com/OpenGVLab/InternVideo.git      && git -C InternVideo checkout 3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3
git clone https://github.com/OpenGVLab/VideoMAEv2.git       && git -C VideoMAEv2  checkout 29eab1e8a588d1b3ec0cdec7b03a86cca491b74b
git clone https://github.com/TimSchoonbeek/IndustReal.git   && git -C IndustReal  checkout ad86a35d4b5125739e24af677e5d7b55c74af945
git clone https://github.com/ChinaYi/ASFormer.git           && git -C ASFormer    checkout e1bbe4f3ed083748f91467c51a63ac2a8b9277ad
```

What each is used for:

| Repo | Role |
|---|---|
| **DiffAct** | The diffusion action-segmentation head — the offline PSR/TAS model that is actually trained |
| **VideoMAEv2** | Frozen feature encoder A (`vit_giant_patch14_224`, SSv2-finetuned → 1408-d) |
| **InternVideo** | Frozen feature encoder B (InternVideo2 `base_patch14_224`, K710 → 768-d) |
| **IndustReal** | The reference dataset / label format the pipeline is compatible with |
| **ASFormer** | Transformer TAS baseline (used for comparison experiments) |

## 2. Apply the first-party DiffAct patch

Two small robustness fixes (constant-class TYPE sequences produce no interior
boundaries → avoid an assert and a divide-by-zero NaN):

```bash
git -C DiffAct apply ../DiffAct.patch
```

## 3. Copy in the experiment configs

The configs in [`configs/`](configs/) are first-party and define every trained
head (input_dim 2176 fusion features, encoder/decoder depths, diffusion
schedule, epochs):

```bash
cp configs/*.json DiffAct/configs/
```

- `IndustReal-Fusion-S1.json` — main step head on IndustReal (11 classes)
- `IndustReal-Type-Fusion-S1.json` — TYPE head on IndustReal (4 classes)
- `IndustReal-Fusion-SMOKE.json` — tiny smoke-test config
- `Copilot-Fusion-S1.json` — main step head on the in-house turbofan dataset
- `Copilot-Type-Fusion-S1.json` — TYPE head on the turbofan dataset

## 4. Pretrained encoder weights

The extraction scripts expect these frozen weights (a few GB, download from
Hugging Face):

| File | Source |
|---|---|
| `psr_tas/weights/vit_g_ssv2_ft.pth` | VideoMAEv2 ViT-giant, SSv2-finetuned (OpenGVLab / VideoMAEv2 releases) |
| `psr_tas/fusion/weights/iv2_b14_k710.bin` | `OpenGVLab/InternVideo2_distillation_models` — InternVideo2-B14, K710 |

On the original machine these were symlinks into `~/.cache/huggingface/hub`;
place the real files (or your own symlinks) at those paths.

Finally, `scripts/03_prepare_diffact.py` creates
`DiffAct/datasets/<Dataset>-Fusion/` with features symlinked to
`psr_tas/fusion/data/features/` — run it after feature extraction.
