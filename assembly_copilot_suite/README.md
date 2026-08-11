# Assembly Copilot — Project Suite

An end-to-end system for building an **AI assembly copilot**: software that
watches a person assemble (or disassemble) a turbofan engine model through a
first-person camera and understands the procedure as it happens — which step is
in progress, when each step completes, and which part comes next.

This monorepo contains the four tools that make up the full pipeline, from data
capture to a live demo. Each subproject is self-contained with its own detailed
README.

```
 ┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────────────┐
 │ 1. assembly_video_  │     │ 2. assembly_video_  │     │ 4. assembly_copilot       │
 │    recorder          │────▶│    labeler           │────▶│    · PSR model training   │
 │  record egocentric   │     │  mark when each      │     │    · live demos           │
 │  videos with an      │     │  assembly step       │     │    · part detector        │
 │  iPhone              │     │  completes           │     │                           │
 └─────────────────────┘     └─────────────────────┘     └──────────▲───────────────┘
                                                                     │
 ┌─────────────────────┐                                             │
 │ 3. box_part_labeler │  draw part bounding boxes on video frames   │
 │                      │─────────────────────────────────────────────┘
 └─────────────────────┘
```

## The four subprojects

### 1. [`assembly_video_recorder/`](assembly_video_recorder) — capture the data

Turns an iPhone into a wireless egocentric camera. A pure-Python HTTPS server
runs on a host computer; the phone opens a web page in Safari and becomes the
camera, streaming a live WebRTC preview to the host and recording full-quality
video straight into the host's `recordings/` folder over Wi-Fi. No app
installation, no cloud, no dependencies beyond the Python standard library.

**Output:** `.mp4` recordings of assembly/disassembly runs.

### 2. [`assembly_video_labeler/`](assembly_video_labeler) — annotate the steps

A lightweight FastAPI web app for **Procedure Step Recognition (PSR)
annotation** in the style of the IndustReal benchmark: you scrub through a
recording and press a number key at the exact frame where each of the 10
assembly steps completes. Supports multiple annotators at once (file-based
video claiming), 480p proxies for smooth scrubbing, and resumable sessions.

**Output:** per-video `PSR_labels.csv` (IndustReal-compatible),
`segments.csv`, and a resumable `labels.json`.

### 3. [`box_part_labeler/`](box_part_labeler) — annotate the parts

A single-file, zero-dependency browser tool for drawing bounding boxes around
the 10 engine parts on still frames. Guided labeling flow (auto-advances
through the part checklist), autosave to IndexedDB, validation, and one-click
export to **COCO JSON + YOLO** format. Its label vocabulary is kept in sync
with the video labeler's parts list so detections and step events join cleanly.

**Output:** COCO/YOLO annotation bundles used to train the part detector.

### 4. [`assembly_copilot/`](assembly_copilot) — the models and the demos

The core of the project, with three strands:

- **PSR model training (`psr_tas/`)** — frozen VideoMAEv2-giant +
  InternVideo2 encoders fused into 2176-d features, with a DiffAct diffusion
  action-segmentation head on top. Trained both on the public IndustReal
  benchmark (Acc 84.2, F1@50 79.9) and on the in-house turbofan dataset
  recorded and labeled with tools 1–2.
- **Live system (`live/`, `live_app/`)** — a causal TCN distilled from the
  same features runs in real time: a browser demo that replays recordings as
  live streams (with prediction timeline, attention saliency, and next-part
  overlays), and a true-live version where a phone camera streams frames to
  the model at 10 fps.
- **Part detection (`detector/` + labeling flow)** — a YOLO11s detector for
  the 10 parts, trained on frames sampled from the recordings and annotated
  with tool 3 / Label Studio, powering the demo's "next part" highlight.

## The dataset (not in this repo)

The recorded corpus — 40 curated egocentric recordings (~9 GB) by 3 operators,
in an IndustReal-compatible layout, plus raw footage — lives on the lab data
drive and is **deliberately excluded from git**. What *is* committed are all
annotations and metadata: PSR label CSVs, per-frame ground truth, bounding-box
labels, split definitions, and a full
[dataset card](assembly_copilot/dataset/prod_dataset/README.md). Anyone with
the videos can drop them into the documented folders and reproduce the
pipeline end to end.

## Getting started

Each subproject README has complete setup and usage instructions. A sensible
reading order for someone new to the codebase:

1. **This README** — the big picture.
2. **`assembly_copilot/README.md`** — the pipeline diagram and how the pieces
   connect (data → features → training → live demo).
3. The tool READMEs (1–3) as needed — each tool also stands alone and is
   reusable outside this project (the recorder is a general egocentric-video
   recorder; the labelers work for any IndustReal-style task).

### Requirements at a glance

| Subproject | Needs |
|---|---|
| assembly_video_recorder | Python 3 (stdlib only), an iPhone, shared Wi-Fi |
| assembly_video_labeler | Python 3.8+, `fastapi` + `uvicorn` |
| box_part_labeler | Any modern desktop browser — nothing else |
| assembly_copilot | CUDA GPU, PyTorch, `ultralytics`, five pinned third-party repos (see its `psr_tas/extern/README.md`), the dataset |
