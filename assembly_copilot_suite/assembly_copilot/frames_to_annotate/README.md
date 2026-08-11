# frames_to_annotate — 140 real video frames for bbox labelling

## Why these exist
A YOLO trained only on the 95 studio photos scored **mAP50 0.995** on held-out
photos but detected only **2 of 10 classes** on real video (mean confidence 0.39).
It memorised the capture conditions. These frames supply what the photos lack:
head-mounted viewpoint, close range, motion blur, hands in shot, and tables that
are partly emptied as parts get consumed.

## What is here
- `images/` — **140 frames, the priority set**, 1920x1080 (the live resolution)
- `optional_extra/` — 60 more, same sampling, if you want to go further later
- `manifest.json` — provenance per frame: recording, split, frame index, kind,
  subject, direction, and for pickup frames the part being reached for

Sampling covers **all 40 recordings**, all 3 operators, both directions:
- `pickup` (80) — 2 s before a step completes: the part is still on the table and
  a hand is reaching for it. This is precisely the moment the live copilot must
  draw its box, and it provides hand occlusion for free.
- `progress15/45/80` (20 each) — full, half-emptied and nearly-bare tables. The
  photo dataset only ever shows a full table.

Blurry frames were deliberately kept: motion blur is the deployment condition.

## Labelling rules (match the existing 95-photo dataset)
1. Box **every visible part**, all 10 classes — not just the one of interest.
   Anything left unboxed trains as background, which actively unteaches the class.
2. Partially occluded (e.g. behind a hand) -> box the visible extent.
3. Under roughly 25% visible -> skip, but be consistent.
4. Once a part is **installed into the assembly**, stop boxing it. The copilot only
   ever needs loose parts on the table.
5. Tight boxes. Use the same class names as `classes.txt` in aiops_parts_detection.

## After labelling
Export YOLO format, then merge with the 95 photos and retrain:
    /media/lm-ciss/LM_4TB/assembly_copilot/detector/train.sh
The photos stay useful — they are clean, well-lit examples of every part.
