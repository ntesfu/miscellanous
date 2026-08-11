#!/usr/bin/env python
"""Pull a stratified set of REAL video frames for bbox annotation.

Why these frames and not random ones: the detector trained on 95 studio photos
scored mAP50 0.995 on held-out photos but found only 2 of 10 classes on video --
it memorised the capture conditions. The gap is viewpoint, scale, motion blur,
hand occlusion, and partially-emptied tables, so the sample deliberately covers
exactly those.

Sampling plan (~140 frames):
  * every one of the 40 recordings contributes, so all 3 operators and both
    directions are represented
  * PICKUP frames -- 2 s before a step's completion, when the part is still on the
    table and a hand is reaching for it. This is the exact moment the live copilot
    must draw its box, and it supplies hand occlusion for free.
  * PROGRESS frames -- at 15%, 45% and 80% through each recording, so the model
    sees full, half-emptied and nearly-bare tables (it has never seen the latter).
  * blurry frames are kept, not filtered: motion blur is the deployment condition.

Frames are written at native 1920x1080 -- the resolution the live system runs at.
"""
import csv, glob, json, os, random
from decord import VideoReader, cpu
from PIL import Image

LIB = '/media/lm-ciss/LM_4TB/assembly_copilot/dataset/prod_dataset'
OUT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(OUT, 'images')
PICKUP_LEAD = 1.0          # seconds AFTER a segment starts: the operator turns to
                           # the table and reaches for the part. The first version
                           # anchored 2 s before COMPLETION -- but by then the part
                           # has long been picked up and is being fastened; a contact
                           # sheet showed 'pickup' frames full of in-hand close-ups
                           # with nothing left on the table to box.
random.seed(1337)


def main():
    os.makedirs(IMG, exist_ok=True)
    manifest, n = [], 0
    recs = sorted(glob.glob(f'{LIB}/recordings/*/*/'))
    for d in recs:
        rec = os.path.basename(d.rstrip('/'))
        split = d.split('/recordings/')[1].split('/')[0]
        mp4 = os.path.join(LIB, rec + '.mp4')
        if not os.path.exists(mp4):
            continue
        segs = list(csv.DictReader(open(os.path.join(d, 'segments.csv'))))
        vr = VideoReader(mp4, ctx=cpu(0), num_threads=4)
        N = len(vr)

        want = []
        # two pickup moments per recording, chosen from different steps each time
        for s in random.sample(segs, min(2, len(segs))):
            f = int(float(s['start_sec']) * 30 + PICKUP_LEAD * 30)
            if 0 <= f < N:
                want.append((f, 'pickup', s['part']))
        # table-state progression
        for frac in (0.15, 0.45, 0.80):
            f = int(N * frac)
            if 0 <= f < N:
                want.append((f, f'progress{int(frac*100)}', ''))

        for f, kind, part in want:
            img = Image.fromarray(vr[f].asnumpy())
            name = f'{rec}__f{f:06d}__{kind}.jpg'
            img.save(os.path.join(IMG, name), quality=92)
            manifest.append(dict(file=name, rec=rec, split=split, frame=f,
                                 kind=kind, near_part=part,
                                 subject=rec.split('-')[0],
                                 direction='disassembly' if 'disAssembled' in rec else 'assembly'))
            n += 1
        del vr
    json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
    print(f'wrote {n} frames -> {IMG}')
    import collections
    print('  by kind   :', dict(collections.Counter(m["kind"] for m in manifest)))
    print('  by subject:', dict(collections.Counter(m["subject"] for m in manifest)))
    print('  by dirn   :', dict(collections.Counter(m["direction"] for m in manifest)))
    print('  recordings covered:', len({m["rec"] for m in manifest}), '/ 40')


if __name__ == '__main__':
    main()
