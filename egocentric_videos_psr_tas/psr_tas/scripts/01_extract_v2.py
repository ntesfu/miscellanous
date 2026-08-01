#!/usr/bin/env python
"""Extract VideoMAEv2-giant (SSv2) clip features -> data_v2/features/<rec>.npy [T,1408].

Faithful to VideoMAEv2/extract_tad_feature.py: vit_giant_patch14_224, 16-frame clips,
tubelet 2, use_mean_pooling=True, transform = ToFloatTensorInZeroOne + Resize(224)
(input in [0,1], no ImageNet mean/std). Clip stride from HANDOFF (2). One feature per
clip start in range(0, N-15, stride); *_starts.npy records clip start frames so labels
can be aligned to feature length.
"""
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
VMAE = os.path.join(ROOT, "extern", "VideoMAEv2")
sys.path.insert(0, VMAE)
import timm.models._registry as _r          # timm 1.x: shim old registry path (HANDOFF §9)
sys.modules['timm.models.registry'] = _r
from models.modeling_finetune import vit_giant_patch14_224  # noqa: E402
from decord import VideoReader, cpu          # noqa: E402


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh: fh.write(line + "\n"); fh.flush()


# SSv2+fusion path: resize to 224 (NO crop) + ImageNet mean/std (per author, Q3).
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def to_clip_tensor(frames_u8):
    # frames_u8: [16,H,W,3] uint8 -> [3,16,224,224] float, ImageNet-normalized
    x = torch.from_numpy(frames_u8).float().div_(255.0).permute(3, 0, 1, 2)  # [3,16,H,W]
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                        align_corners=False)                 # [3,16,224,224]
    x = (x - _MEAN) / _STD
    return x


def build_model(ckpt_path):
    m = vit_giant_patch14_224(img_size=224, num_classes=174, all_frames=16,
                              tubelet_size=2, drop_path_rate=0.3, use_mean_pooling=True)
    ck = torch.load(ckpt_path, map_location="cpu")
    for k in ("model", "module"):
        if k in ck: ck = ck[k]; break
    m.load_state_dict(ck, strict=True)
    return m.eval().cuda()


def recordings(dataset, splits):
    out = []
    for s in splits:
        sdir = os.path.join(dataset, "recordings", s)
        if not os.path.isdir(sdir):
            continue
        for r in sorted(os.listdir(sdir)):
            if os.path.isdir(os.path.join(sdir, r)):
                out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.normpath(os.path.join(ROOT, "..", "dataset")))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "weights", "vit_g_ssv2_ft.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_v2", "features"))
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16, help="clips per forward pass")
    ap.add_argument("--log", default=os.path.join(ROOT, "logs", "extract_giant.log"))
    ap.add_argument("--limit", type=int, default=0, help="process only first N recs (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    fh = open(args.log, "a")
    recs = recordings(args.dataset, args.splits.split(","))
    if args.limit:
        recs = recs[:args.limit]
    log(f"== giant SSv2 extraction: {len(recs)} recs, stride={args.stride}, batch={args.batch} ==", fh)
    model = build_model(args.ckpt)
    log("model built + loaded (strict).", fh)

    t_all = time.time()
    for i, rec in enumerate(recs, 1):
        out_npy = os.path.join(args.out, rec + ".npy")
        if os.path.exists(out_npy):
            log(f"[{i}/{len(recs)}] {rec}: exists, skip", fh); continue
        mp4 = os.path.join(args.dataset, rec + ".mp4")
        vr = VideoReader(mp4, ctx=cpu(0), num_threads=4)
        N = len(vr)
        starts = list(range(0, max(1, N - 15), args.stride))
        t0 = time.time()
        feats = []
        for b in range(0, len(starts), args.batch):
            chunk = starts[b:b + args.batch]
            batch = []
            for s in chunk:
                idx = np.arange(s, s + 16)
                frames = vr.get_batch(idx).asnumpy()      # [16,H,W,3] uint8
                batch.append(to_clip_tensor(frames))
            x = torch.stack(batch).cuda()                 # [B,3,16,224,224]
            with torch.no_grad(), torch.cuda.amp.autocast():
                f = model.forward_features(x)             # [B,1408]
            feats.append(f.float().cpu().numpy())
        arr = np.vstack(feats)                            # [T,1408]
        np.save(out_npy, arr)
        np.save(os.path.join(args.out, rec + "_starts.npy"), np.array(starts))
        log(f"[{i}/{len(recs)}] {rec}: N={N} T={arr.shape[0]} dim={arr.shape[1]} "
            f"({time.time()-t0:.1f}s)", fh)
    log(f"== DONE {len(recs)} recs in {(time.time()-t_all)/60:.1f} min ==", fh)
    fh.close()


if __name__ == "__main__":
    main()
