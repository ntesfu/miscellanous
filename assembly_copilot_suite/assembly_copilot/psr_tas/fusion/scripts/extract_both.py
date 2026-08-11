#!/usr/bin/env python
"""Extract VideoMAEv2-giant (1408) AND InternVideo2-B14 (768) in ONE decode pass.

Why this exists
---------------
01_extract_v2.py / extract_iv2.py call vr.get_batch(scattered indices) once per
clip. At stride 6 / frame_gap 3 consecutive clips overlap by 42 of their 48 frames,
so decord re-seeks and re-decodes the same frames ~8x. Measured on this dataset:
548 ms/clip -> 11.2 h per stream. Unusable.

Two observations collapse that:

  1. Every frame any clip needs sits at an index = 0 (mod frame_gap). Decoding just
     those, sequentially, is one cheap pass (~30 s/video at 1080p) instead of
     thousands of scattered seeks.
  2. Because stride % frame_gap == 0, a clip is 16 CONTIGUOUS entries of that
     subsampled buffer: source frame s+k*gap  <->  buffer index s/gap + k.
     The giant wants all 16; IV2 wants every 2nd (8 frames) of the SAME window,
     which is exactly what extract_iv2.py sampled. So both models read one buffer
     and are clip-aligned by construction -- no *_starts.npy handshake needed.

Pixel semantics are unchanged. The originals do
    uint8 -> /255 -> interpolate(size=(224,224), bilinear) -> (x-mean)/std
and interpolate on [C,T,H,W] resizes each frame independently, so doing the resize
during buffering is mathematically identical to doing it per clip.

Outputs (same names/shapes the rest of the pipeline expects):
    <giant_out>/<rec>.npy         [T,1408]
    <giant_out>/<rec>_starts.npy  [T]  source frame index of each clip start
    <iv2_out>/<rec>.npy           [T, 768]
"""
import argparse, importlib.util, os, sys, time, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))          # psr_tas
sys.path.insert(0, os.path.join(ROOT, "extern", "VideoMAEv2"))
import timm.models._registry as _r
sys.modules['timm.models.registry'] = _r
from models.modeling_finetune import vit_giant_patch14_224       # noqa: E402
from decord import VideoReader, cpu                              # noqa: E402

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def build_giant(ckpt):
    m = vit_giant_patch14_224(img_size=224, num_classes=174, all_frames=16,
                              tubelet_size=2, drop_path_rate=0.3, use_mean_pooling=True)
    ck = torch.load(ckpt, map_location="cpu")
    for k in ("model", "module"):
        if k in ck:
            ck = ck[k]; break
    m.load_state_dict(ck, strict=True)
    return m.eval().cuda()


def build_iv2(ckpt):
    """Reuse extract_iv2.py's loader verbatim rather than reimplementing it: it
    builds a synthetic `iv2m` package so internvideo2.py's relative imports resolve
    without pulling in models/__init__.py, and it passes num_classes=710 before
    swapping the head for Identity. Any drift between the two would silently change
    the 768-d features."""
    spec = importlib.util.spec_from_file_location(
        "extract_iv2", os.path.join(HERE, "extract_iv2.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extract_iv2"] = mod
    spec.loader.exec_module(mod)
    return mod.build_model(ckpt).cuda()


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


def decode_buffer(mp4, gap, threads, chunk=240):
    """Every gap-th frame, decoded sequentially, resized to 224 and kept as fp16.

    Returns [n, 3, 224, 224] where entry j is source frame j*gap, already /255 and
    resized -- i.e. everything the original transform does before normalization.
    """
    vr = VideoReader(mp4, ctx=cpu(0), num_threads=threads)
    N = len(vr)
    need = np.arange(0, N, gap)
    buf = torch.empty((len(need), 3, 224, 224), dtype=torch.float16)
    for i in range(0, len(need), chunk):
        frames = vr.get_batch(need[i:i + chunk]).asnumpy()        # [n,H,W,3] uint8
        x = torch.from_numpy(frames).float().div_(255.0).permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                            align_corners=False)
        buf[i:i + chunk] = x.half()
    return buf, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--giant_ckpt", default=os.path.join(ROOT, "weights", "vit_g_ssv2_ft.pth"))
    ap.add_argument("--iv2_ckpt", default=os.path.join(ROOT, "fusion", "weights", "iv2_b14_k710.bin"))
    ap.add_argument("--giant_out", default=os.path.join(ROOT, "data_v2", "features_copilot"))
    ap.add_argument("--iv2_out", default=os.path.join(ROOT, "fusion", "data", "features_iv2_copilot"))
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--frame_gap", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--log", default=os.path.join(ROOT, "logs", "copilot_extract_both.log"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.stride % args.frame_gap:
        sys.exit(f"stride ({args.stride}) must be a multiple of frame_gap "
                 f"({args.frame_gap}) -- otherwise clip starts do not land on "
                 f"buffer entries and the contiguous-window trick is invalid.")

    os.makedirs(args.giant_out, exist_ok=True)
    os.makedirs(args.iv2_out, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    fh = open(args.log, "a")

    recs = recordings(args.dataset, args.splits.split(","))
    if args.limit:
        recs = recs[:args.limit]
    gap, stride, span = args.frame_gap, args.stride, 16 * args.frame_gap
    jstep = stride // gap                       # buffer entries between clip starts

    log(f"== dual extraction: {len(recs)} recs, stride={stride}, frame_gap={gap} "
        f"(clip spans {span} src frames), batch={args.batch} ==", fh)
    giant = build_giant(args.giant_ckpt)
    iv2 = build_iv2(args.iv2_ckpt)
    log("both models loaded", fh)
    mean, std = _MEAN.cuda(), _STD.cuda()

    t_all = time.time()
    for i, rec in enumerate(recs, 1):
        gp = os.path.join(args.giant_out, rec + ".npy")
        ip = os.path.join(args.iv2_out, rec + ".npy")
        if os.path.exists(gp) and os.path.exists(ip):
            log(f"[{i}/{len(recs)}] {rec}: exists, skip", fh); continue
        t0 = time.time()
        buf, N = decode_buffer(os.path.join(args.dataset, rec + ".mp4"), gap, args.threads)
        t_dec = time.time() - t0

        # clip j covers buffer[j : j+16] == source frames j*gap .. j*gap+span-1
        js = list(range(0, max(1, len(buf) - 16 + 1), jstep))
        starts = [j * gap for j in js]
        gf, vf = [], []
        for b in range(0, len(js), args.batch):
            chunk = js[b:b + args.batch]
            clips = torch.stack([buf[j:j + 16] for j in chunk])        # [B,16,3,224,224]
            x = clips.cuda(non_blocking=True).permute(0, 2, 1, 3, 4).float()  # [B,3,16,H,W]
            x = (x - mean) / std
            with torch.no_grad(), torch.cuda.amp.autocast():
                gf.append(giant.forward_features(x).float().cpu().numpy())
                vf.append(iv2(x[:, :, ::2]).float().cpu().numpy())      # same window, 8 frames
        g = np.vstack(gf); v = np.vstack(vf)
        np.save(gp, g)
        np.save(os.path.join(args.giant_out, rec + "_starts.npy"), np.array(starts))
        np.save(ip, v)
        del buf
        log(f"[{i}/{len(recs)}] {rec}: N={N} T={g.shape[0]} giant={g.shape[1]} "
            f"iv2={v.shape[1]} (decode {t_dec:.0f}s, total {time.time()-t0:.0f}s)", fh)
    log(f"== DONE {len(recs)} recs in {(time.time()-t_all)/60:.1f} min ==", fh)


if __name__ == "__main__":
    main()
