#!/usr/bin/env python
"""Extract InternVideo2-B14 (K710) clip features -> fusion/data/features_iv2/<rec>.npy [T,768].

Uses the InternVideo2 single_modality backbone `internvideo2_base_patch14_224`
(embed 768, depth 12, patch14, tubelet 1, num_frames 8, clip_embed 768). The
B14_ft_k710_f8 checkpoint expects 8 frames; we sample 8 frames uniformly across the
SAME 16-frame window used by the giant extractor (starts s, frames s+[0,2,..,14]) so
the two feature streams are clip-aligned for fusion. Feature = fc_norm(clip_projector(x))
(head set to Identity), i.e. the 768-d pooled embedding before the K710 head.

flash_attn is stubbed (we build with use_flash_attn/fused=False per HANDOFF §9), and the
backbone module is loaded directly to avoid models/__init__.py (which needs teacher weights).
Normalization: ImageNet mean/std (matches single_modality datasets), input resized to 224.
"""
import argparse, importlib.util, os, sys, time, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))          # psr_tas
SM = os.path.join(ROOT, "extern", "InternVideo", "InternVideo2", "single_modality")
MODELS = os.path.join(SM, "models")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def _stub_flash_attn():
    """Satisfy unconditional `import flash_attn` in the backbone (unused when flags False)."""
    for name in ["flash_attn", "flash_attn.modules", "flash_attn.modules.mlp",
                 "flash_attn.ops", "flash_attn.ops.rms_norm",
                 "flash_attn.flash_attn_interface", "flash_attn.bert_padding"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["flash_attn.modules.mlp"].FusedMLP = object
    sys.modules["flash_attn.ops.rms_norm"].DropoutAddRMSNorm = object
    sys.modules["flash_attn.flash_attn_interface"].flash_attn_varlen_qkvpacked_func = None
    sys.modules["flash_attn.bert_padding"].unpad_input = None
    sys.modules["flash_attn.bert_padding"].pad_input = None


def load_backbone_builder():
    _stub_flash_attn()
    import timm.models._registry as _r
    sys.modules.setdefault("timm.models.registry", _r)
    # synthetic package so internvideo2.py's relative imports (.pos_embed) resolve,
    # without triggering models/__init__.py (teacher/CLIP deps).
    pkg = types.ModuleType("iv2m"); pkg.__path__ = [MODELS]; sys.modules["iv2m"] = pkg
    spec = importlib.util.spec_from_file_location("iv2m.internvideo2",
                                                  os.path.join(MODELS, "internvideo2.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules["iv2m.internvideo2"] = mod
    spec.loader.exec_module(mod)
    return mod.internvideo2_base_patch14_224


def build_model(ckpt_path):
    builder = load_backbone_builder()
    m = builder(num_frames=8, tubelet_size=1, num_classes=710,
                use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False)
    ck = torch.load(ckpt_path, map_location="cpu")
    for k in ("model", "module", "state_dict"):
        if isinstance(ck, dict) and k in ck: ck = ck[k]; break
    miss, unexp = m.load_state_dict(ck, strict=False)   # head kept but unused
    m.head = torch.nn.Identity()                         # forward now returns 768-d feature
    print(f"  IV2-B14 loaded (missing={len(miss)} unexpected={len(unexp)})", flush=True)
    return m.eval()


def to_clip(frames_u8, device):
    # frames_u8: [8,H,W,3] uint8 -> [1,3,8,224,224] float, ImageNet-normalized
    x = torch.from_numpy(frames_u8).float().div_(255.).permute(3, 0, 1, 2)  # [3,8,H,W]
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                        align_corners=False).unsqueeze(0)     # [1,3,8,224,224]
    x = x.to(device)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh: fh.write(line + "\n"); fh.flush()


def recordings(dataset, splits):
    out = []
    for s in splits:
        sdir = os.path.join(dataset, "recordings", s)
        if os.path.isdir(sdir):
            out += [r for r in sorted(os.listdir(sdir)) if os.path.isdir(os.path.join(sdir, r))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.normpath(os.path.join(ROOT, "..", "dataset")))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "fusion", "weights", "iv2_b14_k710.bin"))
    ap.add_argument("--out", default=os.path.join(ROOT, "fusion", "data", "features_iv2"))
    ap.add_argument("--giant_feat", default=os.path.join(ROOT, "data_v2", "features"),
                    help="reuse giant *_starts.npy so clips align exactly")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--stride", type=int, default=2, help="fallback if no giant starts")
    ap.add_argument("--frame_gap", type=int, default=1,
                    help="MUST match the giant extractor's --frame_gap: the 8 frames are "
                         "drawn from the same 16*gap-frame window, so a mismatch silently "
                         "misaligns the two streams that fuse.py concatenates")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--log", default=os.path.join(ROOT, "logs", "extract_iv2.log"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from decord import VideoReader, cpu
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    fh = open(args.log, "a")
    recs = recordings(args.dataset, args.splits.split(","))
    if args.limit: recs = recs[:args.limit]
    log(f"== IV2-B14 extraction: {len(recs)} recs, batch={args.batch}, device={args.device} ==", fh)
    dev = torch.device(args.device)
    model = build_model(args.ckpt).to(dev)
    global IMAGENET_MEAN, IMAGENET_STD
    IMAGENET_MEAN, IMAGENET_STD = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    log("model ready.", fh)

    t_all = time.time()
    for i, rec in enumerate(recs, 1):
        out_npy = os.path.join(args.out, rec + ".npy")
        if os.path.exists(out_npy):
            log(f"[{i}/{len(recs)}] {rec}: exists, skip", fh); continue
        vr = VideoReader(os.path.join(args.dataset, rec + ".mp4"), ctx=cpu(0), num_threads=4)
        N = len(vr)
        starts_path = os.path.join(args.giant_feat, rec + "_starts.npy")
        span = 16 * args.frame_gap
        starts = (np.load(starts_path).tolist() if os.path.exists(starts_path)
                  else list(range(0, max(1, N - (span - 1)), args.stride)))
        t0 = time.time()
        feats = []
        for b in range(0, len(starts), args.batch):
            batch = []
            for s in starts[b:b + args.batch]:
                idx = np.clip(np.arange(s, s + span, 2 * args.frame_gap), 0, N - 1)  # 8 frames
                batch.append(to_clip(vr.get_batch(idx).asnumpy(), dev))
            x = torch.cat(batch, 0)                       # [B,3,8,224,224]
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                f = model(x)                              # [B,768]
            feats.append(f.float().cpu().numpy())
        arr = np.vstack(feats)
        np.save(out_npy, arr)
        log(f"[{i}/{len(recs)}] {rec}: N={N} T={arr.shape[0]} dim={arr.shape[1]} ({time.time()-t0:.1f}s)", fh)
    log(f"== DONE {len(recs)} recs in {(time.time()-t_all)/60:.1f} min ==", fh)
    fh.close()


if __name__ == "__main__":
    main()
