#!/usr/bin/env python
"""Isolate what in the streaming loop grows without bound.

The demo server reached 23.8 GB RSS and was OOM-killed four times, taking the user's
VS Code session with it. Shrinking the decode chunk did not fix it, so this walks the
same loop stage by stage and prints host RSS, GPU allocation, and the size of each
accumulator every 25 clips. Whatever climbs is the leak.
"""
import os, sys, time
from collections import deque
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
PSR = "/media/lm-ciss/LM_4TB/egocentric_videos/ego_psr_repro/industReal/psr_tas"
DS = os.path.join(PSR, "extern", "DiffAct", "datasets", "Copilot-Fusion")
from nets.causal_tcn import CausalTCN                                    # noqa: E402

GAP, STRIDE, CHUNK = 3, 6, 24
VIDEO = "/media/lm-ciss/LM_4TB/assembly_copilot/dataset/prod_dataset/DA-disAssembled-6.mp4"


def rss_gb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS"):
                return int(l.split()[1]) / 1048576
    return 0.0


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eb", os.path.join(PSR, "fusion", "scripts", "extract_both.py"))
    eb = importlib.util.module_from_spec(spec); sys.modules["eb"] = eb
    spec.loader.exec_module(eb)
    dev = "cuda"
    classes = [l.split(maxsplit=1)[1].strip() for l in open(os.path.join(DS, "mapping.txt"))]
    giant = eb.build_giant(os.path.join(PSR, "weights", "vit_g_ssv2_ft.pth"))
    iv2 = eb.build_iv2(os.path.join(PSR, "fusion", "weights", "iv2_b14_k710.bin"))
    tcn = CausalTCN(2176, len(classes), 9, 128, 3, 0.25).to(dev).eval()
    mean, std = eb._MEAN.to(dev), eb._STD.to(dev)
    print(f"after model load: RSS {rss_gb():.2f} GB  GPU {torch.cuda.memory_allocated()/2**30:.2f} GB",
          flush=True)

    from decord import VideoReader, cpu
    vr = VideoReader(VIDEO, ctx=cpu(0), num_threads=8)
    N = len(vr); need = list(range(0, N, GAP))
    print(f"video {N} frames -> {len(need)} subsampled", flush=True)

    buf, feats = deque(maxlen=16), []
    j, done, t0 = -1, 0, time.time()
    for bi in range(0, len(need), CHUNK):
        chunk = vr.get_batch(need[bi:bi + CHUNK]).asnumpy()
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).float().div_(255.0)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                            align_corners=False).half()
        del chunk
        for f in x:
            buf.append(f); j += 1
            if j < 15 or (j - 15) % (STRIDE // GAP):
                continue
            clip = torch.stack(list(buf)).unsqueeze(0).to(dev).permute(0, 2, 1, 3, 4).float()
            clip = (clip - mean) / std
            with torch.no_grad(), torch.amp.autocast("cuda"):
                g = giant.forward_features(clip)
                v = iv2(clip[:, :, ::2])
            gn = torch.nn.functional.normalize(g.float(), dim=1)
            vn = torch.nn.functional.normalize(v.float(), dim=1)
            feats.append(torch.cat([gn, vn], 1)[0])
            seq = torch.stack(feats, 1).unsqueeze(0)
            with torch.no_grad():
                lg = tcn(seq)[0, :, -1]
            lp = torch.log_softmax(lg, 0).cpu().numpy()
            done += 1
            if done % 25 == 0:
                print(f"clip {done:5d}  RSS {rss_gb():6.2f} GB   "
                      f"GPU alloc {torch.cuda.memory_allocated()/2**30:5.2f} / "
                      f"reserved {torch.cuda.memory_reserved()/2**30:5.2f} GB   "
                      f"feats {len(feats)}  seqT {seq.shape[2]}  "
                      f"{done/(time.time()-t0):.2f} clip/s", flush=True)
            if done >= 300:
                print("stopping at 300 clips"); return


if __name__ == "__main__":
    main()
