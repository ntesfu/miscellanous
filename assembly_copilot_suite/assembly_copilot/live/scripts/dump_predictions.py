#!/usr/bin/env python
"""Write causal-head predictions in DiffAct's prediction format.

Purpose is strictly comparability: the offline head was scored by
psr_tas/scripts/copilot_metrics.py, and the only defensible way to state a
causal-vs-offline gap is to run that SAME scorer over both. The latency figure
printed during training is a different quantity (signed delay until the model
stops asserting a step) and must not be compared against the offline
completion-error figure, which is an unsigned distance to the nearest matching
predicted boundary and can be satisfied by a boundary predicted EARLY.
"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
from nets.causal_tcn import CausalTCN                        # noqa: E402

DS = ("/media/lm-ciss/LM_4TB/egocentric_videos/ego_psr_repro/industReal/"
      "psr_tas/extern/DiffAct/datasets/Copilot-Fusion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DS)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "runs", "causal_step", "final.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "causal_step", "prediction"))
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--fmaps", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    classes = [l.split(maxsplit=1)[1].strip()
               for l in open(os.path.join(args.dataset, "mapping.txt"))]
    model = CausalTCN(2176, len(classes), args.layers, args.fmaps, 3, 0.5).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    n = 0
    for line in open(os.path.join(args.dataset, "splits", "test.split1.bundle")).read().split():
        rec = line[:-4]
        f = np.load(os.path.join(args.dataset, "features", rec + ".npy"))
        with torch.no_grad():
            pred = model(torch.from_numpy(f).float().unsqueeze(0).to(dev))[0].argmax(0).cpu().numpy()
        with open(os.path.join(args.out, rec + ".txt"), "w") as fh:
            fh.write("### Frame level recognition: ###\n")
            fh.write(" ".join(classes[i] for i in pred) + "\n")
        n += 1
    print(f"wrote {n} predictions -> {args.out}")


if __name__ == "__main__":
    main()
