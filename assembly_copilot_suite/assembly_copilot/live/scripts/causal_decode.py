#!/usr/bin/env python
"""Procedure-aware causal decoding: the v4 'Viterbi decode' box, in streaming form.

Debounce alone trades flicker for latency one-for-one (K ticks of hysteresis = K*0.2 s
of lag) and tops out around F1@50 82 at 3 s of lag. It suppresses spurious segments by
WAITING them out, which is why it costs time.

A transition prior suppresses them by knowing they are illegal. This procedure is
near-deterministic -- from part k you stay in k or advance to k+1 (assembly) / k-1
(disassembly) -- so most of the 432 predicted segments are transitions the procedure
does not permit at all. Rejecting those costs no latency.

Two decode modes, both causal:
  lag=0   pure forward filtering. Emits argmax of the Viterbi score at t using only
          observations <= t. Zero added latency.
  lag>0   fixed-lag smoothing: run the forward pass, then backtrack `lag` steps and
          emit that (now better-informed) state. Costs exactly lag*0.2 s, and unlike
          debounce it uses the extra time to REVISE rather than merely to wait.

self_bias adds a constant to the log-diagonal: a stickiness prior controlling how much
evidence is needed to leave the current state. It is the one knob worth sweeping.
"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
from nets.causal_tcn import CausalTCN                        # noqa: E402

DS = ("/media/lm-ciss/LM_4TB/egocentric_videos/ego_psr_repro/industReal/"
      "psr_tas/extern/DiffAct/datasets/Copilot-Fusion")
SEC = 6 / 30.0


def learn_transitions(gts, classes, smooth=0.1):
    """Per-timestep transition counts from TRAIN ground truth only.

    Laplace `smooth` keeps unseen transitions merely very unlikely rather than
    impossible -- a hard zero would make the decoder unable to ever recover from a
    wrong commitment, which on a 30-recording training set is over-confident.
    """
    K = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    A = np.full((K, K), smooth)
    for gt in gts:
        seq = [idx[c] for c in gt]
        for a, b in zip(seq[:-1], seq[1:]):
            A[a, b] += 1
    A /= A.sum(1, keepdims=True)
    return np.log(A)


def decode(logprob, logA, lag=0, self_bias=0.0):
    """Causal Viterbi. Returns one state per timestep, emitted at that timestep."""
    T, K = logprob.shape
    A = logA.copy()
    np.fill_diagonal(A, np.diag(A) + self_bias)
    delta = logprob[0].copy()
    bp = np.zeros((T, K), dtype=np.int32)
    out = np.zeros(T, dtype=np.int32)
    hist = [delta.argmax()]
    for t in range(1, T):
        m = delta[:, None] + A                    # [prev, cur]
        bp[t] = m.argmax(0)
        delta = m.max(0) + logprob[t]
        delta -= delta.max()                      # stability only; argmax unchanged
        cur = int(delta.argmax())
        hist.append(cur)
        if lag == 0:
            out[t] = cur
        else:
            # backtrack `lag` steps from the current best state
            s = cur
            for u in range(t, max(t - lag, 0), -1):
                s = bp[u][s]
            out[max(t - lag, 0)] = s
    if lag == 0:
        out[0] = hist[0]
    else:
        # flush the tail: nothing left to smooth with, emit the running best
        for t in range(max(T - lag, 0), T):
            out[t] = hist[t]
        out[0] = hist[0]
    return out.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DS)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "runs", "causal_step", "final.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "causal_step", "logits"))
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

    # log-probs for the TEST split (decoded later), transitions from TRAIN only
    for split in ("test", "train"):
        for line in open(os.path.join(args.dataset, "splits",
                                      f"{split}.split1.bundle")).read().split():
            rec = line[:-4]
            if split == "train":
                continue
            f = np.load(os.path.join(args.dataset, "features", rec + ".npy"))
            with torch.no_grad():
                lg = model(torch.from_numpy(f).float().unsqueeze(0).to(dev))[0]
                lp = torch.log_softmax(lg, 0).T.cpu().numpy()          # [T, K]
            np.save(os.path.join(args.out, rec + ".npy"), lp.astype(np.float32))
    print(f"wrote test log-probs -> {args.out}")


if __name__ == "__main__":
    main()
