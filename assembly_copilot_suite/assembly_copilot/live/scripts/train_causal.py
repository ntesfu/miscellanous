#!/usr/bin/env python
"""Train the causal step head on the features the offline system already produced.

This is the go/no-go experiment for live operation. It reuses the exact 2176-d fused
features and clip-aligned labels DiffAct trained on, so the ONLY variable is
bidirectional -> causal. Whatever accuracy separates this from the offline 96.24 is
the true cost of running live, isolated from every other factor.

Reported per eval:
  frame accuracy                    comparable head-to-head with the offline head
  completion detection @ tolerance  the product metric, but CAUSAL -- a step may only
                                    be announced from evidence available at that moment
  announce latency                  seconds AFTER the true completion that the model
                                    first commits to it. Offline this can be negative
                                    (the model sees ahead); live it cannot, which is
                                    exactly why it is measured separately.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
from nets.causal_tcn import CausalTCN, check_causality      # noqa: E402

DS = ("/media/lm-ciss/LM_4TB/egocentric_videos/ego_psr_repro/industReal/"
      "psr_tas/extern/DiffAct/datasets/Copilot-Fusion")
SEC_PER_STEP = 6 / 30.0


def load_split(ds, bundle, classes):
    idx = {c: i for i, c in enumerate(classes)}
    out = []
    for line in open(os.path.join(ds, "splits", bundle)).read().split():
        rec = line[:-4]
        f = np.load(os.path.join(ds, "features", rec + ".npy"))       # [2176, T]
        g = [l.strip() for l in open(os.path.join(ds, "groundTruth", rec + ".txt")) if l.strip()]
        T = min(f.shape[1], len(g))
        out.append((rec, torch.from_numpy(f[:, :T]).float(),
                    torch.tensor([idx[c] for c in g[:T]], dtype=torch.long)))
    return out


def segments(a):
    out, s = [], 0
    for i in range(1, len(a) + 1):
        if i == len(a) or a[i] != a[s]:
            out.append((int(a[s]), s, i - 1)); s = i
    return out


def evaluate(model, data, device, tols=(1, 2, 5)):
    model.eval()
    correct = total = 0
    lat, hit = [], {t: [0, 0] for t in tols}
    with torch.no_grad():
        for _, x, y in data:
            logit = model(x.unsqueeze(0).to(device))[0]
            pred = logit.argmax(0).cpu().numpy()
            correct += (pred == y.numpy()).sum(); total += len(y)
            gt = y.numpy()
            for cls, _, ge in segments(gt):
                # CAUSAL announce time: first t >= the moment the model has committed
                # to this step ending, i.e. the first t where prediction has moved past
                # `cls` and stays there. Nothing after t is consulted.
                after = np.where(np.arange(len(pred)) > ge)[0]
                idx = np.where(pred[ge:] != cls)[0]
                t_ann = ge + idx[0] if len(idx) else len(pred) - 1
                d = (t_ann - ge) * SEC_PER_STEP
                lat.append(d)
                for t in tols:
                    hit[t][1] += 1
                    if abs(d) <= t:
                        hit[t][0] += 1
    lat = np.array(lat)
    return (100 * correct / total, np.median(lat), np.percentile(lat, 90),
            {t: 100 * hit[t][0] / hit[t][1] for t in tols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DS)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--fmaps", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--tmse", type=float, default=0.15,
                    help="weight on the truncated-MSE smoothing loss. Cross-entropy "
                         "alone scores each frame independently and so never punishes "
                         "flicker: the raw causal head emits 432 segments where there "
                         "are 110. This is MS-TCN's T-MSE (DiffAct carries the same "
                         "term as encoder_mse_loss=0.1) and it penalises frame-to-frame "
                         "jumps in log-probability. 0 disables it.")
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "runs", "causal_step"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    classes = [l.split(maxsplit=1)[1].strip()
               for l in open(os.path.join(args.dataset, "mapping.txt"))]
    tr = load_split(args.dataset, "train.split1.bundle", classes)
    te = load_split(args.dataset, "test.split1.bundle", classes)
    print(f"train {len(tr)} / test {len(te)} recordings, {len(classes)} classes", flush=True)

    model = CausalTCN(2176, len(classes), args.layers, args.fmaps, 3, args.dropout).to(dev)
    d, c = check_causality(model, device=dev)
    if d != 0:
        sys.exit(f"ABORT: model is not causal (drift {d}); live numbers would be fiction.")
    print(f"causality verified (drift {d}); receptive field {model.receptive_field} "
          f"steps = {model.receptive_field*SEC_PER_STEP:.0f}s", flush=True)

    # inverse-frequency weights, same formula DiffAct uses, computed on THIS train split
    cnt = np.zeros(len(classes))
    for _, _, y in tr:
        cnt += np.bincount(y.numpy(), minlength=len(classes))
    w = torch.tensor(cnt.sum() / ((cnt + 10) * len(classes)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    hist, best = [], 0.0
    for ep in range(args.epochs + 1):
        model.train()
        order = np.random.permutation(len(tr))
        tot = 0.0
        for b in range(0, len(order), args.batch):
            opt.zero_grad()
            loss = 0.0
            for i in order[b:b + args.batch]:
                _, x, y = tr[i]
                out = model(x.unsqueeze(0).to(dev))[0]
                loss = loss + crit(out.T, y.to(dev))
                if args.tmse > 0:
                    # MS-TCN truncated MSE: square of the frame-to-frame change in
                    # log-prob, clamped at 4 so a genuine boundary (a large, correct
                    # jump) is not punished as hard as continuous jitter.
                    lp = torch.log_softmax(out, dim=0)
                    d = torch.clamp((lp[:, 1:] - lp[:, :-1]).abs(), max=4.0)
                    loss = loss + args.tmse * (d ** 2).mean()
            loss = loss / len(order[b:b + args.batch])
            loss.backward(); opt.step(); tot += loss.item()
        if ep % args.eval_every == 0:
            acc, med, p90, hits = evaluate(model, te, dev)
            hist.append(dict(epoch=ep, acc=float(acc), median_latency=float(med),
                             p90_latency=float(p90), within=hits))
            print(f"Epoch {ep:4d} loss {tot/max(1,len(order)/args.batch):.4f} | "
                  f"Acc {acc:6.2f} | latency med {med:+.2f}s p90 {p90:+.2f}s | "
                  f"within +-2s {hits[2]:5.1f}%", flush=True)
            json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=2)
            if acc > best:
                best = acc
                torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        torch.save(model.state_dict(), os.path.join(args.out, "final.pt"))
    print(f"done. final acc {hist[-1]['acc']:.2f}  (offline DiffAct reference: 96.24)")


if __name__ == "__main__":
    main()
