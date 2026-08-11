#!/usr/bin/env python
"""Build per-frame STEP + TYPE ground-truth from IndustReal PSR annotations.

Reformulates IndustReal PSR as Temporal Action Segmentation (the psr_tas recipe):
each frame gets a STEP label (which part is being worked on; 10 parts + background)
and a TYPE label (none/correct/incorrect/remove). Emits MS-TCN/ASFormer-style files:

    data/mapping.txt              step  class_id -> name   (11 classes)
    data/mapping_type.txt         type  class_id -> name   (4 classes)
    data/groundTruth/<rec>.txt    one STEP name per frame
    data/groundTruth_type/<rec>.txt  one TYPE name per frame
    data/splits/{train,test}.split1.bundle   list of <rec>.txt

Taxonomy is taken verbatim from the official IndustReal `procedure_info.json`:
33 actions = 11 state_idx x {0:correct-install, 1:incorrect-install, 2:remove}.
state_idx 0 ("base") is never a procedure step -> 10 parts (state_idx 1..10) + background.

BESPOKE CHOICE (the one piece psr_tas's 00_build_labels invented and we cannot
recover exactly): densifying the SPARSE completion events into per-frame labels.
We use "run-up assignment": the frames leading up to a step's completion frame are
labelled with that step (segment = (prev_completion, this_completion]); frames before
the first completion are background. This is a documented, swappable policy
(--densify) — expect it to move the absolute numbers vs the original.
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
PROC_INFO = os.path.join(ROOT, "extern", "IndustReal", "PSR", "procedure_info.json")

TYPE_NAMES = ["none", "correct", "incorrect", "remove"]  # class 0..3
TYPE_BY_MOD = {0: "correct", 1: "incorrect", 2: "remove"}  # action_id % 3


def load_proc_info(path=PROC_INFO):
    with open(path) as f:
        return json.load(f)


def part_names(proc_info):
    """state_idx (1..10) -> class name, from each 'Install <part>' action.

    Whitespace is collapsed to underscores because class names are round-tripped
    through np.loadtxt(dtype=str) in two places -- 03_prepare_diffact.py reading
    groundTruth and DiffAct's main.py reading mapping.txt -- and loadtxt splits on
    whitespace, so a name with a space becomes several columns and blows up with
    "number of columns changed". IndustReal's descriptions ('front chassis') are
    spaced too, and its committed mapping.txt is underscored, so this reproduces
    the existing artifact rather than changing it.
    """
    names = {}
    for a in proc_info:
        if a["id"] % 3 == 0:  # the correct-install action names the part
            names[a["state_idx"]] = "_".join(a["description"]
                                             .replace("Install ", "").split())
    return names


def load_completions(rec_dir, with_errors):
    """Sparse (frame, action_id) completion events, ordered by frame."""
    fname = "PSR_labels_with_errors.csv" if with_errors else "PSR_labels.csv"
    path = os.path.join(rec_dir, fname)
    events = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            events.append((int(row[0][:-4]), int(row[1])))  # frame index, action_id
    events.sort()
    return events


def n_frames(rec_name, dataset_dir, rec_dir):
    """Frame count for the recording = decoded video frames (authoritative for extraction)."""
    mp4 = os.path.join(dataset_dir, rec_name + ".mp4")
    if os.path.exists(mp4):
        # fast: container nb_frames tag (no full decode). Reconciled to feature length later.
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", mp4],
            capture_output=True, text=True).stdout.strip()
        if out.isdigit() and int(out) > 0:
            return int(out)
        # some containers carry no nb_frames tag (phone-recorded fragmented mp4).
        # duration * avg_frame_rate keeps the trailing background that the keyframe
        # fallback below would silently truncate away. Must be avg_ and not
        # r_frame_rate: on VFR captures r_ is the nominal peak (one recording here
        # reports 47 fps against a true average of 30) and would inflate N by half.
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration,avg_frame_rate", "-of", "csv=p=0", mp4],
            capture_output=True, text=True).stdout.strip()
        try:
            rate, dur = out.split(",")
            num, den = rate.split("/")
            n = int(float(dur) * float(num) / float(den))
            if n > 0:
                return n
        except (ValueError, ZeroDivisionError):
            pass
    # fallback: last labelled keyframe + 1
    raw = os.path.join(rec_dir, "PSR_labels_raw.csv")
    last = 0
    with open(raw) as f:
        for row in csv.reader(f):
            if row:
                last = int(row[0][:-4])
    return last + 1


def densify_runup(events, N, part_of):
    """(prev_completion, this_completion] labelled with the step being worked toward.
    Returns (step_seq, type_seq) as lists of length N."""
    step = ["background"] * N
    typ = ["none"] * N
    prev = 0
    for frame, aid in events:
        part = part_of.get(aid // 3)     # state_idx -> part name
        if part is None:                 # base (state_idx 0) etc. -> skip, stays background
            prev = frame
            continue
        end = min(frame, N - 1)
        for t in range(prev, end + 1):
            step[t] = part
            typ[t] = TYPE_BY_MOD[aid % 3]
        prev = frame + 1
    return step, typ


DENSIFY = {"runup": densify_runup}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"),
                    help="IndustReal root (contains recordings/{train,val,test} + <rec>.mp4)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--proc_info", default=PROC_INFO,
                    help="procedure taxonomy; step class NAMES come from here, so a "
                         "dataset with different parts must pass its own or the classes "
                         "come out named after IndustReal's chassis")
    ap.add_argument("--with_errors", action="store_true", default=True,
                    help="use PSR_labels_with_errors.csv (needed for the incorrect TYPE class)")
    ap.add_argument("--densify", choices=list(DENSIFY), default="runup")
    args = ap.parse_args()

    proc_info = load_proc_info(args.proc_info)
    parts = part_names(proc_info)  # {1:'front chassis', ...}
    # step classes: background + the 10 real parts in state_idx order
    step_classes = ["background"] + [parts[i] for i in sorted(parts) if i != 0]

    gt_dir = os.path.join(args.out, "groundTruth")
    gtt_dir = os.path.join(args.out, "groundTruth_type")
    split_dir = os.path.join(args.out, "splits")
    for d in (gt_dir, gtt_dir, split_dir):
        os.makedirs(d, exist_ok=True)

    # mapping.txt / mapping_type.txt are written AFTER the build loop, because a
    # class is only emitted if the data actually contains it -- see the support
    # audit below. DiffAct takes num_classes from len(mapping), so a class listed
    # here but absent from the data becomes a dead softmax unit that also shrinks
    # every real class weight (the weight formula divides by class_num).

    rec_root = os.path.join(args.dataset, "recordings")
    split_recs = {"train": [], "val": [], "test": []}
    stats = {"frames": 0, "recs": 0, "incorrect_frames": 0}
    support = {"step": {s: collections.Counter() for s in ("train", "test")},
               "type": {s: collections.Counter() for s in ("train", "test")}}
    for split in ("train", "val", "test"):
        sdir = os.path.join(rec_root, split)
        if not os.path.isdir(sdir):
            print(f"  [skip] no {split} split at {sdir}")
            continue
        for rec in sorted(os.listdir(sdir)):
            rec_dir = os.path.join(sdir, rec)
            if not os.path.isdir(rec_dir):
                continue
            events = load_completions(rec_dir, args.with_errors)
            N = n_frames(rec, args.dataset, rec_dir)
            step_seq, type_seq = DENSIFY[args.densify](events, N, parts)
            with open(os.path.join(gt_dir, rec + ".txt"), "w") as f:
                f.write("\n".join(step_seq) + "\n")
            with open(os.path.join(gtt_dir, rec + ".txt"), "w") as f:
                f.write("\n".join(type_seq) + "\n")
            split_recs[split].append(rec + ".txt")
            stats["frames"] += N
            stats["recs"] += 1
            stats["incorrect_frames"] += type_seq.count("incorrect")
            bucket = "train" if split == "train" else "test"
            support["step"][bucket].update(step_seq)
            support["type"][bucket].update(type_seq)

    def audit(kind, declared, fname):
        """Print per-class support and emit a mapping holding only classes the data has.

        Two distinct failure modes, treated differently:
          train=0, test=0  -> harmless-looking but not free: a dead output unit, and
                              because class weights divide by class_num every REAL
                              class gets weighted down. Dropped.
          train=0, test>0  -> fatal. The model can never learn to emit that class, so
                              those test frames are guaranteed wrong, and DiffAct would
                              index a label that is not in mapping.txt. Refuse to build.
        """
        tr, te = support[kind]["train"], support[kind]["test"]
        total = sum(tr.values()) + sum(te.values())
        print(f"\n  {kind} class support (train / test frames):")
        keep, dropped, fatal = [], [], []
        for c in declared:
            pct = 100 * (tr[c] + te[c]) / total if total else 0
            mark = ""
            if tr[c] == 0 and te[c] == 0:
                dropped.append(c); mark = "   <- NO DATA, dropped from mapping"
            elif tr[c] == 0:
                fatal.append(c); mark = "   <- NO TRAINING DATA but present in test"
            else:
                keep.append(c)
            print(f"    {c:24s} {tr[c]:8d} / {te[c]:<8d} {pct:5.2f}%{mark}")
        if fatal:
            sys.exit(f"\n  ABORT: {fatal} appear in test but never in train. "
                     f"Fix the split or the labels before training.")
        if dropped:
            print(f"    -> emitting {len(keep)} of {len(declared)} declared "
                  f"{kind} classes; dropped {dropped}")
        with open(os.path.join(args.out, fname), "w") as f:
            for i, c in enumerate(keep):
                f.write(f"{i} {c}\n")
        return keep

    audit("step", step_classes, "mapping.txt")
    audit("type", TYPE_NAMES, "mapping_type.txt")

    # ASFormer convention: train.bundle = train split, test.bundle = test split.
    # No test recordings present -> fall back to val as the held-out set so the
    # pipeline is runnable end-to-end (flagged loudly).
    with open(os.path.join(split_dir, "train.split1.bundle"), "w") as f:
        f.write("\n".join(split_recs["train"]) + "\n")
    heldout = "test" if split_recs["test"] else "val"
    with open(os.path.join(split_dir, "test.split1.bundle"), "w") as f:
        f.write("\n".join(split_recs[heldout]) + "\n")

    print(f"\n== built labels ==")
    print(f"  step classes ({len(step_classes)}): {step_classes}")
    print(f"  type classes ({len(TYPE_NAMES)}): {TYPE_NAMES}")
    print(f"  recs={stats['recs']}  total frames={stats['frames']}")
    print(f"  frames labelled 'incorrect' (fault signal): {stats['incorrect_frames']}")
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(split_recs[s])} recs")
    if not split_recs["test"]:
        print("  WARNING: no TEST recordings on disk -> test.bundle uses VAL. "
              "Slide numbers are on the real test set; these will not be comparable "
              "until the 32 test recordings are present.")


if __name__ == "__main__":
    main()
