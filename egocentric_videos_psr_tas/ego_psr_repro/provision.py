#!/usr/bin/env python
"""Check (and optionally fetch) the datasets + backbone weights the pipeline needs.

--check (default) prints a presence table with the exact fetch or side-load command.
--fetch downloads the reachable assets (HF); proxy-blocked / license-gated ones are
reported with instructions and never faked.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EGO = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.join(EGO, "industReal", "psr_tas")
MECC = os.path.join(EGO, "MECCANO")

# fetch: shell command (run from cwd) that downloads it, or None if not auto-fetchable.
ASSETS = [
    dict(group="industreal", name="IndustReal raw dataset", size="51 GB",
         path=os.path.join(EGO, "industReal", "dataset", "test"),
         fetch=None, cwd=EGO,
         note="4tu.nl (DOI 10.4121/c.6104020) is NOT on the proxy allowlist. Side-load the "
              "51 GB dataset to industReal/dataset/{train,val,test}."),
    dict(group="industreal", name="VideoMAEv2-Huge (K710, 1280-d)", size="2.4 GB",
         path=os.path.join(ROOT, "weights", "VideoMAEv2-Huge"),
         fetch="huggingface-cli download OpenGVLab/VideoMAEv2-Huge --local-dir weights/VideoMAEv2-Huge",
         cwd=ROOT, note="HF (allowed). Used by v1_huge."),
    dict(group="industreal", name="VideoMAEv2 ViT-B distilled (K710, 768-d)", size="166 MB",
         path=os.path.join(ROOT, "weights", "vit_b_k710_dl_from_giant.pth"),
         fetch=("python -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download('OpenGVLab/VideoMAE2','distill/vit_b_k710_dl_from_giant.pth', local_dir='weights')\" "
                "&& mv weights/distill/vit_b_k710_dl_from_giant.pth weights/"),
         cwd=ROOT, note="HF (allowed). RT / streaming backbone."),
    dict(group="industreal", name="VideoMAEv2 ViT-giant SSv2 (1408-d)", size="1.9 GB",
         path=os.path.join(ROOT, "weights", "vit_g_ssv2_ft.pth"),
         fetch=None, cwd=ROOT,
         note="FORM-GATED (Google form; vit_g_hybrid_pt_1200e_ssv2_ft). Side-load. Backbone for "
              "v2 / fusion / DiffAct / MECCANO."),
    dict(group="industreal", name="InternVideo2 ViT-B/14 (K710, 768-d)", size="172 MB",
         path=os.path.join(ROOT, "fusion", "weights", "iv2_b14_k710.bin"),
         fetch=None, cwd=ROOT, note="License-gated (OpenGVLab InternVideo2). Side-load. Fusion appearance stream."),
    dict(group="industreal", name="InternVideo2 ViT-L/14 (K710, 768-d)", size="591 MB",
         path=os.path.join(ROOT, "fusion", "weights", "iv2_l14_k710.bin"),
         fetch=None, cwd=ROOT, note="License-gated. Side-load. fusion_l14 only."),
    dict(group="meccano", name="MECCANO RGB videos", size="11.9 GB",
         path=os.path.join(MECC, "dataset", "MECCANO_RGB_Videos"),
         fetch="huggingface-cli download ketanmore/MECCANO --repo-type dataset --local-dir dataset "
               "&& unzip -q -o dataset/*RGB_Videos*.zip -d dataset",
         cwd=MECC, note="HF mirror (your upload). Original iplab.dmi.unict.it is blocked."),
    dict(group="meccano", name="MECCANO PSR annotations", size="small",
         path=os.path.join(MECC, "PSR-annotations"),
         fetch="git clone https://github.com/TimSchoonbeek/PSR-annotations PSR-annotations",
         cwd=MECC, note="GitHub (allowed)."),
]


def present(a):
    p = a["path"]
    return os.path.exists(p) and (not os.path.isdir(p) or len(os.listdir(p)) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--group", choices=["industreal", "meccano", "all"], default="all")
    args = ap.parse_args()
    assets = [a for a in ASSETS if args.group in ("all", a["group"])]

    print(f"=== Provisioning check  (group={args.group}) ===\n")
    print(f"  {'asset':<42s} {'size':>8s}  {'status':<8s} source")
    print("  " + "-" * 96)
    missing_fetchable, missing_manual = [], []
    for a in assets:
        ok = present(a)
        src = "on disk" if ok else ("auto (HF/git)" if a["fetch"] else "SIDE-LOAD")
        print(f"  {a['name']:<42s} {a['size']:>8s}  {('OK' if ok else 'MISSING'):<8s} {src}")
        if not ok:
            (missing_fetchable if a["fetch"] else missing_manual).append(a)

    if missing_manual:
        print("\n  Side-load required (cannot auto-download here):")
        for a in missing_manual:
            print(f"    - {a['name']}: {a['note']}")
            print(f"        -> place at: {a['path']}")
    if missing_fetchable:
        print("\n  Auto-downloadable (use --fetch, or run manually):")
        for a in missing_fetchable:
            print(f"    - {a['name']}:  (cd {os.path.relpath(a['cwd'], EGO)}) {a['fetch']}")

    if args.fetch and missing_fetchable:
        print("\n=== Fetching downloadable assets ===")
        for a in missing_fetchable:
            print(f"\n--- {a['name']} ---")
            rc = subprocess.run(["bash", "-lc", a["fetch"]], cwd=a["cwd"]).returncode
            print(f"  {'OK' if rc == 0 else 'FAILED (rc=%d)' % rc}: {a['name']}")

    if not missing_fetchable and not missing_manual:
        print("\n  All assets present.")
    elif not args.fetch and missing_fetchable:
        print("\n  (run with --fetch to download the auto-downloadable ones)")

    # Fail-fast so this can gate the SLURM DAG: a still-missing side-load (non-fetchable)
    # asset means the pipeline cannot proceed. Re-check fetchables after a --fetch attempt.
    still_missing = [a for a in assets if not present(a)]
    blocking = [a for a in still_missing if not a["fetch"]]
    if blocking:
        print(f"\n  BLOCKING: {len(blocking)} required asset(s) missing and not auto-downloadable.")
        sys.exit(1)


if __name__ == "__main__":
    main()
