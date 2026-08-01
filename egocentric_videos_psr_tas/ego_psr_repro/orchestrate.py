#!/usr/bin/env python
"""Reproduction DAG for every PSR architecture: download -> extract -> finetune -> eval.

Chains the project's EXISTING sbatch scripts + CPU steps with correct dependencies,
running each shared stage (labels, giant features, fusion) exactly once. Idempotent:
stages whose outputs already exist are marked DONE and skipped.

  --dry-run (default)  print the ordered plan; submit NOTHING
  --submit             submit the SLURM DAG (sbatch --dependency) + run CPU steps as cpu jobs
  --arch all|<name>[,<name>]   select architectures (see --list)
  --stages a,b,c       restrict to these stage kinds: download,labels,extract,fuse,train,eval

Nothing in the project is modified; this only submits/echoes commands over existing scripts.
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EGO = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.join(EGO, "industReal", "psr_tas")
MECC = os.path.join(EGO, "MECCANO")
EVAL = os.path.join(EGO, "ego_psr_eval", "run.sh")
CONDA = "/vast/users/fahad.khan/miniconda3/etc/profile.d/conda.sh"
PENV = os.path.join(ROOT, "psr_env")

# SLURM knobs shared by generated CPU jobs
CPU_SB = ("--partition=faculty --qos=gtqos --cpus-per-task=8 --mem=32G "
          "--time=02:00:00 --exclude=auh7-1b-gpu-199")


def r(*p):
    return os.path.join(ROOT, *p)


# --- stage DAG ---------------------------------------------------------------
# kind: prov | cpu | sbatch | eval    gpu: informational
# done: a path or glob whose existence means the stage is already complete
STAGES = {
    # provisioning (asset presence / fetch)
    "prov_industreal": dict(desc="Provision IndustReal dataset + VideoMAEv2/InternVideo2 weights",
        kind="prov", cmd="provision.py --group industreal", cwd=HERE, gpu=False, deps=[],
        done=r("..", "dataset", "test")),
    "prov_meccano": dict(desc="Provision MECCANO videos + PSR annotations",
        kind="prov", cmd="provision.py --group meccano", cwd=HERE, gpu=False, deps=[],
        done=os.path.join(MECC, "dataset", "MECCANO_RGB_Videos")),

    # shared IndustReal spine
    "labels": dict(desc="Build dense step/type labels (CPU, seconds)",
        kind="cpu", cmd="python scripts/00_build_labels.py --dataset ../dataset --out data",
        cwd=ROOT, gpu=False, deps=["prov_industreal"], done=r("data", "mapping.txt")),

    "extract_v1": dict(desc="VideoMAEv2-Huge features [1280] (array 0-7)",
        kind="sbatch", cmd="sbatch slurm/extract.sbatch", cwd=ROOT, gpu=True,
        deps=["labels"], done=r("data", "features")),
    "train_v1": dict(desc="ASFormer step+type on Huge (120 ep) + inline predict",
        kind="sbatch", cmd="sbatch slurm/train.sbatch", cwd=ROOT, gpu=True,
        deps=["extract_v1"], done=r("models", "step", "epoch-120.model")),

    "extract_v2": dict(desc="VideoMAEv2-giant SSv2 features [1408] — SHARED S1 (array 0-7)",
        kind="sbatch", cmd="sbatch slurm/extract_v2.sbatch", cwd=ROOT, gpu=True,
        deps=["labels"], done=r("data_v2", "features")),
    "train_v2": dict(desc="ASFormer step+type on SSv2 (60 ep)",
        kind="sbatch", cmd="sbatch slurm/train_v2.sbatch", cwd=ROOT, gpu=True,
        deps=["extract_v2"], done=r("models", "step_ssv2", "epoch-60.model")),
    "diffact_v2": dict(desc="DiffAct on SSv2 [1408] (1200 ep, inline eval)",
        kind="sbatch", cmd="sbatch slurm/diffact_v2.sbatch", cwd=ROOT, gpu=True,
        deps=["extract_v2"], done=r("extern", "DiffAct", "result", "IndustReal-S1", "prediction")),

    "extract_iv2_b14": dict(desc="InternVideo2-B14 features [768] aligned to S1 (array 0-7)",
        kind="sbatch", cmd="sbatch fusion/slurm/extract_iv2.sbatch", cwd=ROOT, gpu=True,
        deps=["extract_v2"], done=r("fusion", "data", "features_iv2")),
    "fuse_b14": dict(desc="L2-norm + concat giant|B14 -> fusion [2176] — SHARED S2 (CPU)",
        kind="cpu", cmd="python fusion/scripts/fuse.py", cwd=ROOT, gpu=False,
        deps=["extract_iv2_b14"], done=r("fusion", "data", "features")),
    "train_fusion_b14": dict(desc="ASFormer step+type on fusion-B14 (60 ep)",
        kind="sbatch", cmd="sbatch fusion/slurm/train_fusion.sbatch", cwd=ROOT, gpu=True,
        deps=["fuse_b14"], done=r("models", "step_fusion", "epoch-60.model")),
    "diffact_v4": dict(desc="DiffAct on fusion-B14 [2176] — v4 NEW BEST (1200 ep, inline eval)",
        kind="sbatch", cmd="sbatch slurm/diffact_v4.sbatch", cwd=ROOT, gpu=True,
        deps=["fuse_b14"], done=r("extern", "DiffAct", "result", "IndustReal-Fusion-S1", "prediction")),

    "extract_iv2_l14": dict(desc="InternVideo2-L14 features [768] aligned to S1 (array 0-7)",
        kind="sbatch", cmd="sbatch fusion/slurm/extract_iv2_l14.sbatch", cwd=ROOT, gpu=True,
        deps=["extract_v2"], done=r("fusion", "data", "features_iv2_l14")),
    "fuse_l14": dict(desc="L2-norm + concat giant|L14 -> fusion_l14 [2176] (CPU)",
        kind="cpu", cmd="python fusion/scripts/fuse.py --iv2_name features_iv2_l14 --out_name data_l14",
        cwd=ROOT, gpu=False, deps=["extract_iv2_l14"], done=r("fusion", "data_l14", "features")),
    "train_fusion_l14": dict(desc="ASFormer step+type on fusion-L14 (60 ep)",
        kind="sbatch", cmd="sbatch fusion/slurm/train_fusion_l14.sbatch", cwd=ROOT, gpu=True,
        deps=["fuse_l14"], done=r("models", "step_fusionl14", "epoch-60.model")),

    "rt_extract": dict(desc="Causal ViT-B features [768] streaming (array 0-7)",
        kind="sbatch", cmd="sbatch rt/slurm/extract_causal.sbatch", cwd=ROOT, gpu=True,
        deps=["labels"], done=r("rt", "data", "features")),
    "train_rt_gru": dict(desc="Causal GRU step+type heads (streaming) + inline eval",
        kind="sbatch", cmd="sbatch rt/slurm/train_rt.sbatch", cwd=ROOT, gpu=True,
        deps=["rt_extract"], done=r("rt", "models", "step", "model.pt")),
    "train_rt_testra": dict(desc="Causal TeSTra step+type heads + A/B eval",
        kind="sbatch", cmd="sbatch rt/slurm/train_testra.sbatch", cwd=ROOT, gpu=True,
        deps=["rt_extract", "train_rt_gru"], done=r("rt", "models", "step_testra", "model.pt")),

    "mecc_extract": dict(desc="MECCANO fusion features [2176] + labels side-effect (array 0-7)",
        kind="sbatch", cmd="sbatch pipeline/extract_fusion.sbatch", cwd=MECC, gpu=True,
        deps=["prov_meccano", "extract_v2"], done=os.path.join(MECC, "data", "features")),
    "train_meccano": dict(desc="ASFormer step+type on MECCANO (60 ep)",
        kind="sbatch", cmd="sbatch pipeline/train_meccano.sbatch", cwd=MECC, gpu=True,
        deps=["mecc_extract"], done=r("models", "step_meccano", "epoch-*.model")),
}

# eval stages: reuse the ego_psr_eval harness (CPU). arch -> harness names
EVAL_MAP = {
    "v1_huge": ("train_v1", "v1_huge,v1_huge_type"),
    "v2_ssv2": ("train_v2", "v2_ssv2,v2_ssv2_type"),
    "fusion_b14": ("train_fusion_b14", "fusion_b14,fusion_b14_type"),
    "fusion_l14": ("train_fusion_l14", "fusion_l14,fusion_l14_type"),
    "v2_diffact": ("diffact_v2", "v2_diffact"),
    "v4_fusion_diffact": ("diffact_v4", "v4_fusion_diffact"),
    "v3_gru": ("train_rt_gru", "v3_gru"),
    "v3_testra": ("train_rt_testra", "v3_testra"),
    "meccano": ("train_meccano", "meccano_step,meccano_type"),
}
for _arch, (_dep, _names) in EVAL_MAP.items():
    STAGES[f"eval_{_arch}"] = dict(
        desc=f"Evaluate {_arch} via ego_psr_eval harness (CPU)", kind="eval",
        cmd=f"bash {EVAL} --arch {_names} --no-gpu-monitor", cwd=EGO, gpu=False,
        deps=[_dep], done=None)

ARCHES = list(EVAL_MAP)


def done(sid):
    p = STAGES[sid]["done"]
    if p is None:
        return False
    if any(ch in p for ch in "*?["):
        return len(glob.glob(p)) > 0
    if os.path.isdir(p):
        return len(os.listdir(p)) > 0
    return os.path.exists(p)


def resolve(archs):
    """Transitive-closure of the selected archs' eval stages, topologically ordered."""
    want = set()
    stack = [f"eval_{a}" for a in archs]
    while stack:
        s = stack.pop()
        if s in want:
            continue
        want.add(s)
        stack.extend(STAGES[s]["deps"])
    order, seen = [], set()

    def visit(s):
        if s in seen:
            return
        seen.add(s)
        for d in STAGES[s]["deps"]:
            if d in want:
                visit(d)
        order.append(s)

    for s in sorted(want):
        visit(s)
    return order


def category(sid):
    """Which --stages bucket a stage belongs to (extract vs train can't be told by kind)."""
    s = STAGES[sid]
    if s["kind"] == "prov":
        return "download"
    if s["kind"] == "eval":
        return "eval"
    if sid == "labels":
        return "labels"
    if sid.startswith("fuse"):
        return "fuse"
    if "extract" in sid:
        return "extract"
    return "train"  # train_*, diffact_*


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all")
    ap.add_argument("--stages", default=None, help="comma list of: download,labels,extract,fuse,train,eval")
    ap.add_argument("--submit", action="store_true", help="actually submit (default is dry-run)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("Architectures:", ", ".join(ARCHES))
        print("\nStages:")
        for sid, s in STAGES.items():
            print(f"  {sid:<18s} [{s['kind']:<6s}] gpu={int(s['gpu'])}  {s['desc']}")
        return

    archs = ARCHES if args.arch == "all" else [a.strip() for a in args.arch.split(",")]
    for a in archs:
        if a not in ARCHES:
            raise SystemExit(f"unknown arch '{a}'. Known: {ARCHES}")
    order = resolve(archs)

    # stage-kind restriction (e.g. only 'extract,train')
    if args.stages:
        want_names = {x.strip() for x in args.stages.split(",")}
        order = [s for s in order if category(s) in want_names]

    submit = args.submit and not args.dry_run
    mode = "SUBMIT" if submit else "DRY-RUN"
    print(f"=== Reproduction plan [{mode}]  archs: {', '.join(archs)} ===")
    print(f"    {len(order)} stages (shared stages listed once)\n")

    jobids = {}  # sid -> slurm job id (submit mode)
    gpu_jobs = 0
    for i, sid in enumerate(order, 1):
        s = STAGES[sid]
        is_done = done(sid)
        status = "DONE" if is_done else "TODO"
        deps = [d for d in s["deps"] if d in order]
        dep_str = ", ".join(deps) if deps else "-"
        print(f"[{i:>2}/{len(order)}] {sid:<18s} {status:<4s} {'GPU' if s['gpu'] else 'cpu':<3s}  deps: {dep_str}")
        print(f"        {s['desc']}")
        # the concrete command
        if s["kind"] == "sbatch":
            cmd = s["cmd"]
        elif s["kind"] == "prov":
            cmd = f"python {s['cmd']}"
        else:
            cmd = s["cmd"]
        print(f"        $ (cd {os.path.relpath(s['cwd'], EGO)}) {cmd}")
        if s["gpu"]:
            gpu_jobs += 0 if is_done else 1

        if submit and not is_done:
            dep_ids = [jobids[d] for d in deps if d in jobids]
            dep_flag = f"--dependency=afterok:{':'.join(dep_ids)}" if dep_ids else ""
            if s["kind"] == "sbatch":
                full = f"cd {s['cwd']} && sbatch --parsable {dep_flag} {s['cmd'].split(' ',1)[1]}"
            else:  # cpu / prov / eval -> wrap as a cpu SLURM job so deps hold
                inner = (f"source {CONDA}; conda activate {PENV}; cd {s['cwd']}; {cmd}")
                full = (f"sbatch --parsable {dep_flag} {CPU_SB} --job-name=repro_{sid} "
                        f"--wrap {shq(inner)}")
            jid = subprocess.run(["bash", "-lc", full], capture_output=True, text=True).stdout.strip()
            jid = (jid.splitlines()[-1] if jid else "").split(";")[0]  # strip any ;cluster suffix
            jobids[sid] = jid
            print(f"        -> submitted job {jid} {('(dep '+ ':'.join(dep_ids)+')') if dep_ids else ''}")
        elif submit and is_done:
            print("        -> skipped (already done)")
        print()

    todo = sum(1 for s in order if not done(s))
    print(f"=== {mode}: {len(order)-todo} done, {todo} to run "
          f"({gpu_jobs} GPU job(s)) ===")
    if not submit:
        print("    (dry-run: nothing submitted. Re-run with --submit to launch the SLURM DAG.)")


def shq(s):
    return "'" + s.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
