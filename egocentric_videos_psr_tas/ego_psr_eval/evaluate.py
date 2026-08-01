#!/usr/bin/env python
"""Evaluate every trained PSR architecture and collect metrics into results.json.

Reuses the project's OWN eval scripts (eval_step.py / eval_type.py / rt/eval_online.py)
so the numbers are identical to what those scripts produce; DiffAct has no eval-only
entrypoint, so its metrics are recomputed on CPU from its cached predictions via the
DiffAct func_eval (no GPU retrain). Nothing in the project is modified.

Select architectures with --arch:  all | <group> | <name>[,<name>...]
Groups: offline_step, offline_type, streaming, meccano.  List them with --list.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "industReal", "psr_tas"))
SCRIPTS = os.path.join(ROOT, "scripts")
PY = sys.executable  # psr_env python (run.sh activates it)


def cfg(*p):
    return os.path.normpath(os.path.join(ROOT, *p))


# --- the architecture registry -------------------------------------------------
# kinds: step (eval_step.py) | type (eval_type.py) | diffact (cached func_eval) | rt (eval_online.py)
REGISTRY = [
    # ---- offline STEP segmentation ----
    dict(name="v1_huge", label="v1  Huge-K710 + ASFormer", group="offline_step", kind="step",
         config=cfg("configs/default.yaml"), data="data", model_dir="step", epoch=120),
    dict(name="v2_ssv2", label="v2  SSv2-giant + ASFormer", group="offline_step", kind="step",
         config=cfg("configs/default_ssv2.yaml"), data="data_v2", model_dir="step_ssv2", epoch=60),
    dict(name="fusion_b14", label="Fusion (giant+IV2-B14) + ASFormer", group="offline_step", kind="step",
         config=cfg("fusion/configs/fusion.yaml"), data="fusion/data", model_dir="step_fusion", epoch=60),
    dict(name="fusion_l14", label="Fusion (giant+IV2-L14) + ASFormer", group="offline_step", kind="step",
         config=cfg("fusion/configs/fusion_l14.yaml"), data="fusion/data_l14", model_dir="step_fusionl14", epoch=60),
    dict(name="v2_diffact", label="v2  SSv2-giant + DiffAct", group="offline_step", kind="diffact",
         result="IndustReal-S1", dataset="industreal", epoch=1000),
    dict(name="v4_fusion_diffact", label="v4  Fusion + DiffAct  (NEW BEST)", group="offline_step", kind="diffact",
         result="IndustReal-Fusion-S1", dataset="industreal_fusion", epoch=1000),
    # ---- offline TYPE / correctness ----
    dict(name="v1_huge_type", label="v1  Huge-K710 type", group="offline_type", kind="type",
         config=cfg("configs/default.yaml"), model_dir="type", epoch=120),
    dict(name="v2_ssv2_type", label="v2  SSv2-giant type", group="offline_type", kind="type",
         config=cfg("configs/default_ssv2.yaml"), model_dir="type_ssv2", epoch=60),
    dict(name="fusion_b14_type", label="Fusion-B14 type", group="offline_type", kind="type",
         config=cfg("fusion/configs/fusion.yaml"), model_dir="type_fusion", epoch=60),
    dict(name="fusion_l14_type", label="Fusion-L14 type", group="offline_type", kind="type",
         config=cfg("fusion/configs/fusion_l14.yaml"), model_dir="type_fusionl14", epoch=60),
    # ---- real-time / streaming ----
    dict(name="v3_gru", label="v3  causal ViT-B + GRU", group="streaming", kind="rt",
         suffix="", lags="0,8,16,32"),
    dict(name="v3_testra", label="v3  TeSTra head", group="streaming", kind="rt",
         suffix="_testra", lags="0,8,16"),
    # ---- MECCANO (2nd dataset) ----
    dict(name="meccano_step", label="MECCANO Fusion + ASFormer", group="meccano", kind="step",
         config=cfg("..", "..", "MECCANO", "pipeline", "meccano.yaml"), data="../../MECCANO/data",
         model_dir="step_meccano", epoch=60),
    dict(name="meccano_type", label="MECCANO type", group="meccano", kind="type",
         config=cfg("..", "..", "MECCANO", "pipeline", "meccano.yaml"), model_dir="type_meccano", epoch=60),
]
BY_NAME = {e["name"]: e for e in REGISTRY}
GROUPS = sorted({e["group"] for e in REGISTRY})


# --- artifact preflight --------------------------------------------------------
def artifact_path(e):
    """Primary artifact whose presence gates evaluation (for the --check preflight)."""
    if e["kind"] in ("step", "type"):
        return os.path.join(ROOT, "models", e["model_dir"], f"epoch-{e['epoch']}.model")
    if e["kind"] == "diffact":
        return os.path.join(ROOT, "extern", "DiffAct", "result", e["result"], "prediction")
    if e["kind"] == "rt":
        sub = "step" + e["suffix"]
        return os.path.join(ROOT, "rt", "models", sub, "model.pt")
    return None


def preflight(entries):
    print("  preflight — required artifacts:")
    ok = True
    for e in entries:
        p = artifact_path(e)
        present = p is not None and os.path.exists(p)
        ok = ok and present
        print(f"    [{'OK ' if present else 'MISS'}] {e['name']:<20s} {p}")
    return ok


# --- per-kind evaluators (return metrics dict, raw stdout) ---------------------
_FLOAT = r"([-+]?\d+\.?\d*)"


def run_cmd(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def eval_step(e):
    cmd = [PY, "eval_step.py", "--config", e["config"], "--data", e["data"],
           "--model_dir", e["model_dir"], "--split", "test", "--epoch", str(e["epoch"]),
           "--penalties", "0,25,50,100,150"]
    rc, out = run_cmd(cmd, SCRIPTS)
    if rc != 0:
        return None, out
    rows = {}
    for m in re.finditer(r"^\s*(raw argmax|viterbi p=[\d.]+)\s+" + r"\s+".join([_FLOAT] * 5),
                         out, re.M):
        name = m.group(1).strip()
        rows[name] = dict(zip(["Acc", "Edit", "F1@10", "F1@25", "F1@50"],
                              [float(m.group(i)) for i in range(2, 7)]))
    if not rows:
        return None, out
    best = max(rows.items(), key=lambda kv: kv[1]["F1@50"])
    return {"rows": rows, "best_decode": best[0], "best": best[1],
            "raw": rows.get("raw argmax")}, out


def eval_type(e):
    cmd = [PY, "eval_type.py", "--config", e["config"], "--model_dir", e["model_dir"],
           "--split", "test", "--epoch", str(e["epoch"])]
    rc, out = run_cmd(cmd, SCRIPTS)
    if rc != 0:
        return None, out
    metrics = {}
    mo = re.search(r"overall frame acc:\s*" + _FLOAT, out)
    if mo:
        metrics["overall_acc"] = float(mo.group(1))
    for cls in ("none", "correct", "incorrect", "remove"):
        m = re.search(rf"^{cls}\s+(\d+)\s+" + r"\s+".join([_FLOAT] * 3), out, re.M)
        if m:
            metrics[f"{cls}_support"] = int(m.group(1))
            metrics[f"{cls}_prec"] = float(m.group(2))
            metrics[f"{cls}_recall"] = float(m.group(3))
            metrics[f"{cls}_f1"] = float(m.group(4))
    return (metrics or None), out


def eval_diffact(e):
    dda = os.path.join(ROOT, "extern", "DiffAct")
    result_dir = os.path.join(dda, "result", e["result"])
    pred_dir = os.path.join(result_dir, "prediction")
    label_dir = os.path.join(dda, "datasets", e["dataset"], "groundTruth")
    bundle = os.path.join(dda, "datasets", e["dataset"], "splits", "test.split1.bundle")
    snippet = (
        "import sys,json\n"
        f"sys.path.insert(0,{dda!r})\n"
        "from utils import func_eval\n"
        f"vids=[l.strip().split('.')[0] for l in open({bundle!r}) if l.strip()]\n"
        f"a,e,f=func_eval({label_dir!r},{pred_dir!r},vids)\n"
        "print('RESULT_JSON',json.dumps({'Acc':a,'Edit':e,'F1@10':f[0],'F1@25':f[1],'F1@50':f[2],'n':len(vids)}))\n"
    )
    rc, out = run_cmd([PY, "-c", snippet], dda)
    m = re.search(r"RESULT_JSON (\{.*\})", out)
    if m:
        d = json.loads(m.group(1))
        return {"rows": {"decoder-agg": d}, "best_decode": "decoder-agg", "best": d,
                "source": "recomputed from cached predictions"}, out
    # fallback: read the metrics DiffAct saved during training
    import numpy as np
    npy = os.path.join(result_dir, f"test_results_decoder-agg_epoch{e['epoch']}.npy")
    if os.path.exists(npy):
        d = {k: float(v) for k, v in np.load(npy, allow_pickle=True).item().items()}
        return {"rows": {"decoder-agg": d}, "best_decode": "decoder-agg", "best": d,
                "source": "cached test_results npy"}, out + f"\n[fallback] loaded {npy}"
    return None, out


def eval_rt(e):
    cmd = [PY, "rt/scripts/eval_online.py", "--split", "test",
           "--lags", e["lags"]]
    if e["suffix"]:
        cmd += ["--suffix", e["suffix"]]
    rc, out = run_cmd(cmd, ROOT)
    if rc != 0:
        return None, out
    lags = []
    for m in re.finditer(r"^\s*(\d+)\s+" + _FLOAT + r"s\s+" + r"\s+".join([_FLOAT] * 5), out, re.M):
        lags.append({"L": int(m.group(1)), "latency_s": float(m.group(2)),
                     "Acc": float(m.group(3)), "Edit": float(m.group(4)),
                     "F1@10": float(m.group(5)), "F1@25": float(m.group(6)),
                     "F1@50": float(m.group(7))})
    if not lags:
        return None, out
    best = max(lags, key=lambda r: r["F1@50"])
    return {"lags": lags, "best": best}, out


EVALUATORS = {"step": eval_step, "type": eval_type, "diffact": eval_diffact, "rt": eval_rt}


def select(arch):
    if arch == "all":
        return list(REGISTRY)
    out = []
    for tok in arch.split(","):
        tok = tok.strip()
        if tok in BY_NAME:
            out.append(BY_NAME[tok])
        elif tok in GROUPS:
            out += [e for e in REGISTRY if e["group"] == tok]
        else:
            raise SystemExit(f"unknown arch/group '{tok}'. Known: {list(BY_NAME)} or groups {GROUPS}")
    # de-dup preserving order
    seen, uniq = set(), []
    for e in out:
        if e["name"] not in seen:
            seen.add(e["name"]); uniq.append(e)
    return uniq


def headline(name, res):
    """One-line human summary of an arch's key metric."""
    e = BY_NAME[name]
    if e["kind"] in ("step", "diffact"):
        b = res["metrics"]["best"]
        return f"F1@50 {b['F1@50']:.1f} (Acc {b['Acc']:.1f}, Edit {b['Edit']:.1f}, {res['metrics'].get('best_decode','')})"
    if e["kind"] == "type":
        m = res["metrics"]
        return f"incorrect-recall {m.get('incorrect_recall', float('nan')):.1f}% | remove-recall {m.get('remove_recall', float('nan')):.1f}%"
    if e["kind"] == "rt":
        b = res["metrics"]["best"]
        return f"best F1@50 {b['F1@50']:.1f} @ L={b['L']} ({b['latency_s']:.2f}s)"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "results.json"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("Architectures (group · name · kind):")
        for e in REGISTRY:
            print(f"  {e['group']:<13s} {e['name']:<20s} {e['kind']:<8s} {e['label']}")
        print(f"\nGroups: {', '.join(GROUPS)} | 'all'")
        return

    entries = select(args.arch)
    log_dir = os.path.join(os.path.dirname(args.out), "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n=== Evaluating {len(entries)} architecture(s): {', '.join(e['name'] for e in entries)} ===")
    all_ok = preflight(entries)
    if args.check:
        print("  (--check) preflight only; not running evals.")
        return
    if not all_ok:
        print("  WARNING: some artifacts missing; those will be marked 'missing' below.\n")

    results = {}
    for i, e in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}] {e['label']}  ({e['group']} · {e['kind']})")
        art = artifact_path(e)
        if art is not None and not os.path.exists(art):
            print(f"    SKIP — missing artifact: {art}")
            results[e["name"]] = {"label": e["label"], "group": e["group"], "kind": e["kind"],
                                  "status": "missing", "artifact": art}
            continue
        t0 = time.time()
        try:
            metrics, raw = EVALUATORS[e["kind"]](e)
        except Exception as ex:  # keep going; record the failure
            metrics, raw = None, f"EXCEPTION: {ex!r}"
        dt = time.time() - t0
        with open(os.path.join(log_dir, f"{e['name']}.log"), "w") as fh:
            fh.write(raw)
        if metrics is None:
            print(f"    FAILED to parse metrics ({dt:.1f}s) — see logs/{e['name']}.log")
            results[e["name"]] = {"label": e["label"], "group": e["group"], "kind": e["kind"],
                                  "status": "error", "seconds": round(dt, 1)}
            continue
        results[e["name"]] = {"label": e["label"], "group": e["group"], "kind": e["kind"],
                              "status": "ok", "seconds": round(dt, 1), "metrics": metrics}
        print(f"    OK ({dt:.1f}s) — {headline(e['name'], results[e['name']])}")

    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "root": ROOT, "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n=== wrote {args.out} ({sum(1 for r in results.values() if r['status']=='ok')}/{len(results)} ok) ===")


if __name__ == "__main__":
    main()
