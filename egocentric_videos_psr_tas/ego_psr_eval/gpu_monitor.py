#!/usr/bin/env python
"""Live GPU-usage sampler -> CSV, until it receives SIGTERM/SIGINT.

NVIDIA (`nvidia-smi`) is primary; AMD ROCm (`rocm-smi`) is a fallback for
portability. Safe on a host with no GPU: it still writes rows (util/mem blank)
so the timeline chart shows an honest flat line.

CSV columns: elapsed_s, wall_time, gpu, util_pct, mem_pct, mem_used_mb, mem_total_mb, source
"""
import argparse
import csv
import json
import re
import signal
import subprocess
import sys
import time

_STOP = False


def _stop(signum, frame):
    global _STOP
    _STOP = True


def _run(cmd, timeout=8):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return None


def sample_rocm():
    """Return list of per-GPU dicts via `rocm-smi --json` (stable per-card keys),
    or None if rocm-smi is unavailable / reports no GPUs."""
    out = _run(["rocm-smi", "--showuse", "--showmemuse", "--showmeminfo", "vram", "--json"])
    if not out:
        return None
    try:
        data = json.loads(out)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    rows = []
    for card, d in sorted(data.items()):
        if not card.lower().startswith("card") or not isinstance(d, dict):
            continue
        m = re.search(r"(\d+)", card)
        idx = m.group(1) if m else card

        def pick(*subs):
            for k, v in d.items():
                kl = k.lower()
                if all(s in kl for s in subs):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None
            return None

        used_b, total_b = pick("used memory", "(b)"), pick("total memory", "(b)")
        rows.append({
            "gpu": idx,
            "util_pct": pick("use", "(%)"),
            "mem_pct": pick("vram%"),
            "mem_used_mb": round(used_b / 1e6, 1) if used_b is not None else None,
            "mem_total_mb": round(total_b / 1e6, 1) if total_b is not None else None,
            "source": "rocm-smi",
        })
    return rows or None


def sample_nvidia():
    out = _run(["nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"])
    if not out:
        return None
    rows = []
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 4:
            continue
        try:
            used, total = float(p[2]), float(p[3])
            rows.append({"gpu": p[0], "util_pct": float(p[1]),
                         "mem_pct": round(100 * used / total, 1) if total else None,
                         "mem_used_mb": used, "mem_total_mb": total, "source": "nvidia-smi"})
        except ValueError:
            continue
    return rows or None


def sample():
    # NVIDIA first (this harness targets NVIDIA GPUs); ROCm as a fallback.
    return sample_nvidia() or sample_rocm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    cols = ["elapsed_s", "wall_time", "gpu", "util_pct", "mem_pct",
            "mem_used_mb", "mem_total_mb", "source"]
    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        fh.flush()
        n_active = 0
        while not _STOP:
            rows = sample()
            now = time.time()
            stamp = time.strftime("%H:%M:%S", time.localtime(now))
            if rows:
                for r in rows:
                    r = dict(r)
                    r["elapsed_s"] = round(now - t0, 1)
                    r["wall_time"] = stamp
                    w.writerow(r)
                    if r.get("util_pct"):
                        n_active += 1
            else:
                w.writerow({"elapsed_s": round(now - t0, 1), "wall_time": stamp,
                            "gpu": "", "util_pct": "", "mem_pct": "",
                            "mem_used_mb": "", "mem_total_mb": "", "source": "none"})
            fh.flush()
            # sleep in small steps so SIGTERM is honoured promptly
            slept = 0.0
            while slept < args.interval and not _STOP:
                time.sleep(min(0.5, args.interval - slept))
                slept += 0.5
    sys.exit(0)


if __name__ == "__main__":
    main()
