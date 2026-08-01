#!/usr/bin/env bash
# ============================================================================
#  ego_psr_eval — one command to evaluate EVERY PSR architecture.
#
#  Usage:
#     ./run.sh                         # evaluate all architectures
#     ./run.sh --arch v4_fusion_diffact
#     ./run.sh --arch offline_step     # a whole group
#     ./run.sh --arch v2_ssv2,v4_fusion_diffact
#     ./run.sh --list                  # list architectures + groups
#     ./run.sh --check                 # preflight (artifacts present?) only
#     ./run.sh --no-gpu-monitor        # skip the live GPU sampler
#
#  It sets up the env, verifies artifacts, runs the real eval scripts, samples
#  GPU usage live, and renders charts — all from local, already-trained models.
#  (Feature re-extraction from raw video needs GPUs and is out of scope here.)
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../industReal/psr_tas" 2>/dev/null && pwd || true)"
PSR_ENV="${ROOT:-}/psr_env"
CONDA_SH="/vast/users/fahad.khan/miniconda3/etc/profile.d/conda.sh"
RESULTS="$HERE/results"
mkdir -p "$RESULTS/logs" "$RESULTS/charts"

ARCH="all"; GPU_MON=1; GPU_INT=3; PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--arch)        ARCH="$2"; shift 2 ;;
    --list)           PASS+=(--list); shift ;;
    --check)          PASS+=(--check); shift ;;
    --no-gpu-monitor) GPU_MON=0; shift ;;
    --gpu-interval)   GPU_INT="$2"; shift 2 ;;
    -h|--help)        sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
done

banner() { printf '\n\033[1;36m%s\033[0m\n' "== $* =="; }

# ---- 1. environment -------------------------------------------------------
banner "1/5  Environment setup"
if [[ -z "${ROOT:-}" || ! -d "$PSR_ENV" ]]; then
  echo "  ERROR: project env not found at $PSR_ENV"
  echo "  This harness reuses the project's conda env (psr_env). Create it first (see README)."
  exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$PSR_ENV"
PY="$PSR_ENV/bin/python"
echo "  python : $($PY --version 2>&1)  @ $PSR_ENV"
echo "  project: $ROOT"
# ensure plotting dep (already present in psr_env; install only if missing)
$PY -c "import matplotlib" 2>/dev/null || { echo "  installing matplotlib..."; $PY -m pip install -q matplotlib; }

# list/check short-circuit (no GPU monitor, no plots)
if [[ " ${PASS[*]:-} " == *"--list"* || " ${PASS[*]:-} " == *"--check"* ]]; then
  $PY "$HERE/evaluate.py" --arch "$ARCH" "${PASS[@]}"
  exit $?
fi

# ---- 2. live GPU monitor --------------------------------------------------
banner "2/5  Live GPU monitor"
MON_PID=""
if [[ $GPU_MON -eq 1 ]]; then
  $PY "$HERE/gpu_monitor.py" --out "$RESULTS/gpu_usage.csv" --interval "$GPU_INT" \
      >"$RESULTS/logs/gpu_monitor.log" 2>&1 &
  MON_PID=$!
  echo "  sampling GPU every ${GPU_INT}s via nvidia-smi (pid $MON_PID) -> results/gpu_usage.csv"
  echo "  (the evals run on CPU; the monitor still records live NVIDIA GPU state on the host)"
else
  echo "  skipped (--no-gpu-monitor)"
fi
cleanup() { [[ -n "$MON_PID" ]] && kill "$MON_PID" 2>/dev/null; }
trap cleanup EXIT

# ---- 3. evaluate ----------------------------------------------------------
banner "3/5  Evaluating architectures  (arch=$ARCH)"
$PY "$HERE/evaluate.py" --arch "$ARCH" --out "$RESULTS/results.json"
EV_RC=$?

# ---- 4. stop monitor + plot ----------------------------------------------
banner "4/5  Charts"
cleanup; MON_PID=""
$PY "$HERE/plot.py" --results "$RESULTS/results.json" --gpu "$RESULTS/gpu_usage.csv" \
    --outdir "$RESULTS/charts"

# ---- 5. summary -----------------------------------------------------------
banner "5/5  Summary"
$PY - "$RESULTS/results.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))["results"]
def line(name, r):
    m = r.get("metrics", {})
    if r["kind"] in ("step", "diffact") and "best" in m:
        b = m["best"]; return f"F1@50 {b['F1@50']:6.1f}  Edit {b['Edit']:6.1f}  Acc {b['Acc']:6.1f}"
    if r["kind"] == "type":
        return f"incorrect-rec {m.get('incorrect_recall',float('nan')):5.1f}%  remove-rec {m.get('remove_recall',float('nan')):5.1f}%"
    if r["kind"] == "rt" and "best" in m:
        b = m["best"]; return f"best F1@50 {b['F1@50']:5.1f} @ L={b['L']} ({b['latency_s']:.2f}s)"
    return r["status"]
print(f"  {'architecture':<34s} {'status':<8s} key metric")
print("  " + "-" * 78)
for name, r in d.items():
    st = r["status"]
    print(f"  {r['label']:<34s} {st:<8s} {line(name, r) if st=='ok' else ''}")
PYEOF
echo
echo "  results : $RESULTS/results.json"
echo "  charts  : $RESULTS/charts/"
echo "  logs    : $RESULTS/logs/"
exit ${EV_RC:-0}
