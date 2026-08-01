#!/usr/bin/env bash
# ============================================================================
#  ego_psr_repro — reproduce EVERY PSR architecture end-to-end:
#     provision (download) -> extract features -> finetune heads -> evaluate.
#
#  Chains the project's existing SLURM scripts with correct dependencies and
#  runs each shared stage (labels, giant features, fusion) exactly once.
#
#  DEFAULT IS DRY-RUN — it prints the full plan and submits NOTHING.
#
#  Usage:
#     ./repro.sh                          # dry-run the whole pipeline (all archs)
#     ./repro.sh --arch v4_fusion_diffact # dry-run one architecture's chain
#     ./repro.sh --stages extract,train   # restrict to some stage kinds
#     ./repro.sh --provision              # check datasets + weights presence
#     ./repro.sh --provision --fetch      # download the auto-downloadable assets
#     ./repro.sh --status                 # what's already built vs to-do
#     ./repro.sh --list                   # list architectures + stages
#     ./repro.sh --submit                 # ACTUALLY submit the SLURM DAG (GPUs!)
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../industReal/psr_tas" 2>/dev/null && pwd || true)"
PENV="${ROOT:-}/psr_env"
CONDA_SH="/vast/users/fahad.khan/miniconda3/etc/profile.d/conda.sh"

if [[ -z "${ROOT:-}" || ! -d "$PENV" ]]; then
  echo "ERROR: project env not found at $PENV (reuses the project's psr_env)."; exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$PENV"
PY="$PENV/bin/python"

MODE="dryrun"; PROV_FETCH=0; PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --provision) MODE="provision"; shift ;;
    --fetch)     PROV_FETCH=1; shift ;;
    --status)    MODE="status"; shift ;;
    --list)      MODE="list"; shift ;;
    --submit)    PASS+=(--submit); shift ;;
    --arch|--stages) PASS+=("$1" "$2"); shift 2 ;;
    --dry-run)   PASS+=(--dry-run); shift ;;
    -h|--help)   sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
done

case "$MODE" in
  provision) $PY "$HERE/provision.py" --check $([[ $PROV_FETCH -eq 1 ]] && echo --fetch) ;;
  status)    $PY "$HERE/status.py" "${PASS[@]}" ;;
  list)      $PY "$HERE/orchestrate.py" --list ;;
  *)         $PY "$HERE/orchestrate.py" "${PASS[@]}" ;;
esac
