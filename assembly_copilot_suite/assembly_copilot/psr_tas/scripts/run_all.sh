#!/bin/bash
# Fully autonomous v4 reproduction: features -> DiffAct step head -> metrics.
# Resumable (each stage skips completed work). Detached via setsid+nohup so it
# survives the interactive session ending.
source /home/aiops/miniconda3/etc/profile.d/conda.sh && conda activate psr_env
cd /home/aiops/Desktop/ego_psr_repro/industReal/psr_tas
LOG=logs/run_all.log
say(){ echo "[run_all $(date +%F_%H:%M:%S)] $*" | tee -a "$LOG"; }
SPLITS="train,test,val"

say "================ RUN_ALL START ================"

say "STAGE A1: giant SSv2 extraction (ImageNet norm, stride 2)"
python scripts/01_extract_v2.py --splits $SPLITS --stride 2 --batch 8 --log logs/extract_giant.log \
  || { say "GIANT FAILED"; exit 1; }

say "STAGE A2: IV2-B14 extraction (8-frame, aligned)"
python fusion/scripts/extract_iv2.py --splits $SPLITS --batch 32 --log logs/extract_iv2.log \
  || { say "IV2 FAILED"; exit 1; }

say "STAGE A3: fuse -> 2176-d [D,T]"
python fusion/scripts/fuse.py > logs/fuse.log 2>&1 || { say "FUSE FAILED"; exit 1; }

say "STAGE A4: prepare DiffAct IndustReal-Fusion dataset"
python scripts/03_prepare_diffact.py > logs/prep.log 2>&1 || { say "PREP FAILED"; exit 1; }
say "features ready: giant=$(ls data_v2/features/*.npy|grep -v starts|wc -l) fused=$(ls fusion/data/features/*.npy|wc -l)"

say "STAGE B: DiffAct training (1200 ep, eval@200..1000) -> step F1@50 on TEST"
( cd extern/DiffAct && python main.py --config configs/IndustReal-Fusion-S1.json --device 0 ) \
  > logs/diffact_train.log 2>&1 || { say "DIFFACT FAILED (see logs/diffact_train.log)"; exit 1; }

say "STAGE C: best TEST metrics per eval epoch (vs slide: Acc74.9 Edit79.5 F1@10 83.1 F1@25 79.1 F1@50 70.1)"
grep -E "decoder-agg-Test" logs/diffact_train.log | tee -a "$LOG"

say "================ RUN_ALL COMPLETE ================"
