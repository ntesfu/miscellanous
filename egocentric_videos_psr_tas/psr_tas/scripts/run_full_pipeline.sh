#!/bin/bash
# Full v4 feature pipeline: giant(1408) -> IV2-B14(768) -> fuse(2176) -> DiffAct dataset prep.
# All 84 recs (train,test,val order so train+test finish first). Resumable per-rec.
set -e
source /home/aiops/miniconda3/etc/profile.d/conda.sh && conda activate psr_env
cd /home/aiops/Desktop/ego_psr_repro/industReal/psr_tas
SPLITS="train,test,val"
LOG=logs/pipeline.log
say(){ echo "[pipeline $(date +%H:%M:%S)] $*" | tee -a $LOG; }

say "STAGE 1/4 giant SSv2 extraction (ImageNet norm, stride 2)"
python scripts/01_extract_v2.py --splits $SPLITS --stride 2 --batch 8 --log logs/extract_giant.log

say "STAGE 2/4 IV2-B14 extraction (8-frame, aligned to giant starts)"
python fusion/scripts/extract_iv2.py --splits $SPLITS --batch 32 --log logs/extract_iv2.log

say "STAGE 3/4 fuse -> 2176-d [D,T]"
python fusion/scripts/fuse.py 2>&1 | tee -a $LOG | tail -3

say "STAGE 4/4 prepare DiffAct IndustReal-Fusion dataset (clip-aligned labels)"
python scripts/03_prepare_diffact.py 2>&1 | tee -a $LOG | tail -4

say "PIPELINE COMPLETE — ready to train DiffAct (configs/IndustReal-Fusion-S1.json)"
