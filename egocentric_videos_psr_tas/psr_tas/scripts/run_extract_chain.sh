#!/bin/bash
# Runs IV2-B14 extraction then fusion, after the giant extractor process has exited.
set -e
source /home/aiops/miniconda3/etc/profile.d/conda.sh && conda activate psr_env
cd /home/aiops/Desktop/ego_psr_repro/industReal/psr_tas
LOG=logs/extract_chain.log
echo "[chain] waiting for giant extractor to finish..." | tee -a $LOG
while pgrep -f "01_extract_v2.py" >/dev/null 2>&1; do sleep 30; done
echo "[chain] giant done -> starting IV2-B14 extraction $(date +%H:%M:%S)" | tee -a $LOG
python fusion/scripts/extract_iv2.py --splits train,val --batch 32 --log logs/extract_iv2.log 2>&1 | tail -3 | tee -a $LOG
echo "[chain] IV2 done -> fusing $(date +%H:%M:%S)" | tee -a $LOG
python fusion/scripts/fuse.py 2>&1 | tail -5 | tee -a $LOG
echo "[chain] FUSION COMPLETE $(date +%H:%M:%S)" | tee -a $LOG
