#!/bin/bash
# Grid over the two knobs that plausibly control flicker, evaluating each with the
# best causal decoder found so far (viterbi lag=10 bias=6, i.e. 2 s of lag).
#
#   tmse    strength of the smoothing loss. 0.15 took raw-argmax F1@50 from 43.6 to
#           59.3, so it is the lever that works; the question is where it saturates
#           or starts costing frame accuracy.
#   dropout 0.5 was inherited from DiffAct's config, which was tuned for a
#           bidirectional model on a different dataset. High dropout can make a
#           causal model's frame-to-frame output less stable, which is precisely the
#           failure mode here -- worth one test rather than assuming.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source /home/aiops/miniconda3/etc/profile.d/conda.sh && conda activate psr_env
RES=logs/grid2_results.txt
: > $RES

for cfg in "1.0 0.25" "0.7 0.5" "2.0 0.5"; do
    set -- $cfg; TM=$1; DR=$2
    TAG="tmse${TM}_drop${DR}"
    echo "=== $TAG ===" | tee -a $RES
    python scripts/train_causal.py --epochs 400 --eval_every 100 --tmse $TM \
        --dropout $DR --out runs/$TAG > logs/train_$TAG.log 2>&1
    tail -2 logs/train_$TAG.log | tee -a $RES
    python scripts/causal_decode.py --ckpt runs/$TAG/final.pt --out runs/$TAG/logits >/dev/null 2>&1
    python scripts/sweep_decode.py --logits runs/$TAG/logits 2>/dev/null \
        | grep -E "raw argmax|lag=10 bias=6.0|lag=25 bias=6.0" | tee -a $RES
    echo | tee -a $RES
done
echo "GRID COMPLETE" | tee -a $RES
