#!/bin/bash
# Feature pipeline for the assembly_copilot dataset (30 fps, 1080p, 40 recs).
#
# Differs from run_full_pipeline.sh (IndustReal, 10 fps) in exactly two knobs:
#
#   --frame_gap 3   16 sampled frames spaced 3 apart, so a clip spans 48 source
#                   frames = 1.60 s -- the same wall-clock motion IndustReal's 16
#                   consecutive frames cover at 10 fps. The SSv2-finetuned giant
#                   needs seconds of motion, not frames; 16 consecutive frames at
#                   30 fps is 0.53 s and out of distribution.
#   --stride 6      one clip every 6 source frames = 0.2 s, again matching
#                   IndustReal's stride 2 at 10 fps. Keeps T in the 860-2779 range
#                   (IndustReal: 557-2631) and keeps boundary_smooth=20 (4 s) and
#                   purge=3 (0.6 s) meaning what they were tuned to mean.
#
# Both extractors MUST use the same --frame_gap or the two streams fuse misaligned.
set -e
source /home/aiops/miniconda3/etc/profile.d/conda.sh && conda activate psr_env
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

DATASET=/media/lm-ciss/LM_4TB/assembly_copilot/dataset/prod_dataset
SPLITS="train,test"
GAP=3
STRIDE=6
LOG=logs/copilot_pipeline.log
mkdir -p logs
say(){ echo "[copilot $(date +%H:%M:%S)] $*" | tee -a $LOG; }

# Stages 1 and 2 run CONCURRENTLY on one GPU (giant ~10.3 GB + B14 ~3 GB of 24 GB).
# IV2 normally reuses the giant's *_starts.npy for clip alignment; in parallel that
# file may not exist yet or may be half-written, so we point --giant_feat at a
# nonexistent path to force its fallback. That fallback is
#   range(0, max(1, N - (16*gap - 1)), stride)
# which is byte-for-byte what the giant computes given the same N (same decord, same
# file), the same span and the same stride -- so the two streams stay clip-aligned.
# fuse.py truncates to min(len) as a second safety net.
say "STAGE 1+2/4 giant SSv2 || IV2-B14 extraction (stride=$STRIDE, frame_gap=$GAP)"
python scripts/01_extract_v2.py --dataset $DATASET --splits $SPLITS \
    --stride $STRIDE --frame_gap $GAP --batch 8 \
    --out data_v2/features_copilot --log logs/copilot_extract_giant.log &
PID_GIANT=$!
python fusion/scripts/extract_iv2.py --dataset $DATASET --splits $SPLITS \
    --stride $STRIDE --frame_gap $GAP --batch 16 \
    --giant_feat /nonexistent-force-deterministic-fallback \
    --out fusion/data/features_iv2_copilot --log logs/copilot_extract_iv2.log &
PID_IV2=$!
wait $PID_GIANT || { say "giant extraction FAILED"; exit 1; }
wait $PID_IV2   || { say "IV2 extraction FAILED";   exit 1; }

say "STAGE 3/4 fuse -> 2176-d [D,T]"
python fusion/scripts/fuse.py --giant data_v2/features_copilot \
    --iv2 fusion/data/features_iv2_copilot \
    --out fusion/data/features_copilot 2>&1 | tee -a $LOG | tail -3

say "STAGE 4/5 build labels (this procedure's taxonomy, NOT IndustReal's)"
python scripts/00_build_labels.py --dataset $DATASET \
    --proc_info $DATASET/procedure_info.json \
    --out data_copilot 2>&1 | tee -a $LOG | tail -4

# --center is the frame offset of the clip CENTRE, used to pick which per-frame
# label a clip inherits. It defaults to 8 = middle of a 16-CONSECUTIVE-frame clip.
# At frame_gap=3 a clip spans 16*3=48 source frames, so the centre is 48/2=24.
# Leaving it at 8 would label every clip from a third of the way into its window,
# shifting every boundary ~0.5 s early.
CENTER=$((16 * GAP / 2))
say "STAGE 5/5 assemble DiffAct datasets (step + type), --center $CENTER"
for head in "Copilot-Fusion:groundTruth" "Copilot-Type-Fusion:groundTruth_type"; do
    name=${head%%:*}; gt=${head##*:}
    python scripts/03_prepare_diffact.py \
        --fused fusion/data/features_copilot \
        --starts data_v2/features_copilot \
        --gt data_copilot/$gt --data data_copilot --center $CENTER \
        --out extern/DiffAct/datasets/$name 2>&1 | tee -a $LOG | tail -3
done
# the type head needs the 4-class mapping, not the 11-class step mapping
cp data_copilot/mapping_type.txt extern/DiffAct/datasets/Copilot-Type-Fusion/mapping.txt

say "DONE -- train with:"
say "  cd extern/DiffAct && python main.py --config configs/Copilot-Fusion-S1.json --device 0"
say "  cd extern/DiffAct && python main.py --config configs/Copilot-Type-Fusion-S1.json --device 0"
say "  Report the FINAL epoch (1199 step / 400 type), not the best -- there is no val split."
