#!/bin/bash
# Launch the copilot run inside tmux so it survives disconnects, with every stage
# writing a tailable log.
#
#   ./scripts/tmux_copilot.sh extract    stages 1+2: both encoders, one decode pass
#   ./scripts/tmux_copilot.sh post       stages 3-5 (fuse -> labels -> DiffAct dirs)
#   ./scripts/tmux_copilot.sh train      both heads, sequentially, in one window
#   ./scripts/tmux_copilot.sh status     what is alive + last line of each log
#
# Attach:  tmux attach -t copilot        (detach again with Ctrl-b d)
# Tail:    tail -f logs/copilot_extract_both.log
#
# Everything is resumable: extraction skips recordings whose .npy already exists,
# so a killed job can simply be relaunched.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs

SESSION=copilot
DATASET=/media/lm-ciss/LM_4TB/assembly_copilot/dataset/prod_dataset
SPLITS="train,test"
GAP=3
STRIDE=6
CENTER=$((16 * GAP / 2))
CONDA="source /home/aiops/miniconda3/etc/profile.d/conda.sh && conda activate psr_env"

win() {  # win <name> <command>  -- run in its own tmux window, keep it open on exit
    local name=$1 cmd=$2
    # Drop any window already holding this name, addressing it by INDEX: once two
    # windows share a name, every name-based target ("$SESSION:$name") is ambiguous
    # and both kill-window and send-keys fail with "can't find window".
    for idx in $(tmux list-windows -t $SESSION -F '#{window_index} #W' 2>/dev/null \
                 | awk -v n="$name" '$2==n {print $1}' | sort -rn); do
        tmux kill-window -t "$SESSION:$idx" 2>/dev/null
    done
    tmux new-session -d -s $SESSION -n "$name" 2>/dev/null \
        || tmux new-window -t $SESSION -n "$name"
    # capture the id of the window we just made, so send-keys can never be ambiguous
    local wid
    wid=$(tmux list-windows -t $SESSION -F '#{window_index} #W' | awk -v n="$name" '$2==n {print $1}' | tail -1)
    tmux send-keys -t "$SESSION:$wid" "cd $PWD && $CONDA && $cmd; echo; echo '=== $name finished (rc=\$?) ==='" C-m
}

case "${1:-status}" in

extract)
    # ONE process, ONE decode pass, both models -- see extract_both.py for why.
    # Running the two original extractors in parallel measured 11.2 h EACH: at
    # stride 6 / gap 3 consecutive clips overlap by 42 of 48 frames and decord
    # re-seeks per clip. Decoding every 3rd frame once and slicing 16-entry
    # contiguous windows drops decode to ~41 s/video, leaving the run GPU-bound at
    # ~6 clips/s (ViT-giant saturates the 4090; batch 8/16/24 all measure the same).
    # Expect ~3 h for 40 recordings.
    win extract "python fusion/scripts/extract_both.py --dataset $DATASET \
        --splits $SPLITS --stride $STRIDE --frame_gap $GAP --batch 8 --threads 8 \
        --giant_out data_v2/features_copilot \
        --iv2_out fusion/data/features_iv2_copilot \
        --log logs/copilot_extract_both.log 2>&1 | tee -a logs/copilot_extract.stdout"
    echo "launched. tail -f logs/copilot_extract_both.log"
    ;;

post)
    # NB the whole chain is wrapped in { } before the redirect: in `a && b | tee`
    # the pipe binds to b alone, so only the last command would be logged.
    win post "{ python fusion/scripts/fuse.py --giant data_v2/features_copilot \
        --iv2 fusion/data/features_iv2_copilot --out fusion/data/features_copilot && \
      python scripts/00_build_labels.py --dataset $DATASET \
        --proc_info $DATASET/procedure_info.json --out data_copilot && \
      python scripts/03_prepare_diffact.py --fused fusion/data/features_copilot \
        --starts data_v2/features_copilot --gt data_copilot/groundTruth \
        --data data_copilot --center $CENTER \
        --out extern/DiffAct/datasets/Copilot-Fusion && \
      python scripts/03_prepare_diffact.py --fused fusion/data/features_copilot \
        --starts data_v2/features_copilot --gt data_copilot/groundTruth_type \
        --data data_copilot --center $CENTER \
        --out extern/DiffAct/datasets/Copilot-Type-Fusion && \
      cp data_copilot/mapping_type.txt extern/DiffAct/datasets/Copilot-Type-Fusion/mapping.txt; \
      } 2>&1 | tee -a logs/copilot_post.log"
    echo "launched. tail -f logs/copilot_post.log"
    ;;

train)
    # Sequential, not parallel: two DiffAct runs would contend for the same GPU and
    # neither is the bottleneck here (1.2M params). Step first -- it is the result.
    win train "cd extern/DiffAct && \
      python main.py --config configs/Copilot-Fusion-S1.json --device 0 \
        2>&1 | tee -a ../../logs/copilot_train_step.log && \
      python main.py --config configs/Copilot-Type-Fusion-S1.json --device 0 \
        2>&1 | tee -a ../../logs/copilot_train_type.log"
    echo "launched. tail -f logs/copilot_train_step.log"
    ;;

status)
    echo "=== tmux ==="; tmux ls 2>/dev/null | grep -E "^$SESSION" || echo "  no '$SESSION' session"
    tmux list-windows -t $SESSION 2>/dev/null | sed 's/^/  /'
    echo "=== processes ==="
    pgrep -af "extract_both\.py|DiffAct/main\.py" | sed 's/ --.*//;s/^/  /' || echo "  none"
    echo "=== gpu ==="; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
    echo "=== features ==="
    # find|wc always prints exactly one number, including 0 -- `ls|grep -c` does not
    # (grep exits 1 on no match, so a `|| echo 0` guard would print a second zero)
    nf() { find "$1" -maxdepth 1 -name '*.npy' ! -name '*_starts.npy' 2>/dev/null | wc -l; }
    printf "  giant %3d/40   iv2 %3d/40   fused %3d/40\n" \
        "$(nf data_v2/features_copilot)" \
        "$(nf fusion/data/features_iv2_copilot)" \
        "$(nf fusion/data/features_copilot)"
    echo "=== last log lines ==="
    for f in logs/copilot_extract_both.log \
             logs/copilot_post.log logs/copilot_train_step.log logs/copilot_train_type.log; do
        [ -f "$f" ] && printf "  %-34s %s\n" "$(basename $f)" "$(tail -1 "$f" | cut -c1-90)"
    done
    ;;

*) echo "usage: $0 {extract|post|train|status}"; exit 1 ;;
esac
