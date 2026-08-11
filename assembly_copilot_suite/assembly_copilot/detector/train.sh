#!/bin/bash
# Train YOLO11s on the aiops parts dataset.
#
# imgsz=1280: the source photos are 4032x3024 and several parts occupy well under
# 1% of the frame (Propeller Cone Tip averages 0.71%). At the default 640 those
# collapse to a handful of pixels; 1280 keeps them detectable.
#
# The GPU is shared with the live app (~5 GB resident), so batch is kept modest.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY=/home/aiops/miniconda3/envs/psr_env/bin/python
mkdir -p logs runs
setsid nohup $PY -c "
from ultralytics import YOLO
m = YOLO('yolo11s.pt')
m.train(data='cfg/parts.yaml', imgsz=1280, epochs=100, batch=4,
        project='runs', name='parts_y11s', exist_ok=True,
        patience=25, seed=1337, val=True, plots=True, verbose=True)
" > logs/train.log 2>&1 < /dev/null &
echo $! > logs/train.pid
disown 2>/dev/null
echo "training started (pid $(cat logs/train.pid))"
echo "  tail -f /media/lm-ciss/LM_4TB/assembly_copilot/detector/logs/train.log"
