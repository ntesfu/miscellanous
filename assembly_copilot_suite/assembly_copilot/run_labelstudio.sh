#!/bin/bash
# Label Studio for turbofan part bounding boxes.
#
# Isolated venv on purpose: label-studio pulls Django and a large dependency tree,
# and psr_env is running the live app -- installing into it risked breaking a
# working system to add a labelling tool.
#
#   ./run_labelstudio.sh start|stop|status
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
LS=./labelenv/bin/label-studio
PORT=8080
PIDF=logs_ls.pid
mkdir -p detection/images
case "${1:-start}" in
  start)
    if [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "already running"; exit 0; fi
    # LOCAL_FILES_* lets a project reference images straight off this disk instead
    # of uploading copies -- important when the frames are large or numerous.
    export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/detection"
    setsid nohup $LS start --port $PORT --host "http://0.0.0.0:$PORT" \
        > label_studio.log 2>&1 < /dev/null &
    echo $! > $PIDF; disown 2>/dev/null
    echo "starting on http://localhost:$PORT (first run takes ~30s to init its DB)"
    ;;
  stop)
    [ -f $PIDF ] && kill -9 "$(cat $PIDF)" 2>/dev/null
    for p in $(ps -eo pid,cmd | grep "label-studio" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
    rm -f $PIDF; echo stopped ;;
  status)
    curl -s -m 3 -o /dev/null -w "HTTP %{http_code}\n" http://localhost:$PORT/ 2>/dev/null || echo DOWN ;;
esac
