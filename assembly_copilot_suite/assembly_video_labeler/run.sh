#!/bin/bash
# ============================================================================
#   Assembly Labeler — standalone launcher (no cluster, no GPU)
#   ./run.sh          start   (then open http://localhost:7862)
#   ./run.sh stop     stop
#   ./run.sh status   is it up?
#
#   Point it at your data with env vars if it's not in ./videos:
#   LABEL_VIDEOS=/path/to/recordings ./run.sh
# ============================================================================
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LABEL_PORT:-7862}"; export LABEL_PORT="$PORT"
export LABEL_VIDEOS="${LABEL_VIDEOS:-$DIR/videos}"
export LABEL_PROXY="${LABEL_PROXY:-$DIR/videos_proxy}"
export LABEL_OUT="${LABEL_OUT:-$DIR/labels_out}"
PY="${PYTHON:-python}"
PID="$DIR/.labeler.pid"
mkdir -p "$LABEL_VIDEOS" "$LABEL_PROXY" "$LABEL_OUT"

stop(){ [ -f "$PID" ] && kill "$(cat "$PID")" 2>/dev/null; pkill -f "$DIR/label_serve.py" 2>/dev/null; rm -f "$PID"; }
status(){ c=$(curl -s -o /dev/null -m 2 -w "%{http_code}" "http://localhost:$PORT/api/config" 2>/dev/null)
  [ "$c" = "200" ] && echo "  UP  → http://localhost:$PORT" || echo "  down (HTTP ${c:-000})"; }

case "${1:-start}" in
  stop)   stop; echo "  stopped."; exit 0 ;;
  status) status; exit 0 ;;
  start)  ;;
  *) echo "usage: ./run.sh [start|stop|status]"; exit 2 ;;
esac

stop >/dev/null 2>&1
echo "  videos : $LABEL_VIDEOS"
echo "  labels : $LABEL_OUT"
nohup "$PY" "$DIR/label_serve.py" > "$DIR/labeler.log" 2>&1 &
echo $! > "$PID"
for i in $(seq 1 20); do
  c=$(curl -s -o /dev/null -m 2 -w "%{http_code}" "http://localhost:$PORT/api/config" 2>/dev/null)
  [ "$c" = "200" ] && break; sleep 0.5
done
echo "=============================================================="
status
echo "  (forward port $PORT if you're on a remote machine)"
echo "=============================================================="
