#!/bin/bash
# Supervised launcher for the demo server.
#
# The server kept vanishing with no traceback -- it reaches "ready", serves for a
# while, then the process is simply gone. Rather than guess, this wrapper records the
# exit status every time and brings it straight back up, so the UI stays reachable and
# the cause is captured in logs/supervisor.log instead of being lost.
#
#   ./run_demo.sh start | stop | status | log
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY=/home/aiops/miniconda3/envs/psr_env/bin/python
PORT=8099
SUP=logs/supervisor.log
PIDF=logs/demo.pid

start() {
    if [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
        echo "already running (pid $(cat $PIDF))"; return
    fi
    mkdir -p logs
    setsid nohup bash -c '
        while true; do
            echo "[$(date +%F\ %T)] starting server" >> '"$SUP"'
            '"$PY"' serve_demo.py --port '"$PORT"' --preload >> logs/demo_server.log 2>&1
            echo "[$(date +%F\ %T)] server EXITED rc=$? -- restarting in 5s" >> '"$SUP"'
            sleep 5
        done' > /dev/null 2>&1 < /dev/null &
    echo $! > $PIDF
    disown 2>/dev/null
    echo "supervisor started (pid $(cat $PIDF)); models take ~90s to load"
}

stop() {
    [ -f $PIDF ] && kill -9 "$(cat $PIDF)" 2>/dev/null
    for p in $(ps -eo pid,cmd | grep "serve_demo\.py" | grep -v grep | awk '{print $1}'); do
        kill -9 "$p" 2>/dev/null
    done
    rm -f $PIDF
    echo stopped
}

case "${1:-status}" in
  start) start ;;
  stop)  stop ;;
  log)   tail -20 $SUP ;;
  status)
    if curl -s -m 3 "localhost:$PORT/api/status" 2>/dev/null | grep -q loaded; then
        echo "UP   $(curl -s -m 3 localhost:$PORT/api/status)"
    else
        echo "DOWN (or still loading models)"
    fi
    ps -eo pid,etime,cmd | grep "serve_demo\.py" | grep -v grep | awk '{print "  server pid",$1,"up",$2}'
    [ -f $SUP ] && echo "  last supervisor events:" && tail -3 $SUP | sed 's/^/    /'
    ;;
  *) echo "usage: $0 {start|stop|status|log}" ;;
esac
