#!/bin/bash
# Supervised launcher for the LIVE app (same pattern as live/run_demo.sh).
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY=/home/aiops/miniconda3/envs/psr_env/bin/python
SUP=logs/supervisor.log; PIDF=logs/app.pid; PORT=8444
start(){
  if [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "already running"; return; fi
  mkdir -p logs
  setsid nohup bash -c '
    while true; do
      echo "[$(date +%F\ %T)] starting" >> '"$SUP"'
      '"$PY"' server.py >> logs/server.log 2>&1
      echo "[$(date +%F\ %T)] EXITED rc=$? restart in 5s" >> '"$SUP"'
      sleep 5
    done' > /dev/null 2>&1 < /dev/null &
  echo $! > $PIDF; disown 2>/dev/null
  echo "started (models ~90s). dashboard https://localhost:$PORT/  phone https://<LAN-IP>:$PORT/phone"
}
stop(){
  [ -f $PIDF ] && kill -9 "$(cat $PIDF)" 2>/dev/null
  for p in $(ps -eo pid,cmd | grep "live_app/server\.py\|[ ]server\.py" | grep -v grep | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null; done
  rm -f $PIDF; echo stopped
}
case "${1:-status}" in
  start) start;; stop) stop;;
  status) curl -sk -m 3 "https://localhost:$PORT/api/status" && echo || echo DOWN
          tail -2 $SUP 2>/dev/null;;
  log) tail -20 logs/server.log;;
  *) echo "usage: $0 {start|stop|status|log}";;
esac
