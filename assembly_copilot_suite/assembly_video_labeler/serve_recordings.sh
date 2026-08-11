#!/bin/bash
# ============================================================================
#  Start the Assembly Labeler serving the egocentric recordings on the WiFi/LAN.
#  Bakes in the video path so anyone can (re)start it with one command.
#
#    ./serve_recordings.sh          start  -> http://192.168.20.148:7862
#    ./serve_recordings.sh status   is it up?
#    ./serve_recordings.sh stop     stop
#
#  Others on the same WiFi open:   http://192.168.20.148:7862
#  (needs a one-time firewall rule:  sudo ufw allow from 192.168.20.0/24 to any port 7862 proto tcp)
# ============================================================================
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
export LABEL_VIDEOS="/media/lm-ciss/LM_4TB/egocentric_videos/web_app/web_app/recordings"
export LABEL_OUT="$PWD/labels_out"        # labels persist on the 4TB drive
export LABEL_PROXY="$PWD/videos_proxy"    # optional 480p proxies (see make_proxies.sh)
export LABEL_PORT="7862"
export PYTHON="python"                    # miniconda base (has fastapi + uvicorn)
exec ./run.sh "${1:-start}"
