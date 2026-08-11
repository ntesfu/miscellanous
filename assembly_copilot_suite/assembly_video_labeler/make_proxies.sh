#!/bin/bash
# ============================================================================
#   Build lightweight 480p proxies for smooth scrubbing (OPTIONAL).
#   Only needed if labelling high-res video feels laggy. Proxies keep the same
#   fps + duration, so frame numbers map 1:1 to the originals (labels stay valid).
#
#   ./make_proxies.sh
#   Uses system ffmpeg, or `pip install imageio-ffmpeg` for a bundled one.
# ============================================================================
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${LABEL_VIDEOS:-$DIR/videos}"
DST="${LABEL_PROXY:-$DIR/videos_proxy}"
PAR="${PROXY_JOBS:-4}"
mkdir -p "$DST"
FF="$(command -v ffmpeg 2>/dev/null || ${PYTHON:-python} -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null)"
[ -z "$FF" ] && { echo "no ffmpeg found — install it or: pip install imageio-ffmpeg"; exit 1; }
echo "ffmpeg: $FF"; echo "src: $SRC  ->  dst: $DST  (parallel=$PAR)"
export FF DST
ls "$SRC"/*.mp4 "$SRC"/*.mov "$SRC"/*.MP4 "$SRC"/*.MOV 2>/dev/null | sort -u | xargs -P "$PAR" -I{} bash -c '
  f="{}"; b=$(basename "$f"); out="$DST/$b"
  [ -f "$out" ] && { echo "skip $b"; exit 0; }
  "$FF" -y -nostdin -loglevel error -i "$f" -vf "scale=-2:480" \
    -c:v libx264 -preset veryfast -crf 26 -g 30 -pix_fmt yuv420p -an \
    -movflags +faststart "$out.part.mp4" && mv "$out.part.mp4" "$out" && echo "done $b"'
echo "proxies ready in $DST ($(ls "$DST"/*.mp4 2>/dev/null | wc -l) files)"
