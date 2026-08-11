#!/bin/bash
# 480p playback proxies for the demo UI.
#
# The library videos are 1080p, mean 231 MB, largest 502 MB. That is far more than
# playback needs and it is what makes the player stall. Re-encoding to 480p keeps the
# SAME fps and frame count -- so every completion timestamp the model emits still lines
# up with the picture -- at roughly a tenth of the bytes.
#
# Inference is unaffected: it reads the ORIGINAL files, never these.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
SRC=/media/lm-ciss/LM_4TB/assembly_copilot/dataset/prod_dataset
DST=proxies
JOBS=${JOBS:-6}
FF=$(command -v ffmpeg || /home/aiops/miniconda3/envs/psr_env/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null)
[ -z "$FF" ] && { echo "no ffmpeg"; exit 1; }
mkdir -p $DST
export FF DST
ls $SRC/*.mp4 | xargs -P "$JOBS" -I{} bash -c '
  b=$(basename "{}"); out="$DST/$b"
  [ -f "$out" ] && exit 0
  "$FF" -y -nostdin -loglevel error -i "{}" -vf scale=-2:480 \
    -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p -an \
    -movflags +faststart "$out.part.mp4" && mv "$out.part.mp4" "$out"'
echo "proxies: $(ls $DST/*.mp4 2>/dev/null | wc -l)/40  ($(du -sh $DST | cut -f1))"
