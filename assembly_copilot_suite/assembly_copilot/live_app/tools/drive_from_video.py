#!/usr/bin/env python
"""Pretend to be the phone: stream a recording's frames over the ingest WebSocket
at TRUE camera pace.

This is the honest pre-phone test of the live app. It exercises the entire live
path -- JPEG over the network, decode, encoders, head, decoder, SSE out -- with
the only simulated element being the camera itself. Frames are sent at exactly
10 fps wall-clock (the phone page's rate); if the server cannot keep up, its own
backlog/drop counters will say so.

    python tools/drive_from_video.py --video DA-disAssembled-6 [--seconds 120]
"""
import argparse, asyncio, io, json, os, ssl, sys, time, urllib.request

import numpy as np
import websockets
from PIL import Image

LIB = "/media/lm-ciss/LM_4TB/assembly_copilot/dataset/prod_dataset"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE          # self-signed LAN cert


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="DA-disAssembled-6")
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--host", default="localhost:8444")
    args = ap.parse_args()

    from decord import VideoReader, cpu
    vr = VideoReader(os.path.join(LIB, args.video + ".mp4"), ctx=cpu(0),
                     width=512, height=288, num_threads=2)
    N = min(len(vr), int(args.seconds * 30))
    idx = list(range(0, N, 3))                       # the phone's 10 fps cadence
    print(f"streaming {args.video}: {len(idx)} frames at 10 fps "
          f"({len(idx)/10:.0f}s wall-clock)", flush=True)

    async with websockets.connect(f"wss://{args.host}/ws/ingest", ssl=CTX,
                                  max_size=None) as ws:
        t0 = time.monotonic()
        for k, fi in enumerate(idx):
            fr = vr[fi].asnumpy()
            buf = io.BytesIO()
            Image.fromarray(fr).save(buf, format="JPEG", quality=70)
            await ws.send(buf.getvalue())
            # true camera pacing: frame k belongs at t0 + k/10
            lag = t0 + (k + 1) / 10.0 - time.monotonic()
            if lag > 0:
                await asyncio.sleep(lag)
            elif k % 100 == 0 and lag < -0.5:
                print(f"  driver itself behind by {-lag:.1f}s (encode too slow)", flush=True)
            if k % 300 == 299:
                print(f"  sent {k+1}/{len(idx)} ({(k+1)/10:.0f}s)", flush=True)
        print("driver done", flush=True)
        await asyncio.sleep(3)                       # let the tail flush


if __name__ == "__main__":
    asyncio.run(main())
