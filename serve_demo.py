#!/usr/bin/env python
"""Replay demo server for step detection on recorded video.

Stored features use the same causal TCN and fixed-lag decoder as live inference;
uploads can be processed through the live encoding path.

The precomputed-feature replay path (process_precomputed / _copilot_follow) never
touches the vision encoders -- it only runs the small TCN over features that are
already on disk -- so it defaults to CPU (see REPLAY_DEVICE / --replay-device)
rather than contending for the GPU with other services (e.g. the HoloLens app).
The upload path (process()), which does need the encoders, is untouched and still
uses CUDA when available. serve_demo_cuda_backup.py preserves the previous
implementation where replay also ran on CUDA, for rollback.
"""
import argparse, asyncio, json, os, queue, re, sys, threading, time, uuid
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_LIB = os.path.join(os.path.dirname(HERE), "copilot_model")
sys.path.insert(0, MODEL_LIB)
# encoders.py adds external model paths inside _boot() to avoid import name clashes.

import config as MCFG                                                     # noqa: E402
from nets.causal_tcn import CausalTCN                                   # noqa: E402
from causal_decode import learn_transitions, OnlineViterbi                # noqa: E402
from fastapi import FastAPI, UploadFile, File, Request                             # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response  # noqa: E402
import uvicorn                                                            # noqa: E402

DS = MCFG.DATA
UPLOADS = os.path.join(HERE, "uploads")
GAP, STRIDE, SPAN = 3, 6, 48
DECODE_CHUNK = 24        # frames decoded+resized at once; see the OOM note in process()
SEC_PER_STEP = STRIDE / 30.0
LAG, SELF_BIAS = 10, 6.0

def _rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1048576
    return 0.0


def _memory_watchdog(limit_gb=10.0, poll=2.0):
    """Monitor RSS and exit before the process reaches the system OOM killer."""
    def loop():
        while True:
            rss = _rss_gb()
            if rss > limit_gb:
                sys.stderr.write(f"\n*** MEMORY WATCHDOG: RSS {rss:.1f} GB > {limit_gb} GB"
                                 f" -- exiting before the OOM killer fires ***\n")
                sys.stderr.flush()
                os._exit(17)
            time.sleep(poll)
    threading.Thread(target=loop, daemon=True).start()
    print(f"memory watchdog armed at {limit_gb} GB RSS", flush=True)


PROXY_DIR = os.path.join(HERE, "proxies")
RANGE_SLICE = 4 << 20          # cap one 206 response at 4 MB


def _serve_video(path, request):
    """Serve video with HTTP Range support, preferring aligned 480p proxies."""
    base = os.path.basename(path)
    prox = os.path.join(PROXY_DIR, base)
    p = prox if os.path.exists(prox) else path
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    size = os.path.getsize(p)
    rng = request.headers.get("range") or request.headers.get("Range")
    if rng:
        m = re.match(r"bytes=(\d+)-(\d*)", rng.strip())
        start = int(m.group(1)) if m else 0
        req_end = int(m.group(2)) if (m and m.group(2)) else size - 1
        end = min(req_end, size - 1, start + RANGE_SLICE - 1)
        length = end - start + 1

        def gen():
            with open(p, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    b = f.read(min(262144, left))
                    if not b:
                        break
                    left -= len(b)
                    yield b
        return StreamingResponse(gen(), status_code=206, media_type="video/mp4",
                                 headers={"Content-Range": f"bytes {start}-{end}/{size}",
                                          "Accept-Ranges": "bytes",
                                          "Content-Length": str(length)})

    def full():
        with open(p, "rb") as f:
            while True:
                b = f.read(262144)
                if not b:
                    break
                yield b
    return StreamingResponse(full(), media_type="video/mp4",
                             headers={"Accept-Ranges": "bytes",
                                      "Content-Length": str(size)})


app = FastAPI()
JOBS = {}          # job_id -> queue of event dicts (producer side)
JOBLOG = {}        # job_id -> list of already-delivered events, for SSE replay


def _new_job():
    job = uuid.uuid4().hex[:12]
    JOBS[job] = queue.Queue()
    JOBLOG[job] = []
    while len(JOBS) > 6:                       # retain a few finished jobs for replay
        old = next(iter(JOBS))
        if old == job:
            break
        JOBS.pop(old, None); JOBLOG.pop(old, None)
    return job
STATE = {}         # models, loaded once
REPLAY_DEVICE = "cpu"   # device for the precomputed-feature replay TCN; overridden
                        # from --replay-device in __main__ before any request can boot it
WORK = queue.Queue()   # (path, job_id) awaiting the single GPU worker
# Only the newest analysis job runs. Older queued jobs are skipped and active jobs
# stop at their next iteration; saliency and detection jobs use separate generations.
LATEST = {"job": None}


def _superseded(job_id, q):
    if LATEST["job"] == job_id:
        return False
    q.put(dict(type="cancelled", message="superseded by a newer analysis"))
    return True

# Analysis jobs share one worker so PyTorch work does not starve uvicorn and compete
# for the single GPU.


# ----------------------------------------------------------------- model loading


def _boot(ckpt):
    """Load all networks once. Deferred so the page serves before CUDA warms up.

    Two TCN instances are kept: `tcn` on `dev` (cuda if available), used by the
    live upload path alongside the vision encoders below; and `tcn_replay`
    pinned to REPLAY_DEVICE (cpu by default), used by the precomputed-feature
    replay path, which never touches the encoders and so does not need the GPU.
    """
    import encoders as eb

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = MCFG.classes()
    giant = eb.build_giant()
    iv2 = eb.build_iv2()
    tcn = CausalTCN(2176, len(classes), 9, 128, 3, 0.25).to(dev)
    # copilot_model/checkpoints/deployed.pt is a SYMLINK naming the model in
    # service (currently robust_v1; see copilot_model/README.md for the swap and
    # rollback procedure).
    tcn.load_state_dict(torch.load(MCFG.DEPLOYED, map_location=dev))
    tcn.eval()

    if REPLAY_DEVICE == dev:
        tcn_replay = tcn                       # same device: no need for a second copy
    else:
        tcn_replay = CausalTCN(2176, len(classes), 9, 128, 3, 0.25).to(REPLAY_DEVICE)
        tcn_replay.load_state_dict(torch.load(MCFG.DEPLOYED, map_location=REPLAY_DEVICE))
        tcn_replay.eval()

    tr_gt = [[l.strip() for l in open(os.path.join(MCFG.GROUND_TRUTH, r[:-4] + ".txt")) if l.strip()]
             for r in open(os.path.join(MCFG.SPLITS, "train.split1.bundle")).read().split()]
    logA = learn_transitions(tr_gt, classes)
    logA = logA.copy()
    np.fill_diagonal(logA, np.diag(logA) + SELF_BIAS)

    STATE.update(dict(eb=eb, giant=giant, iv2=iv2, tcn=tcn, dev=dev,
                      tcn_replay=tcn_replay, replay_dev=REPLAY_DEVICE,
                      classes=classes, logA=logA, ckpt=MCFG.deployed_name(),
                      mean=eb._MEAN.to(dev), std=eb._STD.to(dev)))
    return STATE


def _video_plan(rec, classes):
    """Build the canonical procedure and its transition prior."""
    steps = [c for c in classes if c != "background"]
    plan = steps[::-1] if "disAssembled" in rec else steps
    seq = []
    for s in plan:
        seq += [s] * 150
    logA = learn_transitions([seq], classes).copy()
    np.fill_diagonal(logA, np.diag(logA) + SELF_BIAS)
    return plan, logA


def _copilot_follow(q, job_id, feat, plan, classes, S,
                    margin=0.8, h=12.0, h2=20.0, skip_dwell=12.0):
    """Follow the expected procedure while using model evidence to time each step."""
    T = feat.shape[1]
    tcn, dev = S["tcn_replay"], S["replay_dev"]
    x = torch.from_numpy(feat).float().unsqueeze(0).to(dev)
    idx = {c: i for i, c in enumerate(classes)}
    pos, s1, s2, st1, st2 = 0, 0.0, 0.0, 0.0, 0.0
    t0 = time.time()
    q.put(dict(type="enter", step=plan[0], at=0.0, idx=idx[plan[0]]))
    for t in range(T):
        if t % 10 == 0 and _superseded(job_id, q):
            return
        with torch.no_grad():
            lg = tcn(x[:, :, :t + 1])[0, :, -1]
        lp = torch.log_softmax(lg, 0).cpu().numpy()
        time.sleep(0.004)                     # yield the GIL (see the free loop)
        t_video = t * SEC_PER_STEP + (SPAN / 2) / 30.0   # clip centre; no decode lag
        now_s = time.time() - t0
        if pos < len(plan) - 1:
            cur, n1 = idx[plan[pos]], idx[plan[pos + 1]]
            inc1 = lp[n1] - lp[cur] - margin
            if s1 <= 0 and inc1 > 0:
                st1 = t_video
            s1 = max(0.0, s1 + inc1)
            advanced = False
            if s1 > h:
                q.put(dict(type="complete", step=plan[pos], at=round(st1, 2),
                           announced_after=round(max(0.0, t_video - st1), 2)))
                pos += 1
                q.put(dict(type="enter", step=plan[pos], at=round(st1, 2),
                           idx=idx[plan[pos]]))
                s1 = s2 = 0.0
                advanced = True
            elif pos + 2 < len(plan):
                n2 = idx[plan[pos + 2]]
                inc2 = lp[n2] - lp[cur] - margin
                if s2 <= 0 and inc2 > 0:
                    st2 = t_video
                s2 = max(0.0, s2 + inc2)
                if s2 > h2 and t_video - st2 >= skip_dwell:
                    q.put(dict(type="complete", step=plan[pos], at=round(st2, 2),
                               announced_after=round(max(0.0, t_video - st2), 2)))
                    q.put(dict(type="skipped", step=plan[pos + 1], at=round(st2, 2)))
                    pos += 2
                    q.put(dict(type="enter", step=plan[pos], at=round(st2, 2),
                               idx=idx[plan[pos]]))
                    s1 = s2 = 0.0
        if t % 5 == 0 or t == T - 1:
            el = time.time() - t0
            q.put(dict(type="tick", done=t + 1, total=T, t_video=round(t_video, 2),
                       rate=round((t + 1) / max(el, 1e-6), 2),
                       realtime=round(((t + 1) * SEC_PER_STEP) / max(el, 1e-6), 2),
                       current=plan[pos], conf=round(float(np.exp(lp.max())), 3)))
    q.put(dict(type="complete", step=plan[pos], at=round(T * SEC_PER_STEP, 2),
               announced_after=0.0))
    q.put(dict(type="done", elapsed=round(time.time() - t0, 1)))


def process_precomputed(rec, job_id, mode="free"):
    """Replay precomputed features through the live causal head and decoder.

    Runs on S["replay_dev"] (CPU by default, see REPLAY_DEVICE) since the vision
    encoders are never touched here -- the features are already extracted, so the
    GPU is not required for this path.
    """
    q = JOBS[job_id]
    try:
        S = STATE if STATE else _boot(None)
        classes, dev, tcn = S["classes"], S["replay_dev"], S["tcn_replay"]
        feat = np.load(os.path.join(DS, "features", rec + ".npy"))       # [2176, T]
        T = feat.shape[1]
        plan, planA = _video_plan(rec, classes)
        q.put(dict(type="meta", frames=T * STRIDE, seconds=T * SEC_PER_STEP, clips=T,
                   classes=classes, lag_s=LAG * SEC_PER_STEP, source="precomputed",
                   plan=plan, mode=mode if plan else "free", device=dev))

        # Free mode reports model order; copilot mode follows the plan and uses the
        # model only to time planned steps.
        if mode == "copilot" and plan and len(set(plan)) == len(plan):
            _copilot_follow(q, job_id, feat, plan, classes, S)
            return
        vit = OnlineViterbi(planA if planA is not None else S["logA"], LAG)
        x = torch.from_numpy(feat).float().unsqueeze(0).to(dev)
        committed, t0 = None, time.time()
        for t in range(T):
            if t % 10 == 0 and _superseded(job_id, q):
                return
            # strictly causal: only the prefix up to t is ever passed to the model
            with torch.no_grad():
                lg = tcn(x[:, :, :t + 1])[0, :, -1]
            lp = torch.log_softmax(lg, 0).cpu().numpy()
            state = vit.step(lp)
            time.sleep(0.004)   # yield the GIL: this tight loop otherwise starves
                                # Yield to uvicorn while it serves video chunks.
            # `at` is event time: clip centre minus decoder lag. `announced_after`
            # records the delay until that event was emitted.
            t_video = max(0.0, (t - LAG) * SEC_PER_STEP + (SPAN / 2) / 30.0)
            if state != committed:
                if committed is not None and classes[committed] != "background":
                    q.put(dict(type="complete", step=classes[committed],
                               at=round(t_video, 2), idx=committed,
                               announced_after=round(LAG * SEC_PER_STEP, 2)))
                committed = state
                q.put(dict(type="enter", step=classes[state], at=round(t_video, 2), idx=state))
            if t % 5 == 0 or t == T - 1:
                el = time.time() - t0
                q.put(dict(type="tick", done=t + 1, total=T, t_video=round(t_video, 2),
                           rate=round((t + 1) / max(el, 1e-6), 2),
                           realtime=round(((t + 1) * SEC_PER_STEP) / max(el, 1e-6), 2),
                           current=classes[committed] if committed is not None else "-",
                           conf=round(float(np.exp(lp.max())), 3)))
        if committed is not None and classes[committed] != "background":
            q.put(dict(type="complete", step=classes[committed],
                       at=round(T * SEC_PER_STEP, 2), idx=committed))
        q.put(dict(type="done", elapsed=round(time.time() - t0, 1)))
    except Exception as e:
        import traceback
        q.put(dict(type="error", message=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc()[-800:]))
    finally:
        q.put(None)


# ----------------------------------------------------------------- the worker
def process(path, job_id):
    q = JOBS[job_id]
    try:
        S = STATE if STATE else _boot(None)
        eb, dev = S["eb"], S["dev"]
        classes, mean, std = S["classes"], S["mean"], S["std"]
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0), num_threads=8)
        N = len(vr)
        need = list(range(0, N, GAP))
        n_clips = max(0, (len(need) - 16) // (STRIDE // GAP) + 1)
        q.put(dict(type="meta", frames=N, seconds=N / 30.0, clips=n_clips,
                   classes=classes, lag_s=LAG * SEC_PER_STEP))

        vit = OnlineViterbi(S["logA"], LAG)
        # Keep only the last 16 frames; use `j` for cadence because the buffer is capped.
        from collections import deque
        buf, feats = deque(maxlen=16), []
        committed, t0, done, j = None, time.time(), 0, -1

        # Keep decode chunks small because interpolation creates additional frame copies.
        for bi in range(0, len(need), DECODE_CHUNK):
            if _superseded(job_id, q):
                return
            chunk = vr.get_batch(need[bi:bi + DECODE_CHUNK]).asnumpy()
            x = torch.from_numpy(chunk).permute(0, 3, 1, 2).float().div_(255.0)
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                                align_corners=False).half()
            del chunk
            for f in x:
                buf.append(f); j += 1
                # clip s spans buffer entries s..s+15 and starts every STRIDE/GAP=2
                if j < 15 or (j - 15) % (STRIDE // GAP):
                    continue
                clip = torch.stack(list(buf)).unsqueeze(0).to(dev)
                clip = clip.permute(0, 2, 1, 3, 4).float()
                clip = (clip - mean) / std
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                    g = S["giant"].forward_features(clip)
                    v = S["iv2"](clip[:, :, ::2])
                gn = torch.nn.functional.normalize(g.float(), dim=1)
                vn = torch.nn.functional.normalize(v.float(), dim=1)
                feats.append(torch.cat([gn, vn], 1)[0])

                seq = torch.stack(feats, 1).unsqueeze(0)          # [1, 2176, t]
                with torch.no_grad():
                    lg = S["tcn"](seq)[0, :, -1]
                lp = torch.log_softmax(lg, 0).cpu().numpy()
                state = vit.step(lp)

                done += 1
                t_video = max(0.0, (done - 1 - LAG) * SEC_PER_STEP + (SPAN / 2) / 30.0)
                if state != committed:
                    if committed is not None and classes[committed] != "background":
                        q.put(dict(type="complete", step=classes[committed],
                                   at=round(t_video, 2), idx=committed))
                    committed = state
                    q.put(dict(type="enter", step=classes[state], at=round(t_video, 2),
                               idx=state))
                if done % 5 == 0 or done == n_clips:
                    el = time.time() - t0
                    q.put(dict(type="tick", done=done, total=n_clips,
                               t_video=round(t_video, 2),
                               rate=round(done / max(el, 1e-6), 2),
                               realtime=round((done * SEC_PER_STEP) / max(el, 1e-6), 2),
                               current=classes[committed] if committed is not None else "-",
                               conf=round(float(np.exp(lp.max())), 3)))
        if committed is not None and classes[committed] != "background":
            q.put(dict(type="complete", step=classes[committed],
                       at=round(N / 30.0, 2), idx=committed))
        q.put(dict(type="done", elapsed=round(time.time() - t0, 1)))
    except Exception as e:                                    # surface, never hang the UI
        import traceback
        q.put(dict(type="error", message=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc()[-800:]))
    finally:
        q.put(None)


# ----------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def index():
    # Always serve the inline script fresh so clients do not use stale UI logic.
    return HTMLResponse(open(os.path.join(HERE, "web", "demo.html")).read(),
                        headers={"Cache-Control": "no-store"})


LIBRARY = MCFG.LIBRARY


@app.get("/assets/{name}")
def asset(name: str):
    """Serve a static branding asset."""
    p = os.path.join(HERE, "web", "assets", os.path.basename(name))
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    ext = os.path.splitext(p)[1].lower()
    ctype = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".png": "image/png", ".svg": "image/svg+xml"}.get(ext, "application/octet-stream")
    return Response(open(p, "rb").read(), media_type=ctype,
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/status")
def status():
    return JSONResponse(dict(loaded=bool(STATE), busy=WORK.qsize(),
                             model=STATE.get("ckpt", "loading"),
                             device=STATE.get("replay_dev", REPLAY_DEVICE),
                             encoder_device="cuda" if torch.cuda.is_available() else "cpu"))


@app.get("/api/library")
def library():
    """Return annotated recordings tagged by train/test split."""
    out = []
    for split in ("test", "train"):
        d = os.path.join(LIBRARY, "recordings", split)
        if not os.path.isdir(d):
            continue
        for rec in sorted(os.listdir(d)):
            if os.path.exists(os.path.join(LIBRARY, rec + ".mp4")):
                out.append(dict(name=rec, split=split,
                                dir="disassembly" if "disAssembled" in rec else "assembly"))
    return JSONResponse(out)


SAL_LOCK = threading.Lock()


def _sal_hook(module, inp, out):
    """Capture post-softmax attention from the giant encoder for patch saliency."""
    STATE["sal_attn"] = out.detach()


SAL_QUEUE = queue.Queue()
SAL_GEN = {"n": 0}      # bumping the generation cancels any older track job


def sal_track(path, job_id, gen):
    """Precompute attention maps on a fixed time grid and stream them to the UI."""
    q = JOBS[job_id]
    try:
        S = STATE if STATE else _boot(None)
        dev = S["dev"]
        if not S.get("sal_hooked"):
            S["giant"].blocks[-1].attn.attn_drop.register_forward_hook(_sal_hook)
            S["sal_hooked"] = True
        from decord import VideoReader, cpu
        from collections import deque
        vr = VideoReader(path, ctx=cpu(0), width=224, height=224, num_threads=4)
        N = len(vr)
        need = list(range(0, N, GAP))
        CAD = 6                     # buffer entries between map centres = 0.6 s of video
        est = max(0, (len(need) - 16) // CAD + 1)
        q.put(dict(type="sal_meta", count=est, cadence=CAD * GAP / 30.0))

        buf, j = deque(maxlen=16), -1
        wins, tcs = [], []

        def flush():
            nonlocal wins, tcs
            if not wins:
                return
            x = torch.stack(wins).to(dev).permute(0, 2, 1, 3, 4)   # (B,3,16,224,224)
            x = (x - S["mean"]) / S["std"]
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                S["giant"].forward_features(x)
            a = STATE.pop("sal_attn").float()      # (B, heads, Nq, Nk)
            a = a.mean(1).mean(1)                  # heads, then queries -> received per key
            P = 8 * 16 * 16
            if a.shape[1] == P + 1:
                a = a[:, 1:]
            g = a.reshape(-1, 8, 16, 16).mean(1).cpu().numpy()
            for tc, gr in zip(tcs, g):
                q.put(dict(type="sal", t=round(tc, 2),
                           grid=[round(float(v), 6) for v in gr.flatten()]))
            wins, tcs = [], []
            time.sleep(0.12)   # GIL air: this loop shares the process with uvicorn,
                               # Keep uvicorn responsive while the user watches playback.

        for bi in range(0, len(need), 48):
            if SAL_GEN["n"] != gen:
                break
            fr = vr.get_batch(need[bi:bi + 48]).asnumpy()
            xb = torch.from_numpy(fr).permute(0, 3, 1, 2).float().div_(255.0)
            for f in xb:
                buf.append(f); j += 1
                if j < 15 or (j - 15) % CAD:
                    continue
                wins.append(torch.stack(list(buf)))
                time.sleep(0.002)                  # per-window GIL yield
                tcs.append(((j - 15) * GAP + SPAN / 2) / 30.0)
                if len(wins) >= 8:
                    flush()
        if SAL_GEN["n"] == gen:
            flush()
        q.put(dict(type="sal_done"))
    except Exception as e:
        import traceback
        q.put(dict(type="error", message=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc()[-800:]))
    finally:
        q.put(None)


def sal_worker():
    while True:
        path, job, gen = SAL_QUEUE.get()
        if SAL_GEN["n"] != gen:                   # superseded while still queued
            if job in JOBS:
                JOBS[job].put(None)
            continue
        try:
            sal_track(path, job, gen)
        except Exception:
            import traceback; traceback.print_exc()


@app.get("/api/saliency_track")
def saliency_track(video: str):
    base = os.path.basename(video)
    for root in (LIBRARY, UPLOADS):
        p = os.path.join(root, base)
        if os.path.exists(p):
            break
    else:
        return JSONResponse({"error": "not found"}, status_code=404)
    SAL_GEN["n"] += 1
    job = _new_job()
    SAL_QUEUE.put((p, job, SAL_GEN["n"]))
    return JSONResponse(dict(job=job))


DET_QUEUE = queue.Queue()
DET_GEN = {"n": 0}
DET_WEIGHTS = ("/media/lm-ciss/LM_4TB/assembly_copilot/detector/"
               "runs/detect/runs/parts_y11s_v2/weights/best.pt")


def det_track(path, job_id, gen):
    """Precompute part detections on a fixed time grid for the UI overlay."""
    q = JOBS[job_id]
    try:
        if "yolo" not in STATE:
            from ultralytics import YOLO
            STATE["yolo"] = YOLO(DET_WEIGHTS)
        yolo = STATE["yolo"]
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0), num_threads=4)
        N = len(vr)
        CADF = 2                                   # every 2nd frame (15 fps).
        # Sample at 15 fps and interpolate so boxes follow the current frame without
        # making detection slower than playback.
        idxs = list(range(0, N, CADF))
        q.put(dict(type="det_meta", count=len(idxs), cadence=CADF / 30.0,
                   names=[yolo.names[i] for i in sorted(yolo.names)]))
        B = 24
        for bi in range(0, len(idxs), B):
            if DET_GEN["n"] != gen:
                break
            frames = list(vr.get_batch(idxs[bi:bi + B]).asnumpy())
            res = yolo.predict(frames, imgsz=1280, conf=0.30, verbose=False)
            for k, r in enumerate(res):
                boxes = [dict(c=int(c), s=round(float(cf), 3),
                              b=[round(v, 4) for v in bx])
                         for c, cf, bx in zip(r.boxes.cls, r.boxes.conf,
                                              r.boxes.xyxyn.tolist())]
                q.put(dict(type="det", t=round(idxs[bi + k] / 30.0, 2), boxes=boxes))
            time.sleep(0.05)                       # GIL air for uvicorn
        q.put(dict(type="det_done"))
    except Exception as e:
        import traceback
        q.put(dict(type="error", message=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc()[-800:]))
    finally:
        q.put(None)


def det_worker():
    while True:
        path, job, gen = DET_QUEUE.get()
        if DET_GEN["n"] != gen:
            if job in JOBS:
                JOBS[job].put(None)
            continue
        try:
            det_track(path, job, gen)
        except Exception:
            import traceback; traceback.print_exc()


@app.get("/api/detect_track")
def detect_track(video: str):
    base = os.path.basename(video)
    for root in (LIBRARY, UPLOADS):
        p = os.path.join(root, base)
        if os.path.exists(p):
            break
    else:
        return JSONResponse({"error": "not found"}, status_code=404)
    DET_GEN["n"] += 1
    job = _new_job()
    DET_QUEUE.put((p, job, DET_GEN["n"]))
    return JSONResponse(dict(job=job))


@app.get("/api/saliency")
def saliency(video: str, t: float):
    """Compute an attention map for the original clip centred at ``t``."""
    if not STATE:
        return JSONResponse({"error": "models not loaded"}, status_code=503)
    if not SAL_LOCK.acquire(blocking=False):
        return JSONResponse({"busy": True}, status_code=429)
    try:
        base = os.path.basename(video)
        for root in (LIBRARY, UPLOADS):
            p = os.path.join(root, base)
            if os.path.exists(p):
                break
        else:
            return JSONResponse({"error": "not found"}, status_code=404)
        if not STATE.get("sal_hooked"):
            STATE["giant"].blocks[-1].attn.attn_drop.register_forward_hook(_sal_hook)
            STATE["sal_hooked"] = True

        from decord import VideoReader, cpu
        vr = VideoReader(p, ctx=cpu(0), num_threads=2)
        N = len(vr)
        c = int(t * 30)
        # 8 frames spanning the 1.6 s window CENTRED at t -- the same convention the
        # labels use (--center 24), so the map matches what the model sees "now"
        idx = np.clip(np.arange(c - SPAN // 2, c + SPAN // 2, GAP), 0, N - 1)
        fr = vr.get_batch(idx).asnumpy()
        x = torch.from_numpy(fr).permute(0, 3, 1, 2).float().div_(255.0)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                            align_corners=False)
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4).to(STATE["dev"])
        x = (x - STATE["mean"]) / STATE["std"]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(STATE["dev"] == "cuda")):
            STATE["giant"].forward_features(x)
        a = STATE.pop("sal_attn", None)
        if a is None:
            return JSONResponse({"error": "hook captured nothing"}, status_code=500)
        a = a.float().mean(1)[0].mean(0)         # heads, then queries -> received per key
        P = 8 * 16 * 16                          # (T/tubelet, H/14, W/14) tokens
        if a.numel() == P + 1:
            a = a[1:]                            # drop cls token if present
        if a.numel() != P:
            return JSONResponse({"error": f"unexpected token count {a.numel()}"},
                                status_code=500)
        g = a.reshape(8, 16, 16).mean(0).cpu().numpy()   # average time -> 16x16
        return JSONResponse({"t": t, "gw": 16, "gh": 16,
                             "grid": [round(float(v), 6) for v in g.flatten()]})
    finally:
        SAL_LOCK.release()


@app.get("/api/gt/{name}")
def gt(name: str):
    """Return annotated segments with labels aligned to model class names."""
    import csv as _csv
    import glob as _glob
    rec = os.path.basename(name)
    hits = _glob.glob(os.path.join(LIBRARY, "recordings", "*", rec, "segments.csv"))
    if not hits:
        return JSONResponse({"segments": None})
    segs = []
    with open(hits[0]) as f:
        for row in _csv.DictReader(f):
            segs.append(dict(step=row["part"].replace(" ", "_"), action=row["action"],
                             start=float(row["start_sec"]), end=float(row["end_sec"])))
    return JSONResponse({"segments": segs})


@app.post("/api/run/{name}")
def run_library(name: str):
    rec = os.path.basename(name)
    path = os.path.join(LIBRARY, rec + ".mp4")
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    job = _new_job()
    LATEST["job"] = job
    JOBS[job].put(dict(type="queued", ahead=WORK.qsize()))
    # ("precomputed", (rec, mode)) marks the feature-replay path
    WORK.put(("precomputed", rec, job))
    return JSONResponse(dict(job=job, video=f"/library/{rec}.mp4",
                             ahead=WORK.qsize() - 1))


@app.get("/library/{name}")
def library_media(name: str, request: Request):
    return _serve_video(os.path.join(LIBRARY, os.path.basename(name)), request)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(UPLOADS, exist_ok=True)
    job = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    path = os.path.join(UPLOADS, job + ext)
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    JOBS[job] = queue.Queue()
    JOBLOG[job] = []
    LATEST["job"] = job
    JOBS[job].put(dict(type="queued", ahead=WORK.qsize()))
    WORK.put(("live", path, job))
    return JSONResponse(dict(job=job, video=f"/media/{os.path.basename(path)}",
                             ahead=WORK.qsize() - 1))


@app.get("/media/{name}")
def media(name: str, request: Request):
    return _serve_video(os.path.join(UPLOADS, os.path.basename(name)), request)


@app.get("/api/events/{job}")
async def events(job: str, request: Request):
    q = JOBS.get(job)
    if q is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    elog = JOBLOG.setdefault(job, [])
    # Keep a replay log so reconnecting clients can resume with Last-Event-ID.
    try:
        start = int(request.headers.get("last-event-id", "-1")) + 1
    except ValueError:
        start = 0

    async def gen():
        loop = asyncio.get_event_loop()
        i = start
        while True:
            if i < len(elog):
                item = elog[i]
                if item is None:
                    break
                yield f"id: {i}\ndata: {json.dumps(item)}\n\n"
                i += 1
                continue
            try:
                item = await loop.run_in_executor(None, q.get, True, 1.0)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            elog.append(item)          # drain into the log; emitted by the branch above
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def worker():
    """Drain WORK forever, one job at a time; stale jobs are skipped unrun."""
    while True:
        kind, arg, job = WORK.get()
        q = JOBS.get(job)
        if q is not None and _superseded(job, q):
            q.put(None)
            continue
        try:
            if kind == "precomputed":
                rec, mode = arg if isinstance(arg, tuple) else (arg, "free")
                process_precomputed(rec, job, mode)
            else:
                process(arg, job)
        except Exception:
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--preload", action="store_true", help="load models at startup")
    ap.add_argument("--replay-device", default="cpu", choices=["cpu", "cuda", "auto"],
                    help="device for the precomputed-feature replay TCN (default: cpu -- "
                         "replay never runs the vision encoders, so the GPU is not needed; "
                         "pass cuda or auto to roll back to GPU-backed replay)")
    a = ap.parse_args()
    if a.replay_device == "auto":
        REPLAY_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    elif a.replay_device == "cuda" and not torch.cuda.is_available():
        print("--replay-device cuda requested but CUDA is unavailable; using cpu", flush=True)
        REPLAY_DEVICE = "cpu"
    else:
        REPLAY_DEVICE = a.replay_device
    if a.preload:
        print(f"loading models ... (replay device: {REPLAY_DEVICE})", flush=True)
        _boot(None)
        print("ready", flush=True)
    _memory_watchdog(10.0)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=sal_worker, daemon=True).start()
    threading.Thread(target=det_worker, daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
