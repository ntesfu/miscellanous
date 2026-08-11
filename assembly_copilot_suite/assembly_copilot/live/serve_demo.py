#!/usr/bin/env python
"""Live demo server: upload a recording, watch step completions appear as they would live.

The demo replays a file rather than reading a camera, but the results are not a
simulation. The head is strictly causal and the decoder is fixed-lag, so the state
emitted for timestep t is computed from t and earlier ONLY -- byte-identical to what
a HoloLens stream would have produced at that moment. Nothing here peeks ahead.

Concretely, each tick:
  1. append the next subsampled frame to a 16-entry ring buffer
  2. once full, form a clip (48 source frames = 1.60 s) and encode it
  3. append the fused 2176-d vector and run the causal TCN over the prefix so far
  4. feed the last timestep's log-probs to the online Viterbi
  5. emit whatever the decoder has now committed, `lag` steps behind the head

Running the TCN over the whole prefix each tick is O(T^2) overall but exactly correct
by prefix-equivalence, and costs nothing next to the encoders (165 ms/clip vs ~5 ms).

    python serve_demo.py [--port 8099]   ->  http://localhost:8099
"""
import argparse, asyncio, json, os, queue, re, sys, threading, time, uuid
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

PSR = ("/media/lm-ciss/LM_4TB/egocentric_videos/ego_psr_repro/industReal/psr_tas")
# NB: do NOT put VideoMAEv2 on sys.path here -- it ships its own top-level `models`
# package which would shadow ours and make `models.causal_tcn` unimportable.
# extract_both.py adds that path itself, inside _boot(), after we are done importing.

from nets.causal_tcn import CausalTCN                                   # noqa: E402
from causal_decode import learn_transitions                               # noqa: E402
from fastapi import FastAPI, UploadFile, File, Request                             # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse  # noqa: E402
import uvicorn                                                            # noqa: E402

DS = os.path.join(PSR, "extern", "DiffAct", "datasets", "Copilot-Fusion")
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
    """Exit before the kernel's OOM killer has to choose a victim.

    The live-encode path grows to ~23 GB of RSS on this 31 GB box. When the kernel
    ran out it killed processes globally -- taking the user's VS Code session down
    along with this server, four times. The growth is still unexplained: decord, each
    model, the decode path and malloc arenas are all flat in isolation.

    RLIMIT_AS is the wrong tool here -- CUDA maps ~48 GB of VIRTUAL address space, so
    an address-space cap fails at init. RSS is what the OOM killer actually scores on,
    so watch that and exit cleanly first. run_demo.sh restarts us, so the blast radius
    is one job instead of the whole desktop.
    """
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
    """Range-capable video response.

    The first version streamed the whole file with a plain 200 and no Content-Length,
    so the browser could neither seek nor size its buffer -- it had to pull all
    231 MB (up to 502 MB) linearly through Python, which reads exactly like a slow
    network. <video> relies on HTTP Range; without 206 replies playback stalls.

    A 480p proxy is preferred when one exists: same fps and frame count, so every
    timestamp the model emits still lines up, at roughly a tenth of the bytes.
    """
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
WORK = queue.Queue()   # (path, job_id) awaiting the single GPU worker

# Jobs run ONE AT A TIME on a single worker thread. Spawning a thread per upload
# starves uvicorn's event loop: each job is a tight Python/PyTorch loop that holds the
# GIL far more than it releases it, so with two or three in flight the server stops
# answering requests entirely -- it stays alive but every endpoint hangs. There is also
# only one GPU, so concurrency would just make each job proportionally slower.


# ----------------------------------------------------------------- model loading


def _boot(ckpt):
    """Load all three networks once. Deferred so the page serves before CUDA warms up."""
    spec_path = os.path.join(PSR, "fusion", "scripts", "extract_both.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("eb", spec_path)
    eb = importlib.util.module_from_spec(spec)
    sys.modules["eb"] = eb
    spec.loader.exec_module(eb)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = [l.split(maxsplit=1)[1].strip() for l in open(os.path.join(DS, "mapping.txt"))]
    giant = eb.build_giant(os.path.join(PSR, "weights", "vit_g_ssv2_ft.pth"))
    iv2 = eb.build_iv2(os.path.join(PSR, "fusion", "weights", "iv2_b14_k710.bin"))
    tcn = CausalTCN(2176, len(classes), 9, 128, 3, 0.25).to(dev)
    tcn.load_state_dict(torch.load(os.path.join(HERE, "runs", "tmse1.0_drop0.25", "final.pt"),
                                   map_location=dev))
    tcn.eval()

    tr_gt = [[l.strip() for l in open(os.path.join(DS, "groundTruth", r[:-4] + ".txt")) if l.strip()]
             for r in open(os.path.join(DS, "splits", "train.split1.bundle")).read().split()]
    logA = learn_transitions(tr_gt, classes)
    logA = logA.copy()
    np.fill_diagonal(logA, np.diag(logA) + SELF_BIAS)

    STATE.update(dict(eb=eb, giant=giant, iv2=iv2, tcn=tcn, dev=dev,
                      classes=classes, logA=logA,
                      mean=eb._MEAN.to(dev), std=eb._STD.to(dev)))
    return STATE


# ----------------------------------------------------------------- online decode
class OnlineViterbi:
    """Fixed-lag causal Viterbi. step() consumes one timestep, returns the committed
    state `lag` ticks back -- the same value sweep_decode.py produces offline."""

    def __init__(self, logA, lag):
        self.A, self.lag = logA, lag
        self.delta, self.bp = None, []

    def step(self, logprob):
        if self.delta is None:
            self.delta = logprob.copy()
            self.bp.append(np.arange(len(logprob)))
            return int(self.delta.argmax())
        m = self.delta[:, None] + self.A
        self.bp.append(m.argmax(0))
        self.delta = m.max(0) + logprob
        self.delta -= self.delta.max()
        s = int(self.delta.argmax())
        for u in range(len(self.bp) - 1, max(len(self.bp) - 1 - self.lag, 0), -1):
            s = int(self.bp[u][s])
        return s


def process_precomputed(rec, job_id):
    """Stream a library recording from its already-extracted 2176-d features.

    The encoders are deterministic and frozen, so replaying stored features produces
    exactly the vectors a live encode would produce for the same clips -- the causal
    head and the online Viterbi are driven identically, one timestep at a time, and
    the emitted events are the same. What it skips is only the (currently
    memory-unstable) re-encoding, which for these 40 recordings is redundant work.

    Uploads still take the live path; this is for the library only.
    """
    q = JOBS[job_id]
    try:
        S = STATE if STATE else _boot(None)
        classes, dev = S["classes"], S["dev"]
        feat = np.load(os.path.join(DS, "features", rec + ".npy"))       # [2176, T]
        T = feat.shape[1]
        q.put(dict(type="meta", frames=T * STRIDE, seconds=T * SEC_PER_STEP, clips=T,
                   classes=classes, lag_s=LAG * SEC_PER_STEP, source="precomputed"))

        vit = OnlineViterbi(S["logA"], LAG)
        x = torch.from_numpy(feat).float().unsqueeze(0).to(dev)
        committed, t0 = None, time.time()
        for t in range(T):
            # strictly causal: only the prefix up to t is ever passed to the model
            with torch.no_grad():
                lg = S["tcn"](x[:, :, :t + 1])[0, :, -1]
            lp = torch.log_softmax(lg, 0).cpu().numpy()
            state = vit.step(lp)
            time.sleep(0.004)   # yield the GIL: this tight loop otherwise starves
                                # uvicorn, which serves video chunks from the same
                                # process -- the player buffers during analysis
            # Two distinct times, and conflating them put every event ~2.9 s late:
            #   the clip's CENTRE is (SPAN/2)/30 = 0.8 s past its start -- the same
            #   convention 03_prepare_diffact.py uses via --center 24; and the state
            #   returned at tick t is the state at t-LAG, because the decoder smooths
            #   with a fixed lag. So the step HAPPENED at (t-LAG), even though we only
            #   learn of it now. `at` is when it happened; `announced_after` is the lag.
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
        # true ring buffer: only the last 16 subsampled frames can ever be needed.
        # `j` counts every frame ever appended -- the clip cadence must key off that,
        # not off len(buf), which is clamped by maxlen.
        from collections import deque
        buf, feats = deque(maxlen=16), []
        committed, t0, done, j = None, time.time(), 0, -1

        # DECODE_CHUNK must stay small. At 1920x1080 each frame is 6.2 MB as uint8 and
        # 24.9 MB as float32, and interpolate() on the permuted (non-contiguous) tensor
        # materialises another copy. A 240-frame chunk therefore peaks near 14 GB, which
        # on this 31 GB box drove the kernel OOM killer -- it killed the server AND the
        # user's VS Code session. 24 frames caps the peak at roughly 1.4 GB.
        for bi in range(0, len(need), DECODE_CHUNK):
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
    # no-store: the UI is a single inline-script page, and a cached stale script
    # silently reproduces already-fixed bugs (events outrunning the video). The
    # page is ~12 KB; re-fetching it every load costs nothing.
    return HTMLResponse(open(os.path.join(HERE, "web", "demo.html")).read(),
                        headers={"Cache-Control": "no-store"})


LIBRARY = "/media/lm-ciss/LM_4TB/assembly_copilot/dataset/prod_dataset"


@app.get("/api/status")
def status():
    return JSONResponse(dict(loaded=bool(STATE), busy=WORK.qsize(),
                             device="cuda" if torch.cuda.is_available() else "cpu"))


@app.get("/api/library")
def library():
    """The 40 annotated recordings, tagged train/test so a demo can be honest about
    whether the model has seen this clip before."""
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
    """Forward hook on clip_projector.cross_attn.attn_drop.

    Same trick, now on the GIANT's last block: its Attention computes attn.softmax
    then immediately attn_drop (identity in eval), so the hook output IS the
    post-softmax attention, (B, heads, N, N). The giant mean-pools (no CLS), so
    per-patch saliency = attention RECEIVED: average over heads and queries of each
    key's weight -- how much the rest of the clip looked at that patch. This is the
    motion (SSv2) encoder, the stream that actually drives step discrimination.
    """
    STATE["sal_attn"] = out.detach()


SAL_QUEUE = queue.Queue()
SAL_GEN = {"n": 0}      # bumping the generation cancels any older track job


def sal_track(path, job_id, gen):
    """Compute the giant's attention map for a window centred every 0.6 s of video,
    streaming each as a {t, grid} event.

    Why a precomputed track instead of the old per-request endpoint: a request per
    poll costs a decode-seek plus a giant forward (~0.5-1.2 s), so the overlay both
    lagged the playhead and snapped between maps. Here the video is decoded ONCE
    sequentially at 224p (decord resizes on decode), windows are batched 8 at a time
    through the giant (~6 maps/s = ~3.6x realtime at this cadence), and the browser
    interpolates between bracketing maps on every animation frame -- the render loop
    never waits on the network or the GPU.
    """
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
                               # which is serving the video the user is WATCHING.
                               # Starve it and playback stalls -- and the timeline is
                               # driven by the video clock, so the whole demo freezes.

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
    """Background part-detection track: one YOLO pass every 0.6 s of video,
    streamed as {t, boxes} events.

    Same architecture as the saliency track and for the same reason: per-request
    detection would put network+GPU latency inside the render loop. The UI buffers
    the track and, driven by the video clock, draws ONE box -- the part the step
    head says is currently being worked on. The detector proposes, the procedure
    disposes: a missed detection means no box this tick, never a wrong part.
    """
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
        # The overlay must reflect the CURRENT frame: long persistence produced
        # ghost boxes after the part left the view. 15 fps + interpolation between
        # adjacent samples is per-frame placement in practice; full 30 fps would
        # compute slower than playback and leave the first steps uncovered.
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
    """Attention map of VideoMAEv2-giant for the clip window centred at time t.

    Computed on demand from the ORIGINAL frames (the precomputed-feature path never
    touches pixels, so this is the only honest source). One giant forward (~180 ms)
    + 16 decoded frames per call; the UI polls ~every 1.5 s. Single-flight via SAL_LOCK -- a busy
    429 is cheaper than queueing GPU work behind a paused player.
    """
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
    """Ground-truth segments for a library recording, for the UI's GT timeline.

    Read from the labeler's segments.csv (part, action, start/end sec) rather than
    re-deriving from PSR labels -- it is the annotator's own segment view. Part names
    are underscored to match the model's class names so the UI can colour-match
    GT and predicted segments per step.
    """
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
    JOBS[job].put(dict(type="queued", ahead=WORK.qsize()))
    # ("precomputed", rec) marks the feature-replay path
    WORK.put(("precomputed", rec, job))
    return JSONResponse(dict(job=job, video=f"/library/{rec}.mp4", ahead=WORK.qsize() - 1))


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
    # Events used to be consumed destructively from the queue: if the connection
    # dropped mid-stream (VS Code port-forwarding does this), the worker drained the
    # job into a dead socket and the browser's automatic reconnect found an empty
    # queue -- the UI went permanently silent while the server looked healthy.
    # Now every event is appended to a replay log and emitted with an SSE id; the
    # browser sends Last-Event-ID on reconnect and the stream resumes exactly there.
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
    """Drain WORK forever, one job at a time."""
    while True:
        kind, arg, job = WORK.get()
        try:
            (process_precomputed if kind == "precomputed" else process)(arg, job)
        except Exception:
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--preload", action="store_true", help="load models at startup")
    a = ap.parse_args()
    if a.preload:
        print("loading models ...", flush=True); _boot(None); print("ready", flush=True)
    _memory_watchdog(10.0)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=sal_worker, daemon=True).start()
    threading.Thread(target=det_worker, daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
