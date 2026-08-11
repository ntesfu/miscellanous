#!/usr/bin/env python
"""Live Assembly Copilot — a phone camera in, step events out, for real.

This is the genuinely-live test the demo could not be: frames arrive over the
network as they are captured, the model runs behind the camera, and every number
shown (latency included) is measured, not replayed.

Transport (adapted from the ego-recorder reference app that captured the dataset):
  * HTTPS with its self-signed certs -- iOS refuses getUserMedia without TLS.
  * The PHONE is the camera: /phone captures at 512x288 and sends JPEG frames over
    a WebSocket at 10 fps. That rate is chosen to be the model's own cadence
    (every 3rd frame of 30 fps), so the phone does the temporal subsampling and
    the uplink is ~400 KB/s instead of a video stream.
  * The dashboard (/) watches the same session: live preview, current step,
    completion timeline, and the honest throughput/latency numbers.

Inference is the same stack validated offline: frozen giant+B14 -> 2176-d fusion
-> causal TCN (prefix-equivalent) -> online fixed-lag Viterbi. The TCN input is
windowed to its receptive field (1023 steps), so per-tick cost is constant.

    ./run.sh start      ->  https://<this-box>:8444/        (dashboard)
                            https://<this-box>:8444/phone   (open on the phone)
"""
import asyncio, io, json, os, queue, sys, threading, time
from collections import deque

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = "/media/lm-ciss/LM_4TB/assembly_copilot/live"
PSR = "/media/lm-ciss/LM_4TB/egocentric_videos/ego_psr_repro/industReal/psr_tas"
DS = os.path.join(PSR, "extern", "DiffAct", "datasets", "Copilot-Fusion")
CERTS = "/media/lm-ciss/LM_4TB/aiops/web_app/web_app/certs"     # reference app's certs
sys.path.insert(0, LIVE)
sys.path.insert(0, os.path.join(LIVE, "scripts"))

from nets.causal_tcn import CausalTCN                          # noqa: E402
from causal_decode import learn_transitions                    # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request  # noqa: E402
from fastapi.responses import (HTMLResponse, JSONResponse,     # noqa: E402
                               Response, StreamingResponse)
import uvicorn                                                 # noqa: E402

GAP, SPAN, LAG, SELF_BIAS = 3, 48, 10, 6.0
SEC_PER_TICK = 0.2            # one decoder tick = 2 ingest frames at 10 fps
TCN_WINDOW = 1024             # >= receptive field (1023): constant per-tick cost
PORT = int(os.environ.get("LIVE_PORT", "8444"))

app = FastAPI()
STATE = {}
FRAMES = queue.Queue()        # jpeg bytes from the phone (or the driver)
LATEST = {"jpg": None}
EVLOG = []                    # global replayable event log (SSE ids = indices)
EVCV = threading.Condition()
RESET = object()              # sentinel: start a fresh session
STOP = object()               # sentinel: end the session, flush the final step
ARMED = {"on": False}         # the phone may stream anytime; the model runs only
                              # while a session is armed from the dashboard


def emit(ev):
    with EVCV:
        EVLOG.append(ev)
        if len(EVLOG) > 20000:          # ring: drop the oldest half
            del EVLOG[:10000]
        EVCV.notify_all()


def _rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1048576
    return 0.0


def _watchdog(limit=10.0):
    def loop():
        while True:
            if _rss_gb() > limit:
                sys.stderr.write("*** watchdog: RSS over limit, exiting ***\n")
                os._exit(17)
            time.sleep(2)
    threading.Thread(target=loop, daemon=True).start()


class OnlineViterbi:
    """Fixed-lag causal Viterbi (same as the demo's, validated against offline)."""

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


def _boot():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eb", os.path.join(PSR, "fusion", "scripts", "extract_both.py"))
    eb = importlib.util.module_from_spec(spec)
    sys.modules["eb"] = eb
    spec.loader.exec_module(eb)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = [l.split(maxsplit=1)[1].strip() for l in open(os.path.join(DS, "mapping.txt"))]
    giant = eb.build_giant(os.path.join(PSR, "weights", "vit_g_ssv2_ft.pth"))
    iv2 = eb.build_iv2(os.path.join(PSR, "fusion", "weights", "iv2_b14_k710.bin"))
    tcn = CausalTCN(2176, len(classes), 9, 128, 3, 0.25).to(dev)
    tcn.load_state_dict(torch.load(os.path.join(LIVE, "runs", "tmse1.0_drop0.25", "final.pt"),
                                   map_location=dev))
    tcn.eval()
    tr_gt = [[l.strip() for l in open(os.path.join(DS, "groundTruth", r[:-4] + ".txt")) if l.strip()]
             for r in open(os.path.join(DS, "splits", "train.split1.bundle")).read().split()]
    logA = learn_transitions(tr_gt, classes).copy()
    np.fill_diagonal(logA, np.diag(logA) + SELF_BIAS)
    STATE.update(dict(giant=giant, iv2=iv2, tcn=tcn, dev=dev, classes=classes,
                      logA=logA, mean=eb._MEAN.to(dev), std=eb._STD.to(dev)))


def worker():
    _boot()
    emit(dict(type="ready", classes=STATE["classes"]))
    dev, classes = STATE["dev"], STATE["classes"]
    mean, std = STATE["mean"], STATE["std"]

    def fresh():
        return dict(buf=deque(maxlen=16), j=-1, feats=[], vit=OnlineViterbi(STATE["logA"], LAG),
                    committed=None, t0=None, clips=0, drops=0, ema=None)

    s = fresh()
    while True:
        item = FRAMES.get()
        if item is RESET:
            s = fresh()
            emit(dict(type="session", state="reset"))
            continue
        if item is STOP:
            # the session ends by decision, not by a next step arriving -- so the
            # step in progress completes NOW. This also fixes the "last step never
            # completes" behaviour the file-driver test surfaced.
            if s["t0"] is not None and s["committed"] is not None \
                    and classes[s["committed"]] != "background":
                now_s = time.monotonic() - s["t0"]
                emit(dict(type="complete", step=classes[s["committed"]],
                          at=round(now_s, 2), announced_after=0.0))
            emit(dict(type="session", state="stopped"))
            s = fresh()
            continue
        if not ARMED["on"]:
            continue              # preview-only: frames flow, the model does not run
        if s["t0"] is None:
            s["t0"] = time.monotonic()
            emit(dict(type="session", state="started"))

        # backpressure: if we fall behind the camera, skip to the freshest frames
        # and SAY so, rather than silently drifting out of real time
        backlog = FRAMES.qsize()
        if backlog > 30:
            dropped = 0
            while FRAMES.qsize() > 10:
                nxt = FRAMES.get_nowait()
                if nxt is RESET:
                    item = RESET
                    break
                item, dropped = nxt, dropped + 1
            if item is RESET:
                s = fresh()
                emit(dict(type="session", state="reset"))
                continue
            s["drops"] += dropped
            emit(dict(type="warn", message=f"behind camera: dropped {dropped} frames"))

        t_dec = time.monotonic()
        img = Image.open(io.BytesIO(item)).convert("RGB")
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div_(255.0)
        x = torch.nn.functional.interpolate(x[None], size=(224, 224), mode="bilinear",
                                            align_corners=False)[0].half()
        s["buf"].append(x)
        s["j"] += 1
        j = s["j"]
        if j < 15 or (j - 15) % 2:        # one decoder tick per 2 ingest frames = 0.2 s
            continue

        clip = torch.stack(list(s["buf"])).unsqueeze(0).to(dev).permute(0, 2, 1, 3, 4).float()
        clip = (clip - mean) / std
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            g = STATE["giant"].forward_features(clip)
            v = STATE["iv2"](clip[:, :, ::2])
        gn = torch.nn.functional.normalize(g.float(), dim=1)
        vn = torch.nn.functional.normalize(v.float(), dim=1)
        s["feats"].append(torch.cat([gn, vn], 1)[0])
        if len(s["feats"]) > TCN_WINDOW:
            del s["feats"][0]

        seq = torch.stack(s["feats"], 1).unsqueeze(0)
        with torch.no_grad():
            lg = STATE["tcn"](seq)[0, :, -1]
        lp = torch.log_softmax(lg, 0).cpu().numpy()
        state = s["vit"].step(lp)
        s["clips"] += 1
        tick = s["clips"] - 1

        # session-clock times, with the same two corrections the demo validated:
        # the decoder's output is LAG ticks old, and a clip's label sits at its centre
        t_hap = max(0.0, (tick - LAG) * SEC_PER_TICK + (SPAN / 2) / 30.0)
        now_s = time.monotonic() - s["t0"]
        if state != s["committed"]:
            if s["committed"] is not None and classes[s["committed"]] != "background":
                emit(dict(type="complete", step=classes[s["committed"]],
                          at=round(t_hap, 2),
                          announced_after=round(now_s - t_hap, 2)))
            s["committed"] = state
            emit(dict(type="enter", step=classes[state], at=round(t_hap, 2)))

        dt = time.monotonic() - t_dec
        s["ema"] = dt if s["ema"] is None else 0.9 * s["ema"] + 0.1 * dt
        if s["clips"] % 5 == 0:
            rate = 1.0 / max(s["ema"], 1e-6)
            emit(dict(type="stats", clips=s["clips"], clip_ms=round(s["ema"] * 1000),
                      rate=round(rate, 2), realtime=round(rate / 5.0, 2),
                      backlog=FRAMES.qsize(), drops=s["drops"],
                      session_s=round(now_s, 1),
                      current=classes[s["committed"]] if s["committed"] is not None else "-",
                      conf=round(float(np.exp(lp.max())), 3)))


# ------------------------------------------------------------------ endpoints
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(open(os.path.join(HERE, "web", "dashboard.html")).read(),
                        headers={"Cache-Control": "no-store"})


@app.get("/phone", response_class=HTMLResponse)
def phone():
    return HTMLResponse(open(os.path.join(HERE, "web", "phone.html")).read(),
                        headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    return JSONResponse(dict(loaded=bool(STATE), armed=ARMED["on"],
                             backlog=FRAMES.qsize(), events=len(EVLOG)))


@app.post("/api/start")
def start_session():
    FRAMES.put(RESET)
    ARMED["on"] = True
    emit(dict(type="session", state="armed"))
    return JSONResponse(dict(ok=True))


@app.post("/api/stop")
def stop_session():
    ARMED["on"] = False
    FRAMES.put(STOP)
    return JSONResponse(dict(ok=True))


@app.post("/api/reset")
def reset():
    ARMED["on"] = False
    FRAMES.put(RESET)
    return JSONResponse(dict(ok=True))


@app.get("/api/latest.jpg")
def latest():
    if LATEST["jpg"] is None:
        return JSONResponse({"error": "no frames yet"}, status_code=404)
    return Response(LATEST["jpg"], media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- WebRTC signaling
# Ported from the reference ego-recorder: the live PREVIEW rides WebRTC (hardware
# H.264, 30 fps, peer-to-peer) exactly as the reference does, because JPEG stills
# polled at 2.5 fps are unwatchable next to it. The 10 fps JPEG WebSocket remains
# the MODEL's feed -- two channels, two consumers, each at its natural rate.
RTC = {}                     # (role, id) -> queue of envelope dicts
RTC_LOCK = threading.Lock()


@app.get("/rtc/events")
async def rtc_events(role: str, id: str):
    q = queue.Queue()
    with RTC_LOCK:
        RTC[(role, id)] = q

    async def gen():
        loop = asyncio.get_event_loop()
        try:
            while True:
                try:
                    msg = await loop.run_in_executor(None, q.get, True, 1.0)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(msg)}\n\n"
        finally:
            with RTC_LOCK:
                RTC.pop((role, id), None)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/rtc/signal")
async def rtc_signal(request: Request):
    msg = await request.json()
    to, sender_role = msg.get("to"), msg.get("role")
    with RTC_LOCK:
        if to:
            targets = [q for (r, i), q in RTC.items() if i == to]
        else:                     # broadcast to every client of the OTHER role
            targets = [q for (r, i), q in RTC.items() if r != sender_role]
        for q in targets:
            q.put(msg)
    return JSONResponse({"delivered": len(targets)})


@app.websocket("/ws/ingest")
async def ingest(ws: WebSocket):
    await ws.accept()
    emit(dict(type="camera", state="connected"))
    try:
        while True:
            data = await ws.receive_bytes()
            LATEST["jpg"] = data
            FRAMES.put(data)
    except WebSocketDisconnect:
        emit(dict(type="camera", state="disconnected"))


@app.get("/api/live")
async def live(request: Request):
    """Replayable SSE (same design the demo converged on): every event has an id,
    the browser resumes with Last-Event-ID after a drop, silence is impossible."""
    # A reconnecting client resumes exactly where it broke (Last-Event-ID). A FRESH
    # client must NOT get the whole server history: replaying it painted a previous
    # session's completed steps green before this one had even started. Start such a
    # client at the most recent session boundary, so it sees the CURRENT session in
    # progress and nothing older.
    lei = request.headers.get("last-event-id")
    if lei is not None:
        try:
            start = int(lei) + 1
        except ValueError:
            start = 0
    else:
        with EVCV:
            start = 0
            for i in range(len(EVLOG) - 1, -1, -1):
                ev = EVLOG[i]
                if ev.get("type") == "session" and ev.get("state") in ("armed", "reset", "stopped"):
                    start = i
                    break

    async def gen():
        i = max(start, 0)
        loop = asyncio.get_event_loop()
        while True:
            def wait_next(i=i):
                with EVCV:
                    EVCV.wait_for(lambda: len(EVLOG) > i, timeout=1.0)
                    return list(EVLOG[i:i + 50])
            batch = await loop.run_in_executor(None, wait_next)
            if not batch:
                yield ": keepalive\n\n"
                continue
            for ev in batch:
                yield f"id: {i}\ndata: {json.dumps(ev)}\n\n"
                i += 1
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    _watchdog(10.0)
    threading.Thread(target=worker, daemon=True).start()
    cert = os.path.join(CERTS, "cert.pem")
    key = os.path.join(CERTS, "key.pem")
    print(f"dashboard: https://0.0.0.0:{PORT}/   phone: https://<LAN-IP>:{PORT}/phone",
          flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning",
                ssl_certfile=cert, ssl_keyfile=key)
