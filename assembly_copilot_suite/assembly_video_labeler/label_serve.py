#!/usr/bin/env python3
"""
Assembly Labeler — a lightweight, login-node web app to annotate recorded videos
in IndustReal PSR style (mark the frame where each step completes).

  Run:   conda activate ./psr_env  &&  python scripts/label_serve.py
  Port:  LABEL_PORT (default 7862)   Videos: LABEL_VIDEOS   Output: LABEL_OUT

Multi-user: one server instance; every annotator forwards port 7862 and opens the
URL. A file-based claim/lock stops two people labelling the same clip at once.
No GPU / no model — safe on a login node.
"""
import os, json, csv, re, time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
WEB  = os.path.join(HERE, "web")
# --- folder paths (override with env vars when the data lives elsewhere) ---
VIDEOS = os.environ.get("LABEL_VIDEOS", os.path.join(HERE, "videos"))        # your recordings
PROXY  = os.environ.get("LABEL_PROXY",  os.path.join(HERE, "videos_proxy"))  # optional 480p proxies
OUT    = os.environ.get("LABEL_OUT",    os.path.join(HERE, "labels_out"))    # written label files
PORT   = int(os.environ.get("LABEL_PORT", "7862"))
CLAIM_TTL = 120                 # a claim goes stale after this many seconds without a heartbeat
SLICE = 3 * 1024 * 1024         # max bytes per range response (keeps seeking snappy, worker free)
os.makedirs(VIDEOS, exist_ok=True); os.makedirs(PROXY, exist_ok=True); os.makedirs(OUT, exist_ok=True)

PARTS = [
    "Main Drive Planet Gear", "Pop Control Ring Gear", "Pop Control Sun Gear",
    "Compressor Casing", "Exhaust", "Exhaust Casing", "Cowling Bracket",
    "Frame Subassembly", "Propeller Cone plate", "Propeller Cone Tip",
]
def part_id(i): return 3 + 3 * i                     # IndustReal install-state id
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
CTYPE = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
         ".mkv": "video/x-matroska", ".avi": "video/x-msvideo", ".m4v": "video/mp4"}

app = FastAPI()
def stem(n): return os.path.splitext(os.path.basename(n))[0]
def vdir(n): return os.path.join(OUT, stem(n))

def vstatus(name):
    d = vdir(name)
    cb = None; cf = os.path.join(d, ".claim")
    if os.path.exists(cf):
        try:
            c = json.load(open(cf))
            if time.time() - c.get("ts", 0) < CLAIM_TTL: cb = c.get("user")
        except Exception: pass
    return {"labeled": os.path.exists(os.path.join(d, "labels.json")),
            "discarded": os.path.exists(os.path.join(d, "discarded.json")),
            "claimed_by": cb}

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(open(os.path.join(WEB, "label.html")).read(),
                        headers={"Cache-Control": "no-store, must-revalidate"})

@app.get("/api/config")
def config():
    return {"parts": [{"name": p, "id": part_id(i), "index": i} for i, p in enumerate(PARTS)],
            "fps_default": 30, "videos_dir": VIDEOS, "out_dir": OUT}

@app.get("/api/videos")
def videos():
    return {"videos": [{"name": f, **vstatus(f)}
                       for f in sorted(os.listdir(VIDEOS)) if f.lower().endswith(VIDEO_EXT)]}

@app.get("/media/video/{name}")
def media(name: str, request: Request):
    base = os.path.basename(name)
    pp = os.path.join(PROXY, base)               # prefer a lightweight proxy for smooth scrubbing
    p = pp if os.path.exists(pp) else os.path.join(VIDEOS, base)
    if not os.path.exists(p): return JSONResponse({"error": "not found"}, 404)
    size = os.path.getsize(p)
    ctype = CTYPE.get(os.path.splitext(p)[1].lower(), "application/octet-stream")
    rng = request.headers.get("range") or request.headers.get("Range")
    if rng:                                          # HTTP Range -> 206, capped to a small window
        m = re.match(r"bytes=(\d+)-(\d*)", rng.strip())
        start = int(m.group(1)); req_end = int(m.group(2)) if m and m.group(2) else size - 1
        end = min(req_end, size - 1, start + SLICE - 1); length = end - start + 1
        def gen():
            with open(p, "rb") as f:
                f.seek(start); left = length
                while left > 0:
                    b = f.read(min(262144, left))
                    if not b: break
                    left -= len(b); yield b
        return StreamingResponse(gen(), status_code=206, media_type=ctype,
            headers={"Content-Range": f"bytes {start}-{end}/{size}",
                     "Accept-Ranges": "bytes", "Content-Length": str(length)})
    def full():
        with open(p, "rb") as f:
            while True:
                b = f.read(262144)
                if not b: break
                yield b
    return StreamingResponse(full(), media_type=ctype,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(size)})

@app.get("/api/labels/{name}")
def load_labels(name: str):
    p = os.path.join(vdir(name), "labels.json")
    return json.load(open(p)) if os.path.exists(p) else {"video": name, "fps": 30, "marks": []}

# ---- multi-user claim / discard / reset ----
@app.post("/api/claim")
async def claim(req: Request):
    d = await req.json(); name = os.path.basename(d.get("video", "")); user = d.get("user") or "anon"
    if not name: return JSONResponse({"error": "no video"}, 400)
    st = vstatus(name)
    if st["claimed_by"] and st["claimed_by"] != user and not d.get("steal"):
        return JSONResponse({"error": "claimed", "claimed_by": st["claimed_by"]}, 409)
    os.makedirs(vdir(name), exist_ok=True)
    json.dump({"user": user, "ts": time.time()}, open(os.path.join(vdir(name), ".claim"), "w"))
    return {"ok": True, **vstatus(name)}

@app.post("/api/release")
async def release(req: Request):
    d = await req.json(); name = os.path.basename(d.get("video", "")); user = d.get("user")
    cf = os.path.join(vdir(name), ".claim")
    if os.path.exists(cf):
        try:
            if json.load(open(cf)).get("user") == user or d.get("force"): os.remove(cf)
        except Exception:
            try: os.remove(cf)
            except Exception: pass
    return {"ok": True}

@app.post("/api/discard")
async def discard(req: Request):
    d = await req.json(); name = os.path.basename(d.get("video", "")); user = d.get("user") or "anon"
    os.makedirs(vdir(name), exist_ok=True)
    json.dump({"user": user, "ts": time.time(), "reason": d.get("reason", "")},
              open(os.path.join(vdir(name), "discarded.json"), "w"))
    cf = os.path.join(vdir(name), ".claim")
    if os.path.exists(cf): os.remove(cf)
    return {"ok": True}

@app.post("/api/undiscard")
async def undiscard(req: Request):
    d = await req.json(); p = os.path.join(vdir(os.path.basename(d.get("video", ""))), "discarded.json")
    if os.path.exists(p): os.remove(p)
    return {"ok": True}

@app.post("/api/reset")
async def reset(req: Request):
    d = await req.json(); dd = vdir(os.path.basename(d.get("video", "")))
    for fn in ("labels.json", "PSR_labels.csv", "segments.csv"):
        p = os.path.join(dd, fn)
        if os.path.exists(p): os.remove(p)
    return {"ok": True}

@app.post("/api/save")
async def save(req: Request):
    d = await req.json(); video = os.path.basename(d.get("video", "")); fps = float(d.get("fps") or 30)
    direction = d.get("direction", "assembly"); marks = d.get("marks", [])
    if not video: return JSONResponse({"error": "no video"}, 400)
    remove = (direction == "disassembly")
    base = 5 if remove else 3; verb = "Remove " if remove else "Install "   # IndustReal id: install 3+3p, remove 5+3p
    d0 = vdir(video); os.makedirs(d0, exist_ok=True)
    try:  # reject bad marks before writing: a negative index would wrap round to the last part
        clean = [(float(m["t"]), int(m["pi"])) for m in marks]
    except (TypeError, ValueError, KeyError):
        return JSONResponse({"error": "malformed marks"}, 400)
    for t, pi in clean:
        if not (0 <= pi < len(PARTS)) or t != t or t in (float("inf"), float("-inf")):
            return JSONResponse({"error": f"bad mark (t={t}, part index {pi})"}, 400)
    rows = [{"t": t, "frame": round(t * fps), "pi": pi,
             "part": PARTS[pi], "id": base + 3 * pi} for t, pi in clean]
    rows.sort(key=lambda r: r["t"])
    json.dump({"video": video, "fps": fps, "direction": direction, "marks": marks},
              open(os.path.join(d0, "labels.json"), "w"), indent=2)
    with open(os.path.join(d0, "PSR_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for r in sorted(rows, key=lambda r: r["frame"]):
            w.writerow([f"{r['frame']:06d}.jpg", r["id"], verb + r["part"]])
    with open(os.path.join(d0, "segments.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["part", "action", "start_frame", "end_frame", "start_sec", "end_sec"])
        pt, pf = 0.0, 0
        for r in rows:
            w.writerow([r["part"], "remove" if remove else "install", pf, r["frame"], round(pt, 3), round(r["t"], 3)])
            pt, pf = r["t"], r["frame"]
    return {"ok": True, "saved_to": d0, "n": len(rows)}

if __name__ == "__main__":
    print(f"[labeler] videos: {VIDEOS}\n[labeler] output: {OUT}")
    print(f"[labeler] open  : http://localhost:{PORT}   (forward port {PORT} in VS Code)")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
