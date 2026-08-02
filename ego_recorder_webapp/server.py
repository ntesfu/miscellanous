#!/usr/bin/env python3
"""
Ego Recorder — LAN iPhone camera streamer + recorder.

Pure Python standard library. No pip installs.

  * Serves an HTTPS web app.
  * iPhone opens  /phone   -> becomes the camera (sender).
  * Mac opens     /viewer  -> live display + Start/Stop recording controls.
  * Live video streams phone -> Mac peer-to-peer over WebRTC.
  * Recording is captured on the iPhone and streamed to this machine's
    ./recordings folder in real time over HTTP.

Signaling + control run over Server-Sent Events (GET /events) for the
downstream and plain POST (/signal) for the upstream, so no websocket
library is required.

Run:
    ./gen_cert.sh          # once, generates certs/cert.pem + certs/key.pem
    python3 server.py      # then open the printed URLs
"""

import json
import os
import queue
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(ROOT, "public")
RECORDINGS_DIR = os.path.join(ROOT, "recordings")
DISCARDED_DIR = os.path.join(RECORDINGS_DIR, "discarded")
CERT_FILE = os.path.join(ROOT, "certs", "cert.pem")
KEY_FILE = os.path.join(ROOT, "certs", "key.pem")

PORT = int(os.environ.get("EGO_PORT", "8443"))
HOST = os.environ.get("EGO_HOST", "0.0.0.0")

# Upper bounds on POST body sizes so a runaway/forged upload can't exhaust RAM.
MAX_BODY_UPLOAD = 128 * 1024 * 1024   # a recording chunk or snapshot (very generous)
MAX_BODY_JSON = 4 * 1024 * 1024       # signaling / control JSON
SESSION_IDLE_TIMEOUT = 120            # seconds before an abandoned recording is reaped

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(DISCARDED_DIR, exist_ok=True)

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def lan_ip():
    """Best-effort detection of this machine's primary LAN IPv4 address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets are actually sent; this just selects the outbound iface.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# Signaling hub: routes JSON messages between phone(s) and viewer(s) via SSE.
# --------------------------------------------------------------------------- #
class Hub:
    def __init__(self):
        self._lock = threading.Lock()
        # client_id -> {"role": str, "q": queue.Queue}
        self._clients = {}

    def register(self, client_id, role):
        q = queue.Queue()
        with self._lock:
            # An EventSource reconnect reuses the same id. Evict the stale
            # connection (wake its thread so it exits) and take over the id.
            old = self._clients.get(client_id)
            if old:
                old["q"].put({"type": "__evict__"})
            self._clients[client_id] = {"role": role, "q": q}
            peers = [
                {"id": cid, "role": c["role"]}
                for cid, c in self._clients.items()
                if cid != client_id and c["role"] != role
            ]
        # Tell the newcomer who is already here (the other role only).
        q.put({"type": "welcome", "id": client_id, "role": role, "peers": peers})
        # Tell the other role that a new peer joined.
        self._broadcast_other(role, {"type": "peer-joined", "id": client_id, "role": role})
        return q

    def unregister(self, client_id, q):
        # Connection-scoped: only remove the id if it still points at THIS
        # connection's queue, so a dying stale thread can't drop a live reconnect.
        with self._lock:
            info = self._clients.get(client_id)
            if not info or info["q"] is not q:
                return
            self._clients.pop(client_id, None)
        self._broadcast_other(
            info["role"], {"type": "peer-left", "id": client_id, "role": info["role"]}
        )

    def _broadcast_other(self, role, msg):
        with self._lock:
            targets = [c["q"] for cid, c in self._clients.items() if c["role"] != role]
        for q in targets:
            q.put(msg)

    def route(self, msg):
        """Deliver a signaling message. `to` = target id, or None = broadcast
        to every client of the *other* role than the sender."""
        to = msg.get("to")
        with self._lock:
            if to:
                info = self._clients.get(to)
                targets = [info["q"]] if info else []
            else:
                sender_role = msg.get("role")
                targets = [
                    c["q"] for cid, c in self._clients.items() if c["role"] != sender_role
                ]
        for q in targets:
            q.put(msg)
        return len(targets)


# --------------------------------------------------------------------------- #
# Recording normalization: rewrite MediaRecorder captures with clean timing.
# --------------------------------------------------------------------------- #
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _probe(path, entries, stream=None):
    """Small ffprobe helper -> first value of a single entry, or '' on failure."""
    cmd = [FFPROBE, "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "csv=p=0", path]
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=60).stdout.strip().splitlines()
    except Exception:
        return []


def normalize_recording(path):
    """Re-time a finished capture so it plays smoothly, replacing the file.

    iOS Safari's MediaRecorder writes H.264 with temporal layering: only ~half
    the frames are "displayable" in the MP4 sample table and the rest are tagged
    AV_PKT_FLAG_DISCARD, so a normal decode yields ~15 fps out of a real 30 fps
    capture (the choppy/jittery playback). Timestamps are also scrambled
    (duplicate PTS, uneven interleave), so trusting them reproduces the freezes.

    Fix: pull the *raw* H.264 elementary stream (bitstream copy, which has no
    discard concept -> every frame survives), then lay all frames down on a
    perfectly even grid at their true rate (frame_count / duration) and mux the
    original audio back. The stream is I/P only (no B-frames), so decode order ==
    display order and this reconstruction is exact. Always outputs .mp4.
    Returns the final file name, or None on skip/failure (original untouched).
    """
    if not (FFMPEG and FFPROBE):
        return None
    base = os.path.splitext(path)[0]
    raw = base + ".tmp.h264"
    tmp = base + ".tmp.mp4"
    try:
        dur_s = _probe(path, "format=duration")
        dur = float(dur_s[0]) if dur_s else 0.0
        codec = (_probe(path, "stream=codec_name", "v:0") or [""])[0]
        # Total video packets = every real frame, INCLUDING the discard-tagged
        # ones (index read only, no decode -> fast and complete).
        try:
            frames = int(subprocess.run(
                [FFPROBE, "-v", "error", "-select_streams", "v:0",
                 "-count_packets", "-show_entries", "stream=nb_read_packets",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=120).stdout.strip())
        except Exception:
            frames = 0
        if dur <= 0 or frames < 2:
            return None
        rate = max(1.0, min(120.0, frames / dur))

        common_out = [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-video_track_timescale", "90000", "-movflags", "+faststart",
            "-shortest", tmp]

        if codec == "h264":
            # 1) recover ALL frames as raw Annex-B (discard flag can't exist here)
            ex = subprocess.run(
                [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
                 "-map", "0:v:0", "-c", "copy", "-bsf:v", "h264_mp4toannexb", raw],
                capture_output=True, text=True, timeout=600)
            if ex.returncode != 0 or not os.path.exists(raw) or not os.path.getsize(raw):
                raise RuntimeError("annexb extract: " + (ex.stderr.strip()[-200:] or "empty"))
            # 2) even 30fps grid from the raw stream + original audio back in
            enc = subprocess.run(
                [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                 "-r", "%.6f" % rate, "-i", raw, "-i", path,
                 "-map", "0:v:0", "-map", "1:a:0?", "-vsync", "cfr"] + common_out,
                capture_output=True, text=True, timeout=1800)
        else:
            # Non-h264 (e.g. VP8/VP9 webm fallback): re-time the decoded frames
            # evenly; no discard-layer trick applies.
            enc = subprocess.run(
                [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
                 "-map", "0:v:0", "-map", "0:a:0?",
                 "-vf", "setpts=N/(%.6f*TB),format=yuv420p" % rate, "-vsync", "cfr"]
                + common_out,
                capture_output=True, text=True, timeout=1800)

        if enc.returncode != 0 or not os.path.exists(tmp) or not os.path.getsize(tmp):
            raise RuntimeError("encode: " + (enc.stderr.strip()[-200:] or "empty"))

        # The source may have been discarded while we were encoding — if so,
        # don't resurrect it into the library.
        if not os.path.exists(path):
            raise RuntimeError("source vanished during normalize (discarded?)")

        final = base + ".mp4"
        os.replace(tmp, final)
        tmp = None
        if final != path:
            try:
                os.remove(path)  # original was .webm; the .mp4 replaces it
            except OSError:
                pass
        sys.stderr.write("  normalized %s (%d frames @ %.2f fps)\n"
                         % (os.path.basename(final), frames, rate))
        return os.path.basename(final)
    except Exception as e:
        sys.stderr.write("  normalize failed for %s: %s\n"
                         % (os.path.basename(path), e))
        return None
    finally:
        for f in (raw, tmp):
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


def _label_from_meta(meta):
    """Extract a sanitized (person, mode) label chosen in the viewer dropdowns.

    Both are reduced to bare alphanumerics so they can never escape the
    recordings directory or inject path separators. Returns ("", "") when the
    recording carried no label (e.g. started from the phone's own button), in
    which case the caller keeps the default timestamped name.
    """
    # meta and meta["label"] come straight from untrusted client JSON, so a
    # non-dict (string/number/list) must not crash finalization.
    meta = meta if isinstance(meta, dict) else {}
    label = meta.get("label")
    label = label if isinstance(label, dict) else {}
    person = re.sub(r"[^A-Za-z0-9]", "", str(label.get("person", "")))[:24]
    mode = re.sub(r"[^A-Za-z0-9]", "", str(label.get("mode", "")))[:24]
    return person, mode


# --------------------------------------------------------------------------- #
# Recording sessions: append streamed chunks to a file in the right order.
# --------------------------------------------------------------------------- #
class RecStore:
    def __init__(self, hub):
        self._lock = threading.Lock()
        self._sessions = {}  # session_id -> dict
        self._hub = hub
        # Serializes final-name allocation so two recordings finishing at once
        # can't both claim the same {Name}-{Stage}-{N}. Held across the rename
        # so the on-disk file itself reserves the number for the next caller.
        self._name_lock = threading.Lock()

    def start(self, session_id, ext, meta):
        ext = ext if ext in (".mp4", ".webm") else ".mp4"
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_sid = re.sub(r"[^A-Za-z0-9_-]", "", session_id)[:40] or "rec"
        name = f"{ts}_{safe_sid}{ext}"
        path = os.path.join(RECORDINGS_DIR, name)
        f = open(path, "wb")
        with self._lock:
            now = time.time()
            self._sessions[session_id] = {
                "path": path,
                "name": name,
                "file": f,
                "expected": 0,
                "buffer": {},          # seq -> bytes, held until in order
                "bytes": 0,
                "lock": threading.Lock(),
                "meta": meta or {},
                "started": now,
                "last": now,           # last activity, for the idle reaper
            }
        return name

    def chunk(self, session_id, seq, data):
        with self._lock:
            s = self._sessions.get(session_id)
        if not s:
            return None
        with s["lock"]:
            # A late chunk can arrive after finish()/reap() already closed the
            # file (rapid stop, Wi-Fi retry, idle reap). Drop it instead of
            # writing to a closed handle (which would 500 and truncate the tail).
            if s["file"].closed:
                return s["bytes"]
            s["last"] = time.time()
            s["buffer"][seq] = data
            # Flush every contiguous chunk we now have, in order.
            while s["expected"] in s["buffer"]:
                d = s["buffer"].pop(s["expected"])
                s["file"].write(d)
                s["bytes"] += len(d)
                s["expected"] += 1
            s["file"].flush()
            return s["bytes"]

    def reap(self, idle=SESSION_IDLE_TIMEOUT):
        """Finalize sessions abandoned mid-recording (phone crashed/reloaded/
        lost Wi-Fi) so file handles don't leak and the partial file is closed."""
        now = time.time()
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if now - s["last"] > idle]
        for sid in stale:
            self.finish(sid)

    def _next_indexed_base(self, prefix):
        """Return "{prefix}-N" with N one past the highest existing on disk.

        Must be called while holding self._name_lock. Counts both finished
        (.mp4) and still-normalizing (.webm) files so numbers never collide.
        """
        pat = re.compile(re.escape(prefix) + r"-(\d+)\.(?:mp4|webm)$", re.IGNORECASE)
        max_n = 0
        # Scan both live and discarded recordings so a discarded number is never
        # handed out again (avoids two different clips sharing a name).
        for d in (RECORDINGS_DIR, DISCARDED_DIR):
            try:
                for fn in os.listdir(d):
                    m = pat.match(fn)
                    if m:
                        max_n = max(max_n, int(m.group(1)))
            except OSError:
                pass
        return "%s-%d" % (prefix, max_n + 1)

    def _apply_label_name(self, s):
        """Rename a finished capture to {Name}-{Stage}-{N}.<ext> if it carried a
        label. Updates s["path"]/s["name"] in place. No-op (keeps the timestamped
        name) when unlabeled or if the rename fails."""
        person, mode = _label_from_meta(s.get("meta"))
        if not (person and mode):
            return
        ext = os.path.splitext(s["path"])[1].lower() or ".mp4"
        with self._name_lock:
            base = self._next_indexed_base("%s-%s" % (person, mode))
            target = os.path.join(RECORDINGS_DIR, base + ext)
            try:
                os.replace(s["path"], target)
                s["path"] = target
                s["name"] = base + ext
            except OSError as e:
                sys.stderr.write("  label rename failed for %s: %s\n"
                                 % (s["name"], e))

    def finish(self, session_id):
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if not s:
            return None
        with s["lock"]:
            try:
                s["file"].flush()
                s["file"].close()
            except Exception:
                pass

        # Give it the human-readable indexed name chosen in the viewer, before
        # normalization rewrites the (now correctly-named) file in place.
        self._apply_label_name(s)

        result = {
            "name": s["name"],
            "bytes": s["bytes"],
            "url": "/recordings/" + s["name"],
            "durationMs": int((time.time() - s["started"]) * 1000),
        }

        # Normalize timing in the background (MediaRecorder timestamps are
        # unreliable), then notify viewers once the clean file is in place.
        def _finalize():
            name = normalize_recording(s["path"]) or s["name"]
            payload = dict(result, name=name, url="/recordings/" + name)
            try:
                payload["bytes"] = os.path.getsize(
                    os.path.join(RECORDINGS_DIR, name))
            except OSError:
                pass
            self._hub._broadcast_other(
                "phone", {"type": "recording_saved", "payload": payload})

        threading.Thread(target=_finalize, daemon=True).start()
        return result


HUB = Hub()
REC = RecStore(HUB)


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "EgoRecorder"

    # Quieter logging: one line, no default noise.
    def log_message(self, fmt, *args):
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    # ---- small response helpers ------------------------------------------- #
    def _send_bytes(self, status, body, content_type, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_json(self, obj, status=200):
        self._send_bytes(status, json.dumps(obj).encode("utf-8"),
                         "application/json; charset=utf-8")

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return b""
        buf = bytearray()
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                break
            buf.extend(chunk)
            remaining -= len(chunk)
        return bytes(buf)

    # ---- routing ---------------------------------------------------------- #
    def do_GET(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)

        if path == "/events":
            return self._sse(qs)
        if path in ("/", "/index.html"):
            return self._serve_file(os.path.join(PUBLIC_DIR, "index.html"))
        if path == "/phone":
            return self._serve_file(os.path.join(PUBLIC_DIR, "phone.html"))
        if path == "/viewer":
            return self._serve_file(os.path.join(PUBLIC_DIR, "viewer.html"))
        if path == "/health":
            return self._send_json({"ok": True})
        if path == "/api/config":
            return self._send_json({"ip": lan_ip(), "port": PORT})
        if path == "/api/recordings":
            return self._list_recordings()
        if path.startswith("/static/"):
            return self._serve_static(path)
        if path.startswith("/recordings/"):
            return self._serve_recording(path)
        if path == "/favicon.ico":
            return self._send_bytes(204, b"", "image/x-icon")
        return self._send_json({"error": "not found"}, 404)

    def do_HEAD(self):
        # Reuse GET routing for static/recordings; harmless for others.
        self.do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)

        # Reject oversized bodies before reading them into memory.
        clen = int(self.headers.get("Content-Length", 0) or 0)
        limit = MAX_BODY_UPLOAD if path in ("/upload", "/snapshot") else MAX_BODY_JSON
        if clen > limit:
            self.close_connection = True
            return self._send_json({"error": "payload too large"}, 413)

        if path == "/signal":
            body = self._read_body()
            try:
                msg = json.loads(body.decode("utf-8"))
            except Exception:
                return self._send_json({"error": "bad json"}, 400)
            n = HUB.route(msg)
            return self._send_json({"delivered": n})

        if path == "/upload/start":
            body = self._read_body()
            try:
                d = json.loads(body.decode("utf-8"))
            except Exception:
                return self._send_json({"error": "bad json"}, 400)
            name = REC.start(d.get("session", ""), d.get("ext", ".mp4"), d.get("meta"))
            HUB._broadcast_other("phone", {"type": "recording_started",
                                           "payload": {"name": name}})
            return self._send_json({"name": name})

        if path == "/upload":
            session_id = (qs.get("session") or [""])[0]
            try:
                seq = int((qs.get("seq") or ["0"])[0])
            except ValueError:
                return self._send_json({"error": "bad seq"}, 400)
            data = self._read_body()
            total = REC.chunk(session_id, seq, data)
            if total is None:
                return self._send_json({"error": "no session"}, 404)
            return self._send_json({"bytes": total})

        if path == "/upload/finish":
            body = self._read_body()
            try:
                d = json.loads(body.decode("utf-8"))
            except Exception:
                return self._send_json({"error": "bad json"}, 400)
            result = REC.finish(d.get("session", ""))
            if result is None:
                return self._send_json({"error": "no session"}, 404)
            return self._send_json(result)

        if path == "/snapshot":
            return self._save_snapshot(qs)

        if path == "/api/discard":
            body = self._read_body()
            try:
                d = json.loads(body.decode("utf-8"))
            except Exception:
                return self._send_json({"error": "bad json"}, 400)
            return self._discard_recording(d.get("name", ""))

        return self._send_json({"error": "not found"}, 404)

    # ---- Server-Sent Events (signaling downstream) ------------------------ #
    def _sse(self, qs):
        role = (qs.get("role") or ["viewer"])[0]
        client_id = (qs.get("id") or [""])[0]
        if role not in ("phone", "viewer") or not client_id:
            return self._send_json({"error": "bad role/id"}, 400)
        # HEAD must not register a phantom peer or enter the streaming loop.
        if self.command == "HEAD":
            return self._send_bytes(200, b"", "text/event-stream; charset=utf-8")

        q = HUB.register(client_id, role)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        # Must be set AFTER send_header("Connection", ...), which otherwise resets
        # this to False. An SSE stream is never keep-alive-reusable, so when the
        # loop exits (e.g. evicted by a reconnect) we want the socket torn down.
        self.close_connection = True
        # Prime the stream so the browser fires `open` immediately.
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            HUB.unregister(client_id, q)
            return

        try:
            while True:
                try:
                    msg = q.get(timeout=20)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # keep-alive
                    self.wfile.flush()
                    continue
                if msg.get("type") == "__evict__":
                    break  # a newer connection took over this id
                payload = json.dumps(msg)
                self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HUB.unregister(client_id, q)

    # ---- static + files --------------------------------------------------- #
    def _serve_file(self, fpath):
        if not os.path.isfile(fpath):
            return self._send_json({"error": "not found"}, 404)
        ext = os.path.splitext(fpath)[1].lower()
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(fpath, "rb") as f:
            body = f.read()
        self._send_bytes(200, body, ctype, {"Cache-Control": "no-cache"})

    def _serve_static(self, path):
        rel = path[len("/static/"):]
        if ".." in rel or rel.startswith("/"):
            return self._send_json({"error": "forbidden"}, 403)
        fpath = os.path.join(PUBLIC_DIR, "static", rel)
        return self._serve_file(fpath)

    def _list_recordings(self):
        items = []
        for name in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
            fpath = os.path.join(RECORDINGS_DIR, name)
            if not os.path.isfile(fpath) or name.startswith("."):
                continue
            if ".tmp." in name:
                continue  # normalization in progress

            st = os.stat(fpath)
            items.append({
                "name": name,
                "size": st.st_size,
                "mtime": int(st.st_mtime * 1000),
                "url": "/recordings/" + name,
            })
        self._send_json({"recordings": items})

    def _serve_recording(self, path):
        name = path[len("/recordings/"):]
        if not SAFE_NAME.match(name):
            return self._send_json({"error": "forbidden"}, 403)
        fpath = os.path.join(RECORDINGS_DIR, name)
        if not os.path.isfile(fpath):
            return self._send_json({"error": "not found"}, 404)
        ext = os.path.splitext(name)[1].lower()
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        st = os.stat(fpath)
        size = st.st_size

        # Minimal HTTP Range support so the browser can seek/scrub playback.
        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
            if m:
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                if start > end:
                    start = 0
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                if self.command != "HEAD":
                    with open(fpath, "rb") as f:
                        f.seek(start)
                        self._pump(f, length)
                return

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'inline; filename="{name}"')
        self.end_headers()
        if self.command != "HEAD":
            with open(fpath, "rb") as f:
                self._pump(f, size)

    def _pump(self, f, length):
        remaining = length
        try:
            while remaining > 0:
                chunk = f.read(min(remaining, 262144))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _save_snapshot(self, qs):
        data = self._read_body()
        if not data:
            return self._send_json({"error": "empty"}, 400)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"snap_{ts}_{time.time_ns() % 1_000_000:06d}.jpg"  # sub-second uniqueness
        fpath = os.path.join(RECORDINGS_DIR, name)
        with open(fpath, "wb") as f:
            f.write(data)
        result = {"name": name, "url": "/recordings/" + name, "size": len(data)}
        HUB._broadcast_other("phone", {"type": "snapshot_saved", "payload": result})
        return self._send_json(result)

    def _discard_recording(self, name):
        """Move a finished recording out of the library into recordings/discarded/.

        The name is validated against SAFE_NAME (same as when serving), so it
        can't contain a path separator or escape RECORDINGS_DIR. On a name clash
        in the discard folder a numeric suffix is added rather than overwriting.
        """
        if not SAFE_NAME.match(name) or name.startswith("."):
            return self._send_json({"error": "bad name"}, 400)
        src = os.path.join(RECORDINGS_DIR, name)
        # realpath guard: refuse anything that resolves outside RECORDINGS_DIR
        # (defence in depth on top of the SAFE_NAME check).
        if os.path.dirname(os.path.realpath(src)) != os.path.realpath(RECORDINGS_DIR):
            return self._send_json({"error": "forbidden"}, 403)
        if not os.path.isfile(src):
            return self._send_json({"error": "not found"}, 404)
        os.makedirs(DISCARDED_DIR, exist_ok=True)
        dst = os.path.join(DISCARDED_DIR, name)
        if os.path.exists(dst):
            stem, ext = os.path.splitext(name)
            dst = os.path.join(DISCARDED_DIR,
                               f"{stem}_{time.time_ns() % 1_000_000:06d}{ext}")
        try:
            os.replace(src, dst)  # atomic within the same filesystem
        except OSError as e:
            return self._send_json({"error": "move failed: %s" % e}, 500)
        return self._send_json({"ok": True, "name": name,
                                "discarded": os.path.basename(dst)})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # Browsers (Safari especially) open speculative TLS connections and drop
        # them; SSE streams end when a page closes. Those surface as broken-pipe /
        # reset / TLS-EOF errors that are entirely benign — don't dump a traceback.
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError,
                            ConnectionAbortedError, TimeoutError, ssl.SSLError)):
            return
        super().handle_error(request, client_address)


def main():
    if not (os.path.isfile(CERT_FILE) and os.path.isfile(KEY_FILE)):
        print("\n  Missing TLS certificate.")
        print("  Run  ./gen_cert.sh  first (it creates certs/cert.pem + certs/key.pem).\n")
        sys.exit(1)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT_FILE, KEY_FILE)

    httpd = Server((HOST, PORT), Handler)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    # Reap recording sessions abandoned mid-capture (crash/reload/Wi-Fi drop).
    def reaper():
        while True:
            time.sleep(30)
            try:
                REC.reap()
            except Exception:
                pass
    threading.Thread(target=reaper, daemon=True).start()

    ip = lan_ip()
    print("\n  Ego Recorder is running.\n")
    print("  On your iPhone (Safari):   https://%s:%d/phone" % (ip, PORT))
    print("  On this Mac (browser):     https://localhost:%d/viewer" % PORT)
    print("                       or    https://%s:%d/viewer" % (ip, PORT))
    print("\n  Recordings are saved to:   %s" % RECORDINGS_DIR)
    print("  (You'll get a self-signed cert warning the first time — tap through it.)\n")
    print("  Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping.\n")
        httpd.shutdown()


if __name__ == "__main__":
    main()
