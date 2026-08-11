# Assembly Video Recorder (Ego Recorder)

Turn an iPhone into a wireless, first-person (egocentric) camera for a host
computer. The iPhone captures; the host displays the live feed and records to
disk — all over local Wi-Fi, no cloud, no app to install. Pure Python standard
library, works on macOS and Linux.

> Part of the **Assembly Copilot** project. This is how the egocentric assembly
> videos are captured — a phone mounted on the operator's head or chest records
> each assembly/disassembly run into `recordings/`, which is then annotated with
> the [Assembly Video Labeler](../assembly_video_labeler). See the
> [top-level README](../README.md) for the full pipeline.

```
iPhone (Safari)  ──live video (WebRTC, peer-to-peer)──▶  Host (Viewer page)
       │                                                      ▲
       └── records locally, streams chunks over HTTP ─────────┘ ──▶ recordings/
```

## Why a page *on the host*, not "the iPhone's IP"

A web page running inside Safari can't open a listening port, so the iPhone
can't be a server you connect to. Instead the **host runs the server**, the
iPhone opens a page served by it and becomes the camera. Same result: the
phone's view on your screen, files on your disk, over the LAN.

## Setup (one time)

```bash
./gen_cert.sh        # creates a self-signed HTTPS cert with your host's LAN IP
```

Optionally create a virtual environment (the app itself needs no packages):

```bash
python3 -m venv .venv && source .venv/bin/activate
```

## Run

```bash
python3 server.py
```

It prints three URLs. Then:

1. **On the host**, open `https://localhost:8443/viewer` (accept the cert warning once).
2. **On the iPhone** (same Wi-Fi), open `https://<host-ip>:8443/phone` in **Safari**,
   tap through the certificate warning, and **Allow** camera access.
3. The live feed shows up in the Viewer. Hit **Record**.

> **Why HTTPS + a warning?** iOS only grants camera access to secure origins. The cert
> is self-signed, so Safari warns once — tap *Show Details → visit this website*.

## Features

- **Live preview** on the host with sub-second latency (WebRTC, peer-to-peer on the LAN).
- **Start / Stop recording** from the host. The iPhone encodes at full quality and streams
  the file to the host's `recordings/` folder as it records.
- **Lens picker** — Ultra-Wide (0.5×) / Wide (1×) / Tele (2×) / Front, chosen on the phone
  or remotely from the Viewer. Ultra-wide is great for a wide egocentric field of view.
- **Quality**: 720p / 1080p / 1080p60 / 4K with adjustable bitrate.
- **Torch**, **photo snapshot**, **mirror**, **fullscreen**.
- **Recordings library** in the Viewer — play back (with scrubbing), download, or discard.
- **Screen-wake-lock** on the phone so it doesn't sleep mid-capture.
- Live metrics: resolution, latency (RTT), active camera.
- Keyboard shortcuts in the Viewer: `R` record · `S` photo · `C` flip · `F` fullscreen.

## Configuration

Environment variables:

| Var        | Default   | Meaning                          |
|------------|-----------|----------------------------------|
| `EGO_PORT` | `8443`    | HTTPS port                       |
| `EGO_HOST` | `0.0.0.0` | Bind address                     |

## Notes & limitations

- The server is open to anyone on your LAN — fine for a home/lab network. Don't expose
  it to the public internet.
- iOS Safari records **MP4/H.264**; other browsers fall back to WebM. Files are named
  `YYYYMMDD_HHMMSS_<session>.<ext>` (rename them afterwards to the
  `<Person>-<Assembled|disAssembled>-<n>.mp4` convention used by the labeler).
- Lens/quality changes are blocked while recording so the file isn't corrupted.
- If your host's IP changes (new network), re-run `./gen_cert.sh` and restart.

## Files

```
server.py            HTTPS server: static hosting, SSE signaling, chunk uploads
gen_cert.sh          Self-signed cert generator (bakes in your LAN IP)
public/
  index.html         Landing page with links + setup steps
  phone.html         iPhone camera page
  viewer.html        Host display + controls
  static/
    common.js        Signaling client (SSE down, POST up)
    phone.js         Camera capture, WebRTC sender, recorder → upload
    viewer.js        WebRTC receiver, controls, recordings library
    style.css        UI
certs/               Generated TLS cert + key (created by gen_cert.sh, not committed)
recordings/          Saved videos & photos (created at runtime, not committed)
```
