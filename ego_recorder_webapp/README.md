# 🎥 Ego Recorder

Turn your iPhone into a wireless, first-person (ego-centric) camera for your Mac.
The iPhone captures; your Mac displays the live feed and records to disk — all over
your local Wi-Fi, no cloud, no app to install. Pure Python standard library.

```
iPhone (Safari)  ──live video (WebRTC, peer-to-peer)──▶  Mac (Viewer page)
       │                                                      ▲
       └── records locally, streams chunks over HTTP ─────────┘ ──▶ recordings/
```

## Why a page *on the Mac*, not "the iPhone's IP"

A web page running inside Safari can't open a listening port, so the iPhone can't be
a server you connect to. Instead the **Mac runs the server**, the iPhone opens a page
served by it and becomes the camera. Same result: the phone's view on your Mac, files
on your Mac, over the LAN.

## Setup (one time)

From this directory:

```bash
./gen_cert.sh        # creates a self-signed HTTPS cert with this machine's LAN IP
```

`gen_cert.sh` detects the LAN IP via macOS's `ipconfig getifaddr en0`/`en1`. On
Linux, either run it with that call replaced (e.g. `hostname -I | awk '{print $1}'`)
or generate the cert by hand with `openssl` using `certs/openssl.cnf` as a template.

The app itself needs no packages (pure Python standard library) — a venv is
optional:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

**Optional but recommended:** install `ffmpeg` (`brew install ffmpeg` /
`apt install ffmpeg`). If present, the server re-times each finished recording
in the background so it plays back smoothly (iOS Safari's H.264 capture has
scrambled timestamps and a discard-frame layer — see `normalize_recording()`
in `server.py`). Without ffmpeg, recordings are still saved fine, just with
choppier native playback in some players.

## Run

```bash
python3 server.py
```

It prints three URLs. Then:

1. **On the Mac**, open `https://localhost:8443/viewer` (accept the cert warning once).
2. **On the iPhone** (same Wi-Fi), open `https://<mac-ip>:8443/phone` in **Safari**,
   tap through the certificate warning, and **Allow** camera access.
3. The live feed shows up in the Viewer. Hit **Record**.

> **Why HTTPS + a warning?** iOS only grants camera access to secure origins. The cert
> is self-signed, so Safari warns once — tap *Show Details → visit this website*.

## Labeling a recording

The Viewer has **Name** and **Stage** dropdowns above the Record button. Pick
both before you hit Record and the finished file is auto-named
`{Name}-{Stage}-{N}.mp4` (the trailing number auto-increments so repeats never
collide) instead of the default `YYYYMMDD_HHMMSS_<session>.mp4`. This is how a
batch of recordings ends up self-organized by person and assembly stage.

The dropdown options (`Ketan`/`DA`/`Nahom` for Name, `Assembled`/`disAssembled`
for Stage) are hardcoded in `public/viewer.html` — edit the `<option>` lists
there for a different set of people/stages.

If you record without picking a label (e.g. from the phone's own button), the
file just keeps the default timestamped name.

## Reviewing recordings

The Viewer's **Recordings** panel lists every saved file with playback
(scrubbing supported) and a **Discard** action, which moves a file into
`recordings/discarded/` (not deleted, just out of the main library) rather
than removing it outright.

## Features

- **Live preview** on the Mac with sub-second latency (WebRTC, peer-to-peer on the LAN).
- **Start / Stop recording** from the Mac. The iPhone encodes at full quality and streams
  the file to the Mac's `recordings/` folder as it records.
- **Lens picker** — Ultra-Wide (0.5×) / Wide (1×) / Tele (2×) / Front, chosen on the phone
  or remotely from the Viewer. Ultra-wide is great for a wide ego-centric field of view.
- **Quality**: 720p / 1080p / 1080p60 / 4K with adjustable bitrate.
- **Torch**, **photo snapshot**, **mirror**, **fullscreen**.
- **Recordings library** in the Viewer — play back (with scrubbing) or download.
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
  `YYYYMMDD_HHMMSS_<session>.<ext>`.
- Lens/quality changes are blocked while recording so the file isn't corrupted.
- If your Mac's IP changes (new network), re-run `./gen_cert.sh` and restart.

## Files

```
server.py            HTTPS server: static hosting, SSE signaling, chunk uploads
gen_cert.sh          Self-signed cert generator (bakes in your LAN IP)
.gitignore           excludes certs/, recordings/, .venv/, __pycache__/
public/
  index.html         Landing page with links + setup steps
  phone.html         iPhone camera page
  viewer.html        Mac display + controls
  static/
    common.js        Signaling client (SSE down, POST up)
    phone.js         Camera capture, WebRTC sender, recorder → upload
    viewer.js        WebRTC receiver, controls, recordings library
    style.css        UI
```

Two directories are created at runtime and are **not** part of this repo
(gitignored — the cert is machine-specific, and recordings are captured video,
not source):

- `certs/` — `cert.pem` + `key.pem` from `./gen_cert.sh`. **Never commit
  `key.pem`** — it's a private key.
- `recordings/` — saved videos/photos, plus `recordings/discarded/` for
  discarded ones.
