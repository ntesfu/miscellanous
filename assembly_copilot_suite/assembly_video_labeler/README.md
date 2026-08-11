# Assembly Video Labeler

A small, self-contained web app to annotate assembly/disassembly videos in
**IndustReal PSR style** — you mark the frame where each step completes and it
saves IndustReal-compatible label files. No GPU, no model — just a lightweight
video server plus a single-page UI.

> Part of the **Assembly Copilot** project. Videos recorded with the
> [Assembly Video Recorder](../assembly_video_recorder) are annotated here; the
> resulting `PSR_labels.csv` files are what the PSR model in
> [assembly_copilot](../assembly_copilot) trains on. See the
> [top-level README](../README.md) for the full pipeline.

This whole folder is portable. Copy it to any machine, point it at your videos, run it.

```
assembly_video_labeler/
├── label_serve.py        # the server (FastAPI)
├── web/label.html        # the UI (single file)
├── run.sh                # start / stop / status
├── serve_recordings.sh   # example launcher: serves the lab's recordings folder on the LAN
├── make_proxies.sh       # optional: build 480p proxies for smooth scrubbing
├── requirements.txt
├── videos/               # ← put your recordings here (or set LABEL_VIDEOS) — not committed
├── videos_proxy/         # ← optional proxies (auto-used if present) — not committed
└── labels_out/           # ← label files are written here — not committed
```

---

## 1. Setup (once)

```bash
pip install -r requirements.txt          # fastapi + uvicorn (+ imageio-ffmpeg for proxies)
```
Python 3.8+.

## 2. Put your videos where it can find them

Either drop them in `videos/`, **or** point at an existing folder:
```bash
LABEL_VIDEOS=/path/to/your/recordings ./run.sh
```
Supported: `.mp4 .mov .avi .mkv .webm .m4v`.

## 3. Run

```bash
./run.sh            # start   → prints the URL
./run.sh status     # is it up?
./run.sh stop       # stop
```
Open **http://localhost:7862** (forward the port first if the machine is remote).

`serve_recordings.sh` is a thin wrapper around `run.sh` that bakes in the lab's
recordings path and LAN port so anyone on the same Wi-Fi can (re)start the
labeler with one command — edit the paths at the top to match your machine.

## 4. (Optional) Smooth scrubbing on high-res video

If labelling 1080p feels laggy, build lightweight 480p proxies once — the server
serves them automatically, and frame numbers stay identical so labels remain valid:
```bash
./make_proxies.sh
```

---

## Folder paths — what to set, and which endpoints use them

**All three folders are configured by environment variables.** Set them before `./run.sh`
(they default to sub-folders of this package):

| Env var | Default | What it holds | Endpoints that use it |
|---|---|---|---|
| `LABEL_VIDEOS` | `./videos` | your source recordings | `GET /api/videos` (lists them), `GET /media/video/{name}` (streams them) |
| `LABEL_PROXY` | `./videos_proxy` | optional 480p proxies | `GET /media/video/{name}` (**preferred** over the original if a same-named file exists) |
| `LABEL_OUT` | `./labels_out` | all written label + lock files | `POST /api/save`, `GET /api/labels/{name}`, `POST /api/claim`, `/api/release`, `/api/discard`, `/api/reset` |
| `LABEL_PORT` | `7862` | port to serve on | — |

Example — data on another disk, labels somewhere else, custom port:
```bash
LABEL_VIDEOS=/data/recordings \
LABEL_PROXY=/data/recordings_480p \
LABEL_OUT=/data/labels \
LABEL_PORT=8000 \
./run.sh
```

### Full API (for reference)
| Method | Endpoint | Purpose | Folder |
|---|---|---|---|
| GET | `/` | the labelling page | `web/` |
| GET | `/api/config` | parts list + defaults | — |
| GET | `/api/videos` | list videos + status (labeled / discarded / locked) | `LABEL_VIDEOS`, `LABEL_OUT` |
| GET | `/media/video/{name}` | stream a video (Range-capable) | `LABEL_PROXY` then `LABEL_VIDEOS` |
| GET | `/api/labels/{name}` | load saved marks to resume | `LABEL_OUT` |
| POST | `/api/save` | write the label files | `LABEL_OUT` |
| POST | `/api/claim` `/api/release` | video lock (multi-user) | `LABEL_OUT` |
| POST | `/api/discard` `/api/reset` | discard / clear a video | `LABEL_OUT` |

---

## How to label

1. Enter your **name** (top-left) — used to lock videos so two people don't label the same one.
2. Pick a video. Set **fps** if it isn't 30. Set **Assembly / Disassembly** (auto-guessed from the filename).
3. Scrub / play (**Space**), step frames (**← →** or **, .**), jump 1 s (**Shift+← →**), change **speed** (dropdown or **[ ]**).
4. When a step **finishes**, press its number **1–9, 0** (or click the part). The timeline fills in the step's time frame.
5. **Save** — writes to disk (see output below). Resumable anytime.

**Buttons:** **↺ Reset** clears this video's marks + saved files · **⨯ Discard** flags a video as not-to-be-labelled and skips it · **⬇ CSV** downloads the PSR csv locally.
**Progress bar** (sub-header) shows how many videos are annotated. **Hide done** removes finished/discarded videos from the dropdown.

## What gets saved (per video, into `LABEL_OUT/<video-name>/`)

- **`PSR_labels.csv`** — IndustReal format: `000878.jpg,21,Install Cowling Bracket`
  (id = `3 + 3·partIndex` for **assembly/install**, `5 + 3·partIndex` for **disassembly/remove**)
- **`segments.csv`** — `part, action, start_frame, end_frame, start_sec, end_sec`
- **`labels.json`** — full state (marks, fps, direction) to resume

### The 10 parts → IndustReal ids
| # | part | install id | remove id |
|---|---|---|---|
| 1 | Main Drive Planet Gear | 3 | 5 |
| 2 | Pop Control Ring Gear | 6 | 8 |
| 3 | Pop Control Sun Gear | 9 | 11 |
| 4 | Compressor Casing | 12 | 14 |
| 5 | Exhaust | 15 | 17 |
| 6 | Exhaust Casing | 18 | 20 |
| 7 | Cowling Bracket | 21 | 23 |
| 8 | Frame Subassembly | 24 | 26 |
| 9 | Propeller Cone plate | 27 | 29 |
| 10 (key `0`) | Propeller Cone Tip | 30 | 32 |

To change the parts, edit the `PARTS` list at the top of `label_serve.py`.

## Multi-user notes
Run **one** server; everyone opens/forwards the same port. Opening a video **claims** it
(a 120 s heartbeat keeps the claim alive; it's released when you switch away or close the tab).
Others see it as `🔒 name` and are asked before taking it over. Claims live in `LABEL_OUT`,
so even separate server instances sharing one `LABEL_OUT` folder coordinate correctly.
