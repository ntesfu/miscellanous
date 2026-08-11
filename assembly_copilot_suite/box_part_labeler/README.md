# Box Part Labeler

A standalone, **single-file** browser tool for drawing part bounding boxes on
photographs of the engine-model assembly parts. Everything — UI, storage, and
COCO/YOLO export — runs client-side in one HTML file.

> Part of the **Assembly Copilot** project. Frames extracted from the recorded
> assembly videos (or standalone part photos) are annotated here with per-part
> bounding boxes; the exported YOLO/COCO bundles feed the part detector trained
> in [assembly_copilot](../assembly_copilot). See the
> [top-level README](../README.md) for the full pipeline.

## Run it

No build step, no server, no dependencies. Open the file directly:

```
box_part_labeler/bbox_labeler.html
```

Double-click it, or `File > Open` in any recent desktop browser (Chrome/Edge/Firefox —
uses `createImageBitmap`, `IndexedDB`, and Pointer Events). Everything runs client-side.

## Workflow

1. **Import folder** (or drag a folder of photos onto the drop zone). Images are
   matched by filename; re-importing the same folder relinks instead of duplicating.
2. The **checklist** on the right shows the ~10 part labels for the current image.
   The active label is highlighted — drag on the photo to draw a box for it.
3. Each new box **auto-advances** to the next unboxed label. Once every label has a
   box, the image is marked **complete** (badge in the left tray).
4. Move to the next image with `→` / `]`, previous with `←` / `[`, or click a thumbnail.
5. Any box can be reselected (click it or its row in **Boxes on this image**), dragged,
   resized via its corner/edge handles, relabeled (dropdown, or select + press a number
   key), or deleted (`Delete`/`Backspace`).
6. Work autosaves to the browser's IndexedDB ~500 ms after every edit. **Save project**
   exports a portable native JSON checkpoint; **Import project JSON** restores one
   (then re-import the image folder to relink the photos — file bytes are never stored
   in the project document, only `file_name` + dimensions).
7. **Validate** before exporting — reports incomplete images, out-of-range or
   zero-size boxes, duplicate labels, and unknown `label_id`s.
8. **Export bundle** downloads a ZIP with COCO JSON, per-image YOLO `.txt` +
   `classes.txt` + `data.yaml`, and the native project JSON.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `1`-`9`, `0` | No box selected: set the active label. Box selected: relabel it. |
| `[` / `]`, `←` / `→` | Previous / next image |
| `Delete` / `Backspace` | Delete the selected box |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo (bounded 60-step history) |
| `Escape` | Deselect the current box |

## Label list

Editable under **Project > Label list**, one line per label:

```
id | Display name | boxes
```

`boxes` is optional and defaults to `1` — it's how many boxes that part needs per image.
List order fixes each label's `class_id` for COCO/YOLO export. Changing labels re-stamps
`class_id` on every existing box; a box whose `label_id` no longer exists is kept (not
deleted) and flagged by **Validate**. **Reset to defaults** restores the shipped list —
use it if an older autosave restored a stale vocabulary.

The default 10 parts mirror the `PARTS` list served by the
[Assembly Video Labeler](../assembly_video_labeler)'s `label_serve.py` (its
`/api/config` endpoint), so the two tools stay in sync:

| class_id | id | industreal_id | boxes | name |
|--:|---|--:|--:|---|
| 0 | `main_drive_planet_gear` | 3 | 1 | Main Drive Planet Gear |
| 1 | `pop_control_ring_gear` | 6 | 1 | Pop Control Ring Gear |
| 2 | `pop_control_sun_gear` | 9 | 1 | Pop Control Sun Gear |
| 3 | `compressor_casing` | 12 | 1 | Compressor Casing |
| 4 | `exhaust` | 15 | **2** | Exhaust |
| 5 | `exhaust_casing` | 18 | 1 | Exhaust Casing |
| 6 | `cowling_bracket` | 21 | 1 | Cowling Bracket |
| 7 | `frame_subassembly` | 24 | 1 | Frame Subassembly |
| 8 | `propeller_cone_plate` | 27 | **2** | Propeller Cone plate |
| 9 | `propeller_cone_tip` | 30 | 1 | Propeller Cone Tip |

So a complete image has **12 boxes**, not 10.

### Multi-box parts and instance numbering

`Exhaust` and `Propeller Cone plate` each need **two** boxes. The guided flow **stays on
a multi-box part until all its boxes are drawn** (the checklist shows `1/2`, then `2/2`),
then advances — this holds even if you label out of order.

Every box gets a stable 1-based **instance** number within its label on that image, so
the two boxes of a pair are individually identifiable:

- **On the canvas** the tag reads `Exhaust #1` / `Exhaust #2` (single-box parts stay
  untagged, so nothing gets noisier).
- **In the box list** a `#1` / `#2` chip sits next to the label — so you can see exactly
  which one you're about to delete.
- **In the saved/exported records** each box carries `instance: 2` and
  `instance_key: "exhaust#2"`, in both the native JSON and COCO. `instance_key` is unique
  per image, so you can drop a specific box programmatically:

  ```python
  boxes = [b for b in img["boxes"] if b["instance_key"] != "exhaust#2"]
  ```

Instance numbers are derived from creation order and renumber contiguously if you delete
one — delete `exhaust#1` and the remaining box becomes `exhaust#1`. They're an
*identifier within an image*, not a claim about which physical piece is which.

**YOLO is unaffected** — it has no instance concept; two boxes of the same class on one
image is the normal, correct representation. The instance data lives only in the native
JSON and COCO.

`industreal_id` (`3 + 3 × class_id`, matching the video labeler's `part_id(i)`) is
carried in the **native JSON export only** — it's the join key between a detected box
here and a step-completion event from the video labeler. COCO/YOLO keep the contiguous
`0..N-1` `class_id` they require. **Caveat:** the id is derived from list position, so
it is only meaningful while this list matches the video labeler's `PARTS` in the same
order — if you reorder or insert labels, it silently re-maps.

## Data model

Coordinates are stored normalized `[0,1]` (`bbox_norm: {x,y,w,h}`, top-left + fraction
of width/height) so annotations are exact regardless of display size or window resize —
the same letterboxing math as the video labeler, adapted from video frames to static
photos.

## Design notes worth knowing about

- **Memory**: only the *active* image's full-resolution bitmap is ever decoded at once
  (closed and released on navigation); every image gets a small (~200 px) thumbnail
  generated once at import and cached for the tray. This matters at ~100 photos —
  keeping all of them decoded at full resolution simultaneously would be several GB.
- **ZIP writer**: hand-rolled, store-only (uncompressed) — no JSZip/CDN dependency, so
  the tool works fully offline. It does not set the UTF-8 filename flag, so non-ASCII
  filenames in an exported bundle are a known limitation (not a concern for plain-ASCII
  photo names).
- **Completeness rule**: an image is `complete` when it has ≥1 box for *every* label —
  this assumes all parts are visible in every photo. There's no "N/A / part absent"
  escape hatch — duplicate boxes on the same label are allowed (the guided flow is
  non-blocking) and flagged by Validate rather than prevented.

## Verified

Driven end-to-end in headless Chrome (Playwright) against 35 real engine photos, with
zero console errors and zero uncaught page errors:

- 35 photos imported, thumbnails generated, first photo letterboxed without distortion
- 10 boxes drawn via real mouse drags → checklist ticked, image badged **complete**,
  progress counters updated
- box selected, **resized** by its corner handle, and **relabeled** via the dropdown
- **undo/redo** round-tripped
- **Validate** reported 0 errors (warnings correctly flagged a deliberately duplicated
  label and the 34 not-yet-labeled images)
- **page reload → re-import folder** restored all image records and all boxes at
  their exact original coordinates (autosave + relink round-trip)
- **Export bundle** produced a ZIP that Python's `zipfile` opens and CRC-checks clean

The key numeric check: a box's normalized coords de-normalize to
`[269.04, 299.52, 129.56, 188.16]` px, and the COCO annotation for that same box reads
`[269.04, 299.52, 129.56, 188.16]` — exact match, with the YOLO line carrying the
equivalent center-based normalized form.
