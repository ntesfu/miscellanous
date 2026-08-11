# aiops_parts_detection_frames

YOLO detection dataset built from **video frames** of the engine assembly (egocentric).
Frames are sparse: most show only a few parts, so only frames WITH boxes are included.

- Images: 114 labeled frames (of 140 total; 26 empty frames excluded)
- Boxes: 587 | Classes: 10 | Resolutions: 1920x1080 and 1280x720

## Per-class box counts
- 0 Main Drive Planet Gear: 16
- 1 Pop Control Ring Gear: 14
- 2 Pop Control Sun Gear: 27
- 3 Compressor Casing: 36
- 4 Exhaust: 90
- 5 Exhaust Casing: 60
- 6 Cowling Bracket: 57
- 7 Frame Subassembly: 59
- 8 Propeller Cone plate: 168
- 9 Propeller Cone Tip: 60

## Layout
images/ labels/(YOLO) classes.txt data.yaml splits/{train,val}.txt annotations/{coco,native_project}.json

## Train (Ultralytics)
yolo detect train data=/home/aiops/AIOps/data/raw/aiops_parts_detection_frames/data.yaml model=yolo11s.pt imgsz=1280 epochs=150

Note: the 26 empty frames were excluded per instruction; they can be re-added as
background negatives later if you want to reduce false positives.
