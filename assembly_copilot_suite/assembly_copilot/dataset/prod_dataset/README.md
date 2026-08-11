# prod_dataset — turbofan assembly, IndustReal layout

40 annotated egocentric recordings, laid out exactly like the IndustReal root so
`psr_tas` runs against it unmodified.

```
prod_dataset/
├── <rec>.mp4                       40 videos, flat — the extractors read here
├── procedure_info.json             33-action taxonomy (this procedure's parts)
└── recordings/
    ├── train/<rec>/                30 recordings
    └── test/<rec>/                 10 recordings
        ├── PSR_labels.csv          frame.jpg,action_id,Description
        ├── PSR_labels_with_errors.csv   identical (no error events exist)
        ├── PSR_labels_raw.csv      identical (n_frames() fallback)
        ├── segments.csv            part,action,start/end frame+sec
        └── labels.json             raw marks, fps, direction
```

Split membership is the folder — the extractors and `00_build_labels.py` both
enumerate `recordings/{train,test}/`, so no bundle files are needed here.

## Split

Stratified, seed 1337. Proportional on both axes to the 22/18 direction split and
the 16/8/16 subject split.

| | n | assembly | disassembly | DA | Ketan | Nahom |
|---|---|---|---|---|---|---|
| train | 30 | 16 | 14 | 12 | 6 | 12 |
| test | 10 | 6 | 4 | 4 | 2 | 4 |

**Not subject-disjoint.** Only three people recorded this data, so all three appear
in both splits — unlike IndustReal, whose train/test participant IDs are disjoint.
Test numbers here are therefore optimistic relative to IndustReal's and the two are
not directly comparable. For a generalization figure, hold out one subject
(Nahom → 24/16, both directions present).

## Taxonomy

`procedure_info.json` mirrors IndustReal's schema: 33 actions,
`id = 3*state_idx + {0 install, 1 incorrect, 2 remove}`, `state_idx 0` = base and
never a procedure step → 10 parts, ids 3–32. `00_build_labels.py`'s `aid // 3` and
`aid % 3` arithmetic works unchanged.

Only ids ≡0 and ≡2 (mod 3) occur — **there are no `incorrect` events in this data**,
so the TYPE head's `incorrect` class has zero support. TYPE here is effectively
assembly-vs-disassembly, and because direction is fixed per video it is constant
within a recording. Treat any TYPE metric as a smoke test, not a result.

`00_build_labels.py` hardcodes `PROC_INFO` to IndustReal's copy — point it at this
file (add a `--proc_info` flag, or override the constant) or the step classes come
out named after IndustReal's chassis parts.

## Properties vs IndustReal

| | IndustReal | this dataset |
|---|---|---|
| recordings | 84 (36/32/16) | 40 (30/10) |
| video | 1280×720 @ 10 fps | 1920×1080 @ 30 fps |
| PSR rows | 762 | 398 |
| install / remove / incorrect | 560 / 164 / 38 | 253 / 145 / **0** |
| background after densify | 3.0% | 2.9% |
| segments | ~10 per rec | exactly 10 per rec (2 recs have 9) |

`Ketan-Assembled-3.mp4` carries no container `nb_frames` tag; `PSR_labels_raw.csv`
is present in every recording dir so `n_frames()`'s fallback path works. That
fallback truncates the video at the last labelled frame, dropping trailing
background — prefer patching `n_frames()` to use `duration × r_frame_rate`.

## Provenance

Videos recorded by `web_app` (`{Person}-{Stage}-{N}.mp4`), annotated with
`assembly_labeler`. Source of truth is
`assembly_copilot/dataset/annotated_data/`; the 40 videos here are byte-identical
copies. The 9 discarded recordings are excluded.
