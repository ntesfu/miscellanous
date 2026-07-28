# Procedure Atlas dataset labeling report

The current source is `procedure_atlas_dataset_labeling_report.tex`. Its
screenshots are stored in `figures/`; keep that folder beside the TeX file when
compiling.

Compile with either:

```powershell
tectonic procedure_atlas_dataset_labeling_report.tex
```

or a standard TeX Live / MiKTeX installation:

```powershell
latexmk -pdf procedure_atlas_dataset_labeling_report.tex
```

The paper contains its rendered bibliography directly for dependable one-command
compilation. `references.bib` is also supplied as a reusable citation database.
The screenshots use a 20-second annotation proxy extracted from the supplied
demo recording `dataset/dataset/20260716_012249_HoloLens.mp4`. Video bytes are
not embedded in the LaTeX source or PDF. Labels shown in the paper are
illustrative workflow examples and have not been adjudicated as ground truth.
