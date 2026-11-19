# Mechanism-focused Kaiti presentation

Build from the workspace root:

```bash
PolicyGen/.venv/bin/python data_generation_methods/reports/pipeline_explained_20261021/build_report.py \
  --out /mnt/sdg/datasets/real_sim_human_pipeline/pipeline_explained_20261021/presentation/NEW_EXPORT
```

The report uses existing evidence under `pipeline_explained_20261021/` and the prior
`report_control_20261021/` run. No model calls or simulation occur. All imagery is
actual source frames or explicitly declared presentation crops; no generated
diagram images are used. Shapes, arrows and text are laid out in code.

Chinese font: AR PL UKai CN, extracted from Ubuntu `fonts-arphic-ukai`
0.2.20080216.2-5. Latin font: Liberation Serif regular/bold, upright. The font
directory and bundled licenses are in the SDG run directory. The source package
hash is recorded in `fonts/font_manifest.json`.

Two PPTX files are exported. The portable edition stores each visible text shape
as a high-resolution transparent image with source text in its accessibility
description, so missing local fonts cannot replace Kaiti. Images, connectors and
videos remain separate objects. The editable edition retains native text with
explicit Latin and East Asian font references; the accompanying ZIP contains
the required fonts and licenses. There is no claim that the portable text is
editable or that the fonts are embedded in the native-text edition.

PNG/PDF previews share the same layout and fonts. They are not native PowerPoint
renders, and structural validation does not establish playback compatibility in
every office application. Original videos are embedded and source hashes checked.

Final evidence: `presentation_export_validation.json`, independent review files
under `source_contract/`, `sim_assets/`, and `quality_repair/`, and `delivery.json`.
