# PRIS image tools

The owner's revised [PowerPoint](../SENG-F-12-HLABISA-SANELE.pptx),
[PDF export](../SENG-F-12-HLABISA-SANELE.pdf) and [practice script](../script.txt)
are the current deliverables. Their contents were preserved byte-for-byte during
cleanup. The chart builder never changes them. The final PDF has a narrow Git
ignore exception so it can be committed; general research PDFs remain ignored.

The flat [assets folder](../assets/) contains `search.png`, `comparison.png`,
`gradio_app_qr.png`, `surveillance_operator.png` and `cover_detection.png`.
The cover image is a separate white-background, 3:1 conceptual banner; its
built-in generation [prompt](cover-image-prompt.txt) is retained here.
It illustrates surveillance-event recognition, not measured detection output.
Each chart retains all its
points and category keys, with the graph on the left and the ten highest-accuracy
models in one vertical column on the right. Comparison's transformer key is `ViT`;
the saved Swin model names and metrics are unchanged.

From the paper directory, regenerate only the two charts:

```bash
implementation/.venv/bin/python presentation/pris-presentation/tools/build_slide_charts.py
```

Search uses mean validation accuracy over the four frame/FPS conditions at
32×32. Comparison uses the saved test accuracy at eight frames, 16 FPS, 64×64.
Inputs remain in `implementation/reports/2026-10-16/`. The generated
[data audit](portrait_chart_data.json) contains all rows and the protected hashes
from chart-generation time, before the owner's later deck/script revision; it
is not a validation record for the revised presentation.
The conceptual surveillance illustration is not a dataset sample; its built-in
image-editing [prompt](surveillance-edit-prompt.txt) is retained here.

## Cleanup on 26 October 2026

Unused `docs/`, `preview/`, `source/`, legacy deck-building scripts and their cache
were removed from PRIS into `/tmp/pris-unused-folders-diN2AS/` for temporary recovery.
That backup also contains the previous top-ten charts. The redundant `source/`
copies matched originals that were then present in the parent directory.
An earlier asset cleanup is recoverable from `/tmp/pris-unused-assets-MV3W1o/`.
Temporary backups are not durable archives. No research evidence was deleted.

The subsequent pre-commit cleanup consolidated the latest PPTX/PDF into PRIS;
both revised script copies were identical, so the PRIS copy was kept. The old
example deck and lock, parent-directory artwork, old image prompts, one-off QR
generator, duplicate QR and duplicate script were removed into
`/tmp/pris-precommit-cleanup-9Vgv1m/` for temporary recovery. The five selected
images are unchanged. The PPTX has embedded media and no local external media
dependencies, so removing those old source images does not break it.

Only `assets/` and `tools/` remain as PRIS subdirectories. Nothing was staged,
committed, pushed or uploaded. Check the final staged diff before committing;
tracked files moved from the parent directory may appear as deletions/additions
until Git detects the renames.
