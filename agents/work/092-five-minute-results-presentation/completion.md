# Completion

## Later owner revision and cleanup — 26 October 2026

The owner replaced the original generated deck/script. Current deliverables are
`presentation/pris-presentation/SENG-F-12-HLABISA-SANELE.pptx`, its matching PDF
export, and `script.txt`. The revised deck has four slides; the old three-slide
animation/script checks below are historical and do not validate this revision.
Cleanup preserved current file hashes, checked PPTX ZIP/XML/internal relationships
and the four-page PDF, and found no local external media dependencies. Duplicate
scripts matched exactly. Unused examples, old artwork and parent-directory
duplicates were moved to `/tmp/pris-precommit-cleanup-9Vgv1m/` for temporary recovery.
Current instructions are in the PRIS `tools/README.md`. Nothing was staged,
committed, pushed or uploaded; venue playback and timed rehearsal remain manual.

## Original automated delivery — 25 October 2026

- Status: Done
- Summary: Delivered the three-slide animated PRIS presentation and matching practice script in the requested organized folder. Artifact generation and automated/rendered checks are complete; venue-specific presentation checks remain manual.
- Changes:
  - `papers/001-journal-abnormal-activity-recognition/presentation/pris-presentation/SENG-F-12-HLABISA-SANELE-animated.pptx`: three slides, three native click reveals per slide, editable prose/diagrams, template branding, speaker notes, persistent QR/link and optional silent five-second validation video.
  - `presentation/pris-presentation/script.txt`: matching 478-word spoken script with click cues and optional 20-second demo; estimated about 4:31 at 120 words/minute plus transitions, target 4:35. No timed-rehearsal claim.
  - Supporting folders: `assets/` (three earlier generated illustrations, genuine app screenshot, real AAD media, QR, evidence-backed plots and CSV provenance), `source/` (unchanged template copies), `tools/` (reproducible builders/rendering/verifier), `docs/` (prompts, evidence, checks and editing instructions), `preview/` (all 12 click states and combined rehearsal PDF). Only final PPTX and script are at the new folder's top level.
  - Root/paper READMEs and ticket status now link to the completed presentation. Original presentation files, generic guide, original QR/helper, open guide lock, manuscript, research code/evidence and ASTT were left unchanged. No training, inference, deployment, commit or push.
- Verification:
  - `build_assets.py`: all 32 search-panel rows and all ten comparison rows use saved October 16 CSV values; source hashes unchanged. Distinguishes selected-across-conditions 98.125% validation from panel-best 98.750%, and final 99.38% test accuracy / 99.17% historical macro F1 / 1,336,459 parameters. Log parameter axes and linear accuracy axes labelled.
  - Actual validation video: 150 frames, 5.005 seconds, zero audio streams. Original video hashes retained. OpenCV decoded the QR to the exact approved Hugging Face URL.
  - Public hosted app and wrapper returned HTTP 200; real UI captured 25 October 2026 at 12:00 UTC with Normal example loaded and prediction panel blank. Hosted inference/performance/checkpoint identity were not tested.
  - `build_presentation.py` and `render_previews.py`: generated deck and all four states × three slides; static renders visually reviewed. Corrected cramped labels, title spacing and square video aspect ratio. Myriad Pro face names preserved; local rendering uses font substitution.
  - `/usr/bin/python3 .../tools/verify_presentation.py --libreoffice`: passed ZIP/XML integrity, all internal relationships, slide count/dimensions, unique valid shape/timing IDs, nine main click effects, intended phase exits/entrances, persistent QR, muted object-click media, four approved demo hyperlinks and exact script/notes correspondence. All Python tool sources compile.
  - LibreOffice read-only import of the actual animated deck: one main sequence and three ON_CLICK effects on each slide. This does not substitute for Microsoft PowerPoint Slide Show testing.
  - Source deck SHA256 remains `d1e82c197859e0d8df92af1109e7fdfc6af0b16144c38e0278195b4dce6bd1ca`; guide remains `799d97713e84b53b4a81d0e4b8c62f8e384a8dd263bbe80f268e6f2918bd5edc`. Original branding shapes/media/masters/layouts/theme preserved.
  - `git diff --check`: passed. Detailed reproducible results: presentation `docs/verification.json` and `assets/evidence/asset_provenance.json`.
- Remaining issues: Replace the approved supervisor/co-supervisor/email placeholders before presenting. Check Microsoft PowerPoint Slide Show and movie playback, venue font availability, physical phone QR scan, hosted prediction and an actual timed rehearsal. These checks were not performed here; embedded screenshot/stills provide offline fallbacks. No known unresolved package or static-layout error.
