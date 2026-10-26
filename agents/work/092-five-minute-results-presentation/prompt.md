# Task Prompt

- Ticket: `092-five-minute-results-presentation`
- Status: Done
- Later owner revision (26 October 2026): the owner supplied a new four-slide PPTX/PDF and revised script, then requested cleanup before their commit. These are now consolidated under `presentation/pris-presentation/`; unused examples and duplicates were recoverably removed. The original animated-plan details below are historical. See the revision note in `completion.md` and current `presentation/pris-presentation/tools/README.md`.
- Aim: Prepare a polished, click-revealed three-slide PRIS talk lasting at most five minutes, with verified results and an optional live demo.
- Scope: Paper 001 `presentation/`; preserve the source `SENG-F-12-HLABISA-SANELE.pptx` and save the revised deck separately as `SENG-F-12-HLABISA-SANELE-animated.pptx`, plus speaker notes, matching script and assets.
- Execution update: 25 October 2026. The owner approved execution of the attached brief and requested an organized `presentation/pris-presentation/` folder, with the final animated deck and practice script at its top level and supporting assets, tools, sources and previews below. Preserve all original files; do not touch ASTT.
- Completion update: The animated deck, matching notes/script, three generated illustrations, two verified plots, real validation media, genuine hosted-app screenshot and 12-state previews are delivered. Package checks and LibreOffice native-animation import passed. PowerPoint Slide Show, physical QR scanning and a timed presenter rehearsal remain explicitly manual; timing is estimated. See `completion.md` and the presentation's `docs/README.md`.
- Dependency: The owner reports the app is deployed at `https://huggingface.co/spaces/sanelehlabisa/lightweight-convlstm-ahar`. Its separately retrained demo checkpoint is not the archived October 16 research checkpoint.
- Changes:
  - Reuse the existing three-slide PRIS deck and branding; retain the generic guide.
  - Replace pending-experiment wording with verified October 16 evidence: problem and configurable method, search/comparison findings, then limitations and the working demo.
  - Explain the validation-selected [64,64,128] model, 160/161 test correct and 1,336,459 parameters with a readable results visual. Distinguish the archived research checkpoint from any separately retrained demo.
  - Write concise, natural speaker notes with per-slide timings and transition cues; include any demo within the five-minute limit.
  - Preserve single-seed, clip-split and short-window limitations. Do not claim global optimality, deployment reliability or that a new demo reproduces the archived test accuracy.
  - Include a static screenshot/fallback so the talk does not depend on a live application or network.
- Acceptance criteria:
  - Three editable, readable slides and a matching script form one coherent story.
  - All numerical claims trace to verified evidence; demo statements reflect actual local verification.
  - Timed rehearsal fits within five minutes, including transitions/demo; report estimated timing separately if rehearsal is unavailable.
  - Deck opens/renders correctly and original guide assets remain intact.
- Out of scope: New experiments, model changes, deployment, manuscript revisions, commit or push.
- Open questions: None. Use the owner-approved supervisor/email placeholders below; the owner will replace them later.
- Verification: Check slide count, rendered layouts and extracted text; trace claims to evidence; compare script against slides and demo provenance; time a rehearsal and run `git diff --check`.

## Owner-approved three-slide plan

- Preserve exactly three slides, PRIS branding, slide size, footer, fonts and theme. Leave the original deck, `Generic_Research_Example_3_Slides.pptx` (actual guide filename, no `(1)` suffix), QR script and QR image untouched.
- Presenter: Sanele Hlabisa. Affiliation: **University of KwaZulu-Natal**. Use `[Supervisor name]`, `[Co-supervisor, if applicable]` and `[Presenter email]`; do not invent personal details.
- At most four main clicks per slide. Initially show the title and first message; reveal grouped ideas using subtle Fade/Appear/Wipe, replacing/dimming earlier visuals when needed. No sound or distracting motion. Validate PowerPoint timing/shape references and record manual Slide Show checks separately from static inspection.

### Slide 1 — Problem, existing approaches and question

Initially show the presentation title, presenter/contact block, `Why is this problem important?` and `Manual surveillance monitoring is difficult to scale.` Click 1 reveals the operator artwork with attention/fatigue/camera-burden points. Click 2 replaces it with 3D CNN, Video Transformer and ConvLSTM families, highlighting ConvLSTM; describe capabilities and qualified computational/data trade-offs without claiming all architectures have the same size. Click 3 reveals the compact/configurable ConvLSTM research question.

### Slide 2 — From Dataset to Selected Architecture

Initially contrast broad action datasets (UCF101/Kinetics/HMDB51), small specialised surveillance datasets (AIRTLab), and selected AAD (1,069 clips, 11 classes), with real AAD example frames if useful. Click 1 shows native editable shapes for `Video → Configurable ConvLSTM stack → Global pooling → Activity class`, illustrating depth and per-layer filters. Click 2 shows 32 architectures × four temporal conditions (8/16 frames × 8/16 FPS), 128 controlled search runs at 32px, test locked. Click 3 shows a readable existing search plot and the selected performance–complexity trade-off, not a universal optimum.

Verified callouts: `ConvLSTM[64,64,128]`, **99.38% test accuracy**, **99.17% macro F1**, **1.34M parameters** (exact archived values: 99.379%, 99.172%, 1,336,459; 160/161 correct). Preserve search validation labels, input-setting context and any compressed-axis disclosure. A panel's best point is not necessarily the across-condition selected model.

Use only actual AAD frames/videos, not generated surveillance data. If practical, embed one silent 4–6 second excerpt from the existing normal/fight validation clips with click-to-play and a still fallback. The Pollution example is only ~2.87 seconds; do not silently stretch it.

### Slide 3 — Strong Performance, Smaller Model, Real Deployment

Initially show a ten-model final **test** accuracy-versus-parameters scatter (log parameter axis; model-family keys), highlighting the selected model. Include the five custom finalists, input-adapted reference, R3D-18, MC3-18 and Swin3D-T/S. Use raw saved CSV values; do not use the older validation comparison plot or pooled CSV F1 as macro F1. State highest observed accuracy under this single-seed protocol, not significant/global superiority. Approximately 99% fewer parameters than the adapted reference (134,415,195) is supported.

Click 1 reveals a real app screenshot, existing QR and `Open Live Demo` button, all linked to the supplied Space URL, plus `Upload a video → Process frames → Predict activity`. Keep QR/button visible thereafter. Do not embed a live web app or generate a fake app screenshot. Click 2 reveals multi-label recognition, continuous windows, state retention and measured deployment tests. Click 3 gives the configurable/lightweight/usable takeaway without repeating long paragraphs.

Use concise speaker notes with natural explanations and click cues: monitoring burden → choosing ConvLSTM; AAD → configurable depth/width search → selected model; ten-model comparison → optional ~20-second demo → multi-label/continuous-video limitations → takeaway. Time the entire talk within five minutes. Distinguish archived research metrics from separately retrained hosted weights and avoid unmeasured latency/reliability promises.

## Assets and evidence

- Generated conceptual artwork and exact prompts: `presentation/assets/generated/01_surveillance_operator.png`, `02_video_model_families.png`, `03_research_question.png`; `presentation/image_prompts.md`. These illustrations are not dataset examples or measured model-size evidence.
- Verified metrics/macro F1: `implementation/reports/2026-10-16/README.md` and current manuscript comparison table. Final scatter input: the same folder's `comparison_results.csv` (`model`, `architecture`, `parameters`, `test_accuracy`). Its F1 columns are pooled, not the macro callout.
- Search: `implementation/reports/2026-10-16/search_capacity.png` (all 128 measurements, four panels); `capacity_controls.png` for matched controls. A disclosed single panel has 32 points, not all 128.
- Real silent validation videos: `implementation/demo_assets/examples/normal.mp4` (~6.97s), `fight.mp4` (~6.04s), `pollution.mp4` (~2.87s). Keep original files unchanged.
- Existing QR: `presentation/gradio_app_qr.png`, encoded by the owner's `generate-qr-code.py` with the exact supplied URL.
- No actual Gradio screenshot was found in the repository. Capture one during execution or use a clearly named placeholder and capture instructions. Do not fabricate a screenshot. The web reader could not verify the live Space during planning.

## Additional acceptance and verification

Confirm the new animated deck has exactly three editable slides and no clipped/overlapping text; source/guide hashes are unchanged. Inspect/render each slide, notes, animation click order and hyperlink relationships. Verify all ten plotted test points against saved CSV values. Verify the QR payload and identify any phone-scan, PowerPoint Slide Show, live-demo or timed-rehearsal check not actually performed. Keep a screenshot/media fallback so only the optional live demo needs internet. All missing assets must be explicit. No new experiments, manuscript edits, deployment, commit or push.

## Execution Prompt

Execute ticket `092-five-minute-results-presentation` exactly as written in
`agents/work/092-five-minute-results-presentation/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the status to `Done` only when complete,
and create `completion.md` from `agents/templates/completion.md`. Do not train,
deploy, commit or push.
