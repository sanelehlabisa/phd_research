# Task Prompt

- Ticket: `087-expand-frame-depth-search`
- Status: Ready
- Aim: Prepare 32 custom architectures at fixed 64x64 resolution, varying 8/16 frames and 8/16 FPS, with compact results backed up to Drive.
- Scope: One new JSON profile on the existing runner; minimal frame-aware cache/ranking/report/archive updates, tests and concise guides. Notebook/export integration belongs to [089](../089-colab-campaign-readiness/prompt.md).
- Dependencies: [084](../084-diagnose-transformer-learning/completion.md) is complete; [088](../088-pooled-metrics-everywhere/prompt.md) owns pooled-only active outputs. Reuse [083](../083-comparison-only-recovery/prompt.md) for comparison preparation; this ticket does not require rerunning the historical search.
- Owner decision (15 October 2026): Fix frames at 64x64, retain at least four candidates per depth 1-5, and defer streaming. This replaces the earlier 24-model/two-resolution proposal; no implementation or launch is authorized by this planning revision.
- Evidence: [15 October graph](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-15/README.md). Current 16 architectures have depth counts 1/1/12/1/1 and only eight frames. The two-layer [64,64] averaged 97.500% validation accuracy; compact [48,48,48] averaged 97.8125%. Shallow width/shape and frame-count effects merit testing; these observations do not guarantee improvements.

## Fixed candidate manifest

Preserve the original 16 architectures and earlier planned shallow variants;
add matched flat-width controls across depths. Exactly 32 unique candidates:

| Depth | Filters per layer |
|---|---|
| 1 (5 models) | [32]; [48]; [64]; [96]; [128] |
| 2 (6 models) | [32,32]; [48,48]; [64,64]; [96,96]; [32,64]; [64,32] |
| 3 (13 models) | [96,64,64]; [32,64,64]; [64,96,64]; [48,48,48]; [64,128,64]; [64,64,128]; [128,64,64]; [64,64,32]; [96,96,96]; [16,16,16]; [32,32,32]; [64,64,64]; [128,128,128] |
| 4 (4 models) | [32,32,32,32]; [48,48,48,48]; [64,64,64,64]; [96,96,96,96] |
| 5 (4 models) | [32,32,32,32,32]; [48,48,48,48,48]; [64,64,64,64,64]; [96,96,96,96,96] |

- Exactly **32 models x num_frames [8,16] x frame_sizes [64] x FPS [8,16] = 128 search configurations**. Kernels stay 3x3. Depth counts are 5/6/13/4/4, with flat widths 32/48/64/96 represented at every depth. Disclose uneven/evidence-guided sampling; do not predeclare depth three the winner or claim a global optimum.
- Keep AAD/all discovered classes, the reviewed 748/160/161 split, seed 42, Adam LR 0.001, WD 0, augmentation off and existing scheduler. Search budget remains cap 256/minimum 64/patience 24; validation-loss checkpoint selection. No new LR/WD/seed/augmentation sweep.
- The prior 64-job search took about 10.10 training hours. More jobs and doubled frame counts may require multiple sessions; no overnight/runtime guarantee. Measure full-step memory/throughput at 16 frames/64px and print workload/storage estimates before owner launch. Never silently skip configurations or shrink inputs to fit.

- Changes:
  - Define the manifest and input lists once in the new JSON; reuse `src.experiments --config <json>`. Include frame count in every condition/job/cache/resume identity, not merely in display labels. Preserve old eight-frame protocols/results; reuse older work only when complete configuration and provenance verification genuinely permit it.
  - Process/cache each input condition once, retaining identical split/classes/seed/recipe across all jobs. Rank by equal-weight mean validation accuracy across all four frame-count/FPS conditions, then worst condition, mean loss, parameters and stable name. This cross-condition mean is not macro class averaging. Keep test locked; record temporal coverage and padding. At fixed frame count, changing FPS changes time coverage, not input tensor size.
  - Plot every measured point in four frame-count/FPS panels, all at 64px, preserving readable depth legends. Include matched width/depth/placement comparisons and the compact [48,48,48] control. Do not infer overfitting from accuracy alone; retain matched train/validation histories. Report the best tested accuracy-size trade-off, not proof of global or three-layer optimality.
  - Produce a new validation-selected shortlist/common-input receipt for a future fresh ten-model comparison through the same runner. The historical search/current 083 shortlist remains unchanged. After transformer recipe approval, compare the five new finalists with the adapted paper model, R3D-18, MC3-18 and Swin3D-T/S under one declared common input/approved recipe; freeze before guarded final test. Do not silently mix old comparison scores with the new campaign.
  - Reuse the existing results-only ZIP helper. Include all search and final comparison configs, code/environment/split provenance, ranking/selection/freeze/test receipts, CSV/JSON metrics, class counts/confusion matrices, histories/curves and prediction records/examples. Exclude datasets and weight files by default. Keep necessary runtime checkpoints until selection/evaluation finish; do not delete old weights or invalidate resume guards. A metrics-only ZIP cannot restore missing weights or repeat evaluation without retraining.
  - Save verified compact ZIPs to an editable Google Drive folder after each completed input block, before comparison, after each comparison model/test and on completion/handled failure. Default Colab destination: `/content/drive/MyDrive/phd_research/paper-001/<study-id>/results.zip`. Mount Drive with owner authorization and preflight the destination before expensive work. Use a verified temporary copy then replace only this campaign's backup; print destination, byte count and matching hash. Preserve a local ZIP and report Drive failure clearly; no browser-only delivery guarantee, credentials in Git or forced-timeout safety promise.
- Acceptance criteria:
  - List-plan shows exactly 32 unique models and 128 configurations, all at 64x64, with at least four candidates per depth 1-5 and the original 16 retained. Frame counts/FPS cannot share cache/job keys accidentally; historical evidence/protocols are unchanged.
  - Complete four-condition ranking, shortlist and input selection reproduce from validation artifacts; missing/failed coverage cannot silently become a complete ranking. Each ablation states its controlled factors; new outputs follow 088's pooled/per-class metric contract.
  - Synthetic search/comparison fixtures produce a verified weights-excluded ZIP and matching destination copy through the same export path intended for the real study. Failed/unmounted Drive is diagnosed before expensive launch; handled backup failure preserves local artifacts. Do not claim real results or a live Drive upload from stand-in tests.
  - Local synthetic verification covers JSON/plan, frame-aware cache/resume/ranking, unchanged old profiles, new shortlist handoff, archive contents and Drive-copy success/failure using a temporary filesystem stand-in. Real Colab training/Drive authorization remains owner-run work.
- Out of scope: Executing 128 real trainings, changing old graph values, deleting checkpoints/results, streaming/state carry, notebook rewiring (089), unrestricted model generation, new datasets/families, test tuning, manuscript edits, commit or push during implementation.
- Open questions: None for preparing this bounded profile. The owner may edit the Drive destination before launch; any transformer recipe change requires separate approval from 084. Additional candidates or a smaller budget require an explicit revision, not silent filtering.
- Verification: Manifest/count and one-command list-plan; full-step memory/throughput smoke where available; synthetic four-condition cache/rank/resume/selection checks; ZIP inventory/no-weight assertions; atomic-copy/hash/failure tests; guide links and `git diff --check`. Never claim CPU or synthetic checks establish A100 runtime or Drive durability.

## Execution Prompt

Execute ticket `087-expand-frame-depth-search` exactly as written here after its
dependencies are satisfied. Follow `AGENTS.md`, `agents/rules.md`, and
`agents/config.md`. Implement only the approved preparation, verify every
acceptance criterion, mark `Done`, and create `completion.md` from the template.
Do not launch real search/comparison training, access test, commit or push.
