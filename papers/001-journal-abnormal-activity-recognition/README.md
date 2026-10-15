# Journal Paper: Abnormal Activity Recognition

- Status: 64 AAD search jobs and ten comparison trainings reviewed; final test failed at a dataset-name compatibility check. See the [October 14 preview](implementation/reports/2026-10-14/README.md) for validation plots, metrics and urgent artifact preservation.
- Target submission: December 2026
- Manuscript: [`manuscript/main.tex`](manuscript/main.tex)
- Code and detailed experiment plan: [`implementation/README.md`](implementation/README.md)
- Dated Colab reports: [`implementation/reports/README.md`](implementation/reports/README.md)

AAD is the primary dataset for the active study. VDD remains available as an optional dataset
choice in configuration; its existing run artifacts will be kept, but it is not
part of the current study. Do not treat the earlier validation result as final.
See the [implementation guide](implementation/README.md) for the current code
status and ticket links.

The paper studies lightweight ConvLSTM-based abnormal human-activity recognition
from surveillance video. Earlier width experiments made `64-32-16` a useful
candidate, but not a proven optimum. The revised study will compare the audited
source-paper topology with narrow stacked ConvLSTM models before selecting a
reference architecture for ablation.

## Evidence plan

1. Run 077's 64 balanced validation-only jobs: 16 architectures, 32/64px,
   8/16 FPS and eight frames; cap 256 epochs. Rank across all four conditions,
   preserving the disclosed [96,64,64] anchor slot if outside the numerical top five.
2. Freeze one common size/FPS using the five finalists, retaining LR 0.001/WD 0
   from the successful prior recipe. Train all five customs, the input-adapted
   paper model, two 3D CNNs and two transformers from scratch, cap 512 epochs.
   Full-step checks freeze common batches; the total is 74 trainings, with no
   extra recipe or seed sweep. The older 075 workflow remains available unchanged.
3. Keep one existing split (seed 42); the final comparison uses one
   training seed, 42. Freeze all validation-selected checkpoints and the best
   custom model before ten full test evaluations. Run All passes verified
   paths automatically. The owner confirmed different recordings behind similar
   filenames; this does not establish subject/scene independence.
4. Save compact results during execution and checkpoint evidence separately,
   then review complete artifacts before manuscript claims. This focused run
   uses one seed; input variation does not estimate seed uncertainty or prove
   global optimality, unseen-data generalisation or subject/scene independence.

See the [October 10 interim analysis](implementation/reports/2026-10-10/todays.results.md)
for grouped results, a representative main-paper table and the complete available
search appendix. These are console-derived validation results, not a final test report.
The latest appended review includes two-seed confirmation and WD/temporal tables;
it informs 075's candidate refinement without changing the 64-base-run budget.

## Next tasks

- [x] [084 transformer learning diagnosis](../../agents/work/084-diagnose-transformer-learning/completion.md) — no proven production bug. A separate [lower-LR follow-up](implementation/reports/2026-10-15/transformer-learning/follow-up.md) restores Swin3D-T learning locally (85.625% validation); Swin3D-S and final comparison remain unverified under that recipe.
- [ ] [083 comparison-only JSON/test recovery](../../agents/work/083-comparison-only-recovery/prompt.md) — next, after any recipe change is approved: five existing finalists + paper + two 3D CNNs + two transformers; local 6-GB CUDA preflight, Colab fallback, final test metrics/confusion/examples. No search rerun.
- [ ] [088 pooled metrics everywhere](../../agents/work/088-pooled-metrics-everywhere/prompt.md) — remove macro outputs from all new active modular/campaign results; retain pooled totals, per-class metrics and immutable historical evidence.
- [ ] [087 expanded 64px search](../../agents/work/087-expand-frame-depth-search/prompt.md) — 32 architectures at depths 1-5 (at least four per depth), fixed 64x64, 8/16 frames and 8/16 FPS: 128 jobs. Retain the original candidates; back up compact results to Drive. Planning only.
- [ ] [089 Colab readiness](../../agents/work/089-colab-campaign-readiness/prompt.md) — integrate 088/087 and 083's comparison path into one thin notebook/export, with recipe/preflight checks and backups before launch. Streaming remains deferred.
- [ ] [086 modular-script cleanup plan](../../agents/work/086-simplify-modular-scripts/prompt.md) — later: audit callers and propose exact removals/merges for approval; do not disrupt the comparison work.
- [x] [085 complete search visibility](../../agents/work/085-complete-search-result-visibility/completion.md) — [all 64 plotted measurements](implementation/reports/2026-10-15/README.md), layer-count-only bottom legend, named Best/Middle/Worst panel keys and a disclosed compressed lower y-range. No arrows/zooms/performance tables; results/ranking unchanged.
- [x] [077 focused A100 rerun](../../agents/work/077-focused-aad-colab-rerun/prompt.md) — [Locally verified](../../agents/work/077-focused-aad-colab-rerun/completion.md): 64 balanced jobs, honest top-five/anchor selection and compact backups. All 64 Colab search jobs reviewed on October 14; full artifact recovery pending.
- [x] [076 fixed finalist comparison](../../agents/work/076-aad-shortlist-comparison/prompt.md) — [Locally verified](../../agents/work/076-aad-shortlist-comparison/completion.md): ten fresh comparison trainings completed; validation reviewed. The first test failed before scoring; test metrics/examples remain pending.
- [ ] [078 clearer ablation figures](../../agents/work/078-aad-ablation-figures/prompt.md) — tally-style layer markers and condition/width/depth plots; figures lead the paper narrative, tables support it.
- [x] [075 wider/deeper capacity search](../../agents/work/075-wide-depth-capacity-search/prompt.md) - [Locally verified](../../agents/work/075-wide-depth-capacity-search/completion.md): 32 architectures, depths 1-5; 64 base runs; 256-epoch search cap; 16f/64px final input, selected FPS/recipe, adapted baseline and per-class metrics. Up to 142 trainings overall; real Colab execution pending.
- [x] [074 validated comparison handoff](../../agents/work/074-validated-comparison-handoff/prompt.md) - recipe validation, cache reuse and stable experiment numbering; [70 local tests passed](../../agents/work/074-validated-comparison-handoff/completion.md). Real Colab execution pending.
- [x] [073 focused shape search and Run All](../../agents/work/073-focused-shape-search-run-all/prompt.md) - consolidates 070–072; [locally verified](../../agents/work/073-focused-shape-search-run-all/completion.md), with 128-epoch search, 512-epoch comparison, exact ranking and automatic guarded export. Real Colab execution remains pending.
- [x] [069 staged capacity search](../../agents/work/069-staged-capacity-search/prompt.md) - implemented and locally verified; see [completion](../../agents/work/069-staged-capacity-search/completion.md). Current 068 runs and final-comparison protocol stay intact; actual AAD/A100 runs are separate work.
- [x] [068 multi-resolution search and top-three comparison](../../agents/work/068-multisize-search-top3-comparison/prompt.md) - implementation verified; [36-run Colab search reviewed](implementation/reports/2026-10-09/2026-10-09-multiresolution-search-review.md), comparison/test still pending. Results are clip-level validation evidence, not established source-independent generalization.
- [x] [057 config-driven datasets and classes](../../agents/work/057-configurable-datasets-and-classes/prompt.md)
- [x] [058 separate custom search and model-family comparison](../../agents/work/058-separate-search-and-comparison/prompt.md) — two AAD profiles and one local smoke profile on the same runner.
- [x] [059 retire VDD-only workflow code](../../agents/work/059-retire-vdd-specific-workflow/prompt.md) — VDD remains optional through generic dataset configuration; old runs are preserved.
- [x] [060 align documentation](../../agents/work/060-align-dataset-documentation/prompt.md)
- [x] [061 thin Colab runner and artifact download](../../agents/work/061-colab-script-runner-and-artifacts/prompt.md) — one-cell notebook/export runs the two active JSON profiles and downloads run artifacts.
- [ ] [062 update manuscript after verified AAD results](../../agents/work/062-update-manuscript-after-aad-results/prompt.md) — do not execute before the new study is complete.
- [x] [063 prepare PRIS 2026 flash presentation](../../agents/work/063-pris-2026-flash-presentation/prompt.md) — current three-slide deck and speaker notes are ready. Revisit after AAD results to add a verified results visual and useful model/example images.

After final results, prepare a small Gradio demo for presentation.

## Tasks

- [x] [`049-aad-dataset-auto-resolution`](../../agents/work/049-aad-dataset-auto-resolution/prompt.md) — shared local-first AAD resolver and optional Kaggle download for modular runners and notebook selection.
- [x] [`047-larger-colab-training`](../../agents/work/047-larger-colab-training/prompt.md) — larger, eight-hour-bounded notebook-03 profile verified locally; real Colab run pending and separate from AAD experiments.
- [x] [`042-merge-expanded-kinetics-suite`](../../agents/work/042-merge-expanded-kinetics-suite/prompt.md) — Integrated both laptops' work: Kinetics-600 (>2,000 videos before splitting), 14-model screen, spatial/weight-decay comparisons and two-seed confirmation within an eight-hour deadline. 115 local tests passed; real Colab run pending.
- [x] [`043-download-colab-run-artifacts`](../../agents/work/043-download-colab-run-artifacts/prompt.md) — Downloads one ZIP of the current suite's saved artifacts after completion or a handled time limit.
- [x] [`044-organize-notebook-helpers`](../../agents/work/044-organize-notebook-helpers/prompt.md) — Moved notebook-only helpers to `implementation/notebooks/utils/`; kept core CLI modules in `src/`.
- [x] [`045-stabilize-colab-script-runtime`](../../agents/work/045-stabilize-colab-script-runtime/prompt.md) — Added a fresh-process NumPy/import integrity check, targeted wheel repair, concise restart handling, and quiet cache reuse.
- [ ] `009-define-ablation-config` — superseded by ticket 017; its preserved WIP
  remains stashed and must not be applied to the current runners.
- [x] `010-prepare-architecture-search` — added the faithful paper baseline,
  sequence-returning ConvLSTM, explicit layer specifications, and lightweight
  stacked model.
- [x] `011-simplify-model-api` — kept one ConvLSTM layer plus the paper and
  custom models, removed obsolete architectures, and migrated every caller.
- [x] `012-audit-and-align-baselines` — verified the published ConvLSTM topology
  and separated the source paper's 3D ResNet-50/101/152 comparisons from this
  study's `r3d_18`, `mc3_18`, and `r2plus1d_18` practical baselines.
- [x] `013-fix-video-augmentation` — replaced augmented copies with optional,
  clip-consistent online augmentation and documented runnable AAD commands.
- [x] `014-standardize-run-artifacts` — separated model, training, evaluation,
  and comparison evidence into local timestamped run directories.
- [x] `015-make-splits-reproducible` — seeded the full pipeline and added a
  fixed, stratified 70:15:15 clip manifest.
- [x] `016-fix-selection-and-metrics` — added full-partition metrics, restored
  validation-selected checkpoints, and isolated final test access.
- [x] `017-connect-ablation-config` — connected one validated JSON schema to
  training and comparisons with explicit CLI overrides and checkpoint
  provenance.
- [x] `018-shortlist-architecture-results` — marked historical outputs as
  non-comparable and predeclared six named custom candidates with manifest
  provenance.
- [x] `019-cloud-experiment-notebook` — added one guarded local/Colab/Kaggle
  workflow for preview, screening, confirmation, selected training, validation
  inspection, final evaluation, and artifact export.
- [x] `020-expand-controlled-experiments` — prepared the approved 11-model
  screen, practical-baseline confirmation, separate published topology, and
  one-factor weight-decay, augmentation, spatial-size, and sequence ablations.
- [x] [`021-optional-cloud-architecture-expansion`](../../agents/work/021-optional-cloud-architecture-expansion/prompt.md) — **Skipped:** retain the eleven-candidate screen; no optional five-model block.
- [x] `022-interactive-notebook-foundation` — simplified the Colab VS Code
  setup and displayed AAD metadata, balance, training frames, and augmentation.
- [x] `023-interactive-model-inspection` — added playable native, model-ready,
  and augmented clips, manifest-driven model tables, and one labelled
  random-weight GPU check.
- [x] `024-notebook-experiment-workflow` — added guarded single-stage execution
  and visual validation-only result inspection to the Colab notebook.
- [x] `035-vdd-learnability-check` — retired by ticket 059; VDD is optional and
  uses the generic dataset pipeline, not a VDD-specific diagnostic.
- [ ] `036-kinetics-subset-audit` — inventory the available Kinetics copy and
  propose a small relevant class subset and resource budget; do not train yet.
- [x] `037-modular-colab-notebooks` — split routine Colab work into independent
  dataset, model-inspection, training, and controlled-experiment notebooks while
  preserving the original workflow as the reference.
- [x] `038-notebook-prediction-endings` — ended each modular notebook with five
  partition-labelled examples and gated final test predictions behind validation
  selection and longer confirmation.
- [x] `039-vdd-run-all-notebooks` — shared VDD defaults, playable previews,
  live curves, validation-ranked screen and automatically guarded final testing;
  separate from the controlled AAD study.
- [x] [`040-kinetics-interest-filter`](../../agents/work/040-kinetics-interest-filter/prompt.md) — shared interests, Kinetics-only filtering and selective downloads; VDD results preserved, test access guarded.
- [x] [`041-learning-and-temporal-diagnostics`](../../agents/work/041-learning-and-temporal-diagnostics/prompt.md) — implemented and locally verified: gentle scheduling, temporal audit, 14-model screen, separate native topology and two-seed confirmation under an eight-hour cap. GPU run pending; existing Kinetics test is exploratory, not fresh paper evidence.
- [x] [`041-larger-kinetics-diagnostic`](../../agents/work/041-larger-kinetics-diagnostic/prompt.md) — added selective Kinetics-600 downloads, source-video grouped splits and a more-than-2,000 unique-clip gate. Actual archive count awaits the first Colab preparation.
- [x] [`046-separate-aad-screen-config`](../../agents/work/046-separate-aad-screen-config/prompt.md) — separated the fixed 24-epoch AAD screen from the local longer-training profile.
- [ ] [`025-run-controlled-aad-experiments`](../../agents/work/025-run-controlled-aad-experiments/prompt.md) — Draft/superseded; do not execute the older matrix. Use 052's updated plan.
- [ ] `026-train-selected-model` — blocked on 053's current validation evidence; confirm and freeze the selected architecture,
  input size, augmentation, and regularisation across both planned seeds.
- [ ] `027-evaluate-selected-model` — blocked until 026 freezes the checkpoint; perform the one-time final AAD test and
  export complete metrics, confusion output, and playable prediction examples.
- [ ] `028-validate-second-dataset` — optional future VDD generalisation study
  after AAD model selection; not part of the active experiment sequence.
- [ ] `029-aggregate-ablation-evidence` — tracked by 053/054; produce paper-ready tables, figures,
  uncertainty, efficiency comparisons, and error-analysis inputs.
- [ ] `030-rewrite-experimental-results` — tracked by 054; revise the experiment and discussion
  section using only verified outputs.
- [ ] `031-align-paper-claims` — tracked by 054; align the abstract, contributions, methods,
  limitations, and conclusion with the final evidence.
- [ ] `032-assess-kinetics-transfer` — conditional only if frozen AAD/VDD
  evidence remains weak; define the exact Kinetics variant and use it for
  pretraining/transfer rather than silently replacing the 11-class task.
- [ ] `033-add-stateful-streaming-inference` — later add chunk-by-chunk stateful
  prediction to the custom model without changing the published baseline.

## Planned results-section flow

Planning only: [079 structure](../../agents/work/079-aad-results-section-structure/prompt.md)
→ [080 search figure/appendix](../../agents/work/080-aad-search-figure-and-appendix/prompt.md)
→ [081 ten-model comparison](../../agents/work/081-aad-ten-model-results-comparison/prompt.md)
→ [082 error analysis/synthesis](../../agents/work/082-aad-error-analysis-and-synthesis/prompt.md).
All four are Draft; execution requires approval and verified evidence. Scaffold
timing and strict-top-five versus reserved-anchor wording remain unresolved.

1. Computing environment
2. AAD dataset and data split
3. Training configuration and hyperparameters
4. Evaluation metrics
5. Architecture-search setup
6. Search results: capacity subplots, then trend discussion
7. Five-model selection and reference to the full appendix search table
8. Final ten-model comparison and discussion
9. Best validation-selected custom's test confusion matrix and class analysis
10. Overall interpretation

Future manuscript tables use pooled precision/recall/F1, accuracy/loss and
per-class counts/metrics, not macro averages. These pooled scores equal accuracy
for single-label multiclass classification; historical reports remain unchanged.
Ticket 088 aligns active code paths before a new campaign.

VDD may be used for a separate, explicitly configured generalisation study;
its historical runs are preserved but are not new comparable evidence.
