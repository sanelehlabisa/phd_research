# PhD Research

One workspace for PhD manuscripts and the code that produces their evidence.
Each paper has a numbered directory under `papers/` with a concise guide,
manuscript sources, and—when needed—its implementation.

## Papers

| Paper | Current focus |
|---|---|
| [`001-journal-abnormal-activity-recognition`](papers/001-journal-abnormal-activity-recognition/) | Experimental/results section accepted; remaining manuscript alignment pending |
| [`002-review-abnormal-activity-recognition`](papers/002-review-abnormal-activity-recognition/) | Initial review-paper scaffold |

Paper 001 uses AAD as its primary study, with class labels discovered from the
selected dataset folders. The experiment order is custom ConvLSTM architecture
search, followed by a fixed-setting comparison with the published model and
other model families. VDD remains optional when selected by JSON and pointed at
a local dataset; its historical runs are preserved but are not comparable new
evidence. Kinetics notebook workflows are exploratory diagnostics, not part of
the active paper study. See the
[Paper 001 guide](papers/001-journal-abnormal-activity-recognition/README.md)
and [implementation guide](papers/001-journal-abnormal-activity-recognition/implementation/README.md).
Colab reviews and their evidence are grouped by report date in the
[reports index](papers/001-journal-abnormal-activity-recognition/implementation/reports/README.md).
Latest evidence: the [October 16 campaign review](papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-16/README.md)
checks 128 search jobs, five resolution checks and ten final tests. The
validation-selected 64-64-128 custom scores 160/161 test correct with 1.34M
parameters; single-seed, clip-split and baseline limitations remain explicit.
The verified study now appears in the manuscript's experimental results/discussion
through [079](agents/work/079-aad-results-section-structure/completion.md).
Other sections still require separate alignment; no deployment/generalization
certification is implied.
Notebook-only helpers live under `implementation/notebooks/utils/`; reusable
CLI and model code remains in `src/`. Notebooks use numbered names; their
available Python exports live under `implementation/notebooks/scripts/`. See the
[notebook index](papers/001-journal-abnormal-activity-recognition/implementation/README.md#notebooks-and-exports).
The separate Kinetics winner notebook trains the selected architecture on its
own inferred classes; it does not replace AAD paper evidence. Existing run folders
stay local and are not included in source commits.
[091: the local-only Gradio demo](agents/work/091-local-gradio-winner-demo/prompt.md)
supports uploads/examples and guarded checkpoint loading.
[094](agents/work/094-trained-demo-lfs/prompt.md) now defaults to the owner's
Colab-retrained checkpoint, with one curated weight file prepared in Git LFS.
Its recorded validation accuracy is 98.75%; no new test score is claimed.
The original October 16 checkpoint remains unavailable; explicit random mode
is retained only for UI testing.
[093](agents/work/093-colab-demo-training/prompt.md) adds a
[one-cell GPU training notebook/export](papers/001-journal-abnormal-activity-recognition/implementation/notebooks/scripts/09_train_demo_winner_colab.py)
for that model only, with Drive-saved weights/metadata and validation previews.
The app stays local; no search or test evaluation is rerun.
[095](agents/work/095-simple-demo-spaces/prompt.md) simplifies the UI and prepares
a small, upload-ready Hugging Face Spaces folder. Nothing is deployed automatically.
[096](agents/work/096-spaces-python-compatibility/prompt.md) adds Python 3.10-safe
checkpoint loading and a ZIP with one standalone Spaces requirements file.
[097](agents/work/097-zerogpu-demo/prompt.md) updates that export for ZeroGPU,
with CUDA dependencies and GPU-decorated inference; local CPU use is unchanged.
[098](agents/work/098-demo-example-videos/prompt.md) bundles three small validation
video examples for the local demo and Spaces; no test data or metrics changed.
[092: the five-minute presentation](agents/work/092-five-minute-results-presentation/prompt.md)
is complete: the [deck, practice script and current image tools](papers/001-journal-abnormal-activity-recognition/presentation/pris-presentation/tools/README.md)
are organized under PRIS, including the owner's revised PPTX, PDF and script.
The owner now edits the deck/script manually; five final
images remain in a flat assets folder, with graphs beside vertical top-ten keys. Obsolete
builds/previews were removed; PowerPoint Slide Show and rehearsal remain manual.

Tickets 057–061 established configurable datasets/classes, separate search and
comparison profiles, aligned the guides, and made the Colab notebook a thin
runner with artifact download. The AAD campaign is reviewed and results/discussion
updated; 090 and 062 retain the remaining methodology/whole-paper alignment.

Preserved legacy workflow: [073 focused shape search and Run All](agents/work/073-focused-shape-search-run-all/prompt.md),
[implemented and locally verified](agents/work/073-focused-shape-search-run-all/completion.md).
It consolidates 070–072: 34 custom stacks with 1–3 layers, a 128-epoch search cap,
automatic eight-model comparison with a 512-epoch cap, frozen test and ZIP export.
The owner reviewed the filename concern; automated duplicate checks stay active.
Local verification is not evidence of new AAD accuracy or A100 runtime.

[074 validated comparison handoff](agents/work/074-validated-comparison-handoff/prompt.md)
adds native-input validation of search LR/weight-decay proposals, one shared
comparison recipe, reusable processed-data caches and stable experiment numbers.
That legacy sequence remains automatic with native 50-frame/50x50 final inputs.
See its [completion record](agents/work/074-validated-comparison-handoff/completion.md)
for local verification; real Colab execution is still pending.

[068](agents/work/068-multisize-search-top3-comparison/prompt.md) remains available
unchanged as a legacy profile, alongside 069's separate 200-epoch search.
Their explicit comparison entry points are preserved; 073 adds automatic handoff.

The [October 9 search review](papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-09/2026-10-09-multiresolution-search-review.md)
verifies 36 completed search jobs, but no comparison/test run. Imported evidence
stays local under ignored `implementation/runs/imports/`; nothing is silently resplit.

The [October 10 interim search analysis](papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-10/todays.results.md)
summarises the new console snapshot, provisional architecture ranking and full
appendix. A later appended excerpt reports search completion and a slow first
comparison-recipe check; final ranking artifacts and test results remain unverified.
The later expanded export now permits reconstruction of confirmation and
WD/temporal scores (168 stage rows, 164 unique configurations); see the appended
review. Raw inputs are preserved; these are not independently verified test results.

Preserved workflow: [075 wider/deeper capacity search](agents/work/075-wide-depth-capacity-search/prompt.md),
32 custom architectures at depths 1-5, 64 base runs at 32/64px, then shared
16-frame/64px comparison with validation-selected FPS and per-class reporting.
Search cap 256 epochs; final cap 512. Up to 142 trainings before reuse, with
global experiment progress. Implemented and [locally verified](agents/work/075-wide-depth-capacity-search/completion.md).
Real AAD/A100 execution remains separate from local verification.
The latest ranking refines the planned width-64 neighbourhood with an 80-wide
flat model and single-layer widening to 128, without increasing the run budget.

The [October 13 AAD report](papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-13/latest_colab_results.md)
reviews 131 console-derived search entries without final comparison/test results.
It informed the fresh [077 focused A100 rerun](agents/work/077-focused-aad-colab-rerun/prompt.md):
16 architectures at two sizes/two FPS values, then [076 fixed finalist comparison](agents/work/076-aad-shortlist-comparison/prompt.md).
[078](agents/work/078-aad-ablation-figures/prompt.md) covers clearer ablation figures/tally markers later; manuscript revision (062) follows verified results.
The new focused runner is [07_aad_focused_workflow.ipynb](papers/001-journal-abnormal-activity-recognition/implementation/notebooks/07_aad_focused_workflow.ipynb),
with a matching [copy-paste Python export](papers/001-journal-abnormal-activity-recognition/implementation/notebooks/scripts/07_aad_focused_workflow.py).
It runs 64 search jobs then ten fresh comparison trainings automatically, with
compact results backups. The old notebook/profiles and saved evidence remain unchanged.
Implementation is [locally verified](agents/work/077-focused-aad-colab-rerun/completion.md).
The [October 14 preview](papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-14/README.md)
reviews all 64 search and ten comparison trainings on A100. Best comparison
custom validation accuracy is 98.750%; final testing failed at a dataset-name
compatibility check. Preserve the compact archive and final checkpoints before
Colab closes; no final test claim is available yet.

[085 complete search visibility](agents/work/085-complete-search-result-visibility/completion.md)
is done: the [October 15 report](papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-15/README.md)
plots all 64 measurements with the original layer-count legend, per-panel
highlight keys and a disclosed compressed lower y-range. All 128 image links
remain; no side/performance tables, arrows or zoom plots.
Original values/ranking are unchanged; unavailable run files are disclosed.
[084 transformer diagnosis is complete](agents/work/084-diagnose-transformer-learning/completion.md):
real 6-GB CUDA checks verify updates but reproduce constant-class learning collapse;
no causal code defect was established. See the [diagnosis](papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-15/transformer-learning/README.md)
and [lower-LR Swin3D-T follow-up](papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-15/transformer-learning/follow-up.md),
which restores learning in a separate local diagnostic, not final comparison.
[083 comparison-only JSON/test recovery is locally verified](agents/work/083-comparison-only-recovery/completion.md)
for the same ten models, without rerunning search. Restore missing evidence/data
before execution; original final checkpoints are also needed for test-only recovery.
No new research scores or GPU fit claim. Recipe changes/full training need approval.
[086](agents/work/086-simplify-modular-scripts/prompt.md)
plans later script cleanup and remains planning-only.

[088 pooled metrics everywhere](agents/work/088-pooled-metrics-everywhere/completion.md)
aligns new modular/campaign reports with pooled P/R/F1, per-class counts and
confusion evidence; historical results retain their original contracts.
[087 expanded search is locally verified](agents/work/087-expand-frame-depth-search/completion.md):
one new JSON, **64x64**, 32 architectures x 8/16 frames x 8/16 FPS (128 jobs),
four-condition top-five ranking, frame-aware reuse and verified results-only
Drive-copy safeguards. Existing reports/notebooks remain unchanged; no real
training or Drive upload was performed. See the implementation guide for CLI use.
[089 Colab campaign](agents/work/089-colab-campaign-readiness/completion.md) adds the
[new notebook](papers/001-journal-abnormal-activity-recognition/implementation/notebooks/08_aad_frame_depth_workflow.ipynb)
and matching Python export: authorize Drive, verify a visible test file, run
128 search jobs at 32px/128 epochs, five validation-only 64px checks, checked
family-specific learning rates and ten fresh 64px/256-epoch comparisons,
freeze, test and compact JSON-only backups after each completed job.
Use this new notebook, not the old focused export. Plan roughly 12-18 hours
(uncertain, potentially longer), not one guaranteed overnight Colab session.
Epoch caps/early stopping do not guarantee runtime or convergence.
Results-only backups cannot restore weights after VM loss. Streaming is deferred.

[079 results/discussion is complete](agents/work/079-aad-results-section-structure/completion.md),
owner-reviewed and accepted on 19 October 2026, including the simplified plot titles.
Section flow: setup → search graphs and selection → one ten-model test table with macro P/R/F1 →
discussion and best-custom confusion analysis. Figures and citations build; all other manuscript
sections are unchanged. Draft 080–082 are not marked complete; additional appendix
work remains deferred.
Next, [090 methodology](agents/work/090-configurable-methodology-and-baseline-citations/prompt.md)
explains the configurable stack and cites baseline models; 062 retains broader
alignment of outdated abstract/introduction/conclusion claims. Both require
separate execution approval before implementation.

## Structure

```text
papers/NNN-short-title/
├── README.md
├── manuscript/
└── implementation/    # optional
```

Build a manuscript from its `manuscript/` directory with:

```bash
latexmk -pdf main.tex
```

Datasets, environments, checkpoints, experiment outputs, PDFs, and LaTeX build
files remain local and untracked unless a task explicitly says otherwise.
