# PhD Research

One workspace for PhD manuscripts and the code that produces their evidence.
Each paper has a numbered directory under `papers/` with a concise guide,
manuscript sources, and—when needed—its implementation.

## Papers

| Paper | Current focus |
|---|---|
| [`001-journal-abnormal-activity-recognition`](papers/001-journal-abnormal-activity-recognition/) | Prepare and run controlled ablation studies |
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
Notebook-only helpers live under `implementation/notebooks/utils/`; reusable
CLI and model code remains in `src/`. Existing run folders stay local and are
not included in source commits. A Gradio presentation demo is a later task,
after model selection and verified results.

Tickets 057–061 established configurable datasets/classes, separate search and
comparison profiles, aligned the guides, and made the Colab notebook a thin
runner with artifact download. Next, run and review the AAD profiles; manuscript
changes wait for verified results (062).

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

The [October 9 search review](papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-09-multiresolution-search-review.md)
verifies 36 completed search jobs, but no comparison/test run. Imported evidence
stays local under ignored `implementation/runs/imports/`; nothing is silently resplit.

The [October 10 interim search analysis](papers/001-journal-abnormal-activity-recognition/implementation/reports/todays.results.md)
summarises the new console snapshot, provisional architecture ranking and full
appendix. A later appended excerpt reports search completion and a slow first
comparison-recipe check; final ranking artifacts and test results remain unverified.
The later expanded export now permits reconstruction of confirmation and
WD/temporal scores (168 stage rows, 164 unique configurations); see the appended
review. Raw inputs are preserved; these are not independently verified test results.

Current workflow: [075 wider/deeper capacity search](agents/work/075-wide-depth-capacity-search/prompt.md),
32 custom architectures at depths 1-5, 64 base runs at 32/64px, then shared
16-frame/64px comparison with validation-selected FPS and per-class reporting.
Search cap 256 epochs; final cap 512. Up to 142 trainings before reuse, with
global experiment progress. Implemented and [locally verified](agents/work/075-wide-depth-capacity-search/completion.md).
Real AAD/A100 execution remains separate from local verification.
The latest ranking refines the planned width-64 neighbourhood with an 80-wide
flat model and single-layer widening to 128, without increasing the run budget.

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
