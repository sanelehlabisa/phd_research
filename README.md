# PhD Research

One workspace for PhD manuscripts and the code that produces their evidence.
Each paper has a numbered directory under `papers/` with a concise guide,
manuscript sources, and—when needed—its implementation.

## Papers

| Paper | Current focus |
|---|---|
| [`001-journal-abnormal-activity-recognition`](papers/001-journal-abnormal-activity-recognition/) | Prepare and run controlled ablation studies |
| [`002-review-abnormal-activity-recognition`](papers/002-review-abnormal-activity-recognition/) | Initial review-paper scaffold |

All four modular Colab notebooks default to five Kinetics-600 activity archives.
Preparation requires more than 2,000 unique videos before splitting; the actual
count remains to be observed on Colab. Shared configuration retains VDD and
Kinetics-400 alternatives. Ticket [042](agents/work/042-merge-expanded-kinetics-suite/prompt.md)
integrates the two laptops' work: an eight-hour exploratory suite with 14 matched
models, spatial/temporal/regularization comparisons, a separate native paper
model and two-seed confirmation. See the [implementation guide](papers/001-journal-abnormal-activity-recognition/implementation/README.md).
Saved notebook outputs are historical; the merged suite needs its Colab run.
The controlled AAD study stays separate; no new manuscript claims are made.
Training and experiment notebooks now use fresh filenames to work around stale
VS Code notebook controllers; use the links in the implementation guide above.
Notebook-only helpers now live under `implementation/notebooks/utils/`; reusable
CLI and model code remains in `src/`. Ticket
[045](agents/work/045-stabilize-colab-script-runtime/prompt.md) hardens the
exported one-cell Colab runtime. Next is [ticket
025](agents/work/025-run-controlled-aad-experiments/prompt.md), after the
screening configuration is safely separated from the local training profile.
Separately, [ticket 047](agents/work/047-larger-colab-training/prompt.md) is ready
for approval: larger notebook-03 inputs/model with a 200-epoch, eight-hour cap.

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
