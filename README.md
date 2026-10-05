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
