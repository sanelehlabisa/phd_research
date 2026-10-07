# PhD Research

One workspace for PhD manuscripts and the code that produces their evidence.
Each paper has a numbered directory under `papers/` with a concise guide,
manuscript sources, and—when needed—its implementation.

## Papers

| Paper | Current focus |
|---|---|
| [`001-journal-abnormal-activity-recognition`](papers/001-journal-abnormal-activity-recognition/) | Prepare and run controlled ablation studies |
| [`002-review-abnormal-activity-recognition`](papers/002-review-abnormal-activity-recognition/) | Initial review-paper scaffold |

The Kinetics-600 notebook workflow is a separate diagnostic; controlled paper
experiments use AAD. The latest saved AAD run reached 96.25% validation accuracy
on 160 clips; test data remains locked. See the [Paper 001 guide](papers/001-journal-abnormal-activity-recognition/README.md)
and its [implementation guide](papers/001-journal-abnormal-activity-recognition/implementation/README.md).
Training and experiment notebooks now use fresh filenames to work around stale
VS Code notebook controllers; use the links in the implementation guide above.
Notebook-only helpers live under `implementation/notebooks/utils/`; reusable
CLI and model code remains in `src/`. Ticket 047's larger Kinetics notebook
profile remains a separate in-progress task. Next AAD tickets 049–052 cover
dataset auto-resolution, overall metrics, split-safe prediction examples, and
a staged JSON-controlled comparison. Gradio deployment is planned after final
model selection and results.

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
