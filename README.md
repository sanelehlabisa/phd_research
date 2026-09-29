# PhD Research

One workspace for PhD manuscripts and the code that produces their evidence.
Each paper has a numbered directory under `papers/` with a concise guide,
manuscript sources, and—when needed—its implementation.

## Papers

| Paper | Current focus |
|---|---|
| [`001-journal-abnormal-activity-recognition`](papers/001-journal-abnormal-activity-recognition/) | Prepare and run controlled ablation studies |
| [`002-review-abnormal-activity-recognition`](papers/002-review-abnormal-activity-recognition/) | Initial review-paper scaffold |

Ticket `019-cloud-experiment-notebook` is complete. The next action is running
its controlled architecture and audited-baseline confirmation stages; ticket
020 ablations begin only after that validation evidence is reviewed.

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
