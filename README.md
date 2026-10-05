# PhD Research

One workspace for PhD manuscripts and the code that produces their evidence.
Each paper has a numbered directory under `papers/` with a concise guide,
manuscript sources, and—when needed—its implementation.

## Papers

| Paper | Current focus |
|---|---|
| [`001-journal-abnormal-activity-recognition`](papers/001-journal-abnormal-activity-recognition/) | Prepare and run controlled ablation studies |
| [`002-review-abnormal-activity-recognition`](papers/002-review-abnormal-activity-recognition/) | Initial review-paper scaffold |

All four modular Colab notebooks now default to five Kinetics-600 activity
archives, downloaded selectively. Preparation requires over 2,000 unique clips;
the actual extracted count will appear on the first Colab run. A shared setting
switches back to VDD or the earlier Kinetics-400 subset. Playable previews,
live curves and validation-selected experiments are described in the
[implementation guide](papers/001-journal-abnormal-activity-recognition/implementation/README.md).
The controlled AAD study stays separate; saved VDD notebook outputs are preserved.
Notebook 04 uses editable factor lists: the quick default runs 12 combinations
for four epochs each, then retrains the validation winner for eight epochs.

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
