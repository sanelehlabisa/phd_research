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

Current workflow: [068 multi-resolution search and top-three comparison](agents/work/068-multisize-search-top3-comparison/prompt.md).
One seed, 36 search jobs, then an eight-model final comparison with guarded
test evaluation and verified ZIP export. Real Colab execution/results review remain pending.

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
