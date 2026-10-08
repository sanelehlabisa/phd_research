# PhD Research

One workspace for PhD manuscripts and the code that produces their evidence.
Each paper has a numbered directory under `papers/` with a concise guide,
manuscript sources, and—when needed—its implementation.

## Papers

| Paper | Current focus |
|---|---|
| [`001-journal-abnormal-activity-recognition`](papers/001-journal-abnormal-activity-recognition/) | Prepare and run controlled ablation studies |
| [`002-review-abnormal-activity-recognition`](papers/002-review-abnormal-activity-recognition/) | Initial review-paper scaffold |

Paper 001 will use AAD as its main study. Dataset choice will live in each JSON
config, and class labels will be read from the selected dataset. VDD remains an
optional dataset choice; its historical run artifacts are preserved. The next
work first separates custom ConvLSTM architecture search from cross-family
comparison, then streamlines the Colab notebook/export workflow. See the
[Paper 001 guide](papers/001-journal-abnormal-activity-recognition/README.md)
and [implementation guide](papers/001-journal-abnormal-activity-recognition/implementation/README.md).
Notebook-only helpers live under `implementation/notebooks/utils/`; reusable
CLI and model code remains in `src/`. Existing run folders stay local and are
not included in source commits. A Gradio presentation demo is a later task,
after model selection and verified results.

Next tickets: 057 made dataset selection and class discovery config-driven;
058 simplifies experiment profiles to one local smoke config and two AAD Colab
configs using the same runner; 059
retires VDD-only workflow code while retaining config-selectable VDD; 060 aligns
the docs; 061 makes the Colab notebook a thin script runner with artifact
download. Manuscript changes wait for verified AAD results (062).

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
