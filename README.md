# PhD Research

One workspace for PhD manuscripts and the code that produces their evidence.
Each paper has a numbered directory under `papers/` with a concise guide,
manuscript sources, and—when needed—its implementation.

## Papers

| Paper | Current focus |
|---|---|
| [`001-journal-abnormal-activity-recognition`](papers/001-journal-abnormal-activity-recognition/) | Prepare and run controlled ablation studies |
| [`002-review-abnormal-activity-recognition`](papers/002-review-abnormal-activity-recognition/) | Initial review-paper scaffold |

All four modular Colab notebooks now default to a five-class Kinetics diagnostic
subset: 87 clips, selectively downloaded (ticket `040`). One shared setting
switches back to VDD. Playable previews, live curves and validation-selected
experiments are described in the [implementation guide](papers/001-journal-abnormal-activity-recognition/implementation/README.md).
Ticket `041` adds a budgeted eight-hour learning/temporal diagnostic suite with
14 matched candidates, a separate native paper topology and two-seed confirmation.
Saved outputs are historical; the new suite still needs its Colab run.
Integration ticket [042](agents/work/042-merge-expanded-kinetics-suite/prompt.md)
is ready for approval: merge the other laptop's larger Kinetics source with the
expanded suite and explicit spatial-size comparisons. The agreed dataset size
is more than 2,000 unique videos before splitting, not a maximum.
The controlled AAD study stays separate; previously inspected test scores are exploratory.

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
