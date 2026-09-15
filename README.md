# PhD Research

One workspace for PhD manuscripts and their supporting implementations. Papers
use numbered, stable directories under `papers/`.

## Structure

Each paper contains a concise `README.md`, a `manuscript/` directory, and an
optional `implementation/` directory. The paper README records its status,
build entry point, dependencies, and next ticket.

| Paper | Status |
|---|---|
| [`001-journal-abnormal-activity-recognition`](papers/001-journal-abnormal-activity-recognition/) | Implementation imported; local experiment work pending transfer |
| [`002-review-abnormal-activity-recognition`](papers/002-review-abnormal-activity-recognition/) | Initial review-paper scaffold |

## Build

Run `latexmk` from the relevant manuscript directory:

```bash
cd papers/001-journal-abnormal-activity-recognition/manuscript
latexmk -pdf main.tex
```

```bash
cd papers/002-review-abnormal-activity-recognition/manuscript
latexmk -pdf main.tex
```

Generated PDFs and LaTeX build files remain untracked.
