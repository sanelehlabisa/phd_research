# Project Configuration

- Content: PhD papers and supporting research implementations for surveillance
  video and unusual human-activity recognition.
- Paper root: `papers/NNN-short-title/`.
- Paper guide: `papers/NNN-short-title/README.md`.
- Manuscript source: `papers/NNN-short-title/manuscript/`, using `main.tex` as
  the entry point.
- Optional code: `papers/NNN-short-title/implementation/`.
- Work directory: `agents/work/NNN-short-title/`.
- Templates: `agents/templates/prompt.md` and `agents/templates/completion.md`.
- Ticket numbers: Start at `001` and increase sequentially across the workspace.
- Build: Use the paper README command; otherwise run `latexmk -pdf main.tex`
  from the manuscript directory.
- Code verification: Use the implementation README and project configuration.
- References: Resolve all citations and bibliography entries in the final build.
- Formatting: Keep prose, LaTeX, code, and Markdown readable and remove trailing
  whitespace.
- Generated files: Do not commit PDFs, build artifacts, datasets, environments,
  checkpoints, or generated experiment outputs unless explicitly requested.
