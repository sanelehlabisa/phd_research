# Task Prompt

- Ticket: `090-configurable-methodology-and-baseline-citations`
- Status: Ready
- Aim: Explain the actual configurable ConvLSTM family and cite comparison architectures in Paper 001's methodology after the results revision.
- Scope: Only `Proposed Method` in `papers/001-journal-abnormal-activity-recognition/manuscript/main.tex`, directly affected architecture figure/algorithm assets, necessary verified bibliography metadata and concise task/guide records.
- Dependencies: Review [079](../079-aad-results-section-structure/prompt.md)'s updated results first. Wider abstract/introduction/conclusion alignment remains [062](../062-update-manuscript-after-aad-results/prompt.md).
- Changes:
  - Replace the obsolete hard-coded five-stage/four-number architecture account with the implemented configurable recurrent stack: a list of filters and spatial kernel dimensions, repeated ConvLSTM layers, final hidden features, global spatial pooling and classifier. Distinguish the general model family from the validation-selected [64,64,128] instance reported in results.
  - Explain which depth/width/placement choices are configurable and which settings were fixed in the actual campaign (including 3×3 kernels). Match equations, algorithm, tensor shapes and architecture figure to code without inventing a new component or claiming all configurations were tested.
  - State the actual fixed-window/last-frame-repeat preprocessing and per-call state reset; do not imply full-length streaming or guaranteed long-term memory. Align AAD/class count and applicable method statements with verified evidence, without rerunning experiments.
  - Briefly identify the input-adapted published ConvLSTM, two practical 3D CNNs and two Video Swin sizes, with one primary-paper citation per model: reuse `vrskova2022`, `tran2018closer` and `liu2022videoswin`. Shared family references are sufficient; do not duplicate BibTeX entries or expand literature review.
- Acceptance criteria:
  - General configurable model description, equations/algorithm/figure and selected example agree with the implemented forward path and verified campaign; no stale hard-coded architecture or unimplemented streaming claim remains in Methods.
  - All baseline families have resolving primary citations in Methods; source-paper topology/input adaptation and scratch training are distinguished correctly.
  - Changes stay within methodology and necessary supporting records/assets; no code, raw results, data, model selection or other manuscript sections change.
  - LaTeX builds with labels/citations resolved and affected figure/algorithm remains readable.
- Out of scope: Implementing new models, training/testing/streaming, changing research results, rewriting related work/abstract/conclusion, bibliography-wide cleanup, commit or push.
- Open questions: None. Later execution requires its own approved execution prompt; this is a logged follow-up, not authorization to edit Methods now.
- Verification: Compare `src/model.py`, active temporal sampling and resolved campaign configurations against method statements; audit citations at primary publishers; inspect scoped diff, run `latexmk -pdf main.tex` and `git diff --check`.

## Execution Prompt

Execute ticket `090-configurable-methodology-and-baseline-citations` exactly as
written in `agents/work/090-configurable-methodology-and-baseline-citations/prompt.md`.
Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the
approved changes, verify every acceptance criterion, set the ticket status to
`Done`, and create `completion.md` from `agents/templates/completion.md`.
Do not train, commit or push.
