# Task Prompt

- Ticket: `092-five-minute-results-presentation`
- Status: Ready
- Aim: Update the existing presentation and speaking script for a clear talk lasting no more than five minutes.
- Scope: Paper 001 `presentation/SENG-F-12-HLABISA-SANELE.pptx`, `presentation/script.txt` and concise guide links.
- Dependency: Execute only after `091-local-gradio-winner-demo` works locally; the user prioritised the demo first.
- Changes:
  - Reuse the existing three-slide PRIS deck and branding; retain the generic guide.
  - Replace pending-experiment wording with verified October 16 evidence: problem and configurable method, search/comparison findings, then limitations and the working demo.
  - Explain the validation-selected [64,64,128] model, 160/161 test correct and 1,336,459 parameters with a readable results visual. Distinguish the archived research checkpoint from any separately retrained demo.
  - Write concise, natural speaker notes with per-slide timings and transition cues; include any demo within the five-minute limit.
  - Preserve single-seed, clip-split and short-window limitations. Do not claim global optimality, deployment reliability or that a new demo reproduces the archived test accuracy.
  - Include a static screenshot/fallback so the talk does not depend on a live application or network.
- Acceptance criteria:
  - Three editable, readable slides and a matching script form one coherent story.
  - All numerical claims trace to verified evidence; demo statements reflect actual local verification.
  - Timed rehearsal fits within five minutes, including transitions/demo; report estimated timing separately if rehearsal is unavailable.
  - Deck opens/renders correctly and original guide assets remain intact.
- Out of scope: New experiments, model changes, deployment, manuscript revisions, commit or push.
- Open questions: None; existing three-slide format, maximum five minutes, after the local demo.
- Verification: Check slide count, rendered layouts and extracted text; trace claims to evidence; compare script against slides and demo provenance; time a rehearsal and run `git diff --check`.

## Execution Prompt

After ticket `091-local-gradio-winner-demo` works locally, complete ticket
`092-five-minute-results-presentation`. Follow `AGENTS.md`, `agents/rules.md`
and `agents/config.md`. Verify the deck and script, update status and create
`completion.md` from `agents/templates/completion.md`. Do not change the
manuscript, train, deploy, commit or push.
