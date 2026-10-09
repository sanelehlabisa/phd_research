# Task Prompt

- Ticket: `072-automatic-search-comparison-workflow`
- Status: Draft
- Disposition: Superseded by [073](../073-focused-shape-search-run-all/prompt.md); retained for history, do not execute separately.
- Aim: Make one Colab Run All complete the approved study without manual stage switches or directory entry.
- Scope: Existing AAD orchestration, active notebook/Python export, resume/archive helpers, tests and guides.
- Decisions:
  - User approved automatic search, validation selection, eight-model comparison, frozen-checkpoint test evaluation, examples and ZIP download.
  - Preserve the comparison recipe: top three custom models plus the published topology, R3D18, MC3_18, Swin3D T/S; 50 frames at 50x50, batch 1, up to 256 epochs, seed 42.
  - Use 070's audit disposition and 071's corrected selection. Known overlap or unresolved suspected source overlap blocks automatic final testing; never bypass research guards to finish.
- Changes:
  - Extend existing orchestration with one automatic default sequence. Pass returned run/selection/checkpoint paths internally; remove the need to edit stage toggles or paste directories.
  - Finish the selected search protocol, freeze its validation-only top three, freshly train all eight comparison models, then freeze all validation-selected checkpoints and the best custom model before accessing test.
  - Evaluate each frozen checkpoint on the full test split once; save accuracy, explicitly macro/micro precision/recall/F1, per-class metrics, confusion matrices, parameters, sizes, histories and recorded timing. Export labelled correct/incorrect video examples from the validation-chosen custom model; never choose it using test.
  - Persist stage state, exact config/split/code hashes, audit disposition and attempt receipts. Reuse only complete compatible evidence; never choose an arbitrary latest directory or mix 068 and 069 scores.
  - On resume, reuse completed stages and test outputs; do not repeat a test attempt after a partial failure. Clearly report missing, failed or incompatible evidence and preserve the frozen decision.
  - Save checksum-verified ZIPs at stage boundaries and on handled interruption/failure; automatically offer final download and retain download-only retry. Exclude datasets/caches and preserve all completed/partial run evidence.
  - Keep manual diagnostic entry points optional, saved notebook outputs intact, notebook/export parity and concise stage/progress displays.
- Acceptance criteria:
  - One synthetic Run All reaches archive/export in the declared order without manual paths or toggles.
  - No test access before audit resolution, complete comparison evidence and a validation-only freeze; unsafe/incomplete stages stop with a clear reason.
  - Resume and download-only retry do not repeat training/testing; archived outputs and existing research safeguards remain intact.
- Out of scope: Real Colab execution, changing comparison settings, new datasets/resplits, Drive integration, manuscript claims, commit/push.
- Open questions: None; the full sequence and existing comparison settings were explicitly approved.
- Verification: Mocked end-to-end ordering and path-handoff tests; failure/interruption/resume and exactly-once-test guards; source-audit/config mismatch tests; ZIP inventory checks; notebook/export/output-preservation tests; relevant CPU regressions and `git diff --check`.

## Execution Prompt

Complete ticket `072-automatic-search-comparison-workflow` after 070 resolves the
source audit and 071 is verified. Follow `AGENTS.md`, `agents/rules.md` and
`agents/config.md`. Implement the approved automatic sequence while preserving
test locks, saved outputs and historical evidence. Verify the result, update this
status and create `completion.md` from `agents/templates/completion.md`.
Do not run real experiments or commit/push.
