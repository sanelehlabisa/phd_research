# Task Prompt

- Ticket: `074-validated-comparison-handoff`
- Status: Done
- Aim: Automatically carry validated search training settings into the final comparison, with reusable data and stable experiment numbers.
- Approval: User agreed to retain 50-frame/50x50 comparison inputs and explicitly requested immediate implementation on 2026-10-09.
- Scope: Paper 001 experiment runner, Colab helpers, cache, focused/final profiles, tests and guides.
- Changes:
  - Keep the top-three architecture handoff and all eight final models; retain native inputs, split, seed, frozen-test gates and 512-epoch comparison cap.
  - Use search-calibrated custom LR and top-one WD evidence as proposals. At native comparison inputs, check baseline, LR, then WD one factor at a time (up to three 128-epoch trainings; minimum 64, patience 24). Freeze one validation-selected shared recipe; never merge untested ablation winners.
  - Reuse bounded train/validation caches across models and revisited inputs; keep test uncached until final evaluation and exclude caches from ZIP/Git.
  - Persist stable experiment numbers through search, recipe validation and comparison; distinguish reuse and test evaluations from new trainings.
  - Preserve saved notebook outputs, historical profiles and existing artifacts; complete Run All through metrics, examples and verified ZIP.
- Acceptance criteria:
  - Verified recipe provenance accompanies every comparison; partial/tampered evidence blocks testing and complete reruns do not retrain/retest.
  - Cache tests demonstrate reuse, invalidation, bounded storage and train/validation-only decoding.
  - Counters remain stable across duplicate reuse and resumes; automated end-to-end tests cover the handoff and export.
- Out of scope: Real A100 experiments, new architectures/data, temporal changes, manuscript claims, commit/push.
- Open questions: None.
- Verification: Focused/regression pytest, notebook preservation/export checks, formatting and Git diff checks.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
