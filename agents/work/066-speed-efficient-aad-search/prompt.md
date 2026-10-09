# Task Prompt

- Ticket: `066-speed-efficient-aad-search`
- Status: Done
- Aim: Make the AAD custom-architecture screening practical on Colab by avoiding repeated video decoding and reducing the search to the most informative runs.
- Scope: Paper 001's active AAD search profile, experiment data-loading path, staged-study runner, Colab helper and Python export, focused tests, and implementation README. Keep the comparison profile and Kinetics diagnostic unchanged.
- Changes:
  - Reuse the existing processed-video cache during each candidate run so clips are not decoded from source video on every epoch. Keep memory use bounded/documented and preserve the same AAD split and labels.
  - Keep the eight declared custom architectures, seed `42`, augmentation off, weight decay fixed at `0`, and the same batch size for every candidate. Do not add optimizer or seed sweeps.
  - Use 8 frames at 32×32 for the first architecture screen, with a 64-epoch cap and validation-loss early stopping. Keep test locked.
  - After validation ranks the custom candidates, vary only the selected candidate's input one factor at a time: frame sizes `[32, 48, 64]` and frame counts `[8, 16]`. Do not test the full Cartesian product or weight-decay variants in this screening run.
  - Add a short, recorded batch-16 versus batch-32 throughput check on the same fixed configuration. Choose one batch size from measured throughput and memory safety, then hold it constant across the search; do not treat this check as model-selection evidence.
  - Make the exported AAD workflow run the custom search without automatically starting the expensive six-model comparison. Keep comparison available as an explicit later action using the saved validation-selected configuration.
  - Preserve progress logging and archive the search configuration, selected model, checkpoints, metrics, logs, and prediction examples. Do not edit or delete historical runs.
  - Keep the notebook and Python export aligned, with concise instructions for the search-only workflow.
- Acceptance criteria:
  - The search uses a single AAD dataset/split and seed `42`; augmentation and weight decay are off, and all candidates use the same measured batch size.
  - Eight custom architectures are screened at the fixed quick input, and only the validation-selected model receives the one-factor frame-size/frame-count checks.
  - Reusing cached clips avoids decoding each video again on every training epoch; the cache's memory footprint is logged and the AAD search completes without exhausting Colab RAM.
  - The batch benchmark records elapsed time/throughput for batch sizes 16 and 32; the selected batch size and reason are recorded in the resolved profile.
  - A full search run does not automatically start model-family comparison; the selected configuration remains available for that later stage.
  - Test remains locked during search; the validation winner, run configuration, progress, checkpoints, metrics, and examples are saved and included in the downloadable archive.
  - Notebook/export parity, profile validation, focused tests, and the local smoke profile pass. No generated run artifacts are committed.
- Out of scope: Changing comparison models or their protocol, changing dataset contents/split, adding augmentation or weight-decay sweeps, manuscript claims, or interrupting/reconfiguring an already-running Colab study.
- Open questions: `None`.
- Verification: Validate active JSON and plan listing; test cache reuse across epochs and verify split/hash consistency; test batch benchmark selection; run focused pytest and local smoke; parse notebook JSON and check Python-export parity; confirm search-only flow archives its outputs without starting comparison; run `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
