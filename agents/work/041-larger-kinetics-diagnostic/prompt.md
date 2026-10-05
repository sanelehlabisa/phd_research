# Task Prompt

- Ticket: `041-larger-kinetics-diagnostic`
- Status: Done
- Aim: Replace the tiny notebook diagnostic with more than 2,000 real videos across the same five activities, aiming for roughly 3,000 total before splitting.
- Scope: Paper 001 modular notebook configuration, Kinetics download/data helpers, focused tests and concise guides.
- Approved decisions: Use Kinetics-600; accept `hugging (not baby)`; keep five activity labels, seed 42 and approximately 70:15:15 partitions. Keep the current quick experiment grid, not a larger search.
- Source: [CVDF guide](https://github.com/cvdfoundation/kinetics-dataset) and [Kinetics-600 training archive list](https://s3.amazonaws.com/kinetics/600/train/k600_train_path.txt).
- Context: The incomplete upstream training CSV lists 2,704 clips across these classes. All five training archives responded successfully to read-only checks on 2026-10-05, totalling about 3 GB compressed. Actual extracted counts must be verified; 3,000 is not an exact requirement.
- Changes:
  - Add a distinct `kinetics600-subset` diagnostic source and make it the shared default for notebooks 01–04. Preserve the old Kinetics-400 and VDD selections and their cached evidence.
  - Download only the five training class archives: `headbutting`, `slapping`, `punching person (boxing)`, `hugging (not baby)`, `shaking hands`. Retain exact source labels; do not merge or binarise activities.
  - Cache downloads under ignored `runs/datasets/`; show progress, reuse verified cached files and supplied local roots, and allow interrupted downloads to be retried. Never download the full release or silently fall back to another source.
  - Extract safely: reject traversal, absolute paths, links and unexpected files; verify archive integrity before marking complete. Record URLs, sizes, available upstream identity fields, content hashes and the final clip inventory.
  - Report actual unique clip counts per class and overall. Require all five classes and more than 2,000 total real clips before training; fail clearly if incomplete. No duplicate copies, augmentation-count inflation or mixing with old Kinetics versions.
  - Build a deterministic, approximately stratified 70:15:15 diagnostic split from these upstream training clips only. Keep clips sharing a source-video ID together when filenames/metadata identify it; record and validate grouping. Do not claim subject independence or official benchmark splits.
  - Key manifests and checkpoint provenance by release, source inventory, labels, split and preprocessing. Preserve older manifests/runs and reject incompatible checkpoints. Keep validation-only selection and frozen-winner final-test guards.
  - Preserve pending bootstrap/grid fixes and saved notebook outputs. Keep the 12-combination, four-epoch screen and eight-epoch confirmation; notebook 03 stays at 128 epochs. No additional models or factors in this ticket.
  - Update root, paper and implementation guides with the new default, download cost, actual-count checks and short Colab Run All instructions. Label this exploratory action recognition, not surveillance-paper evidence.
- Acceptance criteria:
  - All four modular notebooks share the larger source; exactly five approved classes and more than 2,000 total unique real clips are required before training.
  - Download/cache/extraction checks reject unsafe or incomplete data and never fetch unrelated class archives. A cached rerun and supplied local root require no unnecessary downloads.
  - Every compared run reuses the same split and source grouping; validation/test remain clean and test stays locked until the existing frozen-model gate.
  - Source changes reject old checkpoints without deleting earlier evidence. Existing VDD/Kinetics-400 workflows and controlled AAD scripts/configuration still pass regression tests.
  - Saved notebook outputs remain unchanged; guides clearly distinguish implemented behaviour from local mocked checks and an actual Colab download.
- Out of scope: Full Kinetics downloads, new activity classes, artificial sample multiplication, remote GPU training here, accuracy promises, manuscript edits, committing or pushing.
- Open questions: None.
- Verification: Mocked archive download/retry/cache tests; unsafe/corrupt archive rejection; source-ID split isolation and reproducibility; count/provenance/checkpoint rejection tests; synthetic-video notebook workflows; full pytest, notebook JSON/code-cell syntax, formatting and `git diff --check`. Avoid downloading gigabytes locally just to run tests; report the real Colab dataset count as pending until observed.

## Execution Prompt

Execute ticket `041-larger-kinetics-diagnostic` exactly as written in `agents/work/041-larger-kinetics-diagnostic/prompt.md`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved changes, preserve pending work and saved notebook outputs, verify every acceptance criterion, set the ticket status to `Done`, and create `completion.md` from `agents/templates/completion.md`. Do not commit or push.
