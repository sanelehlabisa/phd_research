# Task Prompt

- Ticket: `048-trim-aad-architecture-screen`
- Status: Ready
- Aim: Make Paper 001 experiment configs easy to identify, reduce the AAD custom architecture screen to an evidence-led shortlist, and make its run selectable from the plan JSON alone.
- Scope: Paper 001 implementation config filenames/references, experiment-plan validation/command handling, and implementation README.
- Changes:
  - Prefix each config filename with its owning runner or workflow (`train_`, `evaluate_`, or `experiments_`) instead of moving files into new directories. Update every code, plan, README, test, and notebook reference affected by the renames.
  - Reduce the custom candidate manifest from eleven to three previously run architectures: `[32, 16]` (historical high-validation reference), `[16, 32]`, and `[32, 16, 8]`.
  - Set `[32, 16]` as the controlled-plan reference; update confirmation/ablation reference architecture fields so plan validation remains consistent. Keep the PaperConvLSTM and three approved practical 3D-CNN baseline definitions and comparison stages.
  - Align the architecture-screen profile with the best historical validation run: AAD split seed 42, 8 frames at `64x64`, augmentation off, batch size 16, learning rate `0.01`, weight decay `0`, patience 20, and a 160-epoch maximum. Keep test access locked.
  - Record `architecture-screen` as the active stage in the controlled-plan JSON so the run command needs only that JSON path. Preserve optional inspection/listing behavior.
  - Update plan validation, printed plan summary, and README command/count to match the three-model screen and profile.
- Acceptance criteria:
  - Every implementation config filename clearly identifies its runner/workflow, with no stale references to the old names.
  - The manifest contains exactly the three named architectures above, with unique names and valid layer definitions.
  - The plan selects `[32, 16]` as its reference and its confirmation/ablation references validate against that architecture.
  - The screen profile matches the documented historical high-validation settings and validates as a controlled screen.
  - Running `python -m src.experiments --plan-config configs/aad_controlled_experiment_plan.json` selects the JSON-configured architecture-screen stage without extra stage/model/path arguments.
  - Plan inspection reports three custom candidates; the approved PaperConvLSTM and practical 3D-CNN baseline stages remain available.
  - Existing tests pass, and no test-split access is added to training or screening.
- Out of scope: Running training/evaluation, changing the dataset/split, changing baseline architectures, adding new search dimensions, or committing/pushing.
- Open questions: `None`.
- Verification: Validate all JSON configs; run plan inspection and assert the configured architecture-screen command; run the implementation pytest suite and `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
