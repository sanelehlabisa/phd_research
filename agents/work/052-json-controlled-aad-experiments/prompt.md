# Task Prompt

- Ticket: `052-json-controlled-aad-experiments`
- Status: Done
- Aim: Make one concise AAD experiment JSON the source for a small, staged, reproducible model/input/regularization comparison.
- Scope: Paper 001 experiment configuration, model registry, runner, example JSON, commands, and implementation guide.
- Changes:
  - Replace overlapping AAD experiment-plan/candidate inputs with one validated JSON containing the named dataset, model names, custom layer candidates, frame sizes, sequence lengths, weight-decay values, fixed training budget, and seed settings.
  - Run staged comparisons rather than the full Cartesian product: compare declared models at one common screening input, then vary one factor at a time against the validation-selected reference. Keep learning rate and scheduler fixed; do not add more search dimensions.
  - Include the custom ConvLSTM candidates, the three existing practical 3D-CNN baselines (`r3d_18`, `mc3_18`, `r2plus1d_18`), and one scratch-trained Torchvision video transformer (`swin3d_t`, `weights=None`; [Torchvision 0.24 docs](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.video.swin3d_t.html)). Keep the faithful `PaperConvLSTM` as a separately labelled native-topology run, not in the same ranking when its input protocol differs.
  - Record input dimensions, trainable parameter count, runtime, dataset/split identity, and validation metrics for every candidate. Keep the test split locked.
  - Keep Kinetics diagnostic configuration separate and preserve existing historical run artifacts.
- Acceptance criteria:
  - One documented AAD experiment command reads the single JSON; a no-training listing shows every stage, model, input size, weight decay, run count, and command.
  - The runner rejects unknown model names, invalid sizes, and accidental Cartesian expansion beyond the declared stages before training.
  - Each comparison changes one declared factor at a time, uses the same split and training budget, and selects/ranks using validation only.
  - The listed transformer builds with the project's pinned Torchvision version using no pretrained weights; model parameter count is captured.
  - A bounded smoke check covers each model family; no multi-hour experiment is required for verification.
  - README commands and the AAD config match the actual runner behavior.
- Out of scope: Learning-rate search, new datasets, test-set selection, manuscript edits, and long GPU runs.
- Open questions: `None`.
- Verification: Validate the JSON/schema; run the no-training listing; build/forward each declared model on a small synthetic batch; run focused config/registry pytest checks; confirm test access remains locked.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
