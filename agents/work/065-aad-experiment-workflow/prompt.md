# Task Prompt

- Ticket: `065-aad-experiment-workflow`
- Status: Done
- Aim: Make the active AAD Colab workflow run a longer, reproducible custom ConvLSTM search, pass its actual validation-selected model into a fixed family comparison, then evaluate the frozen comparison on test.
- Scope: `implementation/notebooks/aad_experiment_workflow.ipynb` and its Python export; `notebooks/utils/aad_study.py`; the active AAD search/comparison JSON profiles; minimal changes to `src/experiments.py` or `src/evaluate.py` only if needed for correct selection, logging, or post-freeze test evaluation; focused tests and concise README updates. Keep `04_run_experiments.ipynb` (Kinetics diagnostic) unchanged.
- Evidence to use: The 2026-10-08 report found custom models mostly underfit at 16 epochs and identified a search-to-comparison mismatch (`[16,16,16]` was compared instead of the search winner `[16,16,32]`). A successful modular AAD run used `[32,16]`, 8 frames at 64×64, no augmentation, learning rate `0.01`, weight decay `0`, seed `42`, and a 128-epoch cap; its validation-selected checkpoint reached 85% accuracy with 68,347 parameters. Treat this as a promising reference configuration, not a guaranteed result.
- Changes:
  - Keep AAD, the committed 70:15:15 split manifest, clean validation/test data, and seed `42` fixed throughout. Do not add seed sweeps; state in saved summaries that this is single-seed evidence.
  - Configure a focused custom architecture screen with `[16,16]`, `[16,16,16]`, `[24,24,24]`, `[32,32,32]`, the known `[32,16]` reference, and one-layer early/middle/late variants `[32,16,16]`, `[16,32,16]`, `[16,16,32]`. Avoid redundant candidates.
  - Use the successful modular run as the search starting recipe (8 frames, 64×64, batch 16, learning rate `0.01`, weight decay `0`, augmentation off, 128-epoch cap, validation-loss patience 20). Keep a 3-value width comparison (16/24/32). For the selected custom architecture, check frame sizes `[32,64,96]`, frame counts `[8,16,32]`, and weight decay `[0,0.0001,0.001]` one factor at a time; do not run a full Cartesian product.
  - Save a validation-ranked `selected_config.json` with the exact candidate/layers, input settings, seed, validation metrics, parameter count, and source run. Make the comparison stage consume that file, and fail clearly rather than silently falling back to a hard-coded architecture.
  - Compare the selected custom model with `paper_convlstm_published`, `r3d_18`, `mc3_18`, `swin3d_t`, and `swin3d_s`. Keep one fixed comparison protocol (50 frames at 50×50, batch 1, learning rate `0.001`, weight decay `0.0001`, no augmentation, seed `42`), train from scratch, and use a 64-epoch cap with at least 32 epochs before early stopping (validation-loss patience 12). This lets the scheduler act before stopping while ensuring at least twice the prior 16-epoch budget.
  - After the validation comparison and model choice are frozen, evaluate every comparison checkpoint on the test split once, save a test confusion matrix and standard metrics per model, and save playable prediction examples for the validation-selected best model. Never rank or tune from test results.
  - Make the AAD notebook default to the full run, retain an explicit smoke-only switch, resolve/download AAD once per notebook run, and use the same local dataset and split across both stages. Show clear stage/model/epoch progress with live `tqdm`, learning rate, train/validation loss and accuracy, selected epoch, and duration. Preserve the existing no-time-limit behavior and download the complete study archive, including selected config, progress, logs, configs, checkpoints, metrics, confusion matrices, and examples.
  - In the AAD notebook setup cell, show one concise restart instruction and stop cleanly if pinned dependency repair requires a Colab runtime restart; do not attempt an automatic restart. Keep broader setup consistency work in ticket 064.
  - Keep the notebook and checked-in Python export aligned; update the implementation README's active profile descriptions and commands.
- Acceptance criteria:
  - The active AAD notebook defaults to one full run and clearly exposes the smoke-only option; its Python export matches it.
  - The search plan includes the listed custom model families and 16/24/32 width values, uses the proven reference recipe, and varies spatial size, frame count, and weight decay one factor at a time.
  - The saved selected configuration matches the validation winner exactly, and the comparison run metadata/checkpoint proves that same architecture was trained; no hard-coded fallback is used.
  - The comparison includes the selected custom model, published ConvLSTM, two 3D CNNs, and two transformers; it uses the documented shared input/training protocol and completes at least 32 epochs per model unless a run fails.
  - Test is accessed only after the comparison is frozen; every comparison model has test metrics and a confusion matrix, and examples are generated for the validation-selected best model without using test to select it.
  - The dataset is resolved once and both stages use the same AAD directory and split manifest. Progress and all listed result artifacts are visible/saved and included in the downloaded archive.
  - A restart-required Colab setup displays a concise instruction without an alarming traceback or automatic restart; the following run proceeds after the user restarts.
  - JSON/profile validation, notebook/export parity, focused tests, and the local smoke profile pass. No historical run files are rewritten or committed.
- Out of scope: Changing `src/train.py`, the Kinetics diagnostic notebook, dataset contents/augmentation, manuscript results, adding seeds, or committing generated run artifacts.
- Open questions: `None`.
- Verification: Validate all active JSON profiles and `--list-plan` output; run relevant tests and the local smoke profile; inspect notebook JSON and compare its Python export; test selected-config handoff and test-access gating; confirm the archive lists the progress/selection/result files.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
