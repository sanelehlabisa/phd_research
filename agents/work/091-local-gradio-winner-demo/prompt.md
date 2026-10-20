# Task Prompt

- Ticket: `091-local-gradio-winner-demo`
- Status: Done
- Aim: Run a simple, fully local video-classification UI using the selected ConvLSTM architecture, with explicit random weights for the approved immediate UI test.
- Scope: Paper 001 `implementation/demo.py`, minimal reusable inference helpers, one winner-training JSON, pinned demo dependencies, tests and concise usage instructions.
- Approved scope update (20 October 2026): Execute now and prioritise a running local UI. The user explicitly permits random weights for immediate interface testing, provided they are clearly labelled. Train locally if needed; a full training run is not needed to verify this UI-first handoff. Leave the presentation untouched.
- Changes:
  - Convert the existing demo entry point to Gradio: choose a local sample video or upload a video, preview it, and display the predicted class and all class probabilities. No Colab or automatic public sharing.
  - Use `CustomConvLSTM[64,64,128]`, 3x3 kernels, global pooling and no hidden classifier: 1,336,459 parameters. Reuse the October 16 validation selection, not a new test-based ranking.
  - First check for the original checkpoint. Its recorded SHA256 is `fd21edfc7058efaf39c432fb92318e768f22bef3b0103cc42f049a5e8ff6e896`; it was not found locally or in the compact results ZIP during planning. Never silently substitute weights. An explicit `--random-weights` mode is now approved for UI testing only; display the untrained warning on the page and every prediction.
  - Add `configs/experiments/aad_demo_selected_local.json` and verify a train-only `src.experiments` command with `models: ["custom_convlstm_configured"]`; pass `--minimum-epochs 128` on the CLI, not inside the leaf JSON. Recover the original split and recipe from `implementation/reports/2026-10-16/evidence/results.zip`: Adam, LR 0.001, WD 0, batch 32, seed 42, no augmentation, 256-epoch cap, minimum 128, patience 32, plateau factor 0.9/patience 5. Check local memory before a full run; report rather than silently change settings. For the fast UI-first handoff, provide the command without starting long CPU training.
  - Save a recovered or newly trained model with its configuration, ordered class names, preprocessing and provenance in an ignored local directory. A retrained demo checkpoint is a new run, not the original reported 99.379% result. Keep test locked during training.
  - Preserve the winning input contract: eight RGB frames at 16 FPS, 64x64, `timestamps_v1` sampling, last-frame repetition, bilinear resize and float [0,1]. Reuse `src.temporal_sampling` decoding; the old demo's `--fps` only changes playback and is not a valid inference implementation. Clearly disclose that only the initial sampled window is classified, not the whole uploaded video.
  - Validate checkpoint architecture, class identity/order and preprocessing; use inference mode, load once and support local CPU/CUDA. Reject missing/incompatible weights and invalid videos clearly. Show correctness only for examples with known labels; probabilities are not calibrated reliability.
  - Reuse a few suitable local examples without committing datasets/videos/weights. Document local setup, training/output paths and app launch. Keep later Hugging Face model storage/deployment separate.
- Acceptance criteria:
  - Local sample and upload flows work in the explicitly labelled random-weight mode now, with correctly mapped probabilities summing to one and no external service requirement. Compatible trained-checkpoint loading is implemented and separately guarded/tested.
  - Training JSON resolves to the frozen winner recipe; no search, test evaluation or unrelated experiment changes occur.
  - Preprocessing matches campaign inputs; missing weights, class-order mismatch and bad/short videos have tests. Synthetic checks alone do not establish a working trained demo.
  - Hand off the verified JSON/command and record any pending real-checkpoint validation explicitly. UI completion does not mean the trained model or its reported accuracy has been reproduced.
- Out of scope: Colab, Hugging Face uploads/deployment, streaming, manuscript/results edits, presentation edits, commit or push.
- Open questions: None for the implemented UI-first scope. Recovery of original weights or local training remains a prerequisite for later trained-demo verification.
- Verification: Inspect archived winner provenance; validate JSON with `src.experiments --print-config` and `--list-models` without training; test upload preprocessing, checkpoint guards and inference; exercise the live upload API with an original training-split video in explicit random mode; run scoped tests and `git diff --check`. Do not describe synthetic checkpoint checks as trained-model validation.

## Execution Prompt

Complete ticket `091-local-gradio-winner-demo`. Follow `AGENTS.md`,
`agents/rules.md` and `agents/config.md`. Keep everything local, provide the
winner-training JSON and command, verify the implementation, update status and
create `completion.md` from `agents/templates/completion.md`. Follow the approved
UI-first scope update above; launch only on localhost and clearly distinguish
random-weight testing from trained inference. Record pending real-model
verification. Do not deploy, commit or push.
