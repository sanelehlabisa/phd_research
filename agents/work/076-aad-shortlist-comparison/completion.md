# Completion

- Status: Done
- Summary: Implemented the five-finalist comparison following verified 077 search, with automatic validation freeze, test, examples and compact export. No real training. The owner subsequently requested commit and push to master on 2026-10-13.
- Changes:
  - Five customs plus input-adapted paper ConvLSTM, R3D-18, MC3-18 and Swin3D-T/S. All ten train from scratch at eight frames, the selected shared size/FPS, LR 0.001/WD 0 and 512/128/32 epoch limits.
  - Verified selection/split/recipe, measured common final batch and experiment numbers 65–74. Legacy three-custom workflows stay available.
  - All checkpoints and the best custom freeze before ten full tests. Up to five correct/five incorrect examples come only from the validation-selected custom.
  - Complete pooled and per-class metrics, confusion/prediction records, parameters/bytes, curves and timing; compact backups during comparison and separate optional checkpoint export.
- Verification:
  - Shared verification with [077](../077-focused-aad-colab-rerun/completion.md): eight focused and 71 existing regression tests passed across targeted runs.
  - Synthetic Run All completed all 74 trainings and ten evaluations, verified ZIP contents, reused completed work and refused an attempted test. Recipe mismatch/OOM guards, notebook/export parity and formatting passed.
  - Eight-frame/64px adapted paper model: 134,415,195 parameters; native baseline remains 512,197,467 (11 classes, meta-device count, not training evidence).
  - Existing split content matches its source review; LF-normalized SHA-256 matches the prompt. Windows CRLF changes byte hash only; assignments were not changed.
- Remaining issues: Real A100 fit/runtime and results are unverified. Shared settings are custom-selected, not optimal tuning per family. Preserve optimization failures, inspect downloaded archives and retain checkpoints separately. Final report/Table 2 and manuscript work await owner-generated artifacts; no scores were filled in.
