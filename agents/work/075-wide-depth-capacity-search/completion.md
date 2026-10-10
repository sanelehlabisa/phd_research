# Completion

- Status: Done
- Summary: Implemented the revised 075 protocol and latest approval: 256-epoch search cap and campaign-wide progress. No real AAD execution. The owner's subsequent request authorizes publication to `master` and cleanup of disposable generated caches; evidence and legacy protocols are preserved.
- Changes:
  - New versioned search/comparison profiles: 32 architectures, depths 1-5, 64 base runs at 32/64px; at most 142 trainings overall before reuse.
  - Two-seed top-five reranking, top-three handoff, final-input FPS selection and verified sequential/joint recipe reuse. All eight final models use 16f/64px and a 512-epoch cap; the paper baseline is explicitly input-adapted.
  - Full-step preflight freezes common search/final batches; failures stop safely. Global `Experiment N/142 maximum | stage` reaches leaf/epoch progress; reuse keeps its number.
  - Unaveraged per-class metrics/counts, matched-checkpoint train/validation gaps, complete configuration tables and capacity/cost plots. Legacy metrics/readers remain available.
  - Notebook/export, shared helpers and guides updated. Automatic freeze -> test -> examples -> verified ZIP remains intact.
- Verification:
  - CPU regression suite covering capacity, wide search, comparison handoff, multi-size/staged studies and controlled experiments: **109 passed**.
  - Final `tests/test_wide_depth_search.py`: **8 passed**, including exact manifest, five-layer backward pass, adapted registry, OOM/common-batch paths, FPS warnings, raw counts, compatible recipe reuse and legacy-note compatibility.
  - Focused-workflow suite: eight cases passed; its legacy Run All case initially rejected code edits during execution as designed. Stable-code rerun of that case: **1 passed**. Both new and legacy end-to-end workflows verify eight frozen evaluations, ZIP contents, unchanged reruns and repeat-test refusal.
  - New synthetic workflow completed 127 search + 1 fresh recipe + 8 comparison trainings (136 unique IDs), without manual input. Synthetic scores are not research evidence.
  - Real tiny-video training passed with matched evaluation/per-class reporting; headless checks used `MPLBACKEND=Agg`. Local environment: Python 3.14, CPU PyTorch 2.9.1; no A100 measurement.
  - Search `--list-plan`, Black check (14 Python files), local documentation links and `git diff --check`: passed.
  - All 20 protected report/legacy-profile hashes and all six notebooks' saved outputs/metadata unchanged. Active notebook/export ASTs match.
- Publication follow-up: Re-ran the eight wide-search tests, Black and preservation/parity checks successfully before staging. Removed five ignored Python bytecode cache directories (regenerated on import); no source, evidence, configurations or notebook outputs deleted.
- Remaining issues: Real AAD accuracy, A100 fit and runtime remain unmeasured. GPU preflight runs on execution; epoch caps do not promise completed epochs or an eight-hour finish. Use the latest notebook/export from `master`, and save the verified ZIP outside the VM.
