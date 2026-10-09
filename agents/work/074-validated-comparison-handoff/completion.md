# Completion

- Status: Done
- Summary: Automatic validated LR/weight-decay handoff, reusable processed-data caching and stable experiment numbering; no commit/push or real A100 run.
- Changes:
  - Added native-input recipe validation to the existing runner: baseline, proposed LR, then proposed WD at the retained LR. Up to three 128-cap/64-minimum jobs; validation-only decisions, ties keep incumbent, identical settings reuse evidence.
  - All eight final models share the frozen recipe, keep 50-frame/50x50 inputs and train afresh with the 512-epoch cap. Recipes, receipts and checkpoint freezes remain linked in the verified ZIP; altered/interrupted evidence blocks testing.
  - Enabled comparison caching; retained bounded 2-GiB RAM plus checksum-verified disk caches per input under ignored `runs/cache/training/` (16-GiB budget, 2-GiB free-space reserve). Test clips are excluded; insufficient disk space does not delete evidence.
  - Experiment numbers persist through search, recipe checks and comparison; reused jobs retain their identities. Tables include actual LR/WD and input sizes; tests have separate counters.
  - Updated notebook/export required-file checks and guides; saved outputs/metadata and historical profiles preserved.
- Verification:
  - **70 distinct tests passed:** 69 across `test_comparison_handoff`, `test_focused_workflow`, `test_capacity_search`, `test_multisize_study` and `test_colab_aad_workflow`; one additional exact-evidence proposal test passed afterward.
  - Regression command: `pytest tests/test_comparison_handoff.py tests/test_focused_workflow.py tests/test_capacity_search.py tests/test_multisize_study.py tests/test_colab_aad_workflow.py -q -p no:cacheprovider` (isolated temp directories; CPU, single-threaded math, Agg).
  - Synthetic Run All completed 170 search + 2 recipe + 8 comparison trainings and 8 frozen test evaluations. Numbers 1-180 remained stable; completed reruns did not train/test again, and a partial test marker blocked repetition. ZIP included recipe evidence and excluded caches.
  - Cache tests covered RAM/disk reuse, revisited inputs/new-process simulation, source invalidation, corruption, budget fallback and refusal to substitute another clip after decode failure.
  - All six notebooks' outputs/metadata unchanged; notebook/export AST parity, eight-file Black check, 67 local documentation links, Git ignore and whitespace checks passed.
- Remaining issues: Actual AAD accuracy, A100 memory/runtime and Colab execution remain unmeasured. The shared recipe is tuned on the search's top custom model, not independently optimal for each family. Disk/oversize cache limits may require lazy decoding. Commit/push this update before using it on Colab.
