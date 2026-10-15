# Completion

- Status: Done
- Summary: New modular/campaign outputs use pooled P/R/F1, accuracy/loss and full per-class/confusion evidence; historical metrics retain their original definitions.
- Changes:
  - Existing train/evaluate/experiment/reporting paths use `pooled_per_class_v2`; matched-checkpoint training uses `matched_checkpoint_eval_v2`. No new macro/balanced-average calculation, table column or plot.
  - Campaign receipts reconcile scores, class counts, confusion JSON and complete predictions; split/class/checkpoint/selection and test-once guards remain. Explicit legacy verification preserves old scores.
  - Comparison exports support the existing default when older configs omit FPS. Guides and planned manuscript tickets 079–081 are aligned; manuscript/results/notebooks/configurations are unchanged.
- Verification:
  - CPU synthetic core suites (`test_overall_metrics`, `test_capacity_search`, `test_prediction_examples`, `test_multisize_study`): 70 cases passed across the initial run and corrected end-to-end rerun. The latter verifies search → frozen comparison/test → examples → compact ZIP, with no macro fields.
  - Targeted suites (`test_pooled_reporting`, `test_focused_rerun`, `test_focused_workflow`, `test_wide_depth_search`, `test_comparison_handoff`, `test_aad_final`, `test_controlled_experiments`): 88 passed, three slow full legacy campaign cases deselected. An earlier broad run was stopped; it is not counted as passing.
  - Additional final modular/campaign smoke reruns passed. Tests used isolated temporary fixtures, one CPU thread and `MPLBACKEND=Agg` for headless plotting; no real research training/testing.
  - Black check: ten changed Python files clean. `git diff --check` passed. All 179 tracked report/notebook files retained their hashes; combined SHA-256 manifest: `ed6e8555b763180ba3cb5140b5e60a5ed328c10cc0162f365fa4794f4c277f8d`.
- Remaining issues: None for 088. Comparison recovery (083), search expansion (087) and Colab integration (089) remain separate. No commit or push.
