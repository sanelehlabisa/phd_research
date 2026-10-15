# Colab reports

Folders use the report/review date (`YYYY-MM-DD`), not necessarily the execution
date. Keep each review with its raw logs, CSV tables and figures. Preserve older
evidence; add new reviews without overwriting it. Generated run folders stay local.

- [2026-10-15: all 64 search measurements, graph-first](2026-10-15/README.md) —
  unchanged values/ranking, named per-panel highlights, layer-count-only bottom
  legend, disclosed y-axis compression and 128 image links; no arrows/insets/tables;
  current source-file availability and verification limits disclosed. No new training/test.
- [2026-10-15: Swin3D learning diagnosis](2026-10-15/transformer-learning/README.md) —
  both real GPU train-only traces verify parameter updates but reproduce
  constant-class collapse; no proven production defect or held-out evaluation.
  The [lower-LR follow-up](2026-10-15/transformer-learning/follow-up.md) restores
  Swin3D-T learning locally: 85.625% validation in eight epochs at LR 0.00001.
  This is not final test evidence or a Swin3D-S repair; comparison changes remain gated.
- [2026-10-14: focused search and ten-model comparison preview](2026-10-14/README.md) —
  all 64 search jobs checked against imported summaries/histories, a 128-image
  search gallery and ten comparison trainings; validation plots/tables,
  transformer collapse and urgent archive recovery. Final test failed before scoring.
- [2026-10-13: latest AAD search review](2026-10-13/latest_colab_results.md) —
  reviews the 10–11 October console export; final comparison/test results pending.
- [2026-10-10: interim and expanded search review](2026-10-10/todays.results.md) —
  includes both preserved console exports.
- [2026-10-09: multi-resolution search review](2026-10-09/2026-10-09-multiresolution-search-review.md).
- [2026-10-08: earlier Colab study review](2026-10-08/2026-10-08-colab-results-summary.md).

Console-derived validation results are exploratory evidence, not verified final
test results. See each report for its source dates, limitations and next tickets.
