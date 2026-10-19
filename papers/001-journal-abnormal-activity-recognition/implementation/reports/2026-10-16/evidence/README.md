# Evidence and preservation

- `results.zip`: byte-identical copy of the user-supplied `reports/results.zip`;
  original left untouched. 4,187,553 bytes; 310 JSON members (309 inventoried
  members plus inventory), 51,427,326 bytes uncompressed.
- SHA256: `3042f4512825c93cfc831f0f591d0ecd37c51c85dcdfbd058290d5153cbefe6f`.
- `search_console.txt`: unedited attachment from
  `6c1a77f6-51cd-4ca4-816f-47659b31d4e7/Pasted text.txt`.
- `comparison_console.txt`: unedited attachment from
  `601ccf55-986f-4892-8e14-685ab136a67e/Pasted text.txt`.
- The **last** logged Drive receipt has the same ZIP size/hash. Earlier copies
  in the log have different hashes; they are not substituted. Logged Drive path:
  `MyDrive/phd_research/paper-001/20261016_041454_538943_abnormal-activities-dataset_top3_comparison/results.zip`.
  We did not access Drive directly; the matching local file verifies which
  logged backup was supplied, not current cloud availability.

## Audit scope

- `build_report.py` checks ZIP CRC/inventory SHA256s, one shared split,
  all search rankings/conditions and console scores, selected settings,
  transformer-check receipts, resolution-check order, frozen winner/checkpoint
  identities and all 296 train/validation/test prediction partitions.
- Counts, labels, exact source membership, pooled scores, per-class statistics,
  mean loss and confusion matrices reconcile for 131,454 prediction records.
  Every comparison model has 161 unique test records from the fixed test split.
- Both `run.json` records identify revision
  `c8786d1399eb77b326595b2b759745d2f4a61f76`; search retains source-file hashes.
  The final essential-summary `code_identity` field is null. This is disclosed,
  not replaced with invented provenance; the run revision remains available.
- No original videos/labels were manually audited. No weights were loaded.
  Referenced checkpoint hashes agree across metadata; tensor bytes are absent,
  so their contents cannot be independently rehashed or predictions rerun.
- Full epoch histories, original source-audit JSON and per-job receipt files
  are absent. The ZIP supports this metrics review, not a full restorable run.
  Completed run records have null `ended_at`/`duration_seconds`; reported hours
  are sums of per-job training time, not a certified total wall-clock duration.
- Wilson intervals assume independent Bernoulli clips. Exact paired McNemar
  results are post-hoc, unadjusted for nine comparisons, and not seed estimates.
  No post-test model selection or claimed statistically significant winner.

## Preserve before the Colab VM disappears

The current ZIP already contains the results needed for this report. No larger
results ZIP is required. For reproducibility and presentation, additionally keep:

- Best custom checkpoint (~15.31 MiB, including training state):
  `runs/experiments/20261016_051514_502396_abnormal-activities-dataset_top3_comparison/models/custom_top1/checkpoints/best_model.pth`.
- Its five correct and one incorrect MP4 examples under
  `runs/evaluate/20261016_073235_344768_abnormal-activities-dataset_selected-model/predictions/`.
  The ZIP contains their metadata, **not playable files**. Preview metadata
  reports 8 FPS playback although the model sampled at 16 FPS: label these as
  slowed previews, not native/model-rate playback.
- Full per-model `metrics/history.json`, checkpoint metadata sidecars,
  original split/source audit and study receipts. These small JSONs retain
  learning-curve and provenance detail missing from the compact ZIP. Preserve
  other final checkpoints too if you need to reproduce all ten predictions.

Do not delete existing originals or replace them with this report's derived CSVs.
