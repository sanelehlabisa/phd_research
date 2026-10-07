# Completion

- Status: Done
- Summary: All four CLI commands save split-safe, per-category prediction clips.
- Changes:
  - Shared `prediction_samples_per_category`: default 3 per category, 0 disables
    videos; correct/incorrect folders, counts, source paths and label provenance.
  - Restored training/candidate checkpoints export validation examples;
    evaluation alone exports test examples; model smoke is explicitly random-weight.
  - Removed training's eager whole-dataset cache, which decoded locked test clips.
  - Updated active JSON settings and concise usage documentation.
  - Final compatibility review updated the report reader for current category
    limits while preserving historical five-example reports.
- Verification:
  - Full pytest suite: 170 passed; 10 focused prediction tests include actual tiny
    video model/train/experiment/evaluate commands, playable MP4s, count limits,
    missing categories, zero, nested split provenance and guarded test decoding.
  - Evaluation left the training checkpoint's SHA-256 unchanged.
  - Final prediction/report/notebook compatibility suite: 27 passed, including
    zero clips, an absent category and six-clip reports.
  - Black, Python parsing and git diff checks passed; saved notebook bytes,
    historical run artifacts and manuscript hashes unchanged.
- Remaining issues: Real-data/GPU training was not run; finding a rare category
  can require scanning the entire permitted partition.
