# Completion

- Status: Done
- Summary: Added the shared five-activity Kinetics-600 notebook source and its unique-clip and provenance guards.
- Changes:
  - Selectively download/cache the five approved class archives; safely extract and record archive identity, clip hashes and inventory under ignored `runs/datasets/`.
  - Require at least 2,001 unique clips, use source-video-grouped reproducible splits, and include the source inventory in manifests/checkpoints.
  - Keep Kinetics-400 and VDD selectable; update all four modular notebooks, bootstrap checks and guides.
  - Preserve the small experiment grid, 128-epoch training notebook setting and existing saved notebook outputs. No commit or push.
- Verification:
  - `MPLBACKEND=Agg .venv/bin/python -m pytest -q` — 94 passed.
  - `black --check` — all changed Python source/test files formatted.
  - All four notebook files parsed as JSON and their code cells parsed; each bootstrap checks for the new source helper.
  - `git diff --check` — passed.
  - Synthetic tests cover selective archive download, retry/cache, corrupt/unsafe archives, duplicate/low-count rejection, local-root provenance, source-group split repeatability/isolation, locked test clips and checkpoint invalidation after source changes.
- Remaining issues: The real extracted Kinetics-600 clip count and video decodability are not verified until the first Colab preparation. The notebook prints actual class counts and stops before training if fewer than 2,001 unique files are found. The approximately 3 GB of archives was not downloaded locally.
