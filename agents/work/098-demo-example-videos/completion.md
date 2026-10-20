# Completion

- Status: Done
- User verification: Owner confirmed the deployed examples work on 20 October 2026 and requested publication to Git master.
- Summary: Added three small, original validation videos as clickable demo examples.
- Changes:
  - `demo_assets/examples/`: Normal, Fight and Pollution MP4s (84,452 bytes total), with source/split/hash notes. Original cached files preserved.
  - Local trained demo discovers three samples; explicit CLI examples override them. Random-mode preview behavior preserved.
  - Spaces uses those examples without prediction caching; exporter copies only the three named videos and their README.
  - Updated wiring/export tests and deployment guides. ZeroGPU dependencies, model, weights and research evidence unchanged.
- Verification:
  - All three are validation entries in the existing manifest, matching the checkpoint's LF-normalized split hash; no test clips or model-based selection.
  - Copies and ZIP entries match source SHA256 byte-for-byte. All frames decoded successfully: Normal 209, Fight 181, Pollution 86. H.264, 256×256, 29.97 FPS; previews inspected.
  - `pytest -q -p no:cacheprovider tests/test_space_export.py`: 8 passed, including exactly-three export, omitted extra files, missing-file failure and Space example wiring.
  - Eight CLI/discovery tests passed with inference imports stubbed (real discovery/main logic); six Gradio-dependent tests not run because the previous ML environment is incomplete.
  - Black, Python 3.10 syntax parsing, ZIP CRC and staged/unstaged whitespace checks passed.
  - Fresh export: `implementation/runs/spaces_upload/20261020_hq5c9gn7/`; ZIP 14,919,500 bytes, 19 files, three videos and one requirements file. Checkpoint SHA unchanged.
- Remaining issues:
  - Hosted deployment and example verification were performed by the owner, not this agent.
  - No retraining, test evaluation or source video changes.
