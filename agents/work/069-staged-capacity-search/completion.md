# Completion

- Status: Done
- Summary: Implemented the bounded, validation-only staged capacity search; no real AAD/Colab experiments, commit or push.
- Changes:
  - Added `aad_capacity_search_colab.json`, `capacity_config.py` and `capacity_search.py`: LR calibration, 21 flat architectures at three resolutions, focused refinement, two-seed confirmation and separate WD/temporal ablations; 156-job upper bound before dedup/reuse.
  - Added opt-in timestamp sampling, train/validation-only audits/cache keys, source/duplicate checks, common GPU batch preflight and documented model-only inference measurements. Legacy sampling stays unchanged.
  - Replaced console top-five truncation with full JSON/CSV/Markdown tables; added explicit macro/micro metrics, balanced accuracy, plots, seed summaries and separate ablation recommendations.
  - Added checksum-verified receipts, interrupted-job guards, stage ZIPs and packaging-only retry. Confirmed top-three export is compatible with the unchanged eight-model comparison; test stays locked during search.
  - Defaulted the active notebook/export to `capacity_search`; retained the explicit 068 search option, saved outputs/metadata, historical profiles, downloaded `temp/` configs and manuscript.
  - Updated guides and notebook contract tests; added 31 ticket-specific tests.
- Verification:
  - `python -m src.experiments --config configs/experiments/aad_capacity_search_colab.json --list-plan`: complete 63-custom/6-reference static matrix and bounded dependent stages; no data download/training.
  - `pytest tests/test_capacity_search.py -q`: **31 passed**. Includes a synthetic 149-job adaptive workflow, actual CPU model/reference forwards, tiny timestamped-video training and VFR sampling, metric recomputation, shallow-model selection, resume/tamper guards, comparison handoff and verified ZIP retry.
  - CPU regression coverage across batches: initial broad run passed 179 cases; updated two obsolete notebook-default assertions and reran affected/remaining modules, **80 passed**. Together with the new suite, all **274 collected cases** were covered. Rerun modules: `test_colab_aad_workflow`, `test_notebook_workflows`, `test_overall_metrics`, `test_prediction_examples`, `test_smoke_training`, `test_staged_study`, `test_video_reader`.
  - Verification used Python 3.14/PyTorch 2.9.1 CPU, unique temporary pytest directories and `-p no:cacheprovider`. Used `MPLBACKEND=Agg` for missing local Tk support and sequential runs with OMP/MKL/OpenBLAS threads set to 1 after parallel checks exhausted host memory. No training logic was weakened for these environment issues.
  - Black checks, notebook/export parity, saved-output/metadata comparisons, protected-file SHA-256 comparisons and `git diff --check` passed. HEAD remains `2905fca`.
- Remaining issues: Real AAD/A100 accuracy, runtime and GPU memory safety require the separate Colab run; existing AAD split required. Final comparison/protocol review and stateful full-video processing remain later work, not results claimed by this ticket.
