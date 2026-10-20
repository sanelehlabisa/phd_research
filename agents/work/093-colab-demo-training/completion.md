# Completion

- Status: Done
- Summary: One-cell Colab notebook and matching pasteable script train only the selected demo model.
- Changes:
  - `notebooks/train_demo_winner_colab.ipynb` / `.py`: embedded frozen recipe, published shared runner, automatic setup/AAD download and GPU gate.
  - Drive authorization/write check first; checkpoint/history/config saved during training; cache stays local.
  - Validation curves/examples and verified one-run ZIP with weights, class metadata and split. No search/test evaluation.
  - Focused tests and concise links/instructions in the three READMEs.
- Verification:
  - `pytest -q -p no:cacheprovider tests/test_demo_colab.py`: 10 passed, 1 skipped (Windows lacks symlink privilege; Colab uses Linux).
  - `pytest -q -p no:cacheprovider tests/test_demo_training_config.py tests/test_colab_bootstrap.py`: 32 passed.
  - Notebook/export parity, safe imports, recipe, one training, fail-closed gates and cache-free bundle tested offline.
  - Python compilation, Black check and `git -c core.longpaths=true diff --check` passed.
  - Required core files already exist at `origin/master` (`da71a17`); pasted script does not require publishing new demo helpers.
- Remaining issues:
  - No real Colab/GPU/Drive execution here; cloud calls were mocked. Authorize Drive when running.
  - New training does not inherit archived accuracy; rerunning starts fresh, without automatic resume.
  - Interrupted checkpoint writes can still be damaged. Existing evidence, notebook outputs, app and slides were not changed by this ticket. No commit/push.
