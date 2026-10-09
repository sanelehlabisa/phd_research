# Completion

- Status: Done
- Summary: Made the AAD custom search cache-aware, smaller, measurable, and search-only by default.
- Changes:
  - The search profile screens eight custom models at 8×32×32, then checks only the validation winner at 48×48, 64×64, and 16 frames. It keeps seed 42, augmentation off, and weight decay at zero.
  - Training caches only the fixed train and validation clips; the locked test partition is not decoded. The Colab runner measures batch 16 versus 32, records throughput and memory, applies the safer measured choice to the search, and archives the split and run artifacts.
  - The notebook/export defaults to search. Comparison remains a separate explicit action using the saved validation-selected configuration.
- Verification:
  - `pytest -p no:cacheprovider -q`: 226 passed; two existing CUDA/NVML warnings.
  - Local AAD smoke profile completed in 31 seconds on the 1,069-clip dataset; it used the 748/160/161 train/validation/test split and saved validation metrics and examples.
  - Real local batch benchmark cached 748 train + 160 validation clips (0.083 GiB, zero test clips); batch 16 measured 354.37 samples/s and batch 32 336.61 samples/s, so the rule selected 16. Colab will measure its own GPU.
  - Search plan lists 11 runs with one-factor checks only; notebook and Python export parity passed; search-only and explicit-comparison archive tests passed; `git diff --check` passed.
- Remaining issues: None. The A100 batch size will be measured when the Colab workflow runs.
