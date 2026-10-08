# AI Working Guide

## Repository context

- This repository contains numbered PhD papers and their supporting research
  code. Read the root README and the relevant paper and implementation READMEs
  before changing anything.
- Paper 001's current study uses AAD as its primary dataset and proceeds in two
  stages: search custom ConvLSTM architectures, then compare the selected model
  with the published topology and other model families under fixed settings.
  Both stages use `src.experiments --config <json>`; a small local profile checks
  the pipeline. JSON selects the dataset and classes are read from its folders.
- VDD remains an optional dataset when its local class-folder path is selected
  in JSON. Preserve its historical runs, but treat older runs lacking current
  split, seed, metric, selection, and checkpoint provenance as exploratory and
  non-comparable. Kinetics notebook workflows are diagnostics, not Paper 001
  evidence. Do not report manuscript results until verified run artifacts exist.
- The model API consists of `ConvLSTM`, `PaperConvLSTM`, and `CustomConvLSTM`.
  Express custom recurrent layers as `(filters, (kernel_height, kernel_width))`;
  keep the architecture search focused and change one factor at a time.
  Augmentation is one optional clip-consistent online transform per training
  sample. Splits and full-pipeline seeding are reproducible; select checkpoints
  with validation data, and keep test data locked until model selection is frozen.

## Research rules

- Never invent or silently alter results, citations, datasets, or claims.
- Choose models and checkpoints using validation data only. Keep test data
  locked until a comparison or final configuration has been selected.
- Use the same documented, stratified split for comparable runs. Use source
  grouping when the dataset supports it; never claim group independence when it
  does not.
- Seed Python, NumPy, PyTorch, CUDA, data loading, and augmentation where
  applicable. Record the seed and determinism settings.
- Change one ablation factor at a time against a named reference configuration.
  Keep the data split and training budget fixed so the comparison is interpretable.
- Keep faithful published baselines separate from lightweight variants. Never
  simplify a baseline and still describe it as a faithful reproduction.
- Report variation across the planned confirmation seeds; do not present a
  single favourable run as conclusive.
- Every reported run must retain its full configuration, split reference, code
  revision, parameter count, validation history, selected checkpoint, and final
  metrics. Mark older runs as non-comparable when required fields are missing.
- Keep AAD as the current primary study. Select the dataset in each JSON config
  and derive classes from its directory layout. VDD remains optional when
  explicitly selected; retire only its dedicated workflow, not its generic
  dataset support or historical run artifacts.
- Do not rewrite reported manuscript results until versioned experiment outputs
  exist. Preserve the author's academic voice and distinguish observation from
  interpretation.

## Code rules

- Extend the existing runner and model code; do not create one script per
  experiment or replace working experiments unnecessarily.
- Prefer one validated configuration object/file with explicit CLI overrides.
- Keep architecture variants focused, named, and reproducible. Support width
  and depth only where the implementation and research question justify them.
- Keep runs practical for Colab Pro/Pro+. Use short smoke tests before expensive
  runs and never commit datasets, environments, checkpoints, or generated runs.
- Preserve unrelated user changes. Do not delete or overwrite user work.
- Historical checkpoints from deleted model classes remain research artifacts,
  but they are incompatible with the simplified model API. Do not convert,
  overwrite, or silently load them as another architecture.

## Task workflow

1. Check `git status`, inspect the relevant files, and identify unanswered
   decisions. Stop before editing if unrelated uncommitted work is present.
2. For substantial work, capture the agreed scope in the next
   `agents/work/NNN-short-title/prompt.md`; create it only after required
   questions are answered.
3. Execute only after the prompt is approved. Keep its status current, verify
   every acceptance criterion, and create `completion.md` from the template.
4. Keep the root and relevant paper READMEs concise and current when status,
   commands, dependencies, evidence, or the next task changes.
5. Follow `agents/rules.md` and `agents/config.md` for the shared mechanics.
