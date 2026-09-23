# AI Working Guide

## Repository context

- This repository contains numbered PhD papers and the research code supporting
  their claims. Read the root README and the relevant paper and implementation
  READMEs before changing anything.
- Paper 001 studies lightweight ConvLSTM-based abnormal human-activity
  recognition from surveillance video. More than 30 earlier runs mainly varied
  model width and dense-layer size. They are useful for shortlisting, but are
  exploratory until their configurations, splits, seeds, and training protocol
  are shown to be comparable.
- The current candidate is the `64-32-16-64` custom ConvLSTM. Call it the
  **selected lightweight ConvLSTM candidate**, not an optimal model, until the
  controlled evidence supports a stronger claim.
- The model API now consists of `ConvLSTM`, `PaperConvLSTM`, and
  `CustomConvLSTM`. Express each custom recurrent layer as
  `(filters, (kernel_height, kernel_width))`; do not expand this into an
  unrestricted hyperparameter search. The comparison baselines are now audited,
  and augmentation is one optional, clip-consistent online view per training
  sample. The immediate priority is to make the split and random state
  reproducible before comparison training.

## Research rules

- Never invent or silently alter results, citations, datasets, or claims.
- Choose models and checkpoints using validation data only. Keep test data
  locked until a comparison or final configuration has been selected.
- Use the same documented, stratified, group-aware split for comparable runs.
  Prevent related clips or camera views from crossing splits.
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
- Use AAD for architecture screening and most ablations, then evaluate the
  selected configuration on VDD. If the manuscript states a different protocol,
  flag the conflict instead of silently changing the paper or code.
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
