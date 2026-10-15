# Swin3D learning diagnosis — ticket 084

Reviewed and checked locally on 15 October 2026. **The transformers update,
but collapse to nearly input-independent predictions under the checked recipe.**
No causal implementation defect was demonstrated, so the production models,
preprocessing, comparison configs and historical scores were not changed.
This completes diagnosis, not final comparison or a successful learning fix.

Later owner-requested lower-LR checks and modular training are recorded in the
[follow-up](follow-up.md); the original measurements below are preserved.

## Preserved comparison evidence

The [14 October review](../../2026-10-14/README.md#why-the-transformers-require-caution),
its raw console and compact archive report:

| Scratch model | Train correct / 748 | Validation correct / 160 | Train loss | Validation loss |
|---|---:|---:|---:|---:|
| Swin3D-T | 93 | 20 | 2.384147 | 2.384343 |
| Swin3D-S | 93 | 20 | 2.383835 | 2.384090 |

Both predict Begging throughout validation: 12.50% accuracy is exactly its
20/160 support. The training class-prior entropy is 2.383566 nats, almost the
observed losses. That supports a constant/class-prior-like optimization
collapse, **not high-training/low-validation overfitting**. It does not identify
which setting caused it. Custom finalists reached 100% matched training
accuracy and 98.125–98.750% validation; R3D-18/MC3-18 reached 95.625/96.250%
validation under the shared recipe. Their success does not establish that this
recipe is suitable for transformers.

Original Swin per-epoch history files and checkpoint weights are unavailable
in this checkout. The compact ZIP contains JSON summaries/configuration, not
those missing files. Selected/actual epochs (102/134) and endpoint evidence
are available; a complete original LR/gradient trajectory cannot be verified.
Do not reconstruct missing histories as measured data.

## Code-path audit

- **Inputs and sampling:** `AHARDataset` supplies RGB float32 TCHW in [0, 1].
  The wrapper transposes BTCHW → BCTHW exactly once, without changing pixels.
  `timestamps_v1` uses causal timestamp selection and explicit repeat-last
  padding; decode failure raises instead of silently substituting another clip.
  The four diagnostic clips each supplied eight distinct frames, no padding,
  at target 16 FPS, covering approximately 0.434 seconds. This is not full-video
  coverage. No additional mean/std normalization is applied; `weights=None`
  has no pretrained-weight normalization contract. Whether normalization or
  longer temporal coverage helps is untested, not an established bug.
- **Targets/head/initialization:** original integer dataset labels feed an
  11-output head. Both torchvision factories use `weights=None`. Native Swin
  linear layers use truncated-normal initialization; the replacement head
  uses ordinary `nn.Linear` initialization, also used by the shared wrapper.
  This difference was audited but is not demonstrated to cause collapse.
- **Training/loss/updates:** the runner builds the head before creating Adam
  over all model parameters; raw logits enter `CrossEntropyLoss`.
  The shared trainer calls `train`, zeroes gradients, backpropagates and steps;
  evaluation uses `eval`/inference mode. Tests cover both builders, exact axis
  conversion, optimizer coverage and mode transitions. The real check below
  verifies finite gradients and actual patch, attention and head changes.
- **Recipe/scheduler:** archived comparison uses Adam LR 0.001, WD 0,
  augmentation off, batch 32, seed 42 and ReduceLROnPlateau on validation loss
  (factor 0.9, patience 5). Checkpoints use validation loss. The shared
  custom/CNN/transformer training loop is intact; a transformer-specific
  convergence problem is plausible, but its mechanism remains unresolved.

## Real bounded training-only check

[Full trace](diagnosis.json) · [runtime/provenance manifest](run.json).

Both actual registered Swin models ran on the local RTX 4050 Laptop GPU
(6 GB), PyTorch 2.9.1+cu128 / torchvision 0.24.1+cu128. Each received 64 Adam
updates, batch 2, seed 42, LR 0.001, WD 0, no augmentation/pretraining, and
eight 64×64 frames at 16 FPS. The original 11-class head was retained.
This deliberately small diagnostic uses fixed LR, not the original batch-32
plateau schedule; **it is not a reproduction or a held-out performance score**.

Exactly four existing training clips were decoded: dataset indices
`[0, 134, 3, 138]`, alternating two Begging and two Drunkenness clips.
Their paths, SHA-256 hashes and sampling plans are in the trace. The existing
748/160/161 split hash matches the archived split. No validation or test video
was opened; no new split or dataset download occurred.

| Model | Initial → final train-only loss | Final accuracy | Final predictions | Peak allocated VRAM |
|---|---:|---:|---|---:|
| Swin3D-T | 3.535331 → 0.693441 | 50.00% | Begging for all four | 622.77 MiB |
| Swin3D-S | 2.248783 → 0.720270 | 50.00% | Drunkenness for all four | 1,055.85 MiB |

All trainable parameters had present, finite gradients on every step.
The patch projection, first attention QKV and classifier weights changed on
all 64 updates; all optimizer steps were in training mode and optimizer
membership covered every trainable parameter. Nevertheless, mean across-sample
logit standard deviation fell from 0.0334 to 0.00000753 (T) and 0.0477 to
0.00000345 (S): outputs became nearly identical between different clips.
The T loss approached log(2), consistent with the balanced two-class prior.
The trace retains all step losses, periodic clean-subset metrics and per-step
representative gradient/update norms, including this failed discrimination.

An initial instrumentation run gave identical learning values; its local
run is retained, not used as a second independent seed. The final trace adds
all-parameter finite-gradient checks, sampling plans and source hashes.
Small-batch fit is measured; full comparison batch fit is **not** established.
These are instrumented diagnostic memory peaks, including extra evaluations
and parameter snapshots, not production memory/throughput benchmarks.
Failure to memorize four clips in 64 steps does not prove model incapacity.

## Reproduce the diagnostic

From `implementation/`, with existing local AAD and its seed-42 split:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -u -c 'from notebooks.utils.diagnostics import audit_transformer_learning; result = audit_transformer_learning(".", steps=64, batch_size=2); print(result["run_dir"])'
```

The helper reuses `src.experiments.build_registered_model` and
`src.metrics.train_classifier_steps`/`evaluate_classifier`; it is not another
experiment runner. Steps are bounded at 128 and batch at 1 or 2. Outputs go to
a new `runs/experiments/<timestamp>_..._transformer-train-only-audit/`; no weights
are saved. CPU is allowed for regression checks, not claimed as GPU evidence.

## Exact handoff to 083

1. Model wiring and optimizer updates are verified; there is **no proven
   production defect to patch**. Preserve the original scratch failures as the
   shared-recipe result. Do not interpret them as transformer-family inferiority.
2. **Owner approval remains needed** before changing the recipe: first test only
   LR 0.0001 against the recorded 0.001 on the same fixed training subset and
   seed, keeping optimizer, input, WD, augmentation and scratch initialization
   fixed. This is a proposed diagnostic, not a measured improvement. If it
   restores discrimination, approve a bounded validation-only convergence
   confirmation before expensive comparison; never pick LR on test. Any
   transformer-tuned control must have its own protocol label and retain the
   original scores. Do not add pretraining, larger inputs or a sweep implicitly.
3. 083 still owns portable ten-model JSON, local common-batch preflight,
   dataset-alias compatibility and guarded pre-scoring test recovery. Nothing
   here launches those stages, reranks search, freezes a new winner or scores
   test. Existing missing weights remain missing. More epochs alone are not an
   evidence-backed remedy; no accuracy threshold is promised.

## Integrity

The trace records source hashes against base revision `77a0425` plus the
uncommitted diagnostic helper. Historical raw inputs, search values, selection,
configs, images and the completed graph-first report remain unchanged.
The report builder only gained safe coexistence with dated-report subdirectories;
its edited-file and unexpected-root-file protections remain tested.

- Original console SHA-256: `c7609a5ec25cf93399de2f951ad93cb878212c99db88e3356da98d187e2cf98a`.
- Original essential ZIP SHA-256: `f4086085850f1bce64aa3abaf7a968cebc6166d4e5cd304afb6b955bc2081384`.
- Existing split SHA-256: `aada503d63d4433b272237db6cf27d7b1c0eaeb0091d5ba8289343a6111454d8`.
- Diagnostic trace SHA-256: `190e6cdaea794dd8bc5959abc98fff2afb95b86607cb2532eafcf7f7e014d3ba`.

No commit, push, manuscript change, full comparison or test evaluation was run.
