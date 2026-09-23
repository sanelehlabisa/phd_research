# Task Prompt

- Ticket: `015-make-splits-reproducible`
- Status: Done
- Aim: Make every comparable AAD run reuse one committed, stratified 70:15:15
  clip split and reproducible random state.
- Scope:
  - `.gitignore` only if split/runtime exclusions need clarification
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/dataset.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/utils.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/model.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/train.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/evaluate.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiments.py`
  - one AAD manifest under
    `papers/001-journal-abnormal-activity-recognition/implementation/splits/`
  - root, paper, and implementation guidance affected by commands or status
- Changes:
  - Add a shared `--seed` option defaulting to `42` and a shared
    `--split_manifest` option to training, evaluation, and comparison commands.
    Keep the existing model-smoke seed but route it through the same seeding
    helper.
  - Seed Python, NumPy, PyTorch CPU/CUDA, data-loader shuffling/workers, and
    online augmentation. Disable cuDNN benchmarking and record deterministic
    settings; use deterministic algorithms in warning mode so unsupported GPU
    operations are visible without silently changing the experiment.
  - Build one stratified 70:15:15 split from the dataset sample inventory using
    seed `42`. This is a clip-level split because AAD exposes class folders and
    clips but no reliable source-group identifier; do not claim camera or source
    independence.
  - Store relative sample paths, class labels/indices, split membership,
    requested ratios, actual counts, per-class counts, seed, schema version,
    dataset name, and a deterministic inventory hash in the manifest. Commit
    the AAD manifest because it defines the experiment protocol, not a result.
  - Load and validate an existing manifest instead of silently regenerating it.
    Reject missing, duplicated, unknown, relabelled, or omitted samples, changed
    class mappings, wrong ratios or seed, and train/validation/test overlap.
  - Generate a manifest only when the resolved path does not exist, using an
    atomic write. Default its name to `<dataset>_seed<seed>.json` under
    `splits/`; an explicit path remains supported for later datasets.
  - Replace independent `random_split` calls in train, evaluate, and experiments
    with subsets reconstructed from the same validated manifest. Evaluation
    must use only its recorded test members.
  - Record the manifest path, inventory hash, split seed, requested ratios,
    actual counts, and deterministic settings in each command's `run.json` and
    resolved configuration.
  - Update concise AAD commands and guidance with the seed and manifest path.
- Acceptance criteria:
  - The committed AAD manifest covers every current AAD clip exactly once with
    no overlap and an approximately 70:15:15 allocation produced by stratified
    splitting; every class is represented in all three partitions.
  - Train, evaluate, and experiments resolve identical indices for each named
    partition when given the same dataset and manifest.
  - Repeated manifest creation from the same ordered or reordered inventory and
    seed produces identical path membership and inventory hash.
  - A changed dataset inventory, class mapping, ratio, or seed causes a clear
    validation error rather than a new implicit split.
  - Data-loader order and clip-consistent online augmentation are repeatable for
    the same seed, including worker processes, while a different seed changes
    the stochastic sequence.
  - Run metadata identifies the exact manifest and determinism settings used.
  - No model architecture, frame sampling, augmentation policy, optimizer,
    scheduler, loss, metric definition, checkpoint policy, or manuscript result
    is changed.
  - Active Python files compile, formatting checks pass, and `git diff --check`
    reports no errors.
- Out of scope:
  - Source-group splitting without trustworthy group metadata.
  - Trying alternative split ratios or choosing models from test performance.
  - Metric aggregation and validation-selected checkpoint restoration; ticket
    016 owns those changes.
  - Connecting the stashed ablation configuration, running full experiments,
    stateful streaming, or manuscript result changes.
  - Moving, deleting, committing, or interrupting active local training runs,
    models, checkpoints, datasets, or ignored run artifacts.
- Open questions: None.
- Verification:
  - Format and compile every modified Python file; run Pylint error-only and
    `git diff --check`.
  - Verify parser defaults and shared `--seed`/`--split_manifest` options for
    train, evaluate, and experiments, plus the shared seed path for model smoke.
  - Exercise manifest creation and validation with a temporary balanced sample
    inventory, including reordered input and intentional corruption cases.
  - Generate and validate the real AAD manifest without decoding video; report
    total and per-class train/validation/test counts and overlap checks.
  - Confirm all three consumers reconstruct exactly the manifest indices and
    that their declared split metadata is written below a temporary run root.
  - Test deterministic data-loader ordering and augmentation with the same and
    different seeds using small synthetic tensors; do not train a model.
  - Inspect the final diff to confirm split/seeding changes only, the manuscript
    is untouched, ignored artifacts remain ignored, active jobs remain running,
    and the ticket-009 stash is unchanged.

## Execution Prompt

Execute ticket `015-make-splits-reproducible` exactly as written in
`agents/work/015-make-splits-reproducible/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
