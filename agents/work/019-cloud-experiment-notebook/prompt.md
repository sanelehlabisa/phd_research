# Task Prompt

- Ticket: `019-cloud-experiment-notebook`
- Status: Done
- Aim: Add one Colab-first, portable Paper 001 notebook that orchestrates the
  existing AAD scripts without duplicating research logic or weakening test
  isolation.
- Scope:
  - one new notebook at
    `papers/001-journal-abnormal-activity-recognition/implementation/notebooks/aad_experiment_workflow.ipynb`
  - `AGENTS.md` and the root, Paper 001, and implementation READMEs only where
    notebook usage, status, or the next task changes
  - `.gitignore` only if notebook-specific generated files need an explicit rule
  - no implementation-module changes unless a current CLI cannot support an
    approved notebook stage; stop and report that blocker before expanding scope
- Changes:
  - Create one concise notebook with this manual workflow:
    1. configure paths and stage controls;
    2. detect the platform, Python, GPU, CUDA, PyTorch, and free disk space;
    3. locate an existing repository or clone the public repository only when it
       is absent—never pull over or overwrite an existing checkout;
    4. validate the AAD and split-manifest paths;
    5. preview clean and augmented dataset samples through `src.dataset`;
    6. smoke-test model construction and predictions through `src.model`;
    7. list the approved architecture candidates and optionally run
       `src.experiments` for architecture screening;
    8. optionally run the audited baseline comparison after a custom candidate
       has been chosen;
    9. train one explicit selected custom architecture through `src.train`;
    10. inspect validation summaries, histories, checkpoints, and confusion
        matrices; and
    11. evaluate one frozen checkpoint through `src.evaluate` only after the
        user explicitly unlocks the test stage, then archive the chosen run.
  - Keep the existing Python modules and committed JSON files as the single
    source of truth. Notebook cells must build and print normal command lists,
    then call them with `subprocess.run(..., check=True)` and `sys.executable`;
    do not reimplement datasets, models, training loops, metrics, splitting,
    checkpoint selection, or evaluation in notebook cells.
  - Add one obvious parameter cell containing at least Colab-ready repository,
    dataset, and runs paths; the public AAD Kaggle handle; config and
    candidate-manifest paths; selected ConvLSTM layers; seed; sequence length;
    spatial size; batch size; epochs; early-stopping patience; workers; and
    checkpoint path.
  - Default the explicit custom stack to the user's current preliminary
    `8-8-8`, all with `3x3` kernels, but label it a validation-screen candidate,
    not a final or optimal model. Make every architecture layer visible in the
    generated command.
  - Separate comparable screening/confirmation settings from optional extended
    selected-model training. Clearly state that changing frame count, spatial
    size, batch size, or budget creates a new protocol and must not be compared
    directly with the original `16 x 32 x 32` screen as if only architecture
    changed.
  - Make all costly or state-changing stages opt-in with clearly named booleans,
    defaulting to false. Dataset/path validation, hardware reporting, config
    printing, and model listing may be safe defaults; no training, experiment,
    evaluation, clone, install, or archive action may run merely by opening the
    notebook.
  - Keep test access locked by default. The evaluation cell must require both an
    explicit enable flag and an existing validation-selected checkpoint; it
    must never discover and evaluate every checkpoint automatically.
  - Use Colab-ready `/content/...` defaults while retaining local Jupyter and
    Kaggle compatibility through user-editable paths. Warn that Colab storage is
    temporary and direct Kaggle outputs to a writable working directory.
  - Add a separately guarded KaggleHub stage that downloads only the public AAD
    folder from `sanelehlabisa/abnormal-activities-dataset`. Keep it disabled by
    default, print the resolved path, and never embed credentials or tokens.
    Explain the optional `KAGGLE_API_TOKEN` Colab secret if Kaggle requests
    authentication or consent.
  - Provide an optional dependency-install cell that installs only missing
    project packages and does not silently replace the cloud runtime's
    accelerator-enabled `torch`, `torchvision`, or `torchaudio`. Print installed
    versions so each generated run can be interpreted against its environment.
  - Add lightweight result-inspection cells that read existing JSON/PNG
    artifacts without selecting a model from test metrics. Keep run artifacts
    outside Git and offer an explicit opt-in archive command for download or
    persistent storage.
  - Keep the notebook's committed outputs empty, execution counts null, code
    cells short, explanations plain, and section order usable top-to-bottom.
  - Document how to upload/open the notebook in Colab and Kaggle, where to set
    dataset and output paths, which cells are expensive, how to resume from a
    checkpoint, and why the test cell stays disabled until selection is frozen.
  - Keep roadmap ticket 019 associated with cloud-assisted architecture
    confirmation; the full expensive runs remain user-operated. Do not mark the
    architecture or baseline comparison complete merely because the notebook
    exists.
- Acceptance criteria:
  - The committed file is a valid, output-free nbformat-4 notebook and is linked
    from the implementation README.
  - A single parameter cell adapts the notebook to local Jupyter, Colab, and
    Kaggle without editing the training modules or committed experiment JSON.
  - Every operational stage invokes an existing `src.*` CLI, prints the exact
    command, fails clearly on missing paths, and stops on a non-zero exit code.
  - Safe inspection cells can run without AAD training, while every expensive,
    network, archive, or test action is disabled by default.
  - The notebook preserves the committed split, seed/config provenance,
    validation-only ranking, early stopping, restored checkpoint behavior, and
    final-test isolation already enforced by the scripts.
  - The architecture screen, audited baseline comparison, extended selected
    training, and final evaluation are visibly distinct stages with honest
    comparability warnings.
  - The notebook includes artifact inspection and persistence guidance suitable
    for interrupted Colab sessions and Kaggle outputs.
  - No datasets, credentials, environments, notebook outputs, checkpoints,
    generated runs, or archives are committed.
- Out of scope:
  - Running any full architecture, baseline, training, or evaluation job while
    implementing this ticket.
  - Publishing AAD, automatically mounting the user's Drive, or storing cloud
    credentials.
  - Rewriting the working CLI modules, combining their logic into the notebook,
    changing model architectures, experiment results, manuscript claims, split
    manifests, or ticket-018 configuration files.
  - Claiming that a larger input or longer training budget improves the model
    before controlled evidence exists.
- Open questions: None. The notebook uses Colab-ready paths and the user-approved
  public Kaggle dataset handle while keeping cloning, downloading, installing,
  training, evaluation, and archiving disabled by default.
- Verification:
  - Parse the notebook as JSON; verify nbformat 4, unique cell IDs, null execution
    counts, and empty outputs.
  - Compile every Python code cell and run static checks that all action flags
    default to false, test evaluation has a double guard, commands use
    `sys.executable`, and no credential-like values are present.
  - Match every notebook command and option against the corresponding module's
    `--help`; exercise only safe help, config-printing, and candidate-listing
    paths locally.
  - Use temporary paths/mocks to exercise platform/path validation, command
    printing, missing-path failures, non-zero subprocess failures, summary
    inspection, and archive opt-in without accessing AAD or creating a research
    run.
  - Review Markdown order and warnings for local, Colab, and Kaggle use, then run
    `git diff --check` and a final scope/status review.

## Execution Prompt

Execute ticket `019-cloud-experiment-notebook` exactly as written in
`agents/work/019-cloud-experiment-notebook/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
