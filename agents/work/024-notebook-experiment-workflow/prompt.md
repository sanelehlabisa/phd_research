# Task Prompt

- Ticket: `024-notebook-experiment-workflow`
- Status: Ready
- Aim: Finish the Colab VS Code notebook with a safe final section for running
  controlled experiment stages and visually reviewing validation evidence.
- Scope:
  - the Paper 001 Colab notebook
  - experiment result-display helpers only when existing artifacts cannot be
    presented cleanly from the notebook
  - focused tests and concise root, paper, and implementation guidance
- Changes:
  - Execute only after tickets 022 and 023. Keep all dataset/model inspection
    sections intact and place experiment execution at the end.
  - Reuse the validated plan and `src.experiments` command path as the source of
    truth. Do not copy training loops, metrics, checkpoint selection, splitting,
    augmentation, or model construction into the notebook.
  - First display a readable plan table containing stage, trial, model role,
    changed factor, seed, input size, training budget, and expected run count.
    Preserve the separate architecture-screen, practical-baseline,
    published-topology, and one-factor-ablation roles.
  - Add one obvious parameter cell for the selected plan stage and optional
    validation-selected reference candidate. Validate both before allocating
    data or models.
  - Keep execution opt-in with `RUN_EXPERIMENTS = False` by default. When
    enabled, invoke `python -m src.experiments --run-plan-stage ...` using
    `sys.executable`, the active A100 environment, existing AAD path, and
    existing runs directory. Stream stdout so progress is visible in VS Code.
  - Never automatically run all stages, retry failed trials, select from an
    incomplete screen, or open the test split. Ticket 021's optional candidate
    decision must remain separate and occur before the full screen.
  - After a completed stage, discover only that run's recorded artifacts and
    display validation rankings, configuration/provenance, parameter/runtime
    comparisons, training curves, and confusion matrices when present.
    Clearly distinguish missing, partial, and completed evidence.
  - Use pandas and matplotlib for readable inline output while retaining JSON,
    checkpoints, plots, and run metadata under the normal timestamped `runs/`
    structure. Do not save notebook-only copies as research evidence.
  - End with the next research action and keep final-test evaluation absent.
- Acceptance criteria:
  - Safe plan inspection works without loading AAD or allocating a model.
  - No experiment starts until the explicit flag is enabled and a valid stage
    is selected; the exact command and resolved paths are shown first.
  - Experiment progress appears in the VS Code notebook while the existing
    runner retains all research logic and timestamped artifacts.
  - A completed or partial run can be inspected visually without ranking from
    test metrics or confusing incomplete evidence with a final result.
  - The notebook remains valid, output-free, Colab-only, and top-to-bottom
    usable after kernel restart when the remote checkout and dataset exist.
- Out of scope:
  - Running the expensive suite while implementing the ticket, final test
    evaluation, VDD validation, changing the controlled plan, candidate
    expansion, manuscript rewriting, or committing generated runs.
- Open questions: None. The notebook ends with opt-in plan-stage execution and
  validation-only visualization; final evaluation remains a later ticket.
- Verification:
  - Parse/compile the notebook and verify unique IDs, empty outputs, and a false
    default execution guard.
  - Match the generated command to `src.experiments --help` and exercise safe
    plan listing plus invalid-stage/missing-path failures without training.
  - Test result discovery against representative completed and partial local
    run metadata without selecting from test data.
  - Run the focused suite, `git diff --check`, and a final scope/status review.

## Execution Prompt

Execute ticket `024-notebook-experiment-workflow` exactly as written in
`agents/work/024-notebook-experiment-workflow/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
