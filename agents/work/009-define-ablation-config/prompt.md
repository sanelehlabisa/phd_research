# Task Prompt

- Ticket: `009-define-ablation-config`
- Status: Draft
- Aim: Give Paper 001 one validated experiment configuration so later ablation
  runs can be changed, reproduced, and compared without duplicate scripts.
- Scope:
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/`
  - a concise ablation configuration example and focused tests in the same project
  - implementation and paper READMEs only when commands or task status change
- Changes:
  - Add one typed configuration object covering dataset and split identifiers;
    train, validation, and test ratios; seed; image height and width; sequence
    length; convolution, ConvLSTM, and dense widths; convolution and ConvLSTM
    layer counts; dropout; optimiser and loss settings; learning rate; weight
    decay; scheduler; augmentation; batch size; epochs; early stopping; worker
    count; and pinned memory.
  - Make the ablation runner load a JSON configuration and apply only explicit
    CLI overrides. Existing CLI-only usage and its defaults must remain available.
  - Add a concise example for the agreed 70:15:15 ablation protocol. Do not
    silently make historical runs appear to have used that split.
  - Validate ratios, dimensions, widths, layer counts, probabilities, optimiser
    names, and positive training values with clear errors before a run starts.
  - Represent currently unsupported depth choices in the schema, but reject them
    clearly until ticket 010 implements and tests the architecture change.
  - Print and save the fully resolved configuration in stable JSON form.
  - Refactor shared argument/configuration handling only where needed; do not
    create a separate script for each experiment or change training behaviour.
  - Add focused tests for legacy defaults, JSON loading, CLI precedence,
    validation, round-trip serialisation, and unsupported architecture depth.
- Acceptance criteria:
  - One configuration expresses every planned ablation control without editing
    Python source.
  - Running the ablation command without a configuration retains its existing
    supported CLI defaults; the example explicitly resolves to 70:15:15.
  - A JSON value is used unless the same option is explicitly overridden on the
    CLI.
  - Invalid or unsupported configurations fail before dataset loading or model
    training begins and explain the offending field.
  - The resolved configuration round-trips through JSON without losing values.
  - Existing model variants and historical results remain untouched.
  - Focused tests pass and `git diff --check` reports no errors.
- Out of scope:
  - Changing dataset splits or seeding; ticket 013 owns reproducible data.
  - Correcting metrics, checkpoint selection, or test access; ticket 014 owns
    those changes.
  - Creating run directories or provenance bundles; ticket 015 owns artifacts.
  - Changing model depth; ticket 010 owns model implementation.
  - Running experiments or editing manuscript results.
- Open questions: Deferred until ticket 010 stabilises the stacked-model layer
  specification; revise this ticket before resuming it.
- Verification:
  - Run the focused configuration tests.
  - Resolve legacy defaults, the example JSON, and that JSON with one CLI
    override; inspect their stable JSON output.
  - Run one invalid-value and one unsupported-depth case and confirm they fail
    before dataset access.
  - Confirm `git diff --check` and review the diff for unrelated changes.

## Execution Prompt

Execute ticket `009-define-ablation-config` exactly as written in
`agents/work/009-define-ablation-config/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
