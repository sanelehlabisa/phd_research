# Task Prompt

- Ticket: `023-interactive-model-inspection`
- Status: Ready
- Aim: Extend the Colab VS Code notebook with understandable model, candidate,
  and random-weight prediction displays before expensive experiments begin.
- Scope:
  - the Paper 001 Colab notebook
  - model/candidate display helpers only if the notebook cannot reuse existing
    public functions cleanly
  - focused tests and concise notebook documentation
- Changes:
  - Execute only after ticket 022. Preserve its Colab-only setup and dataset
    walkthrough without restoring multi-platform branches.
  - Reuse `CustomConvLSTM`, `PaperConvLSTM`, parameter counting, model registry,
    candidate-manifest parsing, and the safe patterns demonstrated by
    `src.model.main()` and `src.experiments.main()`; do not reimplement models.
  - Present one compact table containing every controlled custom candidate's
    stable name, ordered filter stack, depth, parameter count when practical,
    and research question. Clearly separate the three practical 3D-CNN
    baselines and the native published PaperConvLSTM topology.
  - Display the selected custom model's readable layer structure, input/output
    tensor shapes, trainable parameter count, and estimated device allocation.
  - Run one small A100 smoke inference on a real displayed clip, showing logits
    or probabilities and top predicted classes. Label it prominently as a
    random-weight software check, never model performance or evidence.
  - Avoid allocating every candidate simultaneously. Release temporary models
    and CUDA memory between demonstrations, and do not run the native
    PaperConvLSTM forward pass merely to display its audited topology.
  - Keep the section visual and junior-friendly: use pandas tables and concise
    text instead of raw configuration dumps where possible.
  - End with a clear selected-reference placeholder for ticket 024; do not
    choose a model using random predictions or test data.
- Acceptance criteria:
  - The notebook shows the custom candidates, practical baselines, and published
    topology with their comparison roles unambiguous.
  - One custom model successfully consumes a real dataset tensor on the A100
    and displays a labelled random-weight prediction without training.
  - Parameter counts and model configuration come from repository code or
    committed manifests rather than hand-maintained notebook copies.
  - The notebook remains output-free, top-to-bottom runnable, and does not start
    training, access test samples for selection, or make a performance claim.
- Out of scope:
  - Architecture selection, experiment execution, checkpoint evaluation,
    model/module redesign, candidate expansion, or manuscript changes.
- Open questions: None. Use one lightweight custom candidate for smoke inference
  and represent the published topology without its expensive forward pass.
- Verification:
  - Parse and compile the notebook, then verify unique IDs and cleared outputs.
  - Compare displayed candidates and roles with the validated manifests and
    safe model/plan listings.
  - Smoke-test the selected custom forward pass on the available device and
    verify temporary CUDA allocations are released.
  - Run relevant tests, `git diff --check`, and a scope/status review.

## Execution Prompt

Execute ticket `023-interactive-model-inspection` exactly as written in
`agents/work/023-interactive-model-inspection/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
