# Completion

- Status: Done
- Summary: Added one validated, explicit six-model CustomConvLSTM architecture
  screen for AAD while preserving the default five audited comparison entries
  and validation-only selection protocol.
- Changes:
  - Added `configs/aad_architecture_candidates.json` with exactly the approved
    names, 3x3 layer stacks, no hidden classifier, and one research question per
    candidate.
  - Added typed strict manifest loading, deterministic JSON, and SHA-256
    provenance with field-specific rejection of invalid definitions.
  - Added `--candidates-config`; manifest mode lists and runs only its custom
    candidates through `CustomConvLSTM`, while no-manifest behavior remains the
    original five-model registry.
  - Stored the shared experiment configuration plus exact manifest content and
    hash in run, resolved, checkpoint, per-model, and summary metadata.
  - Documented the side-effect-free listing command, expensive full AAD command,
    interruption behavior, historical-run limitation, and ticket 019 handoff.
- Verification:
  - Added copy-paste `.venv/bin/python -m src.<module>` examples to all six
    executable modules; the training and experiment examples use the committed
    configuration files and the preprocessing example is an AAD dry run.
  - Historical local JSON metadata was audited read-only and treated as
    non-comparable; no test metrics or generated outputs were reused.
  - Temporary pytest checks: `20 passed`, covering strict validation, stable
    round trips and hashes, configuration resolution, unchanged default registry,
    side-effect-free listing, all six CPU forward passes and distinct parameter
    counts, checkpoint provenance, ticket-016 selection, and ticket-017 config.
  - Temporary mocked comparison: `1 passed`; six completed model metrics and the
    final summary retained shared configuration, manifest content/hash, and
    locked test access without loading AAD.
  - All six CLI help checks, resolved-config printing, candidate listing, Black,
    Python compilation, Pylint error-only, JSON parse, test-isolation scan, and
    `git diff --check` passed.
  - Ticket-009 stash remains preserved; baseline definitions, model classes,
    split manifest, augmentation, evaluation command, and manuscript are unchanged.
- Remaining issues: None. The full AAD screen is intentionally left for the user
  to launch with the documented expensive command.
