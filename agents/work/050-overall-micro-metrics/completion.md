# Completion

- Status: Done
- Summary: New runs report overall micro metrics and rank by validation accuracy,
  loss, then parameter count; checkpoints still select by validation loss.
- Changes:
  - Updated shared metrics, CLI/notebook summaries, ranking and protocol metadata.
  - Preserved read-only support for historical macro-protocol checkpoints.
  - Added hand-computed, batch-boundary, ranking, provenance and saved-JSON tests;
    corrected Windows path/CRLF assertions without altering notebook content.
- Verification:
  - Full implementation suite: `python -m pytest tests -q --tb=short --show-capture=no`
    — 160 passed; Black, Python parsing and `git diff --check` passed.
  - Inspected synthetic training JSON/checkpoint metadata: loss plus overall
    accuracy/precision/recall/F1, micro protocol and locked test access.
  - SHA-256 snapshot confirms all historical runs, configs, manuscript files and
    saved notebooks are unchanged.
- Remaining issues: No real dataset training or GPU run performed for this ticket.
