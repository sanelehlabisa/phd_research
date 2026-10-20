# Task Prompt

- Ticket: `096-spaces-python-compatibility`
- Status: Done
- Aim: Fix Spaces startup and provide a simple, self-contained deployment bundle.
- Scope: Demo hashing, Spaces requirements/exporter, tests and deployment instructions.
- Approval: User requested this fix and one dependency file for deployment.
- Changes:
  - Support Python 3.10 without weakening checkpoint hash or LFS validation.
  - Use the user's compatible pins in one standalone Spaces requirements file.
  - Export a clean folder and ZIP with source, trained weights and metadata.
- Acceptance criteria:
  - Loading works without `hashlib.file_digest`; invalid weights still fail safely.
  - Bundle contains one requirements file; no runtime installation or research data.
  - Existing model, predictions and research dependencies remain unchanged.
- Out of scope: Retraining, paper/results changes, commit/push or remote deployment.
- Open questions: None.
- Verification: Regression tests, dependency compatibility checks, real local checkpoint inference and ZIP contents/hash checks.

## Execution Prompt

Complete the approved work above, follow repository rules, verify the result and create `completion.md`. Preserve existing work; do not commit, push or deploy.
