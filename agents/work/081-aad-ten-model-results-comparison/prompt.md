# Task Prompt

- Ticket: `081-aad-ten-model-results-comparison`
- Status: Draft
- Aim: Present the final ten-model AAD test comparison with traceable performance and efficiency measurements (requested Group 6).
- Scope: Paper 001 final comparison table/discussion and derived evidence tables under `implementation/reports/YYYY-MM-DD/`.
- Dependencies: [079](../079-aad-results-section-structure/prompt.md), [080](../080-aad-search-figure-and-appendix/prompt.md), and complete verified [076](../076-aad-shortlist-comparison/prompt.md) comparison artifacts. This ticket reports runs; it does not launch them.
- Changes:
  - Read five custom configurations from the frozen selection receipt and baseline names from the resolved experiment config. Current baseline IDs: `paper_convlstm_adapted`, `r3d_18`, `mc3_18`, `swin3d_t`, `swin3d_s`.
  - Verify all ten trained from scratch on one frozen split with the documented common input, seed and practical training protocol. Report actual batches, early stopping, checkpoint selection and any failures/deviations; disclose custom-selected hyperparameters, not individually tuned family optima.
  - Add exactly ten rows after shortlist selection: model name, architecture/configuration, parameters, test loss/accuracy, pooled precision/recall/F1, and available model-size/latency measurements with units and measurement conditions. Link per-class counts/metrics and confusion evidence.
  - Use matching-checkpoint, full-test metrics, reconciled with saved predictions. Explain that pooled P/R/F1 equal accuracy in this single-label task; no macro averages (088). Keep unavailable fields missing, historical scores unchanged, and paper-reported numbers separate from retrained results.
  - Label the published ConvLSTM as input-adapted, not an exact reproduction. Follow the table with one paragraph comparing classification performance, parameter savings, gains from larger baselines and the supported proposed-model trade-off.
- Acceptance criteria:
  - Exactly five verified custom finalists and five configured baselines; each value traces to its frozen checkpoint, split and test receipt.
  - No validation number substituted for test, hidden failed model, unmeasured latency claim, or assumed custom-model victory.
  - All checkpoints and the example custom model were selected before test; comparison claims follow measured results.
- Out of scope: Training/retesting, changing the shortlist or hyperparameters, benchmarking new latency measurements, fabricated missing scores, commit or push.
- Open questions: No additional reporting decision; inherits 080's pending shortlist clarification. Complete final comparison evidence is required before execution approval.
- Verification: Reconcile ten receipts/configs and split/checkpoint identities; check metric calculations, parameter reductions, units and missing fields; build LaTeX and run `git diff --check`.

## Execution Prompt

After dependencies, verified artifacts and owner approval, complete only the
reporting work above. Follow `AGENTS.md`, `agents/rules.md`, and
`agents/config.md`. Verify, update status and create `completion.md` from
`agents/templates/completion.md`. Do not train, retest, commit or push.
