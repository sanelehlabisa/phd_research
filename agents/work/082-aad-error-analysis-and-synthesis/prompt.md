# Task Prompt

- Ticket: `082-aad-error-analysis-and-synthesis`
- Status: Draft
- Aim: Close the AAD experimental section with honest class-level analysis and a bounded performance–complexity conclusion (requested Group 7).
- Scope: Paper 001 results-section confusion figure, class analysis and closing paragraph; traceable derived assets under `implementation/reports/YYYY-MM-DD/`.
- Dependencies: [080](../080-aad-search-figure-and-appendix/prompt.md), [081](../081-aad-ten-model-results-comparison/prompt.md), frozen best-custom selection and its saved full-test predictions/confusion matrix.
- Changes:
  - After the final table and comparison paragraph, insert the test confusion matrix of the custom model chosen using validation before test. Do not switch the example model to the highest test scorer.
  - Label actual/predicted axes, class order and counts; explain any normalization. Identify consistently recognized classes and frequent confusions using per-class support/metrics, not visual impression alone.
  - Discuss possible error causes using available saved examples. Separate observations from hypotheses; state when evidence cannot distinguish dataset limitations from model limitations.
  - Close with one concise paragraph connecting search evidence, any supported capacity turning point, the ten-model comparison, quantified parameter reduction, measured retained/improved performance and practical relevance.
  - Limit any best-trade-off claim to evaluated configurations and the declared criterion. If no turning point, superiority or performance retention is supported, say so. State single-seed/clip-split limitations; small parameter count alone does not prove real-time deployment.
  - Keep the narrative order fixed and hand broader abstract/methods/conclusion consistency back to [062](../062-update-manuscript-after-aad-results/prompt.md).
- Acceptance criteria:
  - Confusion totals, class support, accuracy and model/checkpoint identity reconcile with 081's test evidence.
  - Error explanations are evidence-linked or clearly labelled hypotheses; no causal, global-optimality or cross-dataset claim is inferred.
  - Figure follows comparison discussion; class analysis and closing synthesis follow it, with references resolved.
- Out of scope: Rerunning test, selecting models from test results, relabelling data, changing code/protocol, rewriting the whole paper, commit or push.
- Open questions: None beyond upstream decisions and verified artifacts; execution still requires owner approval.
- Verification: Reconstruct confusion counts from saved predictions, audit label order/support and quantitative claims, inspect the compiled figure and manuscript flow, run `latexmk -pdf main.tex` and `git diff --check`.

## Execution Prompt

After upstream tickets, verified evidence and owner approval, complete the work
above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`.
Verify, update status and create `completion.md` from
`agents/templates/completion.md`. Do not train, retest, commit or push.
