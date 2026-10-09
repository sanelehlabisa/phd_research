# Task Prompt

- Ticket: `073-focused-shape-search-run-all`
- Status: Done
- Aim: Run a bounded 1–3-layer architecture search and the full guarded comparison workflow automatically.
- Scope: Paper 001 AAD profiles, existing runners/helpers/notebook/export, tests, guides and imported artifact organization.
- Approval: User explicitly requested implementation on 9 October 2026 after clarification; this consolidates 070–072 and revises the uncommitted 069 work.
- Decisions:
  - Evidence-guided, not unbiased/exhaustive: keep historical successes and failures visible; a two-layer winner does not rule out deeper models.
  - User checked similarly named videos and confirmed independent recordings. Save this as a dataset/split-bound user attestation, not independently verified source/subject separation. Exact duplicates or contrary source IDs still block.
  - No real Colab training, test evaluation, commit or push in this implementation task.
- Changes:
  - New versioned search: 34 distinct custom stacks. Flat widths 4/8/16/24/32 at depths 1/2/3, plus [64,64,64]; two-layer orientations of (8,16), (16,24), (16,32); six non-flat three-layer patterns using 16/32; all six permutations of 8/16/32. Keep 3x3 kernels and adaptive head. No four-layer stacks.
  - Cover all 1/3/13 relative-width patterns at depths 1/2/3, not every numerical combination. Keep [32,32,32] and [64,64,64] as wide controls; no 48-wide grid.
  - Common search cap 128 epochs, minimum 64, patience 24; retain gentle scheduler, LR calibration, 32/48/64 pixels, at-most-16-frame sampling, two-seed confirmation and separate WD/FPS ablations from 069.
  - Upper search budget 177 before exact reuse: 12 calibration + 48 flat + 6 references + 54 shape + 24 confirmation + 18 WD + 15 temporal. Save deterministic manifest/counts and updated memory preflight. Never mix 068/069 scores into the new protocol.
  - Fix new ranking with exact correct/total accuracy and equal resolution/seed weights, then loss/parameters/name. Preserve legacy versioned selection reconstruction and archived results.
  - Default Run All automatically passes verified paths through search, selection, eight-model comparison, freeze, one-time test, labelled best-custom examples and ZIP download. Resume/download retry must not retrain complete jobs or repeat test attempts.
  - New comparison profile: same eight models, 50 frames at 50x50, batch 1, seed 42; maximum 512 epochs, minimum 128, patience 32. Validation alone selects every checkpoint and the custom example model. Legacy comparison profile stays unchanged.
  - Export all results, explicit micro/macro metrics, per-class metrics/confusion, histories, parameters/sizes/timing and exact provenance. Preserve stage/partial ZIPs; no manual directory or stage edits for the default flow.
  - Record the reviewed AAD identity; flag unknown/new source evidence without inferring recording IDs from filename suffixes. No silent resplit or test-frame decoding before freeze.
  - Move the verified imported artifact bundle intact under ignored runs/imports/, update report links, and narrowly ignore downloaded temp configs. No evidence deletion or raw-path rewriting; verify file hashes before/after.
  - Preserve saved notebook outputs/metadata and notebook/export parity. Mark 070–072 superseded, update guides and complete this ticket.
- Acceptance criteria:
  - Exactly 34 distinct 1–3-layer stacks, all width-order patterns and the 177-job bound are verified.
  - New protocol uses 128/512 epoch caps; legacy profiles and saved selections remain readable.
  - Exact accuracy ties reach the declared tie-break; incomplete/tampered evidence cannot select or test.
  - One synthetic Run All reaches export without manual paths; audit/freeze/failure/resume guards hold.
  - Artifact hashes and notebook saved outputs are unchanged; complete metrics and verified ZIPs are available.
- Out of scope: New datasets/resplits, stateful streaming, pretrained tuning, manuscript claims, real GPU experiments, commit/push.
- Open questions: None; bounded candidate choice and final cap follow the user's implementation request.
- Verification: Candidate/count/pattern and exact-ranking unit tests; mocked full workflow and audit/freeze/retry tests; synthetic staged run; legacy comparison and notebook regressions; archive/output checks; formatter and git diff checks.

## Execution Prompt

Complete ticket `073-focused-shape-search-run-all`. Follow `AGENTS.md`,
`agents/rules.md` and `agents/config.md`. Implement the agreed focused search
and automatic guarded workflow, preserve saved evidence and outputs, verify
the result, update status and create `completion.md` from the template.
Do not run real experiments, commit or push.
