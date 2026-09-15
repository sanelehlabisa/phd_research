# Task Prompt

- Ticket: `005-separate-paper-code-roadmaps`
- Status: Done
- Aim: Separate the paper and implementation roadmaps so each README shows a concise, numbered queue for its own repository.
- Scope:
  - `starter_journal/README.md`
  - `/home/it-software/Desktop/projects/2026/convlstm_abnormal_human_activity_recognition/README.md`
- Changes:
  - Refocus the journal README on the paper rather than experiment implementation details.
  - Keep a short current-state note that `Proposed Lightweight ConvLSTM (64-32-16-64)` is provisional until controlled results are available.
  - Keep the implementation repository link and state that the paper must record the exact code commit used for final results.
  - Replace the detailed experiment protocol and implementation task lists in the journal README with one short external dependency pointing to the implementation README.
  - Add the recommended future structure for `Experimental Results and Discussion`:
    1. Experimental Setup
    2. Comparison and Ablation Protocol
    3. Main Comparative Results, with AAD and VDD subsections
    4. Ablation Studies
    5. Error Analysis
    6. Efficiency and Deployment Trade-offs
    7. Overall Discussion and Limitations
  - Add a numbered paper-ticket checklist showing completion and dependencies:
    - checked `003-starter-bib-private-notes`;
    - checked `004-starter-journal-experiment-roadmap`;
    - checked `005-separate-paper-code-roadmaps`;
    - unchecked `006-rewrite-experimental-results`, blocked until the implementation experiment queue is complete;
    - unchecked `007-align-paper-claims`, blocked until ticket 006 is complete.
  - State that ticket 007 updates the abstract, introduction contributions, conclusion, and any methods, limitations, tables, figures, or claims affected by the verified results.
  - Keep useful paper-writing guidance concise: separate baselines from ablations, report variability, support causal claims only with controlled evidence, and include only figures that explain a result.
  - Move ownership of the detailed protocol and experiment plan to the implementation README.
  - In the implementation README, retain its current state and known problems, then replace the plain numbered plan with this sequential checkbox queue:
    - unchecked `EXP-001-repair-experiment-runner`, marked `Next`;
    - unchecked `EXP-002-define-architecture-registry`, blocked by EXP-001;
    - unchecked `EXP-003-screen-aad-architectures`, blocked by EXP-002;
    - unchecked `EXP-004-run-focused-ablations`, blocked by EXP-003;
    - unchecked `EXP-005-confirm-and-export-results`, blocked by EXP-004.
  - Keep the agreed implementation details in the implementation README: fixed group-aware 70:15:15 splits, validation-only selection, locked tests, seeds `42` and `2026`, top five custom models plus the original ConvLSTM and three 3D-CNNs, controlled width/depth variants, focused ablations on the selected model, three input settings, early stopping, and versioned provenance.
  - Keep both READMEs concise and avoid duplicating the detailed implementation plan in the paper repository.
  - Preserve every existing uncommitted implementation change and modify only its `README.md`.
- Acceptance criteria:
  - The journal README is primarily a paper roadmap and contains no detailed implementation experiment matrix.
  - The journal README includes the recommended Results-section flow and numbered paper-ticket checkboxes with clear blockers.
  - The implementation README contains the detailed experiment protocol and a sequential numbered checkbox queue with EXP-001 visibly marked as next.
  - Both READMEs link to the other repository and use consistent model, dataset, and ticket terminology.
  - No LaTeX, bibliography, source code, model metadata, result, or generated curve file is modified.
  - The existing implementation changes and untracked outputs remain untouched.
  - Markdown has no trailing whitespace or obvious broken local structure.
- Out of scope:
  - Creating prompts for tickets 006, 007, or EXP-001 through EXP-005.
  - Defining exact custom architecture configurations or early-stopping thresholds.
  - Modifying code, running experiments, or changing research results and claims.
  - Editing the paper's LaTeX or bibliography.
  - Committing or pushing either repository.
- Open questions: None.
- Verification:
  - Run `git status --short --branch` and `git diff --check` in both repositories.
  - Inspect both README diffs and confirm the journal/implementation separation, checkbox order, blockers, and cross-links.
  - Confirm only the two approved READMEs and ticket records changed beyond the preserved pre-existing state.
  - Confirm the implementation repository's existing modified and untracked files remain untouched.

## Execution Prompt

Execute ticket `005-separate-paper-code-roadmaps` exactly as written in `agents/work/005-separate-paper-code-roadmaps/prompt.md`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved changes, verify every acceptance criterion, set the ticket status to `Done`, and create `completion.md` from `agents/templates/completion.md`.
