# Task Prompt

- Ticket: `004-starter-journal-experiment-roadmap`
- Status: Done
- Aim: Replace the rough starter-journal plan with concise, linked roadmaps for improving the paper and its separate implementation repository.
- Scope:
  - `starter_journal/plan.md`
  - `starter_journal/README.md`
  - `/home/it-software/Desktop/projects/2026/convlstm_abnormal_human_activity_recognition/README.md`
- Changes:
  - Replace `starter_journal/plan.md` with a concise `starter_journal/README.md`; remove `plan.md` as part of the approved rename without losing its useful planning decisions.
  - In the journal README, link to `https://github.com/sanelehlabisa/convlstm_abnormal_human_activity_recognition` and explain that final paper results must identify the exact implementation commit used.
  - Record `Proposed Lightweight ConvLSTM (64-32-16-64)` as the provisional leading candidate from earlier runs, not as a confirmed optimum.
  - Summarize the current paper state and the main validity issues: incomplete reproducibility, possible split leakage, inconsistent split descriptions, final-epoch testing instead of best-validation checkpoint testing, questionable accumulated metrics, unequal augmentation budgets, incomplete provenance, and claims that require controlled evidence.
  - Record the agreed experimental protocol:
    - Keep one fixed, stratified, group-aware 70:15:15 train/validation/test split; do not experiment with split ratios.
    - Keep related camera views or source-video groups in the same split.
    - Use validation data for architecture and training decisions, and keep the test data locked until the final declared comparisons.
    - Use two fixed training seeds, `42` and `2026`, for confirmation runs.
    - Compare one factor at a time under the same data split, seeds, training budget, and evaluation procedure.
    - Use a fixed maximum epoch count with validation-loss early stopping and restore the best validation-loss checkpoint; leave the precise patience and minimum-delta values for the implementation ticket.
  - Record the staged experiment programme:
    1. Repair the experiment runner before new scientific runs.
    2. Define a bounded registry of custom ConvLSTM width and depth variants in the implementation ticket; do not hard-code or invent the exact registry in this documentation task.
    3. Screen custom candidates on AAD using validation results, alongside the original ConvLSTM and the three existing 3D-CNN baselines.
    4. Select the top five custom models by the predeclared validation rule.
    5. Confirm and report nine models on both datasets: five custom models, three 3D-CNN baselines, and the original ConvLSTM.
    6. Apply focused training and input ablations only to the selected proposed model so the full architecture search does not become a factorial grid.
  - Record the focused ablations for the selected model:
    - augmentation enabled versus disabled with an equal training budget;
    - three input settings: 16 frames at 32x32, 32 frames at 32x32, and 16 frames at 64x64;
    - weight decay `0` versus `0.0001`;
    - Adam versus AdamW as a lower-priority comparison, while documenting and controlling the initial learning rate because a scheduler does not replace it;
    - VDD training from scratch, frozen-feature transfer, and full-network fine-tuning from AAD.
  - Add a short sequential dependency list for implementation work: repair runner, define architectures, screen models, run focused ablations and confirmations, then export versioned results.
  - Add a short sequential dependency list for paper work: update Experimental Results and Discussion only after verified results, then update the abstract, contributions, conclusion, and other affected claims.
  - Add a concise section to the implementation README covering its current state, known problems, planned implementation tasks, and a link to `https://github.com/sanelehlabisa/phd_work/tree/master/starter_journal`.
  - Preserve all existing uncommitted implementation changes. Modify only its `README.md`.
- Acceptance criteria:
  - `starter_journal/README.md` exists and `starter_journal/plan.md` no longer exists.
  - Both READMEs link the paper and implementation repositories clearly.
  - The journal README distinguishes provisional historical results from the future controlled results.
  - The protocol, staged model selection, nine-model final comparison, focused ablations, and task dependencies are stated concisely and consistently.
  - The implementation README clearly separates current capabilities, known problems, and sequential work.
  - No LaTeX, bibliography, source code, result, model metadata, or generated curve file is modified.
  - The pre-existing changes to `models/meta.json`, `src/experiments.py`, and the two untracked curve images in the implementation repository remain untouched.
  - Markdown has no trailing whitespace or obvious broken local structure.
- Out of scope:
  - Defining the exact custom width/depth architecture registry.
  - Modifying experiment or model code.
  - Running or fabricating experiments, metrics, figures, or research claims.
  - Editing the paper's LaTeX or bibliography, including changing its stated split at this stage.
  - Committing or pushing either repository.
- Open questions: None.
- Verification:
  - Run `git status --short --branch` in both repositories and confirm only the approved documentation changes were added to the pre-existing state.
  - Run `git diff --check` in both repositories.
  - Confirm `starter_journal/README.md` exists and `starter_journal/plan.md` does not.
  - Inspect both README diffs and verify every acceptance criterion manually.
  - Confirm the existing implementation diffs for `models/meta.json` and `src/experiments.py`, plus both untracked curve images, are unchanged.

## Execution Prompt

Execute ticket `004-starter-journal-experiment-roadmap` exactly as written in `agents/work/004-starter-journal-experiment-roadmap/prompt.md`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved changes, verify every acceptance criterion, set the ticket status to `Done`, and create `completion.md` from `agents/templates/completion.md`.
