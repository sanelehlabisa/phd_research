# Completion

- Status: Done
- Summary: Aligned repository guidance around the AAD-first, two-stage experiment workflow.
- Changes:
  - Updated `AGENTS.md` and the root, Paper 001, and implementation READMEs to describe AAD as primary, VDD as optional via JSON, and historical VDD/Kinetics runs as non-comparable or exploratory.
  - Documented the custom-search and fixed-setting comparison workflow and clarified which notebook/profile paths are current versus legacy.
- Verification:
  - `git diff --check` — passed.
  - Reviewed documented active experiment commands/configs and verified referenced profile/config paths exist; `--list-plan` succeeded for local smoke (8 configurations), custom search (20 runs), and model comparison (12 runs).
  - Searched the four guides for contradictory active VDD/Kinetics instructions; remaining mentions identify optional, historical, or exploratory workflows.
  - Confirmed changes are limited to the approved documentation and ticket files; no code, manuscript results, performance claims, or historical run artifacts changed.
- Remaining issues: `None`.
