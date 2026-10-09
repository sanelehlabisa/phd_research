# Task Prompt

- Ticket: `063-pris-2026-flash-presentation`
- Status: Done
- Aim: Prepare a concise, exactly three-slide PRIS 2026 flash presentation about the lightweight ConvLSTM abnormal-activity-recognition research.
- Scope: Paper 001 presentation materials currently split across `presentaion/` and `presentation/`.
- Changes:
  - Keep the generic example as a guide and use the current named PRIS deck and speaker script as the approved presentation draft; duplicate archival copies are not required.
  - After confirming each supplied file is safely retained in `presentation/`, remove the redundant misspelled `presentaion/` directory.
  - Produce an exactly three-slide editable PowerPoint named `presentation/SENG-F-12-HLABISA-SANELE.pptx`, retaining the official PRIS template branding and using the generic example only for guidance.
  - Shape the story around: (1) surveillance problem, literature gap and research question; (2) method and verified findings; (3) supported takeaway, limitations and next step/demo only if it exists.
  - Revise a concise speaker script in `presentation/script.txt` to match the final slides and fit a flash presentation.
  - Use only claims supported by the manuscript or verified run artifacts. Do not present draft claims such as “30+ experiments,” results across two datasets, strong performance, or a working Gradio demo unless the relevant evidence is confirmed. If final AAD results are unavailable, label the result area as pending rather than inventing values.
  - Add a brief presentation link/status to the Paper 001 README.
- Acceptance criteria:
  - The final deck contains exactly three slides, uses the named PRIS deck as its visual base, and is clearly readable as a short research story.
  - Every numerical result and completed-work claim on the slides and in the script is traceable to verified evidence; unresolved results are clearly marked pending or omitted.
  - The current deck, matching speaker script and generic guide are in `presentation/` beside `manuscript/`; the redundant `presentaion/` directory is gone.
  - The deck opens and renders without PowerPoint structural errors; the script agrees with the final slide content.
- Scope update: The user accepted the current three-slide deck and speaker notes as the presentation draft. Revisit them after AAD experiments to add a compact, verified results visual and useful model/example images.
- Remaining work: Post-results visual update only; do not imply the current results are final.
- Out of scope: Changing the manuscript, running new experiments, changing model/code, adding unsupported results, or deploying a demo.
- Open questions: `None`.
- Verification: Inspect the final slide count, extracted slide text and visual rendering; compare every factual/numeric claim with its cited manuscript section or run artifact; check that source files are unchanged and the speaker script follows the deck.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
