# Task Prompt

- Ticket: `022-interactive-notebook-foundation`
- Status: Done
- Aim: Turn the Paper 001 notebook into a clear Colab-only VS Code walkthrough
  that verifies the runtime and visually inspects AAD before any experiment runs.
- Scope:
  - `papers/001-journal-abnormal-activity-recognition/implementation/notebooks/aad_experiment_workflow.ipynb`
  - the shared video reader in `implementation/src/utils.py` only where required
    for the Colab torchvision version used by the approved notebook stage
  - concise notebook guidance in the root, Paper 001, and implementation READMEs
  - focused notebook validation tests only when useful
- Changes:
  - Treat the current uncommitted notebook and implementation README as
    user-approved work in progress. Preserve useful parts and do not discard or
    commit them while executing this ticket.
  - Target only the official Google Colab VS Code extension. Use the remote
    Colab filesystem and `/content/phd_research`; remove local-Jupyter, browser-
    Colab, Kaggle, Drive, upload, archive, and multi-platform branches.
  - Keep one short setup sequence that:
    1. displays `nvidia-smi`, CUDA/PyTorch state, and the connected GPU;
    2. requires the requested A100 runtime and fails with a plain message when
       it is unavailable;
    3. reuses the existing `/content/phd_research` checkout, cloning only when
       absent and pulling only when the remote checkout is clean;
    4. installs only missing non-PyTorch dependencies with the active kernel's
       Python; and
    5. reuses an already cached AAD directory, or downloads the public dataset
       once with KaggleHub when the current Colab runtime has no cached copy.
  - Use normal Python notebook cells. Do not use notebook shell magic or wrap
    basic inspection in Linux commands when Python provides a clearer result.
  - Import and reuse `AHARDataset`, `VideoAugmentation`, seeding, and related
    helpers from `src.dataset`, following the working examples in its `main()`.
    Do not duplicate video loading, frame sampling, or augmentation logic.
  - Preserve the shared video-reader API but use PyAV directly when torchvision
    does not provide its removed `io.read_video` function.
  - Add a concise dataset overview showing dataset path, sample count, class
    names, class counts, tensor shape, value range, and the committed split
    counts without opening the locked test samples for model selection.
  - Display a small reproducible sample table with source name and label, a
    class-distribution chart, several temporal frames from one clip, and a
    clean-versus-online-augmented comparison. Include an inline playable video
    only when it remains lightweight and does not write a research artifact.
  - Keep cells short, ordered top-to-bottom, rerunnable, and explicit about
    what is inspection versus experimental evidence. Clear all saved outputs.
  - End with a short handoff saying model inspection is ticket 023 and no
    experiment has run yet.
- Acceptance criteria:
  - From a VS Code notebook connected to an A100 Colab runtime, the setup and
    dataset sections run top-to-bottom without browser-only Colab APIs.
  - An existing repository and dataset are reused without redundant clone,
    pull, download, or dependency installation; a missing dataset is downloaded
    once into KaggleHub's normal cache and then reused.
  - The notebook visibly presents dataset metadata, class balance, real frames,
    and a clean/augmented example using the repository's dataset code.
  - A real AAD clip decodes to a `(T, H, W, C)` uint8 tensor with a positive
    frame rate on torchvision versions that omit `io.read_video`.
  - No training, experiment stage, checkpoint evaluation, test-set access, or
    result claim occurs in this ticket.
  - The notebook remains valid nbformat 4 with unique cell IDs, null execution
    counts, empty outputs, and Python cells that compile.
- Out of scope:
  - Kaggle notebooks, browser-only Colab, local Jupyter, Google Drive, repeated
    or custom dataset copying, model inspection, training, experiments, final
    evaluation, and manuscript changes.
  - Refactoring unrelated dataset, writer, preprocessing, or experiment logic.
- Open questions: None. The approved target is the Colab-only VS Code extension;
  reuse cached AAD first and call the public KaggleHub download only when absent.
- Verification:
  - Parse the notebook as JSON and verify nbformat, unique IDs, empty outputs,
    null execution counts, and compilation of every Python cell.
  - Statically verify the absence of Drive, upload, and browser-Colab paths and
    verify that the KaggleHub download is guarded by cache detection.
  - Exercise reusable dataset-display logic locally against a minimal sample or
    the existing dataset without starting training.
  - Decode one real AAD clip through the shared reader and verify shape, dtype,
    frame count, and frame rate without using `torchvision.io.read_video`.
  - Run the focused test suite, `git diff --check`, and a scope/status review.

## Execution Prompt

Execute ticket `022-interactive-notebook-foundation` exactly as written in
`agents/work/022-interactive-notebook-foundation/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
