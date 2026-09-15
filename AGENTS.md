# AI Working Guide

## Repository

- This repository contains PhD papers and their supporting research code for
  video processing and unusual human-activity classification.
- Each paper lives under `papers/NNN-short-title/` with a concise `README.md`, a
  `manuscript/` directory, and an optional `implementation/` directory.
- Keep LaTeX readable and consistently formatted. Keep research code tested,
  reproducible, and scoped to its paper.
- Use the commands documented in the relevant paper README to build manuscripts
  and verify implementations.

## Workflow

1. Read the root README, the relevant paper README and files, and the guidance
   in `agents/`, then check `git status` before setting up the task.
2. If there are pre-existing uncommitted changes, stop and ask the user how to
   handle them. Never discard or overwrite them silently.
3. Discuss substantial work with the user and ask short, focused questions until
   the aim, scope, inputs, decisions, exclusions, acceptance criteria, and
   verification are explicit. Do not silently assume missing requirements.
4. Do not create a work directory or `prompt.md` while any required question is
   unanswered.
5. Once all questions are answered, create the next
   `agents/work/NNN-short-title/` directory and use
   `agents/templates/prompt.md` to record the answers in a `Ready` prompt.
6. Return the completed prompt and its execution prompt to the user.
7. Do not begin substantial edits until the user pastes the execution prompt.
   After that approval, set the prompt status to `Approved`, then `In progress`.
8. Make only the approved changes and verify every acceptance criterion.
9. Set the prompt status to `Done` and create `completion.md` from
   `agents/templates/completion.md`.

Small typo fixes or explanations do not require a ticket unless the user asks for one.

## Rules

- Do not invent research results, citations, datasets, or claims.
- Preserve the author's meaning and academic voice.
- Ask before changing the paper structure, research claims, or bibliography.
- Ask rather than guess when a missing decision could affect the work.
- Keep generated LaTeX files out of source changes unless explicitly requested.
- Never delete user content without explicit approval.
- Keep the root and relevant paper READMEs concise and current whenever
  structure, build commands, implementation status, dependencies, or the next
  ticket changes. Do not duplicate details owned by a lower-level README.
- Follow `agents/rules.md` and `agents/config.md`.
