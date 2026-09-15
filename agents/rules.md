# AI Rules

- Inspect relevant files before defining the work.
- Check `git status` before task setup. If pre-existing uncommitted changes exist,
  stop and ask how to handle them; never discard them silently.
- Ask focused questions until the aim, scope, inputs, decisions, exclusions,
  acceptance criteria, and verification are explicit. Do not silently assume
  missing requirements.
- Do not create a work directory or prompt while a required question is unanswered.
- After all questions are answered, create one folder per substantial task:
  `agents/work/NNN-short-title/`, and generate a `Ready` `prompt.md` from
  `agents/templates/prompt.md` using those answers.
- Return the completed prompt to the user. Do not execute substantial work until
  the user pastes its execution prompt.
- Keep the task's `prompt.md` status current.
- On completion, create `completion.md` from `agents/templates/completion.md`,
  describing changes and verification.
- Never fabricate citations, evidence, results, or claims.
- Preserve unrelated user changes.
- Keep prompts, records, and responses concise.
- Keep the root and relevant paper READMEs concise and current when structure,
  build commands, implementation status, dependencies, or the next ticket
  changes. Link to lower-level details instead of duplicating them.
