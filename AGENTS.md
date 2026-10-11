# Workspace agent instructions

This workspace contains two independent projects:

- `Amazon4U/`: Amazon product recommendations and KG/KG + GNN experiments.
- `Movie4U/`: MovieLens movie recommendations and its application/demo.

## Select the project before editing

1. Identify the project from the user's task, named paths and declared session
   scope. If the user declares a project for the session, keep that scope until
   they explicitly change it. This does not authorize unrelated project changes.
2. If the target project is unclear, ask before creating, editing or deleting files.
3. Read and follow the selected project's local `AGENTS.md` and any applicable
   deeper instructions before making changes.

## Keep edits in the suitable project

- For Amazon4U tasks, edit only files inside `Amazon4U/`.
- For Movie4U tasks, edit only files inside `Movie4U/`.
- Place code, scripts, tests, configuration, documentation and generated artifacts
  in logically appropriate locations within the selected project.
- Do not use the other project as a temporary destination for blocked work, notes,
  handoffs or generated files. Explain the restriction or ask for clarification.
- Do not transfer dataset assumptions, evaluation protocols or configurations
  between projects without checking the selected project's documented decisions.
- Reading sibling-project files for explicitly relevant reference does not grant
  permission to modify them.

## Exceptions and shared files

- Changes to root/shared files (including this file, root `.gitignore` and workspace
  configuration) require an explicit user request or approval covering those files.
- Cross-project changes require explicit approval identifying the affected projects.
  Treat approval as specific to that task, not a permanent expansion of scope.
- Preserve unrelated user changes. Do not stage, commit or push unless requested;
  when requested, respect the specified file/project or staged-change scope.

Before finishing, check changed paths for accidental scope violations and report
which project was changed. Use the selected project's documented test commands.
