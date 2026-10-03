# Agent Notes

Scope: the entire Movie4U project.

Keep the required KG-only recommender, KG + GNN, API, and mobile demo modular. Prefer small, tested changes.

From this directory, run `.venv/bin/python -m pytest -q` and `.venv/bin/python -m ruff check src scripts tests` for Python changes. Run `npm run typecheck` from `mobile/` for mobile changes.

Never modify raw MovieLens files or expose held-out rating edges to training. Check `docs/evaluation.md` before changing splits or metrics. Generated dependencies, caches, data, and build folders do not need their own AGENTS.md.
