#!/usr/bin/env python3
"""Print a parent-session kickoff prompt for Antigravity autonomous research loops."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_prompt(campaign: str, max_iterations: int) -> str:
    return f"""You are the Antigravity ML research coordinator running in {ROOT}.

Read first:
- GEMINI.md
- AGENTS.md
- .gemini/antigravity-cli/AGENTS.md
- research/notes.md
- research/do-not-repeat.md
- research/results.tsv

Your goal is to optimize the HeteroGraphSAGE recommendation model for campaign: "{campaign}".

Autonomous Loop Instructions:
1. Review research/results.tsv and establish the current best eval_score (NDCG@10).
2. Formulate 1 hypothesis at a time (e.g. BPR loss, negative sampling, learning rate, regularizations).
3. Implement the change in train.py (keep prepare.py and evaluate.py read-only).
4. Run:
   uv run scripts/kaggle_job.py launch --mode experiment
5. Follow the logs:
   uv run scripts/kaggle_job.py logs <kernel_slug>
6. Download outputs:
   uv run scripts/kaggle_job.py output <kernel_slug>
7. Record the result in research/results.tsv and summarize in research/notes.md.
8. Loop up to {max_iterations} iterations without stopping prematurely.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Print an Antigravity kickoff prompt.")
    parser.add_argument("--campaign", default="hetero-graphsage-ranking-optimization", help="Research campaign name")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum research iterations")
    args = parser.parse_args()

    print(build_prompt(args.campaign, args.max_iterations))


if __name__ == "__main__":
    main()
