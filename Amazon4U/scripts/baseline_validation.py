"""Musical_Instruments validation for the Popularity and KG-only baselines (protocol v2.1.0).

For each feedback condition, scores Popularity and the KG-only mixtures on the full
validation split with `scripts.kg_only_fast`. Core KG-only selects among its three frozen
mixtures; the labeled `KG-only (extended brand grid)` variant selects among those plus
0.25/0.75 and 0/1 (docs/kg-only-extended-brand-amendment.md). Both use the best
validation NDCG@10, and exact ties keep the earliest grid entry. Every `spot_every`-th
eligible user is re-scored with the accepted per-user `Evaluator`, and the run fails
unless their results are identical. The test split is never read.

Usage: python -m scripts.baseline_validation --data <dataset dir> --manifest-sha256 <sha> --out <new file>
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from scripts.kg_only_fast import EXTENDED_GRID, FastKGOnly, evaluate_kg_only, evaluate_popularity, reference_scorer
from scripts.profiling_data import CONDITIONS, ProfilingData
from scripts.recommendation_core import Evaluator, KGOnlyScorer, PopularityScorer

SCHEMA = 'amazon4u-baseline-validation-v2'


def _summary(report, seconds):
    return {k: report[k] for k in ('audit', 'candidate_shortages', 'overall_metrics')} | {
        'subgroups': {route: {'count': g['count'], 'metrics': g['metrics']} for route, g in report['subgroups'].items()},
        'fallback_by_reason': {r: g['count'] for r, g in report['subgroups']['fallback']['by_reason'].items()},
        'seconds': round(seconds, 1)}


def _spot_check(context, report, targets, scorer, spot_every):
    eligible = {r['user_id'] for r in report['per_user_results']}
    sample = sorted(eligible)[::spot_every]
    chosen = set(sample)
    reference = Evaluator(context).evaluate([t for t in targets if t[0] in chosen], scorer)
    fast = {r['user_id']: r for r in report['per_user_results'] if r['user_id'] in chosen}
    for r in reference['per_user_results']:
        if {k: fast[r['user_id']][k] for k in r} != r:
            raise RuntimeError(f'fast and reference results differ for user {r["user_id"]}')
    return len(sample)


def select(results, grid=KGOnlyScorer.GRID):
    """Best validation NDCG@10 in grid order; an exact tie keeps the earlier mixture."""
    best = None
    for mixture in grid:
        score = results[str(list(mixture))]['overall_metrics']['NDCG@10']
        if best is None or score > best[1]:
            best = (mixture, score)
    return list(best[0])


def run(data_dir, manifest_sha256, *, spot_every=200, log=print):
    data = ProfilingData(data_dir, manifest_sha256)
    targets = data.validation_targets()
    out = {'schema_version': SCHEMA, 'protocol_version': 'amazon4u-protocol-v2.1.0',
           'category': data.manifest['category'], 'dataset_manifest_sha256': manifest_sha256,
           'split': 'validation', 'test_execution': 'not_run_test_split_not_loaded',
           'environment': {'python': platform.python_version(), 'torch': torch.__version__},
           'conditions': {}}
    for condition in CONDITIONS:
        context = data.context(condition)
        t0 = time.monotonic()
        popularity = evaluate_popularity(context, targets)
        checked = _spot_check(context, popularity, targets, PopularityScorer(context), spot_every)
        group = {'Popularity': {**_summary(popularity, time.monotonic() - t0), 'spot_checked_users': checked}}
        log(f'{condition} Popularity NDCG@10={popularity["overall_metrics"]["NDCG@10"]:.5f}')
        fast = FastKGOnly(context, data.product_categories, data.product_brands)
        kg = {}
        for mixture in EXTENDED_GRID:
            t0 = time.monotonic()
            report = evaluate_kg_only(fast, mixture, targets)
            scorer = reference_scorer(context, data.product_categories, data.product_brands, mixture)
            checked = _spot_check(context, report, targets, scorer, spot_every)
            kg[str(list(mixture))] = {**_summary(report, time.monotonic() - t0), 'spot_checked_users': checked}
            log(f'{condition} KG-only {mixture} NDCG@10={report["overall_metrics"]["NDCG@10"]:.5f}')
        rule = 'best validation NDCG@10, exact ties keep the earliest grid entry'
        group['KG-only'] = {'mixtures': {str(list(m)): kg[str(list(m))] for m in KGOnlyScorer.GRID},
                            'selected_mixture': select(kg), 'selection_rule': rule}
        group['KG-only (extended brand grid)'] = {
            'mixtures': kg, 'selected_mixture': select(kg, EXTENDED_GRID), 'selection_rule': rule,
            'label': 'post-validation labeled variant; see docs/kg-only-extended-brand-amendment.md'}
        out['conditions'][condition] = group
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--spot-every', type=int, default=200)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; overwrite refused')
    result = run(args.data, args.manifest_sha256, spot_every=args.spot_every)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
