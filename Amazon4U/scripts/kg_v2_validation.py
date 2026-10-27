"""KG-v2 enrichment attempts A and B for KG-only, on Musical_Instruments validation.

Plan and rules: docs/kg-v2-enrichment-plan.md (agreed 2026-10-27). Per feedback condition:

- **Control.** Popularity and core KG-only on KG-v1 at the mixture core selection chose.
  Both must reproduce the recorded v2.1.0 validation numbers exactly.
- **A.** KG-v1 categories with KG-v2 brands, over the core 3 mixtures.
- **B.** A plus the KG-v2 attribute channel, over `KG_V2_GRID`.

Each attempt selects its mixture by validation NDCG@10, exact ties earliest. Its gain over
the KG-v1 control is a paired bootstrap over eligible validation users (1,000 resamples,
seed 12345, 95% percentile interval). Every `spot_every`-th eligible user is re-scored
with a per-user reference. The test split is never read.

Usage: python -m scripts.kg_v2_validation --data <dataset dir> --manifest-sha256 <sha> \\
    --facts <kg-v2 facts dir> --recorded <validation.json> --out <new file>
"""

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.baseline_validation import _spot_check, _summary, select
from scripts.kg_only_fast import KG_V2_GRID, FastKGOnly, ReferenceKGv2, evaluate_kg_only, evaluate_popularity, \
    reference_scorer
from scripts.kg_v2_facts import load_facts
from scripts.profiling_data import CONDITIONS, ProfilingData
from scripts.recommendation_core import KGOnlyScorer

SCHEMA = 'amazon4u-kg-v2-validation-v1'
RESAMPLES, BOOTSTRAP_SEED = 1000, 12345


def per_user_ndcg(report):
    return {r['user_id']: r['metrics']['NDCG@10'] for r in report['per_user_results']}


def paired_bootstrap(a, c, *, resamples=RESAMPLES, seed=BOOTSTRAP_SEED):
    """Mean per-user NDCG@10 difference, `a` minus `c`, with a 95% percentile interval."""
    if a.keys() != c.keys():
        raise ValueError('attempt and control must rank the same eligible users')
    users = sorted(a)
    diff = torch.tensor([a[u] - c[u] for u in users], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    means = []
    for start in range(0, resamples, 50):
        idx = torch.randint(len(users), (min(50, resamples - start), len(users)), generator=generator)
        means.append(diff[idx].mean(dim=1))
    means = torch.cat(means).sort().values
    return {'mean_difference': float(diff.mean()), 'ci95': [float(means[int(0.025 * resamples)]),
                                                             float(means[int(0.975 * resamples) - 1])],
            'users': len(users), 'resamples': resamples, 'seed': seed}


def _spot_check_v2(context, report, reference, spot_every):
    """Re-rank every `spot_every`-th eligible user with the per-user reference."""
    warm = context.sorted_warm_items
    sample = sorted(r['user_id'] for r in report['per_user_results'])[::spot_every]
    fast = {r['user_id']: r for r in report['per_user_results']}
    for u in sample:
        history = context.all_rated_history.get(u, set())
        candidates = [p for p in warm if p not in history]
        scores, route, reason = reference.score_candidates(u, candidates)
        ranked = sorted(candidates, key=lambda p: (-scores[p], p))
        if (fast[u]['route'], fast[u]['fallback_reason'], fast[u]['top_20']) != (route, reason, ranked[:20]):
            raise RuntimeError(f'fast and reference results differ for user {u}')
    return len(sample)


def _attempt(evaluations, grid, control):
    best = select(evaluations, grid)
    key = str(list(best))
    return {'mixtures': {k: {kk: vv for kk, vv in v.items() if kk != 'ndcg'} for k, v in evaluations.items()},
            'selected_mixture': best,
            'gain_over_kg_v1': paired_bootstrap(evaluations[key]['ndcg'], control)}


def run(data_dir, manifest_sha256, facts_dir, recorded, *, spot_every=200, log=print):
    data = ProfilingData(data_dir, manifest_sha256)
    brands_v2, attributes, facts_manifest = load_facts(facts_dir)
    recorded = json.loads(Path(recorded).read_text())
    targets = data.validation_targets()
    cats = data.product_categories
    out = {'schema_version': SCHEMA, 'plan': 'docs/kg-v2-enrichment-plan.md',
           'category': data.manifest['category'], 'dataset_manifest_sha256': manifest_sha256,
           'facts_sha256': facts_manifest['facts_sha256'], 'facts_counts': facts_manifest['counts'],
           'split': 'validation', 'test_execution': 'not_run_test_split_not_loaded', 'conditions': {}}
    for condition in CONDITIONS:
        context = data.context(condition)
        done = recorded['conditions'][condition]
        group = {}

        t0 = time.monotonic()
        popularity = evaluate_popularity(context, targets)
        if popularity['overall_metrics'] != done['Popularity']['overall_metrics']:
            raise RuntimeError(f'{condition}: Popularity does not reproduce the recorded result')
        group['Popularity'] = _summary(popularity, time.monotonic() - t0)

        core = tuple(done['KG-only']['selected_mixture'])
        t0 = time.monotonic()
        control = evaluate_kg_only(FastKGOnly(context, cats, data.product_brands), core, targets)
        if control['overall_metrics'] != done['KG-only']['mixtures'][str(list(core))]['overall_metrics']:
            raise RuntimeError(f'{condition}: core KG-only does not reproduce the recorded result')
        group['KG-only (KG-v1)'] = {**_summary(control, time.monotonic() - t0), 'mixture': list(core)}
        log(f'{condition} control: Popularity {popularity["overall_metrics"]["NDCG@10"]:.5f}, '
            f'KG-only v1 {control["overall_metrics"]["NDCG@10"]:.5f}')
        control, popularity = per_user_ndcg(control), None

        fast_a = FastKGOnly(context, cats, brands_v2)
        evaluations = {}
        for mixture in KGOnlyScorer.GRID:
            t0 = time.monotonic()
            report = evaluate_kg_only(fast_a, mixture, targets)
            checked = _spot_check(context, report, targets,
                                  reference_scorer(context, cats, brands_v2, mixture), spot_every)
            evaluations[str(list(mixture))] = {**_summary(report, time.monotonic() - t0),
                                               'spot_checked_users': checked, 'ndcg': per_user_ndcg(report)}
            log(f'{condition} A {mixture} NDCG@10={report["overall_metrics"]["NDCG@10"]:.5f}')
        group['A. brand fix'] = _attempt(evaluations, KGOnlyScorer.GRID, control)

        fast_b = FastKGOnly(context, cats, brands_v2, attributes)
        evaluations = {}
        for mixture in KG_V2_GRID:
            t0 = time.monotonic()
            report = evaluate_kg_only(fast_b, mixture, targets)
            checked = _spot_check_v2(context, report,
                                     ReferenceKGv2(context, cats, brands_v2, attributes, mixture), spot_every)
            evaluations[str(list(mixture))] = {**_summary(report, time.monotonic() - t0),
                                               'spot_checked_users': checked, 'ndcg': per_user_ndcg(report)}
            log(f'{condition} B {tuple(round(w, 3) for w in mixture)} NDCG@10={report["overall_metrics"]["NDCG@10"]:.5f}')
        group['B. brand fix + attributes'] = _attempt(evaluations, KG_V2_GRID, control)
        out['conditions'][condition] = group
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    for name in ('--data', '--manifest-sha256', '--facts', '--recorded', '--out'):
        ap.add_argument(name, required=True)
    ap.add_argument('--spot-every', type=int, default=200)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; overwrite refused')
    result = run(args.data, args.manifest_sha256, args.facts, args.recorded, spot_every=args.spot_every)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
