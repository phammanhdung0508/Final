"""KG-only attempt C (accessory links) on Musical_Instruments validation.

Plan: docs/kg-v2-enrichment-plan.md, "Attempt C". Per feedback condition:

- **Controls.** Popularity, KG-only on KG-v1 and attempt B at its selected mixture must
  reproduce the recorded NDCG@10 exactly.
- **C.** KG-v2 plus the accessory channel over `KG_C_GRID`, selected by validation
  NDCG@10, ties earliest. Its gain over B and over KG-v1 is a paired bootstrap
  (`kg_v2_validation.paired_bootstrap`).

Every `spot_every`-th eligible user is re-ranked with the per-user reference. The test
split is never read.

Usage: python -m scripts.kg_c_validation --data <dataset dir> --manifest-sha256 <sha> \\
    --facts <kg-v2 facts dir> --meta '<raw_meta glob>' --recorded-v1 <validation.json> \\
    --recorded-v2 <kg-v2 validation.json> --out <new file>
"""

import argparse
import glob
import json
import time
from pathlib import Path

from scripts.baseline_validation import _summary, select
from scripts.kg_accessory import accessory_tokens, load_roles
from scripts.kg_only_fast import KG_C_GRID, FastKGOnly, ReferenceKGv2, evaluate_kg_only, evaluate_popularity
from scripts.kg_v2_facts import load_facts
from scripts.kg_v2_validation import _spot_check_v2, paired_bootstrap, per_user_ndcg
from scripts.profiling_data import CONDITIONS, ProfilingData

SCHEMA = 'amazon4u-kg-c-validation-v1'


def run(data_dir, manifest_sha256, facts_dir, meta_glob, recorded_v1, recorded_v2, *, spot_every=200, log=print):
    data = ProfilingData(data_dir, manifest_sha256)
    brands_v2, attributes, facts_manifest = load_facts(facts_dir)
    v1 = json.loads(Path(recorded_v1).read_text())['conditions']
    v2 = json.loads(Path(recorded_v2).read_text())['conditions']
    roles = load_roles(sorted(glob.glob(meta_glob)), {p for _, p, *_ in data.train})
    accessory = accessory_tokens(roles)
    targets = data.validation_targets()
    cats = data.product_categories
    out = {'schema_version': SCHEMA, 'plan': 'docs/kg-v2-enrichment-plan.md#attempt-c',
           'category': data.manifest['category'], 'dataset_manifest_sha256': manifest_sha256,
           'facts_sha256': facts_manifest['facts_sha256'],
           'accessory_coverage': {'products': len(roles),
                                  'is_an_instrument': sum(bool(i) for i, _ in roles.values()),
                                  'serves_an_instrument': sum(bool(s) for _, s in roles.values())},
           'split': 'validation', 'test_execution': 'not_run_test_split_not_loaded', 'conditions': {}}

    def check(name, report, expected):
        if report['overall_metrics'] != expected:
            raise RuntimeError(f'{name} does not reproduce its recorded result')

    for condition in CONDITIONS:
        context = data.context(condition)
        group = {}
        popularity = evaluate_popularity(context, targets)
        check(f'{condition} Popularity', popularity, v1[condition]['Popularity']['overall_metrics'])
        group['Popularity'] = _summary(popularity, 0.0)
        popularity = None

        core = tuple(v1[condition]['KG-only']['selected_mixture'])
        kg_v1 = evaluate_kg_only(FastKGOnly(context, cats, data.product_brands), core, targets)
        check(f'{condition} KG-only v1', kg_v1, v1[condition]['KG-only']['mixtures'][str(list(core))]['overall_metrics'])
        group['KG-only (KG-v1)'] = {**_summary(kg_v1, 0.0), 'mixture': list(core)}
        kg_v1 = per_user_ndcg(kg_v1)

        fast = FastKGOnly(context, cats, brands_v2, attributes, accessory)
        b_mix = tuple(v2[condition]['B. brand fix + attributes']['selected_mixture'])
        b = evaluate_kg_only(fast, b_mix, targets)
        check(f'{condition} attempt B', b,
              v2[condition]['B. brand fix + attributes']['mixtures'][str(list(b_mix))]['overall_metrics'])
        group['B (KG-v2)'] = {**_summary(b, 0.0), 'mixture': list(b_mix)}
        b = per_user_ndcg(b)
        log(f'{condition} controls reproduced')

        evaluations = {}
        for mixture in KG_C_GRID:
            t0 = time.monotonic()
            report = evaluate_kg_only(fast, mixture, targets)
            checked = _spot_check_v2(context, report,
                                     ReferenceKGv2(context, cats, brands_v2, attributes, mixture, accessory), spot_every)
            evaluations[str(list(mixture))] = {**_summary(report, time.monotonic() - t0),
                                               'spot_checked_users': checked, 'ndcg': per_user_ndcg(report)}
            log(f'{condition} C {mixture} NDCG@10={report["overall_metrics"]["NDCG@10"]:.5f}')
        best = select(evaluations, KG_C_GRID)
        chosen = evaluations[str(list(best))]['ndcg']
        group['C. accessory links'] = {
            'mixtures': {k: {kk: vv for kk, vv in v.items() if kk != 'ndcg'} for k, v in evaluations.items()},
            'selected_mixture': best,
            'gain_over_B': paired_bootstrap(chosen, b),
            'gain_over_kg_v1': paired_bootstrap(chosen, kg_v1)}
        out['conditions'][condition] = group
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    for name in ('--data', '--manifest-sha256', '--facts', '--meta', '--recorded-v1', '--recorded-v2', '--out'):
        ap.add_argument(name, required=True)
    ap.add_argument('--spot-every', type=int, default=200)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; overwrite refused')
    result = run(args.data, args.manifest_sha256, args.facts, args.meta, args.recorded_v1, args.recorded_v2,
                 spot_every=args.spot_every)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
