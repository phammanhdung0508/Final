"""Batched KG-only and Popularity validation reports equal the accepted reference evaluator."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from test_profiling_data import _csv, _kg_root
from test_profiling_runner import _synthetic
from scripts import baseline_validation, kg_v2_facts, kg_v2_validation
from scripts.profiling_data import MANIFEST_NAME, _sha256, stage_dataset

from scripts.kg_only_fast import (EXTENDED_GRID, KG_V2_GRID, FastKGOnly, ReferenceKGv2, evaluate_kg_only,
                                  evaluate_popularity, reference_scorer)
from scripts.recommendation_core import Evaluator, KGOnlyScorer, PopularityScorer
from scripts.recommendation_evaluation import TrainingContext

CATEGORIES = [['A'], ['A', 'A>g'], ['A', 'A>g', 'A>g>e'], ['A', 'A>g', 'A>g>a'], ['A', 'A>s'],
              ['A', 'A>s', 'A>s>p'], ['B'], ['B', 'B>d']]
BRANDS = ['Fender', 'Boss', 'Ernie', 'Snark']


def world(seed, n_users=60, n_items=40):
    """Training ratings, validation targets and metadata with gaps, twins and fallback users."""
    rng = random.Random(seed)
    cats, brands = {}, {}
    for i in range(n_items):
        p = f'p{i:02d}'
        if i % 7 != 3:
            cats[p] = list(rng.choice(CATEGORIES))
        if i % 5 != 1:
            brands[p] = rng.choice(BRANDS)
    # Identical metadata, so equal scores: the tie rule decides their order.
    cats['p10'], brands['p10'] = list(cats.get('p11', ['A'])), brands.get('p11')
    if brands['p10'] is None:
        del brands['p10']
    train, targets = [], []
    for u in range(n_users):
        user = f'u{u:02d}'
        items = rng.sample(sorted(cats.keys() | brands.keys() | {f'p{i:02d}' for i in range(n_items)}),
                           rng.randint(2, 8))
        low = u % 9 == 0  # every rating below 4: no positive history in the >=4 condition
        no_meta = u % 11 == 5  # only metadata-free items: unusable profile
        if no_meta:
            items = ['p03', 'p17'] + items[:1]
        for p in items[:-1]:
            train.append((user, p, float(rng.choice([1, 2, 3]) if low else rng.choice([2, 4, 5, 5]))))
        targets.append((user, items[-1], float(rng.choice([2, 4, 5]))))
    targets.append(('stranger', 'p00', 5.0))
    return train, targets, cats, brands


def strip_users(report):
    return {k: v for k, v in report.items() if k != 'per_user_results'}, report['per_user_results']


class FastBaselineTests(unittest.TestCase):
    def assertSameReport(self, fast, reference):
        f_summary, f_users = strip_users(fast)
        r_summary, r_users = strip_users(reference)
        self.assertEqual(f_summary, r_summary)
        self.assertEqual(len(f_users), len(r_users))
        for a, b in zip(f_users, r_users):
            self.assertEqual({k: a[k] for k in b}, b, a['user_id'])

    def test_kg_only_reports_equal_the_reference_for_every_mixture_and_condition(self):
        for seed in (1, 2, 3):
            train, targets, cats, brands = world(seed)
            for condition in ('all_ratings', 'rating_ge_4'):
                context = TrainingContext(train, condition=condition)
                fast_scorer = FastKGOnly(context, cats, brands)
                for mixture in EXTENDED_GRID:
                    with self.subTest(seed=seed, condition=condition, mixture=mixture):
                        reference = Evaluator(context).evaluate(
                            targets, reference_scorer(context, cats, brands, mixture))
                        fast = evaluate_kg_only(fast_scorer, mixture, targets, chunk_users=7)
                        self.assertSameReport(fast, reference)

    def test_the_fixture_reaches_every_route(self):
        train, targets, cats, brands = world(1)
        context = TrainingContext(train, condition='rating_ge_4')
        report = evaluate_kg_only(FastKGOnly(context, cats, brands), (1.0, 0.0), targets)
        reasons = {r['fallback_reason'] for r in report['per_user_results']}
        self.assertEqual(reasons, {None, 'empty_positive_history', 'unusable_metadata_profile'})
        self.assertGreater(report['audit']['excluded_targets'], 0)

    def test_popularity_reports_equal_the_reference(self):
        for seed in (1, 2):
            train, targets, _, _ = world(seed)
            for condition in ('all_ratings', 'rating_ge_4'):
                with self.subTest(seed=seed, condition=condition):
                    context = TrainingContext(train, condition=condition)
                    self.assertSameReport(evaluate_popularity(context, targets, chunk_users=5),
                                          Evaluator(context).evaluate(targets, PopularityScorer(context)))

    def test_every_score_is_bit_identical_on_deep_category_paths(self):
        rng = random.Random(7)
        cats, brands, train = {}, {}, []
        for i in range(200):
            path = ['R']
            for _ in range(rng.randint(1, 6)):
                path.append(f'{path[-1]}>{rng.randint(0, 3)}')
            cats[f'p{i:03d}'] = path
            if i % 3:
                brands[f'p{i:03d}'] = rng.choice(BRANDS)
        for u in range(80):
            train += [(f'u{u:02d}', p, 5.0) for p in rng.sample(sorted(cats), rng.choice([2, 3, 4, 12]))]
        context = TrainingContext(train)
        fast = FastKGOnly(context, cats, brands)
        users = sorted(context.users)
        for mixture in EXTENDED_GRID:
            reference = reference_scorer(context, cats, brands, mixture)
            score_users, route_of = fast.ranker(mixture)
            for u, row in zip(users, score_users(users).tolist()):
                expected, route, _ = reference.score_candidates(u, list(fast.items))
                self.assertEqual(route_of(u)[0], route, (mixture, u))
                if route == 'personalized':
                    self.assertEqual(row, [expected[p] for p in fast.items], (mixture, u))

    def test_the_variant_reference_equals_the_core_scorer_on_the_core_grid(self):
        train, targets, cats, brands = world(3)
        context = TrainingContext(train)
        for mixture in KGOnlyScorer.GRID:
            core = Evaluator(context).evaluate(targets, KGOnlyScorer(context, cats, brands, mixture=mixture))
            self.assertEqual(Evaluator(context).evaluate(targets, reference_scorer(context, cats, brands, mixture)), core)
        with self.assertRaises(ValueError):
            reference_scorer(context, cats, brands, (0.1, 0.9))

    def test_a_changed_formula_is_caught(self):
        train, targets, cats, brands = world(2)
        context = TrainingContext(train)
        reference = Evaluator(context).evaluate(targets, KGOnlyScorer(context, cats, brands, mixture=(0.5, 0.5)))
        wrong = evaluate_kg_only(FastKGOnly(context, cats, brands), (0.75, 0.25), targets)
        self.assertNotEqual(strip_users(wrong)[1], strip_users(reference)[1])


def attributes_of(seed, items):
    rng = random.Random(seed)
    values = ['color=black', 'color=red', 'strings=6', 'strings=4', 'power=battery', 'material=wood']
    return {p: rng.sample(values, rng.randint(1, 3)) for p in items if rng.random() < 0.7}


class AttributeChannelTests(unittest.TestCase):
    def test_a_hand_calculated_attribute_score(self):
        # p1 {black}, p2 {black, 6}, p3 {6}: both attributes have df 2 of 3 warm products, so
        # equal IDF; p2's normalized vector is 1/sqrt(2) on each. u rated p1, so u = {black: 1}.
        train = [('u', 'p1', 5.0), ('v', 'p2', 5.0), ('v', 'p3', 5.0)]
        attrs = {'p1': ['color=black'], 'p2': ['color=black', 'strings=6'], 'p3': ['strings=6']}
        context = TrainingContext(train)
        fast = FastKGOnly(context, {}, {}, attrs)
        score_users, route_of = fast.ranker((0.4, 0.4, 0.2))
        row = dict(zip(fast.items, score_users(['u'])[0].tolist()))
        self.assertAlmostEqual(row['p2'], 0.2 * 0.7071067811865476, places=15)
        self.assertEqual(row['p3'], 0.0)
        self.assertEqual(route_of('u'), ('personalized', None))
        self.assertEqual(fast.ranker((0.5, 0.5, 0.0))[1]('u'), ('fallback', 'unusable_metadata_profile'))

    def test_scores_and_routes_equal_the_per_user_reference(self):
        for seed in (1, 2, 3):
            train, _, cats, brands = world(seed)
            attrs = attributes_of(seed, {p for _, p, _ in train})
            for condition in ('all_ratings', 'rating_ge_4'):
                context = TrainingContext(train, condition=condition)
                fast = FastKGOnly(context, cats, brands, attrs)
                users = sorted(context.users)
                for mixture in KG_V2_GRID:
                    reference = ReferenceKGv2(context, cats, brands, attrs, mixture)
                    score_users, route_of = fast.ranker(mixture)
                    for u, row in zip(users, score_users(users).tolist()):
                        expected, route, reason = reference.score_candidates(u, list(fast.items))
                        with self.subTest(seed=seed, condition=condition, mixture=mixture, user=u):
                            if route == 'personalized':
                                self.assertEqual(route_of(u), (route, reason))
                                self.assertEqual(row, [expected[p] for p in fast.items])
                            elif reason == 'unusable_metadata_profile':
                                self.assertEqual(route_of(u), (route, reason))

    def test_zero_attribute_weight_reproduces_the_two_channel_report(self):
        train, targets, cats, brands = world(2)
        context = TrainingContext(train)
        fast = FastKGOnly(context, cats, brands, attributes_of(2, {p for _, p, _ in train}))
        self.assertEqual(evaluate_kg_only(fast, (0.5, 0.5, 0.0), targets),
                         evaluate_kg_only(FastKGOnly(context, cats, brands), (0.5, 0.5), targets))

    def test_three_channel_mixtures_need_attributes(self):
        train, _, cats, brands = world(1)
        with self.assertRaises(ValueError):
            FastKGOnly(TrainingContext(train), cats, brands).ranker((0.4, 0.4, 0.2))


class BaselineValidationRunTests(unittest.TestCase):
    def test_the_run_reports_both_conditions_and_selects_by_the_frozen_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train, valid = _synthetic(seed=5)
            splits = tmp / 'splits'
            splits.mkdir()
            _csv(splits / 'Musical_Instruments.train.csv', train)
            _csv(splits / 'Musical_Instruments.valid.csv', valid)
            stage_dataset(_kg_root(tmp, train=train), splits, tmp / 'data')
            result = baseline_validation.run(tmp / 'data', _sha256(tmp / 'data' / MANIFEST_NAME),
                                             spot_every=1, log=lambda _: None)
        self.assertEqual(result['test_execution'], 'not_run_test_split_not_loaded')
        for group in result['conditions'].values():
            kg = group['KG-only']
            self.assertEqual(kg['selected_mixture'], baseline_validation.select(kg['mixtures']))
            self.assertEqual(len(kg['mixtures']), 3)
            variant = group['KG-only (extended brand grid)']
            self.assertEqual(len(variant['mixtures']), 5)
            self.assertEqual(variant['selected_mixture'],
                             baseline_validation.select(variant['mixtures'], EXTENDED_GRID))
            self.assertEqual(group['Popularity']['spot_checked_users'], group['Popularity']['audit']['eligible_users'])

    def test_selection_keeps_the_earliest_mixture_on_an_exact_tie(self):
        scores = {'[1.0, 0.0]': 0.2, '[0.75, 0.25]': 0.3, '[0.5, 0.5]': 0.3}
        self.assertEqual(baseline_validation.select(
            {k: {'overall_metrics': {'NDCG@10': v}} for k, v in scores.items()}), [0.75, 0.25])
        scores |= {'[0.25, 0.75]': 0.4, '[0.0, 1.0]': 0.4}
        self.assertEqual(baseline_validation.select(
            {k: {'overall_metrics': {'NDCG@10': v}} for k, v in scores.items()}, EXTENDED_GRID), [0.25, 0.75])


class KgV2ValidationRunTests(unittest.TestCase):
    def test_the_run_reproduces_the_controls_and_reports_both_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train, valid = _synthetic(seed=5)
            splits = tmp / 'splits'
            splits.mkdir()
            _csv(splits / 'Musical_Instruments.train.csv', train)
            _csv(splits / 'Musical_Instruments.valid.csv', valid)
            stage_dataset(_kg_root(tmp, train=train), splits, tmp / 'data')
            sha = _sha256(tmp / 'data' / MANIFEST_NAME)
            recorded = tmp / 'recorded.json'
            recorded.write_text(json.dumps(baseline_validation.run(tmp / 'data', sha, log=lambda _: None)))
            products = sorted({p for _, p, *_ in train})
            rows = [(p, json.dumps({'Color': 'Black' if i % 2 else 'Red'}), f'Store{i % 3}')
                    for i, p in enumerate(products)]
            facts, counts = kg_v2_facts.build_facts(rows)
            out = tmp / 'facts'
            out.mkdir()
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq.write_table(pa.table({k: [f[i] for f in facts] for i, k in enumerate(('parent_asin', 'relation', 'value'))}),
                           out / 'facts.parquet')
            (out / 'manifest.json').write_text(json.dumps({'schema_version': kg_v2_facts.SCHEMA, 'counts': counts,
                                                           'facts_sha256': kg_v2_facts._sha256(out / 'facts.parquet')}))
            result = kg_v2_validation.run(tmp / 'data', sha, out, recorded, spot_every=1, log=lambda _: None)
        for group in result['conditions'].values():
            self.assertEqual(set(group), {'Popularity', 'KG-only (KG-v1)', 'A. brand fix', 'B. brand fix + attributes'})
            self.assertEqual(len(group['B. brand fix + attributes']['mixtures']), 3)
            gain = group['A. brand fix']['gain_over_kg_v1']
            self.assertLessEqual(gain['ci95'][0], gain['mean_difference'])
            self.assertLessEqual(gain['mean_difference'], gain['ci95'][1])

    def test_the_bootstrap_interval_brackets_a_known_shift(self):
        control = {f'u{i}': 0.0 for i in range(400)}
        attempt = {u: (0.5 if i % 2 else 0.0) for i, u in enumerate(control)}
        gain = kg_v2_validation.paired_bootstrap(attempt, control)
        self.assertEqual(gain['mean_difference'], 0.25)
        self.assertLess(gain['ci95'][0], 0.25)
        self.assertGreater(gain['ci95'][0], 0.18)
        self.assertEqual(kg_v2_validation.paired_bootstrap(attempt, control), gain)
        with self.assertRaises(ValueError):
            kg_v2_validation.paired_bootstrap(attempt, {'other': 0.0})


if __name__ == '__main__':
    unittest.main()
