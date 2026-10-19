"""Additive H input/graph controls. No optimizer calls in this module."""
import copy
import unittest

import torch

from scripts.checkpoint_identity import (
    CheckpointValidationError, build_canonical_identity,
)
from scripts.differentiable_models import build_hetero_graph
from scripts.learned_models import LightGCNGraph
from scripts.recommendation_core import NegativeSampler
from scripts.training_data import HarnessError, TrainingData, bounded_training_context
from test_checkpoint_payload import (
    _build_coherent_lightgcn_payload, _make_synthetic_resource_policy,
)


IDENTITY_INPUT_FIELDS = (
    'method', 'category', 'feedback_condition', 'seed', 'training_records',
    'positive_history', 'all_rated_history', 'warm_items', 'graph_nodes',
    'graph_edges', 'product_metadata', 'validation_inputs', 'resolved_settings',
)


def changed_validation_identity(serialized):
    """Independently declared heldout mutation, retaining train inputs exactly."""
    fields = copy.deepcopy(serialized)
    validation = copy.deepcopy(fields['validation_inputs'])
    validation['records'] = [
        {'user_id': r['user_id'], 'parent_asin': r['parent_asin'],
         'rating': 4.0 if float(r['rating']) == 5.0 else 5.0,
         'timestamp': r['timestamp'] + 777}
        if isinstance(r, dict) else
        (r[0], r[1], 4.0 if float(r[2]) == 5.0 else 5.0, r[3] + 777)
        for r in validation['records']]
    return build_canonical_identity(**{
        key: (validation if key == 'validation_inputs' else fields[key])
        for key in IDENTITY_INPUT_FIELDS}).to_dict()


class TrainingDataTests(unittest.TestCase):
    def fixture(self, empty_pool=False):
        _, identity, _, _, _ = _build_coherent_lightgcn_payload(
            epoch=0, with_empty_pool=empty_pool)
        policy = _make_synthetic_resource_policy()
        return TrainingData(identity, expected_identity=copy.deepcopy(identity),
                            resource_policy=policy), identity, policy

    def test_independent_valid_input_and_owned_isolation(self):
        data, identity, _ = self.fixture()
        self.assertEqual(data.positive_pairs,
                         (('u1', 'p1'), ('u1', 'p2'), ('u2', 'p2'), ('u2', 'p3')))
        self.assertEqual(data.context.sorted_warm_items, ['p1', 'p2', 'p3'])
        identity['training_records'][0]['rating'] = 1.0
        self.assertEqual(data.identity.training_records[0]['rating'], 4.0)
        self.assertIsInstance(data.identity.validation_inputs['records'][0], dict)
        self.assertEqual(set(data.identity.validation_inputs['records'][0]),
                         {'user_id', 'parent_asin', 'rating', 'timestamp'})
        self.assertIsInstance(data.validation_records[0], tuple)
        self.assertEqual(data.validation_records[0][:3], ('u1', 'p3', 5.0))
        self.assertEqual(data.validation_records,
                         [('u1', 'p3', 5.0, 5000), ('u2', 'p1', 4.0, 6000)])

    def test_holdout_mutations_do_not_enter_context_or_graphs(self):
        data, identity, policy = self.fixture()
        changed = changed_validation_identity(identity)
        other = TrainingData(changed, expected_identity=copy.deepcopy(changed),
                             resource_policy=policy)
        self.assertNotEqual(data.identity.validation_fingerprint,
                            other.identity.validation_fingerprint)
        self.assertEqual(data.context.positive_history, other.context.positive_history)
        self.assertEqual(data.context.all_rated_history, other.context.all_rated_history)
        self.assertEqual(data.identity.graph_fingerprint, other.identity.graph_fingerprint)
        a, b = LightGCNGraph(data.context), LightGCNGraph(other.context)
        self.assertEqual(a.users, b.users)
        self.assertEqual(a.warm_items, b.warm_items)
        ha = build_hetero_graph(data.context, data.product_categories, data.product_brands)
        hb = build_hetero_graph(other.context, other.product_categories, other.product_brands)
        self.assertEqual(ha.node_to_idx, hb.node_to_idx)
        self.assertEqual(data.positive_pairs, other.positive_pairs)

    def test_low_rating_observed_exclusion_and_condition_graph(self):
        data, _, _ = self.fixture(empty_pool=True)
        self.assertIn('p3', data.context.all_rated_history['u3'])
        self.assertNotIn('p3', data.context.positive_history['u3'])
        sampler = NegativeSampler(data.context)
        self.assertEqual(sampler.get_negative_pool('u3'), [])
        self.assertEqual(sampler.get_negative_pool('u1'), ['p3'])
        self.assertEqual(sampler.get_negative_pool('u2'), ['p1', 'p2'])
        graph = LightGCNGraph(data.context)
        self.assertEqual(graph.users, ['u1', 'u2', 'u3'])
        self.assertEqual(len(data.positive_pairs), 5)

    def test_reject_mutated_identity_or_missing_actual_validation_records(self):
        _, identity, policy = self.fixture()
        # Complete valid control before each one-field canonical corruption. These
        # witness D1 structural/content validation, not a ValueError boundary.
        for mutation in ('rating', 'metadata', 'fingerprint'):
            control = TrainingData(copy.deepcopy(identity), expected_identity=identity,
                                   resource_policy=policy)
            self.assertEqual(control.identity.to_dict(), identity)
            bad = copy.deepcopy(identity)
            if mutation == 'rating':
                bad['training_records'][0]['rating'] = 1.0
            elif mutation == 'metadata':
                bad['product_metadata']['p1']['review_text'] = 'excluded'
            else:
                bad['data_fingerprint'] = '0' * 64
            with self.subTest(mutation=mutation), self.assertRaises(CheckpointValidationError):
                TrainingData(bad, expected_identity=identity, resource_policy=policy)
        # Rebuild the independent canonical packet so an obsolete fingerprint
        # cannot stand in for the H guard requiring actual validation records.
        fields = {key: copy.deepcopy(identity[key]) for key in IDENTITY_INPUT_FIELDS}
        del fields['validation_inputs']['records']
        missing_records = build_canonical_identity(**fields).to_dict()
        with self.assertRaisesRegex(HarnessError, 'actual records and target IDs'):
            TrainingData(missing_records, expected_identity=copy.deepcopy(missing_records),
                         resource_policy=policy)

    def test_pre_native_bounds_exact_fields_and_duplicates(self):
        policy = _make_synthetic_resource_policy()
        record = {'user_id': 'u', 'parent_asin': 'p', 'rating': 5.0, 'timestamp': 1}
        valid = bounded_training_context([record], 'all_ratings', policy)
        self.assertEqual(valid.users, {'u'})
        cases = [
            ([record, dict(record)], 'duplicate training pair'),
            ([dict(record, review_text='forbidden')], 'fields are not permitted'),
            ([dict(record, rating=True)], 'finite nonboolean'),
            ([dict(record, user_id=' x')], 'bounded clean strings'),
            ([dict(record, user_id='u' + str(i)) for i in range(9)], 'population exceeds'),
            ([dict(record, parent_asin='p' + str(i)) for i in range(13)], 'population exceeds'),
            ([dict(record)] * 97, '1..96 owned records'),
        ]
        for records, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(HarnessError, reason):
                bounded_training_context(records, 'all_ratings', policy)
        # D2 resource preflight runs before H rating/context guards: non-finite float
        # raises CheckpointValidationError from preflight, not HarnessError.
        for nonfinite in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(nonfinite=str(nonfinite)), self.assertRaisesRegex(
                    CheckpointValidationError, 'contains non-finite float'):
                bounded_training_context([dict(record, rating=nonfinite)], 'all_ratings', policy)
        # Guard-order control: even with invalid H-context fields/duplicates,
        # preflight catches non-finite floats first before H guards are reached.
        with self.assertRaisesRegex(CheckpointValidationError, 'contains non-finite float'):
            bounded_training_context(
                [dict(record, rating=float('nan'), review_text='forbidden')],
                'all_ratings', policy)
        with self.assertRaisesRegex(ValueError, 'Invalid feedback condition'):
            bounded_training_context([record], 'final_test', policy)
        with self.assertRaisesRegex(HarnessError, '1..96 owned records'):
            bounded_training_context(torch.zeros(2, 2), 'all_ratings', policy)

    def test_all_exhausted_is_untrainable_not_an_empty_update(self):
        policy = _make_synthetic_resource_policy()
        trainable = bounded_training_context([
            {'user_id': 'u1', 'parent_asin': 'p1', 'rating': 5.0},
            {'user_id': 'u2', 'parent_asin': 'p2', 'rating': 5.0},
        ], 'all_ratings', policy)
        self.assertEqual(NegativeSampler(trainable).get_negative_pool('u1'), ['p2'])
        context = bounded_training_context([
            {'user_id': 'u1', 'parent_asin': 'p1', 'rating': 5.0},
            {'user_id': 'u2', 'parent_asin': 'p1', 'rating': 5.0},
        ], 'all_ratings', policy)
        with self.assertRaisesRegex(ValueError, 'No trainable positive examples'):
            NegativeSampler(context)
