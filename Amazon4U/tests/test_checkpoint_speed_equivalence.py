"""Readiness C2: the sped-up D1/D2 give the accepted D1/D2's outcome on every input tried.

The accepted code is the pinned copy in tests/accepted_checkpoint/ (checked against the
recorded O-OFFLINE hashes). An outcome is the returned value or the exception's class
name and message. The fingerprint property test backs D2's `content_verified` shortcut:
for any input D1 accepts, the content it retains has the same three fingerprints.
"""

import copy
import random
import unittest

from checkpoint_dual import accepted_modules, outcome
from test_checkpoint_storage import _fixture
import scripts.checkpoint_identity as d1
import scripts.checkpoint_payload as d2

CATEGORIES = ['c:["Music"]', 'c:["Music","Guitars"]', 'c:["Gear"]']
BRANDS = ['b:Acme', 'b:Zed']


class _IterSurrogate(str):
    def __iter__(self):
        return iter('\ud800')


def _validate(module, payload, arguments):
    kwargs = {k: v for k, v in arguments.items() if k != 'io_policy'}
    return outcome(module.validate_checkpoint_payload, payload, **kwargs)


def _random_identity_fields(rng, settings):
    users = [f'u{i}' for i in range(rng.randint(2, 5))]
    items = [f'p{i}' for i in range(rng.randint(2, 5))]
    condition = rng.choice(['all_ratings', 'rating_ge_4'])
    pairs = rng.sample([(u, p) for u in users for p in items], rng.randint(2, len(users) * len(items)))
    rows = [(u, p, rng.choice([1, 2, 3, 4, 5, 4.0, 5.0, 2.5]), rng.randint(0, 9)) for u, p in pairs]
    records = [rng.choice([lambda r: {'user_id': r[0], 'parent_asin': r[1], 'rating': r[2], 'timestamp': r[3]},
                           tuple, list])(r) for r in rows]
    positive = {u: [p for v, p, r, _ in rows if v == u and (condition == 'all_ratings' or r >= 4)] for u in users}
    rated = {u: [p for v, p, _, _ in rows if v == u] for u in users}
    warm = sorted({p for ps in positive.values() for p in ps})
    shape = lambda xs: rng.choice([list, tuple, set, frozenset])(rng.sample(xs, len(xs)))
    nodes = {'User': users, 'Product': items, 'CategoryPath': CATEGORIES, 'Brand': BRANDS}
    possible = [(st, s, rel, dt, d) for rel, (st, dt) in d1.RELATION_SCHEMAS.items()
                for s in nodes[st] for d in nodes[dt]]
    edges = [rng.choice([lambda e: e + (rng.choice([1, 1.0, 0.5]),),
                         lambda e: dict(zip(('src_type', 'src_id', 'relation', 'dst_type', 'dst_id', 'weight'),
                                            e + (1.0,)))])(e)
             for e in rng.sample(possible, rng.randint(1, min(12, len(possible))))]
    metadata = {p: {'categories': shape(rng.sample(CATEGORIES, rng.randint(0, 3))),
                    'brand': rng.choice(BRANDS + [None])} for p in items}
    targets = {u: [rng.choice(items)] for u in users}
    vrecords = [{'user_id': u, 'parent_asin': t[0], 'rating': rng.choice([4, 5.0]), 'timestamp': 99}
                for u, t in targets.items()]
    validation = rng.choice([{'targets': targets}, {'records': vrecords},
                             {'targets': targets, 'records': vrecords},
                             {'targets': targets, 'records': None}])
    return dict(method='LightGCN', category='Musical_Instruments', feedback_condition=condition,
                seed=42, training_records=records,
                positive_history={u: shape(v) for u, v in positive.items()},
                all_rated_history={u: shape(v) for u, v in rated.items()},
                warm_items=rng.sample(warm, len(warm)),
                graph_nodes={k: shape(v) for k, v in nodes.items()}, graph_edges=edges,
                product_metadata=metadata, validation_inputs=validation,
                resolved_settings=copy.deepcopy(settings))


class SpeedEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_d1, cls.old_d2 = accepted_modules()

    def test_clean_text_check_gives_the_accepted_outcome(self):
        values = ['ok', 'B07ZHC81BW', 'naïve', 'x　', '　x', ' pad', 'pad ', '', ' ',
                  '\ud800', 'a\ud800', '\ud800a', 'a\udfffb', '😀', 'x퟿',
                  _IterSurrogate('plain'), 7, None, b'bytes']
        for value in values:
            for allow_empty in (False, True):
                with self.subTest(value=repr(value), allow_empty=allow_empty):
                    self.assertEqual(outcome(d1._validate_clean_text, value, 'n', allow_empty),
                                     outcome(self.old_d1._validate_clean_text, value, 'n', allow_empty))

    def test_empty_pool_pairs_match_the_accepted_computation(self):
        for with_empty_pool in (False, True):
            ident = _fixture_identity(with_empty_pool)
            old = self.old_d2._compute_condition_positive_and_empty_pool_pairs(ident)
            self.assertEqual(d2._compute_condition_positive_and_empty_pool_pairs(ident), old)
        self.assertTrue(old[1], 'the empty-pool fixture has a non-empty K')

    def test_retained_content_has_the_fingerprints_from_dict_checked(self):
        settings = _fixture()[1]['expected_identity']['resolved_settings']
        rng, accepted = random.Random(2026), 0
        for trial in range(400):
            fields = _random_identity_fields(rng, settings)
            try:
                ident = d1.build_canonical_identity(**fields)
            except d1.CheckpointValidationError:
                continue
            accepted += 1
            with self.subTest(trial=trial):
                self.assertEqual(d1.compute_data_fingerprint(
                    ident.feedback_condition, sorted(ident.positive_history), ident.training_records,
                    ident.positive_history, ident.all_rated_history, ident.warm_items),
                    ident.data_fingerprint)
                self.assertEqual(d1.compute_graph_fingerprint(
                    ident.graph_nodes, ident.graph_edges, ident.product_metadata), ident.graph_fingerprint)
                self.assertEqual(d1.compute_validation_fingerprint(ident.validation_inputs),
                                 ident.validation_fingerprint)
                rebuilt = d1.CanonicalResumeIdentity.from_dict(ident.to_dict())
                d1.validate_canonical_identity(rebuilt, ident)
        self.assertGreater(accepted, 200)

    def test_public_identity_validation_still_detects_changed_retained_content(self):
        ident = _fixture_identity(False)
        ident.training_records[0]['rating'] = 1.0 if ident.training_records[0]['rating'] != 1.0 else 5.0
        expected = _fixture_identity(False)
        new = outcome(d1.validate_canonical_identity, ident, expected)
        self.assertEqual(new[:2], ('raised', 'CheckpointIdentityMismatchError'))

    def test_graph_fingerprint_category_checks_give_the_accepted_outcome(self):
        settings = _fixture()[1]['expected_identity']['resolved_settings']
        rng = random.Random(15)
        variants = ['c:["Music"]', 'c:["Music","Guitars"]', 'c: ["Music"]', 'c:["Music" ]', 'c:[]',
                    'c:["Music"', 'c:"Music"', '["Music"]', 'c:[" Music"]', _IterSurrogate('c:["Music"]'),
                    'c:["Unknown"]', 7, None]
        for trial in range(300):
            fields = _random_identity_fields(rng, settings)
            for meta in fields['product_metadata'].values():
                meta['categories'] = [rng.choice(variants) for _ in range(rng.randint(0, 3))]
            if rng.random() < 0.3:
                fields['graph_nodes']['CategoryPath'] = [rng.choice(variants) for _ in range(3)]
            args = (fields['graph_nodes'], fields['graph_edges'], fields['product_metadata'])
            with self.subTest(trial=trial):
                self.assertEqual(outcome(d1.compute_graph_fingerprint, *args),
                                 outcome(self.old_d1.compute_graph_fingerprint, *args))

    def test_resource_walk_gives_the_accepted_outcome(self):
        from test_checkpoint_storage_equivalence import _value
        rng = random.Random(14)
        for trial in range(1500):
            value = _value(rng)
            policy = {'max_input_bytes': rng.choice([0, 9, 40, 300, 1 << 20]),
                      'max_output_bytes': rng.choice([0, 9, 40, 300, 1 << 20]),
                      'max_container_depth': rng.choice([0, 1, 2, 3, 32]),
                      'max_visited_values': rng.choice([0, 1, 5, 20, 1 << 20])}
            with self.subTest(trial=trial):
                self.assertEqual(outcome(d2.preflight_resource_accounting, value, policy, 'payload'),
                                 outcome(self.old_d2.preflight_resource_accounting, value, policy, 'payload'))

    def test_payload_validation_builds_each_identity_once(self):
        from unittest import mock
        payload, arguments = _fixture(hetero=False, initial=False)
        built = []
        original = d1.CanonicalResumeIdentity.from_dict.__func__

        def spy(cls, data):
            built.append(data is arguments['expected_identity'])
            return original(cls, data)
        with mock.patch.object(d2.CanonicalResumeIdentity, 'from_dict', classmethod(spy)):
            self.assertEqual(_validate(d2, payload, arguments)[0], 'ok')
        self.assertEqual(built, [False, True, False], 'raw identity, expected once, owned copy')

    def test_second_pass_compares_the_owned_identity_with_the_expected_one(self):
        from unittest import mock
        payload, arguments = _fixture(hetero=False, initial=False)
        other = _rebuilt(payload['identity'], lambda f: _flip(f['training_records'][0]))
        original = d2._make_owned_output_copy

        def swap(value):
            owned = original(value)
            if type(owned) is dict and 'identity' in owned and 'sampler' in owned:
                owned['identity'] = copy.deepcopy(other)
            return owned
        with mock.patch.object(d2, '_make_owned_output_copy', side_effect=swap):
            result = _validate(d2, payload, arguments)
        self.assertEqual(result[:2], ('raised', 'CheckpointIdentityMismatchError'))

    def test_payload_validation_gives_the_accepted_outcome(self):
        for hetero in (False, True):
            for initial in (False, True):
                payload, arguments = _fixture(hetero=hetero, initial=initial)
                for label, mutate in _MUTATIONS:
                    p, a = copy.deepcopy(payload), copy.deepcopy(arguments)
                    mutate(p, a)
                    with self.subTest(hetero=hetero, initial=initial, mutation=label):
                        new = _validate(d2, p, a)
                        self.assertEqual(new, _validate(self.old_d2, p, a))
                        if label == 'none':
                            self.assertEqual(new[0], 'ok')


def _fixture_identity(with_empty_pool):
    from test_checkpoint_payload import _make_synthetic_identity_and_provenance
    return _make_synthetic_identity_and_provenance(with_empty_pool=with_empty_pool)[0]


def _record(p):
    return p['identity']['training_records'][0]


def _flip(record):
    record['rating'] = 1.0 if record['rating'] != 1.0 else 5.0


def _rebuilt(identity, change):
    """A self-consistent identity whose content differs: from_dict accepts it."""
    fields = {k: copy.deepcopy(identity[k]) for k in (
        'method', 'category', 'feedback_condition', 'seed', 'training_records', 'positive_history',
        'all_rated_history', 'warm_items', 'graph_nodes', 'graph_edges', 'product_metadata',
        'validation_inputs', 'resolved_settings')}
    change(fields)
    return d1.build_canonical_identity(**fields).to_dict()


_MUTATIONS = [
    ('none', lambda p, a: None),
    ('payload rating', lambda p, a: _flip(_record(p))),
    ('expected rating', lambda p, a: _flip(a['expected_identity']['training_records'][0])),
    ('both ratings', lambda p, a: (_flip(_record(p)), _flip(a['expected_identity']['training_records'][0]))),
    ('payload edge weight', lambda p, a: p['identity']['graph_edges'][0].update(weight=2.0)),
    ('payload edge dropped', lambda p, a: p['identity']['graph_edges'].pop()),
    ('payload data fingerprint', lambda p, a: p['identity'].update(data_fingerprint='0' * 64)),
    ('expected graph fingerprint', lambda p, a: a['expected_identity'].update(graph_fingerprint='f' * 64)),
    ('payload identity hash', lambda p, a: p['identity'].update(identity_hash='1' * 64)),
    ('payload validation target', lambda p, a: p['identity']['validation_inputs']['targets'].update(
        {next(iter(p['identity']['validation_inputs']['targets'])): ['p_unknown']})),
    ('lone surrogate in user ID', lambda p, a: _record(p).update(user_id=_record(p)['user_id'] + '\ud800')),
    ('padded item ID', lambda p, a: _record(p).update(parent_asin=' ' + _record(p)['parent_asin'])),
    ('payload warm item dropped', lambda p, a: p['identity']['warm_items'].pop()),
    ('expected history reordered', lambda p, a: a['expected_identity']['positive_history'].update(
        {u: list(reversed(v)) for u, v in a['expected_identity']['positive_history'].items()})),
    ('payload seed', lambda p, a: p['identity'].update(seed=p['identity']['seed'] + 1)),
    ('payload rebuilt with another rating', lambda p, a: p.update(
        identity=_rebuilt(p['identity'], lambda f: _flip(f['training_records'][0])))),
    ('expected rebuilt with another edge weight', lambda p, a: a.update(
        expected_identity=_rebuilt(a['expected_identity'], lambda f: f['graph_edges'][0].update(weight=3.0)))),
    ('expected category metadata', lambda p, a: next(iter(a['expected_identity']['product_metadata'].values()))
     .update(categories=['c:["Other"]'])),
]


if __name__ == '__main__':
    unittest.main()
