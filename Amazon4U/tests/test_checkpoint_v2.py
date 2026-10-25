"""Checkpoint identity v2 (docs/checkpoint-identity-v2-plan.md).

v2 must act as v1 everywhere outside the identity: every mutation of a payload leaf or an
expectation leaf gets the same outcome (class and message) from both. Inside the identity,
v2 must refuse every change, now by comparison with the trusted data. Files of one version
are refused by the other, and resume is exact.
"""

import copy
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from checkpoint_dual import outcome
from checkpoint_v2_derivation import derive
from test_checkpoint_real_data import BINDING, CASES, DEVICE, FP, RealDataCheckpointTests
from test_checkpoint_storage import _fixture
import scripts.checkpoint_payload as d2
import scripts.checkpoint_v2 as v2
from scripts.checkpoint_identity import CanonicalResumeIdentity, CheckpointError, build_canonical_identity
from scripts.checkpoint_real_data import HETERO, LIGHTGCN, build
from scripts.checkpoint_storage_cuda import publish_cuda_checkpoint, retrieve_cuda_checkpoint
from scripts.training_harness import D2_KEYS, owned_digest

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('method', 'category', 'feedback_condition', 'seed', 'training_records', 'positive_history',
          'all_rated_history', 'warm_items', 'graph_nodes', 'graph_edges', 'product_metadata',
          'validation_inputs', 'resolved_settings')


def as_v2(payload, arguments):
    """The v2 form of a v1 fixture: same payload with a reference, trusted identity object."""
    p, a = copy.deepcopy(payload), {k: v for k, v in copy.deepcopy(arguments).items() if k != 'io_policy'}
    p['schema_version'] = v2.PAYLOAD_SCHEMA_VERSION_V2
    p['identity'] = v2.identity_reference(CanonicalResumeIdentity.from_dict(p['identity']))
    a['expected_identity'] = CanonicalResumeIdentity.from_dict(a['expected_identity'])
    return p, a


def _without_identity(fn):
    """The validator's output minus the two fields v2 changes by design."""
    def call(p, **a):
        return {k: v for k, v in fn(p, **a).items() if k not in ('identity', 'schema_version')}
    return call


def v1_call(p, a):
    return outcome(_without_identity(d2.validate_checkpoint_payload), p,
                   **{k: v for k, v in a.items() if k != 'io_policy'})


def v2_call(p, a):
    return outcome(_without_identity(v2.validate_checkpoint_payload_v2), p, **a)


def leaves(value, path=()):
    """Every (path, leaf) under value; tensors and empty containers are leaves."""
    if type(value) is dict and value:
        for k, v in value.items():
            yield from leaves(v, path + (k,))
    elif type(value) in (list, tuple) and value:
        for i, v in enumerate(value):
            yield from leaves(v, path + (i,))
    else:
        yield path, value


def variants(leaf):
    t = type(leaf)
    if t is bool:
        return [not leaf, int(leaf)]
    if t is int:
        return [leaf + 1, float(leaf), str(leaf)]
    if t is float:
        return [leaf + 0.5, float('nan')]
    if t is str:
        return [leaf + 'x', '']
    if leaf is None:
        return [0]
    if t is torch.Tensor:
        changed = leaf.clone()
        if changed.layout == torch.strided and changed.numel():
            flat = changed.view(-1)
            flat[0] = flat[0] + 1 if changed.dtype != torch.bool else ~flat[0]
        return [changed, leaf.to(torch.float64) if leaf.dtype != torch.float64 else leaf.to(torch.float32)]
    return [None]


def set_path(root, path, value):
    node = root
    for key in path[:-1]:
        node = node[key]
    if type(node) is tuple:
        raise TypeError('tuple')
    node[path[-1]] = value


class DerivationTests(unittest.TestCase):
    def test_v2_semantic_checks_are_v1_with_only_the_listed_substitutions(self):
        source = (ROOT / 'scripts/checkpoint_v2.py').read_text()
        start = source.index('# DERIVED from checkpoint_payload._validate_semantic_payload')
        derived = source[source.index('\n', start) + 1:source.index('# END DERIVED')]
        v1 = (ROOT / 'scripts/checkpoint_payload.py').read_text()
        self.assertEqual(derived.strip(), derive(v1).strip())


class EquivalenceOutsideIdentityTests(unittest.TestCase):
    # The full sweep is about 18,500 mutations; A4U_V2_FULL_SWEEP=1 runs all of them. By
    # default a fixed random sample keeps the suite fast.
    SAMPLE = None if os.environ.get('A4U_V2_FULL_SWEEP') == '1' else 2400

    def test_every_payload_and_expectation_leaf_mutation_matches_v1(self):
        compared = refused = 0
        rng = random.Random(21)
        for hetero in (False, True):
            for initial in (False, True):
                payload, arguments = _fixture(hetero=hetero, initial=initial)
                p2, a2 = as_v2(payload, arguments)
                self.assertEqual(v1_call(payload, arguments)[0], 'ok')
                self.assertEqual(v2_call(p2, a2)[0], 'ok')
                targets = [('payload', path, leaf) for path, leaf in leaves(payload) if path[:1] != ('identity',)
                           and path[:1] != ('schema_version',)]
                targets += [(k, path, leaf) for k in ('expected_provenance', 'expected_model', 'expected_runtime')
                            for path, leaf in leaves(arguments[k])]
                todo = [(w, p, v) for w, p, leaf in targets for v in variants(leaf)]
                if self.SAMPLE is not None:
                    todo = rng.sample(todo, min(len(todo), self.SAMPLE // 4))
                for where, path, value in todo:
                    c1 = (copy.deepcopy(payload), copy.deepcopy(arguments))
                    c2 = (copy.deepcopy(p2), copy.deepcopy(a2))
                    try:
                        for p, a in (c1, c2):
                            set_path(p if where == 'payload' else a[where], path, copy.deepcopy(value))
                    except TypeError:
                        continue
                    old, new = v1_call(*c1), v2_call(*c2)
                    with self.subTest(hetero=hetero, initial=initial, where=where, path=path, value=repr(value)[:40]):
                        self.assertEqual(new, old)
                    compared += 1
                    refused += old[0] == 'raised'
        self.assertGreater(compared, 1000)
        if self.SAMPLE is None:
            print(f'full sweep: {compared} compared, {refused} refused by both', file=sys.stderr)
        self.assertGreater(refused, compared // 2)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.payload, arguments = _fixture(hetero=False, initial=False)
        self.p2, self.a2 = as_v2(self.payload, arguments)
        self.fields = {k: copy.deepcopy(arguments['expected_identity'][k]) for k in FIELDS}

    def trusted_with(self, change):
        fields = copy.deepcopy(self.fields)
        change(fields)
        return build_canonical_identity(**fields)

    def test_the_reference_holds_no_data_copy_and_keeps_the_identity_hash(self):
        ref = self.p2['identity']
        self.assertEqual(set(ref), set(v2.REFERENCE_KEYS))
        self.assertEqual(ref['identity_hash'], self.a2['expected_identity'].identity_hash)

    def test_resuming_against_changed_data_is_refused_by_fingerprint(self):
        def rating(f):
            f['training_records'][0]['rating'] = 1.0 if f['training_records'][0]['rating'] != 1.0 else 5.0

        def edge(f):
            f['graph_edges'][0] = {**f['graph_edges'][0], 'weight': 2.0} if isinstance(
                f['graph_edges'][0], dict) else (*f['graph_edges'][0][:5], 2.0)

        def target(f):
            t = f['validation_inputs']['targets']
            u = next(iter(t))
            t[u] = ['p_other']
            f['validation_inputs'].pop('records', None)

        def setting(f):
            f['resolved_settings']['training']['batch_size'] += 1
        for name, change, field in (('rating', rating, 'data_fingerprint'), ('edge', edge, 'graph_fingerprint'),
                                    ('target', target, 'validation_fingerprint'),
                                    ('setting', setting, 'settings_fingerprint'),
                                    ('seed', lambda f: f.update(seed=f['seed'] + 1), 'seed')):
            with self.subTest(name):
                a = {**self.a2, 'expected_identity': self.trusted_with(change)}
                result = v2_call(copy.deepcopy(self.p2), a)
                self.assertEqual(result[:2], ('raised', 'CheckpointIdentityMismatchError'))
                self.assertIn(field, result[2])

    def test_a_tampered_reference_is_refused(self):
        other = v2.identity_reference(self.trusted_with(lambda f: f.update(seed=f['seed'] + 1)))
        edits = {
            'fingerprint without hash': (lambda r: r.update(data_fingerprint='0' * 64),
                                         'Falsified or corrupt identity_hash'),
            'settings without fingerprint': (lambda r: r['resolved_settings']['training'].update(batch_size=3),
                                             'Falsified or corrupt settings_fingerprint'),
            'consistent other identity': (lambda r: r.update(other), 'differs from the trusted data: seed'),
            'data copy added': (lambda r: r.update(training_records=[]), 'payload.identity keys must be exactly'),
            'old schema': (lambda r: r.update(schema_version='amazon4u-checkpoint-identity-v1'),
                           'payload.identity.schema_version'),
            'not a dict': (None, 'payload.identity must be a dict'),
        }
        for name, (edit, message) in edits.items():
            p = copy.deepcopy(self.p2)
            if edit is None:
                p['identity'] = list(p['identity'])
            else:
                edit(p['identity'])
            with self.subTest(name):
                result = v2_call(p, self.a2)
                self.assertEqual(result[0], 'raised')
                self.assertIn(message, result[2])

    def test_the_reference_check_needs_a_trusted_identity_object(self):
        for trusted in (self.a2['expected_identity'].to_dict(), None):
            with self.subTest(type(trusted).__name__):
                result = outcome(v2.validate_identity_reference, copy.deepcopy(self.p2['identity']), trusted)
                self.assertEqual(result[:2], ('raised', 'CheckpointValidationError'))
                self.assertIn('trusted CanonicalResumeIdentity', result[2])

    def test_an_untrusted_expected_identity_is_refused(self):
        a = {**self.a2, 'expected_identity': self.a2['expected_identity'].to_dict()}
        self.assertEqual(v2_call(copy.deepcopy(self.p2), a)[:2], ('raised', 'CheckpointValidationError'))


class RealDataV2Tests(RealDataCheckpointTests):
    """The C1 synthetic-dataset cases, rerun with v2 checkpoints, plus cross-version refusal."""

    def fresh(self, method, condition, version=2):
        return build(self.data, method, condition, batch_size=8, fingerprint=FP, version=version)

    def publish(self, name, trainer, expectations, identity):
        d = {k: expectations[k] for k in D2_KEYS}
        if type(expectations['expected_identity']) is dict:
            return super().publish(name, trainer, expectations, identity)
        return v2.publish_checkpoint_v2(self.root, name, trainer.cuda_payload_v2(expectations, identity, BINDING),
                                        expected_device=DEVICE, **d, io_policy=expectations['io_policy'])

    def retrieve(self, name, expectations):
        d = {k: expectations[k] for k in D2_KEYS}
        if type(expectations['expected_identity']) is dict:
            return super().retrieve(name, expectations)
        return v2.retrieve_checkpoint_v2(self.root, name, expected_device=DEVICE, **d,
                                         io_policy=expectations['io_policy'])

    def test_resume_after_epoch_one_equals_two_uninterrupted_epochs(self):
        for method, condition in CASES:
            with self.subTest(method=method, condition=condition):
                straight, expectations, identity = self.fresh(method, condition)
                logs = [straight.train_epoch() for _ in range(2)]
                first, _, _ = self.fresh(method, condition)
                part = [first.train_epoch()]
                name = self.publish(f'{condition}-{len(method)}.a4ucp', first, expectations, identity)
                resumed, again, _ = self.fresh(method, condition)
                resumed.restore_cuda(self.retrieve(name, again))
                part.append(resumed.train_epoch())
                self.assertEqual(part, logs)
                self.assertEqual(owned_digest(resumed.cuda_payload_v2(expectations, identity, BINDING)),
                                 owned_digest(straight.cuda_payload_v2(expectations, identity, BINDING)))

    def test_identity_follows_the_dataset_and_condition(self):
        _, a, ref_a = self.fresh(LIGHTGCN, 'all_ratings')
        _, b, ref_b = self.fresh(LIGHTGCN, 'rating_ge_4')
        self.assertEqual(len(a['expected_identity'].training_records), len(self.data.train))
        self.assertNotEqual(ref_a['data_fingerprint'], ref_b['data_fingerprint'])
        _, v1_expectations, v1_identity = self.fresh(LIGHTGCN, 'all_ratings', version=1)
        self.assertEqual(ref_a['identity_hash'], v1_identity['identity_hash'])

    def test_a_changed_training_record_is_refused(self):
        trainer, expectations, identity = self.fresh(HETERO, 'rating_ge_4')
        changed = dict(expectations)
        fields = {k: copy.deepcopy(getattr(expectations['expected_identity'], k)) for k in FIELDS}
        fields['training_records'][0]['timestamp'] += 1
        changed['expected_identity'] = build_canonical_identity(**fields)
        with self.assertRaises(CheckpointError):
            self.publish('bad.a4ucp', trainer, changed, identity)

    def test_each_version_refuses_the_other_versions_file(self):
        t1, e1, i1 = self.fresh(LIGHTGCN, 'all_ratings', version=1)
        t2, e2, i2 = self.fresh(LIGHTGCN, 'all_ratings', version=2)
        for t in (t1, t2):
            t.train_epoch()
        self.publish('v1.a4ucp', t1, e1, i1)
        self.publish('v2.a4ucp', t2, e2, i2)
        with self.assertRaisesRegex(CheckpointError, 'not an amazon4u-checkpoint-payload-cuda-v2'):
            self.retrieve('v1.a4ucp', e2)
        with self.assertRaises(CheckpointError):
            self.retrieve('v2.a4ucp', e1)


if __name__ == '__main__':
    unittest.main()
