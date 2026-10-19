"""H-REF additive full-discovery tests. Static maximum: 36 actual Adam calls.

First-step 2; restart 16; empty-pool 2; heldout 4; late-restore 2;
entrypoint 4; budget/evaluator failures 6. Every numerical branch uses <=4.
All remaining methods, including CPU refusals and closure controls, use zero.
"""
import copy
import errno
import hashlib
import os
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import torch

from scripts.checkpoint_identity import (
    CheckpointSecurityError, CheckpointValidationError, build_canonical_identity,
)
from scripts.checkpoint_payload import validate_checkpoint_payload
from scripts.differentiable_models import (
    DifferentiableHeteroGNN, DifferentiableLightGCN, build_hetero_graph,
)
from scripts.learned_models import LightGCNGraph
from scripts.recommendation_core import (
    Evaluator, KGOnlyScorer, NegativeSampler, PopularityScorer, select_kg_only_mixture,
)
from scripts.training_data import HarnessError, TrainingData, bounded_training_context
from scripts.training_entrypoint import execute_fixture, main, validate_run_spec
from scripts.training_harness import (
    ADAM_OPTIONS, BUNDLE_PATHS, CPUBackend, HCPUAdam, Harness, LiveSlot,
    PrivateRNG, Selection, baseline_validation, bundle_manifest, owned_digest,
    validation_score,
)
from test_checkpoint_storage import _fixture
from test_checkpoint_payload import (
    _build_coherent_lightgcn_payload, _make_synthetic_resource_policy,
)


# Independent expected membership: do not obtain the required closure from the
# implementation constant or a checkpoint's self-carried manifest.
EXPECTED_CLOSURE_PATHS = (
    'scripts/checkpoint_identity.py',
    'scripts/learned_models.py',
    'scripts/recommendation_core.py',
    'scripts/recommendation_evaluation.py',
    'scripts/differentiable_models.py',
    'scripts/build_kg.py',
    'scripts/audit_kg_inputs.py',
    'scripts/audit_candidates.py',
    'scripts/validate_kg_splits.py',
    'scripts/visualize_kg.py',
    'scripts/inspect_data.py',
    'scripts/download_5core.py',
    'configs/compute-plan.json',
    'configs/dataset-scope.json',
    'configs/evaluation-metrics.json',
    'configs/kg-v1.json',
    'configs/model-suite.json',
    'configs/protocol-lifecycle.json',
    'configs/reproducibility.json',
    'configs/training-semantics.json',
    'scripts/checkpoint_payload.py',
    'scripts/checkpoint_storage.py',
    'scripts/training_harness.py',
    'scripts/training_data.py',
    'scripts/training_entrypoint.py',
    'tests/test_training_harness.py',
    'tests/test_training_data.py',
    'docs/training-harness.md',
)


def trusted_sources():
    return {path: ('# independently supplied synthetic packet: ' + path + '\n').encode()
            for path in EXPECTED_CLOSURE_PATHS}


def expected_source_manifest():
    files = {
        path: {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        for path, content in trusted_sources().items()
    }
    return {'schema_version': 'amazon4u-H-bundle-v1', 'files': files,
            'sha256': owned_digest(files)}


@contextmanager
def fail_on_cuda_route():
    """Scoped fail-on-call tripwires, never harmless-looking CUDA no-op mocks."""
    with ExitStack() as stack:
        for owner, name in (
            (torch.cuda, 'is_available'),
            (torch.cuda, 'is_current_stream_capturing'),
            (torch._C, '_cuda_getDeviceCount'),
            (torch.optim.Optimizer, '_cuda_graph_capture_health_check'),
        ):
            stack.enter_context(patch.object(
                owner, name, side_effect=AssertionError('forbidden CUDA route: ' + name)))
        yield


def make_case(hetero=False, empty_pool=False, backend=None, heldout_mutation=False):
    if empty_pool:
        _, identity, provenance, model, runtime = _build_coherent_lightgcn_payload(
            epoch=0, with_empty_pool=True)
        _, ordinary = _fixture(initial=True)
        expectations = dict(expected_identity=identity, expected_provenance=provenance,
                            expected_model=model, expected_runtime=runtime,
                            resource_policy=_make_synthetic_resource_policy(),
                            io_policy=ordinary['io_policy'])
    else:
        _, expectations = _fixture(hetero=hetero, initial=True)
    expectations = copy.deepcopy(expectations)
    if heldout_mutation:
        old = expectations['expected_identity']
        validation = copy.deepcopy(old['validation_inputs'])
        validation['records'] = [
            {'user_id': r['user_id'], 'parent_asin': r['parent_asin'],
             'rating': 4.0 if float(r['rating']) == 5.0 else 5.0,
             'timestamp': r['timestamp'] + 999}
            if isinstance(r, dict) else
            (r[0], r[1], 4.0 if float(r[2]) == 5.0 else 5.0, r[3] + 999)
            for r in validation['records']]
        fields = {key: old[key] for key in (
            'method', 'category', 'feedback_condition', 'seed', 'training_records',
            'positive_history', 'all_rated_history', 'warm_items', 'graph_nodes',
            'graph_edges', 'product_metadata', 'validation_inputs', 'resolved_settings')}
        fields['validation_inputs'] = validation
        expectations['expected_identity'] = build_canonical_identity(**fields).to_dict()
    identity = expectations['expected_identity']
    data = TrainingData(copy.deepcopy(identity), expected_identity=identity,
                        resource_policy=expectations['resource_policy'])
    light = None
    if hetero:
        reference = DifferentiableHeteroGNN(
            build_hetero_graph(data.context, data.product_categories, data.product_brands),
            init_seed=42, embedding_dim=64)
        expected_initial = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    else:
        users = len(expectations['expected_model']['row_order']['User'])
        light = {
            'user_embeddings': (torch.arange(users * 64, dtype=torch.float32).reshape(users, 64) + 1) / 4096,
            'item_embeddings': (torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64) + 3) / 3072,
        }
        expected_initial = {
            'user_embeddings': light['user_embeddings'].clone(),
            'item_embeddings': light['item_embeddings'].clone(),
            'adj_user_item': expectations['expected_model']['buffer_state']['adj_user_item'].clone(),
        }
    expected_bundle = expected_source_manifest()
    envelope = {
        'schema_version': 'amazon4u-H-run-v1', 'run_id': 'independent-H-fixture',
        'approval_id': 'explicit-tiny-CPU-approval', 'profile_id': 'H-REF-synthetic-cpu-v1',
        'method': identity['method'], 'identity_hash': identity['identity_hash'],
        'device': 'cpu', 'accelerator': 'none', 'bundle_sha256': expected_bundle['sha256'],
        'initial_state_sha256': owned_digest(expected_initial),
        'determinism_policy': 'private_rng_bounded_cpu_no_global_setters',
    }
    kwargs = {
        'backend': CPUBackend() if backend is None else backend,
        'requested_device': 'cpu', 'accelerator': 'none',
        'run_envelope': copy.deepcopy(envelope), 'expected_run_envelope': envelope,
        'bundle_sources': trusted_sources(), 'expected_bundle': expected_bundle,
        'light_initial': light, 'max_adam_steps': 4,
    }
    return data, expectations, kwargs


def new_harness(**options):
    data, expectations, kwargs = make_case(**options)
    return Harness(data, expectations, **kwargs), data, expectations, kwargs


def spec_and_authority(method='LightGCN'):
    baseline = method in {'CorePopularity', 'KG-only'}
    spec = {
        'schema_version': 'amazon4u-H-spec-v1', 'run_id': 'independent-H-fixture',
        'approval_id': 'explicit-tiny-CPU-approval', 'stage': 'training_from_scratch',
        'mode': 'fixture_verification', 'data_kind': 'owned_synthetic_packet',
        'method': method, 'device': 'cpu', 'accelerator': 'none',
        'profile_id': 'H-REF-synthetic-cpu-v1', 'epochs': 0 if baseline else 2,
        'max_adam_steps': 0 if baseline else 4,
    }
    authority = {
        'schema_version': 'amazon4u-H-authorization-v1',
        'run_id': 'independent-H-fixture', 'approval_id': 'explicit-tiny-CPU-approval',
        'scope': 'tiny_synthetic_cpu_backward_and_private_restore', 'max_adam_steps': 4,
    }
    return spec, authority


class RecordingBackend(CPUBackend):
    def __init__(self):
        self.updates, self.builds = [], 0

    def make_model(self, *args, **kwargs):
        self.builds += 1
        return super().make_model(*args, **kwargs)

    def update(self, model, optimizer, triples):
        self.updates.append(list(triples))
        return super().update(model, optimizer, triples)


class TrainingHarnessTests(unittest.TestCase):
    def assert_tree_equal(self, a, b):
        self.assertEqual(owned_digest(a), owned_digest(b))

    @fail_on_cuda_route()
    def test_initial_both_models_complete_D2_and_owned_snapshots(self):
        for hetero in (False, True):
            with self.subTest(hetero=hetero):
                h, _, expected, _ = new_harness(hetero=hetero)
                self.assertIs(type(h.optimizer), HCPUAdam)
                initial = h.capture_epoch_checkpoint()
                self.assertEqual(initial['progress']['global_step'], 0)
                self.assertEqual(initial['optimizer']['state_dict']['state'], {})
                self.assertIsNone(initial['best_model'])
                self.assertIsNone(initial['sampler']['source_order'])
                validate_checkpoint_payload(initial, **{k: expected[k] for k in h.d2})
                first = initial['current_model']['state_dict']['user_embeddings']
                first.fill_(123)
                self.assertFalse(torch.equal(first, h.model.user_embeddings))

    @fail_on_cuda_route()
    def test_independent_first_Adam_step_oracle_both_models(self):
        # Two actual calls total; oracle uses only arithmetic/backward.
        for hetero in (False, True):
            with self.subTest(hetero=hetero):
                h, _, _, _ = new_harness(hetero=hetero)
                triples = [('u1', 'p1', 'p3'), ('u2', 'p3', 'p1')]
                h.optimizer.zero_grad(set_to_none=True)
                losses = h.model.compute_batch_loss(['u1', 'u2'], ['p1', 'p3'], ['p3', 'p1'])
                self.assertEqual(set(losses), {'total_loss', 'ranking_loss', 'reg_loss'})
                losses['total_loss'].backward()
                before = {n: p.detach().clone() for n, p in h.model.named_parameters()}
                gradients = {n: None if p.grad is None else p.grad.detach().clone()
                             for n, p in h.model.named_parameters()}
                h.backend.update(h.model, h.optimizer, triples)
                for name, parameter in h.model.named_parameters():
                    gradient = gradients[name]
                    state = h.optimizer.state.get(parameter)
                    if gradient is None:
                        self.assertIsNone(state)
                        self.assertTrue(torch.equal(before[name], parameter))
                    else:
                        oracle = before[name] - 0.001 * gradient / (gradient.abs() + 1e-8)
                        torch.testing.assert_close(parameter.detach(), oracle, atol=2e-7, rtol=2e-6)
                        torch.testing.assert_close(state['exp_avg'], 0.1 * gradient,
                                                   atol=1e-8, rtol=2e-6)
                        torch.testing.assert_close(state['exp_avg_sq'], 0.001 * gradient.square(),
                                                   atol=1e-9, rtol=2e-6)
                        self.assertEqual(state['step'].item(), 1)
                    self.assertIsNone(parameter.grad)

    @fail_on_cuda_route()
    def test_both_models_D3_restart_matches_uninterrupted_complete_trajectory(self):
        # Eight calls/model: uninterrupted four; prefix two; resumed two.
        for hetero in (False, True):
            with self.subTest(hetero=hetero), tempfile.TemporaryDirectory(dir=tempfile.tempdir) as directory:
                os.chmod(directory, 0o700)
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    uninterrupted, _, _, kw = new_harness(hetero=hetero)
                    prefix, _, _, _ = new_harness(hetero=hetero)
                    initial_receipt = prefix.publish_epoch_checkpoint(fd, 'initial.a4ucp')
                    self.assertEqual(initial_receipt['name'], 'initial.a4ucp')
                    initial_restart = prefix.retrieve_epoch_checkpoint(
                        fd, 'initial.a4ucp', actual_envelope=kw['run_envelope'],
                        expected_checkpoint_digest=initial_receipt['checkpoint_digest'],
                        bundle_sources=trusted_sources())
                    self.assert_tree_equal(prefix.capture_epoch_checkpoint(),
                                           initial_restart.capture_epoch_checkpoint())
                    uninterrupted.train_epoch()
                    expected_next_epoch = uninterrupted.train_epoch()
                    prefix.train_epoch()
                    saved = prefix.capture_epoch_checkpoint()
                    receipt = prefix.publish_epoch_checkpoint(fd, 'epoch.a4ucp')
                    self.assertEqual(receipt['name'], 'epoch.a4ucp')
                    restarted = prefix.retrieve_epoch_checkpoint(
                        fd, 'epoch.a4ucp', actual_envelope=kw['run_envelope'],
                        expected_checkpoint_digest=receipt['checkpoint_digest'],
                        bundle_sources=trusted_sources())
                    # Valid publication AND retrieval precede the collision test.
                    self.assert_tree_equal(saved, restarted.capture_epoch_checkpoint())
                    with self.assertRaisesRegex(
                            CheckpointSecurityError, 'filesystem operation refused or failed') as collision:
                        prefix.publish_epoch_checkpoint(fd, 'epoch.a4ucp')
                    self.assertIsInstance(collision.exception.__cause__, OSError)
                    self.assertEqual(collision.exception.__cause__.errno, errno.EEXIST)
                    retained = prefix.retrieve_epoch_checkpoint(
                        fd, 'epoch.a4ucp', actual_envelope=kw['run_envelope'],
                        expected_checkpoint_digest=receipt['checkpoint_digest'],
                        bundle_sources=trusted_sources())
                    self.assert_tree_equal(saved, retained.capture_epoch_checkpoint())
                    self.assertIsNot(restarted.model, prefix.model)
                    self.assertIsNot(restarted.optimizer, prefix.optimizer)
                    self.assertIs(type(restarted.optimizer), HCPUAdam)
                    a = PrivateRNG(prefix.data.identity, prefix.d2['expected_runtime'])
                    b = PrivateRNG(restarted.data.identity, restarted.d2['expected_runtime'])
                    a.load_validated(saved['rng'])
                    b.load_validated(restarted.rng.capture())
                    for gid in a.objects:
                        self.assertEqual([a.index(gid, 17) for _ in range(5)],
                                         [b.index(gid, 17) for _ in range(5)])
                    actual_next_epoch = restarted.train_epoch()
                    self.assertEqual(expected_next_epoch, actual_next_epoch)
                    self.assert_tree_equal(uninterrupted.capture_epoch_checkpoint(),
                                           restarted.capture_epoch_checkpoint())
                    self.assert_tree_equal(uninterrupted.selected_validation(),
                                           restarted.selected_validation())
                    snap = restarted.capture_epoch_checkpoint()
                    bindings = snap['optimizer']['parameter_bindings']
                    self.assertTrue(all(0 <= p['updates'] <= 4 for p in bindings))
                    if hetero:
                        inactive = [p for p in bindings if p['updates'] == 0]
                        self.assertTrue(inactive)
                        self.assertTrue(all(p['id'] not in snap['optimizer']['state_dict']['state']
                                            for p in inactive))
                    self.assertEqual(snap['progress']['epoch_optimizer_steps'], [2, 2])
                    self.assertEqual(snap['selection']['selected_epoch'], 2)
                    current = snap['current_model']['state_dict']['user_embeddings']
                    best = snap['best_model']['state_dict']['user_embeddings']
                    self.assertNotEqual(current.data_ptr(), best.data_ptr())
                    self.assertEqual(os.stat(os.path.join(directory, 'epoch.a4ucp')).st_mode & 0o777, 0o600)
                    self.assertLessEqual(sum(os.stat(os.path.join(directory, p)).st_size
                                             for p in os.listdir(directory)), 8388608)
                finally:
                    os.close(fd)

    @fail_on_cuda_route()
    def test_CPU_Adam_refuses_nonCPU_state_and_incompatible_modes_before_mutation(self):
        # No valid optimizer.step: all refusals precede arithmetic/lazy state.
        h, _, _, _ = new_harness()
        original = h.capture_epoch_checkpoint()
        triples = [('u1', 'p1', 'p3')]
        for option, value in (
            ('capturable', True), ('fused', True), ('differentiable', True),
            ('foreach', True),
        ):
            with self.subTest(option=option):
                h.optimizer.param_groups[0][option] = value
                before = copy.deepcopy(h.optimizer.state_dict())
                with patch.object(h.model, 'compute_batch_loss',
                                  side_effect=AssertionError('model called before mode refusal')):
                    with self.assertRaisesRegex(HarnessError, 'options/modes'):
                        h.backend.update(h.model, h.optimizer, triples)
                self.assert_tree_equal(before, h.optimizer.state_dict())
                h.optimizer.param_groups[0][option] = ADAM_OPTIONS[option]
                self.assert_tree_equal(original, h.capture_epoch_checkpoint())
        h.optimizer.defaults['differentiable'] = True
        with self.assertRaisesRegex(HarnessError, 'defaults/modes'):
            h.backend.update(h.model, h.optimizer, triples)
        h.optimizer.defaults['differentiable'] = False
        with patch('scripts.training_harness.torch.compiler.is_compiling', return_value=True):
            with self.assertRaisesRegex(HarnessError, 'eager execution'):
                h.backend.update(h.model, h.optimizer, triples)
        parameter = next(h.model.parameters())
        parameter.grad = torch.ones_like(parameter)
        old_gradient = parameter.grad
        meta_state = torch.empty((), device='meta')
        h.optimizer.state[parameter] = {'step': meta_state}
        with patch.object(h.model, 'compute_batch_loss',
                          side_effect=AssertionError('model called before state refusal')):
            with self.assertRaisesRegex(HarnessError, 'state tensor must remain CPU'):
                h.backend.update(h.model, h.optimizer, triples)
        self.assertIs(parameter.grad, old_gradient)
        self.assertIs(h.optimizer.state[parameter]['step'], meta_state)
        del h.optimizer.state[parameter]
        parameter.grad = None
        self.assert_tree_equal(original, h.capture_epoch_checkpoint())
        meta_parameter = torch.nn.Parameter(torch.empty(1, device='meta'))
        invalid = HCPUAdam([meta_parameter], **copy.deepcopy(ADAM_OPTIONS))
        with self.assertRaisesRegex(HarnessError, 'parameter must remain CPU'):
            invalid.step()
        self.assertEqual(len(invalid.state), 0)
        self.assertEqual(meta_parameter.device.type, 'meta')
        self.assertIsNone(meta_parameter.grad)
        self.assert_tree_equal(original, h.capture_epoch_checkpoint())

    def test_empty_pool_full_P_order_and_no_empty_updates(self):
        backend = RecordingBackend()
        h, _, _, _ = new_harness(empty_pool=True, backend=backend)
        result = h.train_epoch()
        self.assertEqual(result['empty_pool_skips'], 2)
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertEqual(set(result['source_order']), set(h.data.positive_pairs))
        self.assertEqual(len(result['source_order']), 5)
        self.assertEqual([len(batch) for batch in backend.updates], [2, 1])
        self.assertTrue(all(u != 'u3' for batch in backend.updates for u, _, _ in batch))
        snapshot = h.capture_epoch_checkpoint()
        self.assertEqual(snapshot['sampler']['cursor'], 5)
        self.assertEqual(snapshot['progress']['empty_pool_skips'], [2])
        before = owned_digest(snapshot)
        with self.assertRaisesRegex(HarnessError, 'empty optimizer update'):
            h.backend.update(h.model, h.optimizer, [])
        self.assertEqual(before, owned_digest(h.capture_epoch_checkpoint()))

    def test_sampler_independent_pool_index_and_private_alias_controls(self):
        h, _, _, _ = new_harness(empty_pool=True)
        calls = []

        class LastIndex:
            def choice(self, pool):
                calls.append(tuple(pool))
                return pool[-1]
        triples, skipped = h.sampler.prepare_batch(
            [('u2', 'p3'), ('u3', 'p1'), ('u1', 'p1')], LastIndex())
        self.assertEqual(calls, [('p1', 'p2'), ('p3',)])
        self.assertEqual(triples, [('u2', 'p3', 'p2'), ('u1', 'p1', 'p3')])
        self.assertEqual(skipped, 1)
        with self.assertRaisesRegex(ValueError, 'condition-positive training history'):
            h.sampler.prepare_batch([('u1', 'p3')], LastIndex())
        roles = h.d2['expected_runtime']['rng_role_bindings']
        self.assertEqual(roles['initialization'], roles['PyTorch_CPU'])
        self.assertIs(h.rng.objects[roles['initialization'][0]],
                      h.rng.objects[roles['PyTorch_CPU'][0]])
        payload = h.capture_epoch_checkpoint()
        validate_checkpoint_payload(payload, **h.d2)
        negative = copy.deepcopy(payload)
        negative['rng']['roles'][0]['generator_ids'] = []
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(negative, **h.d2)
        self.assert_tree_equal(payload, h.capture_epoch_checkpoint())

    def test_holdout_values_do_not_change_actual_updates(self):
        original, _, _, _ = new_harness()
        mutated, _, _, _ = new_harness(heldout_mutation=True)
        a, b = original.train_epoch(), mutated.train_epoch()
        self.assertEqual(a, b)
        sa, sb = original.capture_epoch_checkpoint(), mutated.capture_epoch_checkpoint()
        for field in ('current_model', 'optimizer', 'rng', 'progress'):
            self.assert_tree_equal(sa[field], sb[field])
        self.assertEqual(sa['sampler']['source_order'], sb['sampler']['source_order'])
        self.assertNotEqual(sa['identity']['identity_hash'], sb['identity']['identity_hash'])

    def test_graph_score_and_loss_composition_both_models(self):
        for hetero in (False, True):
            with self.subTest(hetero=hetero):
                h, _, _, _ = new_harness(hetero=hetero)
                model = h.model
                users, positive, negative = ['u1', 'u2'], ['p1', 'p3'], ['p3', 'p1']
                positive_scores, negative_scores = model.compute_batch_scores(users, positive, negative)
                if hetero:
                    masked = model.forward(mask_pairs={('u1', 'p1'), ('u2', 'p3')})
                    indices = model.graph.node_to_idx
                    ui = [indices['User'][u] for u in users]
                    pi = [indices['Product'][p] for p in positive]
                    ni = [indices['Product'][p] for p in negative]
                    torch.testing.assert_close(positive_scores, (masked['User'][ui] * masked['Product'][pi]).sum(1))
                    torch.testing.assert_close(negative_scores, (masked['User'][ui] * masked['Product'][ni]).sum(1))
                    before = model.forward()
                    model.compute_batch_scores(users, positive, negative)
                    after = model.forward()
                    for kind in before:
                        torch.testing.assert_close(before[kind], after[kind], rtol=0, atol=0)
                else:
                    user_repr, item_repr = model.forward()
                    torch.testing.assert_close(positive_scores,
                                               (user_repr[[0, 1]] * item_repr[[0, 2]]).sum(1))
                    adj = model.adj_user_item.to_dense()
                    x, y = model.user_embeddings, model.item_embeddings
                    xs, ys = [x], [y]
                    for _ in range(3):
                        x, y = adj @ y, adj.t() @ x
                        xs.append(x)
                        ys.append(y)
                    torch.testing.assert_close(user_repr, torch.stack(xs).mean(0))
                    torch.testing.assert_close(item_repr, torch.stack(ys).mean(0))
                    self.assertGreater(adj[0, 0].item(), 0)
                losses = model.compute_batch_loss(users, positive, negative)
                oracle = torch.nn.functional.softplus(negative_scores - positive_scores).mean()
                torch.testing.assert_close(losses['ranking_loss'], oracle)
                torch.testing.assert_close(losses['total_loss'], losses['ranking_loss'] + losses['reg_loss'])

    def test_selection_ties_patience_maximum_and_failure_atomicity(self):
        selection = Selection(2)
        for epoch in range(1, 7):
            selection.advance(epoch, {2: 0.4, 4: 0.5, 6: 0.5}.get(epoch))
        self.assertEqual(selection.state['selected_epoch'], 4)
        self.assertEqual(selection.state['bad_checks'], 1)
        saved = copy.deepcopy(selection.state)
        selection.advance(7)
        for score in (None, float('nan'), 1.1, True):
            with self.assertRaisesRegex(HarnessError, 'finite eligible NDCG'):
                selection.advance(8, score)
            self.assertEqual(selection.state, saved)
            self.assertEqual(selection.epoch, 7)
        selection.advance(8, 0.5)
        for epoch in range(9, 25):
            selection.advance(epoch, 0.5 if epoch % 2 == 0 else None)
        self.assertEqual(selection.state['stop_reason'], 'patience')
        self.assertEqual(selection.epoch, 24)
        with self.assertRaisesRegex(HarnessError, 'terminal selection advance'):
            selection.advance(25)
        maximum = Selection(2)
        for epoch in range(1, 101):
            maximum.advance(epoch, float(epoch) / 200 if epoch % 2 == 0 else None)
        self.assertEqual(maximum.state['stop_reason'], 'max_epochs')
        both = Selection(2)
        for epoch in range(1, 101):
            both.advance(epoch, min(epoch, 80) / 200.0 if epoch % 2 == 0 else None)
        self.assertEqual(both.state['stop_reason'], 'patience_and_max_epochs')
        self.assertEqual(both.state['selected_epoch'], 80)

    def test_wrong_bundle_runtime_digest_and_late_restore_leave_live_unchanged(self):
        h, _, _, kw = new_harness()
        h.train_epoch()
        snapshot = h.capture_epoch_checkpoint()
        digest = owned_digest(snapshot)
        slot = LiveSlot(h)
        model_id, optimizer_id = id(h.model), id(h.optimizer)
        options = dict(actual_envelope=kw['run_envelope'], expected_checkpoint_digest=digest,
                       bundle_sources=trusted_sources())
        valid = h.restore_epoch_checkpoint(snapshot, **options)
        self.assert_tree_equal(snapshot, valid.capture_epoch_checkpoint())
        bad_sources = trusted_sources()
        bad_sources['scripts/training_harness.py'] += b'# altered\n'
        bad_runtime = copy.deepcopy(snapshot)
        bad_runtime['runtime_profile']['torch_version'] = 'wrong'
        bad_rng = copy.deepcopy(snapshot)
        bad_rng['rng']['generators'][-1]['state'] = torch.zeros(7, dtype=torch.uint8)
        bad_moment = copy.deepcopy(snapshot)
        bad_moment['optimizer']['state_dict']['state'][0]['exp_avg_sq'].fill_(-1)
        cases = [
            ('external closure', snapshot, {'bundle_sources': bad_sources}, HarnessError),
            ('independent digest', snapshot, {'expected_checkpoint_digest': '0' * 64}, HarnessError),
            ('D2 runtime binding', bad_runtime, {}, CheckpointValidationError),
            ('D2 native RNG extent', bad_rng, {}, CheckpointValidationError),
            ('D2 nonnegative Adam moment', bad_moment, {}, CheckpointValidationError),
        ]
        for reason, payload, changes, error_family in cases:
            with self.subTest(reason=reason):
                # D2 defects are independently refused by D2 itself, before the
                # downstream digest/native-transfer guards can witness anything.
                if error_family is CheckpointValidationError:
                    with self.assertRaises(CheckpointValidationError):
                        validate_checkpoint_payload(payload, **h.d2)
                with ExitStack() as stack:
                    for method in ('make_model', 'make_optimizer', 'load_model', 'load_optimizer'):
                        stack.enter_context(patch.object(
                            h.backend, method, side_effect=AssertionError('premature native stage')))
                    stack.enter_context(patch.object(
                        PrivateRNG, 'load_validated', side_effect=AssertionError('premature RNG setter')))
                    with self.assertRaises(error_family):
                        slot.restore(payload, **dict(options, **changes))
                self.assertIs(slot.live, h)
                self.assertEqual((id(h.model), id(h.optimizer)), (model_id, optimizer_id))
                self.assertEqual(owned_digest(h.capture_epoch_checkpoint()), digest)
        original_load = h.backend.load_optimizer

        def fail_after_native_load(optimizer, state):
            original_load(optimizer, state)
            raise HarnessError('injected late staged native transfer failure')
        with patch.object(h.backend, 'load_optimizer', side_effect=fail_after_native_load):
            with self.assertRaisesRegex(HarnessError, 'injected late staged native transfer failure'):
                slot.restore(snapshot, **options)
        self.assertIs(slot.live, h)
        self.assertEqual(owned_digest(h.capture_epoch_checkpoint()), digest)
        replacement = slot.restore(snapshot, **options)
        self.assertIs(slot.live, replacement)
        self.assertIsNot(slot.live, h)

    @fail_on_cuda_route()
    def test_external_closure_D2_missing_or_changed_refuses_before_native_stages(self):
        self.assertEqual(set(BUNDLE_PATHS), set(EXPECTED_CLOSURE_PATHS))
        control = trusted_sources()
        expected_manifest = expected_source_manifest()
        self.assertEqual(bundle_manifest(control), expected_manifest)
        backend = RecordingBackend()
        data, expected, kw = make_case(backend=backend)
        # Complete initial capture and staged native restoration are valid controls;
        # neither performs an update. All expectations remain separately retained.
        h = Harness(data, expected, **kw)
        snapshot = h.capture_epoch_checkpoint()
        options = dict(actual_envelope=kw['run_envelope'],
                       expected_checkpoint_digest=owned_digest(snapshot), bundle_sources=control)
        restored = h.restore_epoch_checkpoint(snapshot, **options)
        self.assert_tree_equal(snapshot, restored.capture_epoch_checkpoint())
        self.assertEqual(backend.updates, [])
        target = 'scripts/checkpoint_payload.py'
        missing = dict(control)
        del missing[target]
        changed = dict(control)
        changed[target] = b'!' + changed[target][1:]
        self.assertEqual(len(changed[target]), len(control[target]))
        for name, sources in [('missing D2', missing), ('changed D2', changed)]:
            with self.subTest(name=name):
                for path in control:
                    if path != target:
                        self.assertEqual(sources[path], control[path])
                self.assertEqual(kw['expected_bundle'], expected_manifest)
                self.assertEqual(kw['run_envelope'], kw['expected_run_envelope'])
                before_builds = backend.builds
                with ExitStack() as stack:
                    for method in ('make_model', 'make_optimizer', 'update', 'load_model', 'load_optimizer'):
                        stack.enter_context(patch.object(
                            backend, method, side_effect=AssertionError('closure checked too late')))
                    stack.enter_context(patch.object(
                        PrivateRNG, 'load_validated', side_effect=AssertionError('closure checked after RNG setter')))
                    with self.assertRaisesRegex(HarnessError, 'bundle|closure'):
                        Harness(data, expected, **dict(kw, bundle_sources=sources))
                    with self.assertRaisesRegex(HarnessError, 'bundle|closure'):
                        h.restore_epoch_checkpoint(snapshot, **dict(options, bundle_sources=sources))
                self.assertEqual(backend.builds, before_builds)
                self.assertEqual(backend.updates, [])
                self.assert_tree_equal(snapshot, h.capture_epoch_checkpoint())
        # A self-recomputed manifest does not override the independent expectation.
        self_carried = bundle_manifest(changed)
        self.assertNotEqual(self_carried, expected_manifest)
        with self.assertRaisesRegex(HarnessError, 'independent expected closure'):
            Harness(data, expected, **dict(kw, bundle_sources=changed))
        # Even replacing both presented manifests cannot replace the retained run
        # envelope's independently expected closure digest.
        with self.assertRaisesRegex(HarnessError, 'run/bundle binding mismatch'):
            Harness(data, expected, **dict(kw, bundle_sources=changed, expected_bundle=self_carried))

    def test_device_and_initialization_fail_before_update(self):
        backend = RecordingBackend()
        data, expected, kw = make_case(backend=backend)
        for device, accelerator in [('cuda', 'gpu'), ('cpu', 'gpu'), ('tpu', 'tpu')]:
            with self.subTest(device=device), self.assertRaisesRegex(HarnessError, 'unsupported device/accelerator'):
                Harness(data, expected, **dict(kw, requested_device=device, accelerator=accelerator))
        self.assertEqual(backend.builds, 0)
        self.assertEqual(backend.updates, [])
        bad = copy.deepcopy(expected)
        bad['expected_runtime']['runtime_profile']['numpy_version'] = 'wrong'
        with self.assertRaisesRegex(HarnessError, 'mismatched CPU fixture runtime'):
            Harness(data, bad, **kw)
        self.assertEqual(backend.builds, 0)
        with self.assertRaisesRegex(HarnessError, 'explicit owned fixture tensors'):
            Harness(data, expected, **dict(kw, light_initial=None))
        wrong = copy.deepcopy(kw['light_initial'])
        wrong['user_embeddings'].add_(0.01)
        with self.assertRaisesRegex(HarnessError, 'initialization archive binding mismatch'):
            Harness(data, expected, **dict(kw, light_initial=wrong))
        self.assertEqual(backend.updates, [])

        class Unsupported(CPUBackend):
            def capabilities(self):
                return {'device': 'cpu', 'accelerator': 'none', 'supports_training': False}
        with self.assertRaisesRegex(HarnessError, 'unsupported device/accelerator'):
            Harness(data, expected, **dict(kw, backend=Unsupported()))
        valid = Harness(data, expected, **kw)
        self.assertEqual(valid.progress['global_step'], 0)

    def test_authoritative_reports_fallback_baselines_and_all_NA_refusal(self):
        h, data, _, _ = new_harness()
        direct = Evaluator(data.context).evaluate(data.validation_records, h.model, split='validation')
        self.assert_tree_equal(h.backend.evaluate(data, h.model), direct)
        self.assertEqual(set(direct['overall_metrics']), {
            'NDCG@5', 'NDCG@10', 'NDCG@20', 'HitRate@5', 'HitRate@10',
            'HitRate@20', 'CatalogCoverage@10'})
        self.assertEqual(direct['audit']['eligible_users'], 2)
        self.assertEqual({r['user_id'] for r in direct['per_user_results']}, {'u1', 'u2'})
        # Unknown users are excluded, not an eligible empty-history fallback.
        with patch.object(h.model, 'score_candidates',
                          side_effect=AssertionError('excluded user reached scorer')):
            cold = Evaluator(data.context).evaluate(
                [('cold-user', 'p1', 5.0)], h.model, split='validation')
        self.assertEqual(cold['audit']['eligible_users'], 0)
        self.assertEqual(cold['per_user_results'], [])
        with self.assertRaisesRegex(HarnessError, 'eligible authoritative validation report'):
            validation_score(cold)
        unavailable = Evaluator(data.context).evaluate([('u1', 'cold-item', 5.0)], h.model, split='validation')
        with self.assertRaisesRegex(HarnessError, 'eligible authoritative validation report'):
            validation_score(unavailable)
        popularity = baseline_validation(data, 'CorePopularity')
        expected_pop = Evaluator(data.context).evaluate(
            data.validation_records, PopularityScorer(data.context), split='validation')
        self.assert_tree_equal(popularity['validation_report'], expected_pop)
        mixture, trials = select_kg_only_mixture(
            data.context, data.product_categories, data.product_brands, data.validation_records)
        outcome = baseline_validation(data, 'KG-only')
        self.assertEqual(outcome['mixture'], mixture)
        self.assert_tree_equal(outcome['validation_report'], trials[mixture])
        tie, tie_trials = select_kg_only_mixture(data.context, {}, {}, data.validation_records)
        self.assertEqual(tie, KGOnlyScorer.GRID[0])
        self.assertEqual(len(tie_trials), 3)
        self.assertEqual(h.optimizer.state_dict()['state'], {})

    def test_known_low_rating_only_user_has_evaluator_owned_fallback(self):
        # The known user has ONLY low-rated TRAIN history under rating_ge_4.
        # The relevant target is warm through other users and unobserved by this
        # user. No extra positive is manufactured for the empty-history user.
        context = bounded_training_context([
            {'user_id': 'empty', 'parent_asin': 'low', 'rating': 2.0},
            {'user_id': 'a', 'parent_asin': 'p1', 'rating': 5.0},
            {'user_id': 'a', 'parent_asin': 'p2', 'rating': 5.0},
            {'user_id': 'b', 'parent_asin': 'p2', 'rating': 4.0},
        ], 'rating_ge_4', _make_synthetic_resource_policy())
        self.assertIn('empty', context.users)
        self.assertEqual(context.all_rated_history['empty'], {'low'})
        self.assertEqual(context.positive_history.get('empty', set()), set())
        self.assertEqual(context.sorted_warm_items, ['p1', 'p2'])
        self.assertNotIn('p1', context.all_rated_history['empty'])

        class MustNotScore:
            def score_candidates(self, *args):
                raise AssertionError('evaluator-owned fallback called provider')
        report = Evaluator(context).evaluate(
            [('empty', 'p1', 5.0)], MustNotScore(), split='validation')
        self.assertEqual(report['audit']['eligible_users'], 1)
        self.assertEqual(len(report['per_user_results']), 1)
        row = report['per_user_results'][0]
        self.assertEqual(row['user_id'], 'empty')
        self.assertEqual(row['route'], 'fallback')
        self.assertEqual(row['fallback_reason'], 'empty_positive_history')
        self.assertEqual(row['candidate_count'], 2)
        self.assertEqual(row['top_10'], ['p2', 'p1'])
        self.assertEqual(row['metrics']['HitRate@10'], 1.0)
        self.assertAlmostEqual(validation_score(report), 1.0 / np.log2(3.0))

    def test_entrypoint_authority_modes_and_versioned_result(self):
        data, expected, kw = make_case()
        spec, authority = spec_and_authority()
        validate_run_spec(spec, expected_spec=copy.deepcopy(spec), authorization=authority,
                          resource_policy=expected['resource_policy'], backend=CPUBackend())
        for change in [
            {'mode': 'profiling'}, {'data_kind': 'real_data'}, {'stage': 'SFT'},
            {'stage': 'final_test'}, {'device': 'cuda'}, {'test_path': 'forbidden'},
            {'epochs': True}, {'max_adam_steps': 5},
        ]:
            bad = dict(spec, **change)
            with self.subTest(change=change), self.assertRaises(HarnessError):
                validate_run_spec(bad, expected_spec=bad, authorization=authority,
                                  resource_policy=expected['resource_policy'], backend=CPUBackend())
        with self.assertRaisesRegex(HarnessError, 'explicit independent execution authorization'):
            validate_run_spec(spec, expected_spec=spec, authorization={},
                              resource_policy=expected['resource_policy'], backend=CPUBackend())
        call = {key: kw[key] for key in (
            'run_envelope', 'expected_run_envelope', 'bundle_sources',
            'expected_bundle', 'light_initial', 'backend')}
        result = execute_fixture(spec, expected_spec=copy.deepcopy(spec), authorization=authority,
                                 data=data, expectations=expected, **call)
        self.assertEqual(result['schema_version'], 'amazon4u-H-result-v1')
        self.assertEqual(result['status'], 'fixture_budget_complete')
        self.assertEqual(result['actual_Adam_steps'], 4)
        self.assertEqual(result['selection']['selected_epoch'], 2)
        self.assertEqual(result['deployment_decision'], 'NOT_AUTHORIZED')
        self.assertEqual(result['test_execution'], 'not_authorized')
        self.assertTrue(result['remaining_readiness_gates'])
        for method in ('CorePopularity', 'KG-only'):
            baseline_spec, baseline_authority = spec_and_authority(method)
            envelope = dict(kw['run_envelope'], method=method)
            baseline_call = dict(call, light_initial=None, run_envelope=copy.deepcopy(envelope),
                                 expected_run_envelope=envelope)
            # Patch the actual adapter and its factory, not just the base-class
            # symbol from which a previously defined subclass already inherited.
            with ExitStack() as stack:
                for target in (
                    'scripts.training_harness.CPUBackend.make_optimizer',
                    'scripts.training_harness.HCPUAdam',
                    'scripts.training_harness.torch.optim.Adam',
                    'scripts.training_harness.CPUBackend.update',
                ):
                    stack.enter_context(patch(target, side_effect=AssertionError('baseline optimizer route')))
                baseline = execute_fixture(
                    baseline_spec, expected_spec=copy.deepcopy(baseline_spec),
                    authorization=baseline_authority, data=data, expectations=expected,
                    **baseline_call)
            self.assertEqual(baseline['actual_Adam_steps'], 0)
        with self.assertRaisesRegex(HarnessError, 'CLI/host input acquisition'):
            main()

    def test_epoch_budget_failure_and_evaluator_failure_are_truthful(self):
        h, _, _, _ = new_harness()
        h.train_epoch()
        h.train_epoch()
        before = owned_digest(h.capture_epoch_checkpoint())
        with self.assertRaisesRegex(HarnessError, 'exceed the fixture Adam budget'):
            h.train_epoch()
        self.assertEqual(before, owned_digest(h.capture_epoch_checkpoint()))
        failed, _, _, _ = new_harness()
        failed.selection.frequency = 1
        with patch.object(failed.backend, 'evaluate', side_effect=RuntimeError('scoring failure')):
            with self.assertRaisesRegex(RuntimeError, 'scoring failure'):
                failed.train_epoch()
        self.assertTrue(failed.poisoned)
        self.assertEqual(failed.selection.state['history'], [])
        with self.assertRaisesRegex(HarnessError, 'successful quiescent epoch boundaries'):
            failed.capture_epoch_checkpoint()

    def test_D3_registered_source_caller_private_modes_and_no_D1_extension(self):
        h, _, expected, _ = new_harness()
        # This deliberately partial D1 packet tests only the accepted registered
        # acquisition API. It is NOT the external project runtime closure.
        sources = {'scripts/checkpoint_identity.py': b'# checkpoint identity stub\n',
                   'configs/compute-plan.json': b'{}\n'}
        provenance = expected['expected_provenance']
        environment = {
            'python_version': h.d2['expected_runtime']['runtime_profile']['python_version'],
            'platform_system': 'Linux', 'platform_release': '6.0',
            'platform_machine': 'x86_64', 'cpu_count': 4,
            'torch_version': str(torch.__version__), 'library_versions': {'numpy': np.__version__},
        }
        git = {'commit': 'a' * 40, 'dirty': False, 'tracked_diff': '', 'untracked_sources': []}
        with tempfile.TemporaryDirectory(dir=tempfile.tempdir) as directory:
            os.chmod(directory, 0o700)
            roots = {}
            try:
                for prefix in ('scripts', 'configs'):
                    location = os.path.join(directory, prefix)
                    os.mkdir(location, 0o700)
                    roots[prefix] = os.open(location, os.O_RDONLY | os.O_DIRECTORY)
                for path, content in sources.items():
                    prefix, name = path.split('/')
                    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                                 dir_fd=roots[prefix])
                    try:
                        self.assertEqual(os.write(fd, content), len(content))
                        self.assertEqual(os.fstat(fd).st_mode & 0o777, 0o600)
                    finally:
                        os.close(fd)
                actual = h.acquire_registered_sources(
                    roots, {p: len(v) for p, v in sources.items()},
                    git_evidence=git, environment_records=environment)
                self.assertEqual(actual, provenance)
                with self.assertRaisesRegex(CheckpointSecurityError, 'not in allowlisted source/config set'):
                    h.acquire_registered_sources(
                        roots, {'scripts/training_harness.py': 0},
                        git_evidence=git, environment_records=environment)
                self.assertEqual(len(os.listdir(directory)), 2)
            finally:
                for fd in roots.values():
                    os.close(fd)
