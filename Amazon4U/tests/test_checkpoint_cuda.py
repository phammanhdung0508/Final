"""GR06: the cuda-v1 checkpoint profile over unchanged D1, D2 and D3 (CPU only).

CUDA parts are faked with recorded byte tensors and a CUDA-shaped fingerprint; real
CUDA restore is phase-2 run 3.
"""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from scripts import checkpoint_storage
from scripts.checkpoint_payload import CheckpointValidationError, validate_checkpoint_payload
from scripts.checkpoint_payload_cuda import (
    CAPABILITY, SCHEMA_VERSION, core_view, validate_cuda_checkpoint_payload,
)
from scripts.checkpoint_storage_cuda import publish_cuda_checkpoint, retrieve_cuda_checkpoint
from scripts.gpu_ready_parity import checkpoint_fixture, restart_case
from scripts.gpu_runtime import EXPECTED_DETERMINISM, RuntimeRefusal, environment_fingerprint
from scripts.training_harness import D2_KEYS, owned_digest

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device('cpu')
CPU_FP = environment_fingerprint(CPU)
BINDING = {'bundle_sha256': 'a' * 64, 'run_spec_sha256': 'b' * 64}
CUDA_FP = {**CPU_FP, 'device': 'cuda:0', 'cuda_runtime_version': '12.8', 'cudnn_version': 91002,
           'visible_device_count': 2, 'used_device_count': 1, 'gpu_name': 'Tesla T4',
           'compute_capability': [7, 5], 'driver_version': '560.35.03'}
METHODS = ('LightGCN', 'Heterogeneous KG+GNN')


def trained(method, epochs=2):
    trainer, expectations, identity = checkpoint_fixture(method, CPU, CPU_FP, None)
    for _ in range(epochs):
        trainer.train_epoch()
    return trainer, expectations, identity


def _policy():
    import sys
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_checkpoint_payload import _make_synthetic_resource_policy
    return _make_synthetic_resource_policy()


def d2(expectations):
    return {k: expectations[k] for k in D2_KEYS}


def expected_device(fingerprint=CPU_FP, determinism=None, binding=BINDING):
    return {'fingerprint': fingerprint, 'determinism': determinism, 'run_binding': binding}


def as_cuda(payload):
    """The same payload as a CUDA run would record it, with fake generator bytes."""
    fake = copy.deepcopy(payload)
    fake['amp']['reason'] = 'not_used_cuda_float32'
    fake['device'].update(fingerprint=copy.deepcopy(CUDA_FP), determinism=dict(EXPECTED_DETERMINISM))
    fake['device']['global_rng']['torch_cuda'] = [torch.arange(16, dtype=torch.uint8),
                                                  torch.arange(16, dtype=torch.uint8) + 1]
    return fake


class StorageCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.chmod(tmp.name, 0o700)
        self.fd = os.open(tmp.name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.addCleanup(os.close, self.fd)

    def publish(self, payload, expectations, name='c.a4ucp', **override):
        args = {'expected_device': expected_device(), **d2(expectations),
                'io_policy': expectations['io_policy'], **override}
        return publish_cuda_checkpoint(self.fd, name, payload, **args)

    def retrieve(self, expectations, name='c.a4ucp', **override):
        args = {'expected_device': expected_device(), **d2(expectations),
                'io_policy': expectations['io_policy'], **override}
        return retrieve_cuda_checkpoint(self.fd, name, **args)


class RoundTripTests(StorageCase):
    def test_publish_and_retrieve_round_trip_every_section(self):
        for method in METHODS:
            trainer, expectations, identity = trained(method)
            payload = trainer.cuda_payload(expectations, identity, BINDING)
            name = f'{method[:5]}.a4ucp'
            self.publish(payload, expectations, name)
            owned = self.retrieve(expectations, name)
            self.assertEqual(owned_digest(owned), owned_digest(payload), method)
            self.assertEqual(owned['schema_version'], SCHEMA_VERSION)
            self.assertEqual(owned['capability'], CAPABILITY)

    def test_restart_through_storage_is_bit_for_bit(self):
        for method in METHODS:
            result = restart_case(method, CPU, CPU_FP, None, BINDING)
            self.assertTrue(result['passed'], result)

    def test_cuda_shaped_payload_validates_with_matching_expectation(self):
        trainer, expectations, identity = trained('LightGCN')
        fake = as_cuda(trainer.cuda_payload(expectations, identity, BINDING))
        owned = validate_cuda_checkpoint_payload(
            fake, expected_device=expected_device(CUDA_FP, dict(EXPECTED_DETERMINISM)),
            **d2(expectations))
        self.assertEqual(len(owned['device']['global_rng']['torch_cuda']), 2)


class CrossProfileTests(StorageCase):
    def test_v1_reader_refuses_cuda_file_and_cuda_reader_refuses_v1_file(self):
        trainer, expectations, identity = trained('LightGCN')
        self.publish(trainer.cuda_payload(expectations, identity, BINDING), expectations)
        with self.assertRaises(CheckpointValidationError):
            checkpoint_storage.retrieve_checkpoint(self.fd, 'c.a4ucp', **d2(expectations),
                                                   io_policy=expectations['io_policy'])
        core = trainer.payload_core(expectations, identity)
        checkpoint_storage.publish_checkpoint(self.fd, 'v1.a4ucp', core, **d2(expectations),
                                              io_policy=expectations['io_policy'])
        with self.assertRaises(CheckpointValidationError):
            self.retrieve(expectations, 'v1.a4ucp')

    def test_core_view_is_exactly_a_valid_v1_payload(self):
        trainer, expectations, identity = trained('Heterogeneous KG+GNN')
        payload = trainer.cuda_payload(expectations, identity, BINDING)
        core = trainer.payload_core(expectations, identity)
        self.assertEqual(owned_digest(core_view(payload)), owned_digest(core))
        validate_checkpoint_payload(core, **d2(expectations))


class RefusalTests(StorageCase):
    def setUp(self):
        super().setUp()
        self.trainer, self.expectations, self.identity = trained('LightGCN')
        self.payload = self.trainer.cuda_payload(self.expectations, self.identity, BINDING)

    def refused(self, payload, device=None, **override):
        with self.assertRaises(CheckpointValidationError):
            validate_cuda_checkpoint_payload(
                payload, expected_device=device or expected_device(),
                **{**d2(self.expectations), **override})

    def test_each_fingerprint_field_change_is_refused(self):
        for key, value in CPU_FP.items():
            changed = copy.deepcopy(self.payload)
            changed['device']['fingerprint'][key] = f'changed-{value}'
            self.refused(changed)

    def test_run_binding_and_amp_and_profile_changes_are_refused(self):
        for mutate in (
            lambda p: p['device']['run_binding'].update(bundle_sha256='c' * 64),
            lambda p: p['amp'].update(reason='not_used_cuda_float32'),
            lambda p: p.update(capability='epoch_boundary_validation_only'),
            lambda p: p.update(schema_version='amazon4u-checkpoint-payload-v1'),
            lambda p: p['device'].update(extra=1),
            lambda p: p.pop('device'),
        ):
            changed = copy.deepcopy(self.payload)
            mutate(changed)
            self.refused(changed)

    def test_malformed_cuda_generator_records_are_refused(self):
        good = as_cuda(self.payload)
        cuda = expected_device(CUDA_FP, dict(EXPECTED_DETERMINISM))
        for bad in ([torch.zeros(16, dtype=torch.int64)] * 2,
                    [torch.zeros(16, dtype=torch.uint8)],
                    [torch.zeros(16, dtype=torch.uint8), torch.zeros(8, dtype=torch.uint8)],
                    [torch.zeros(16, dtype=torch.uint8, device='meta')] * 2,
                    [torch.zeros((4, 4), dtype=torch.uint8)] * 2):
            changed = copy.deepcopy(good) if bad[0].device.type != 'meta' else dict(good)
            changed['device'] = {**good['device'], 'global_rng': {**good['device']['global_rng'],
                                                                  'torch_cuda': bad}}
            self.refused(changed, cuda)
        missing = copy.deepcopy(good)
        del missing['device']['global_rng']['torch_cuda']
        self.refused(missing, cuda)

    def test_cuda_expectation_requires_the_fixed_determinism_policy(self):
        relaxed = {**EXPECTED_DETERMINISM, 'cudnn_benchmark': True}
        fake = as_cuda(self.payload)
        fake['device']['determinism'] = relaxed
        self.refused(fake, expected_device(CUDA_FP, relaxed))

    def test_non_cpu_tensor_in_the_core_is_refused(self):
        changed = dict(self.payload)
        model = dict(changed['current_model'])
        state = dict(model['state_dict'])
        state['user_embeddings'] = torch.zeros(state['user_embeddings'].shape, device='meta')
        model['state_dict'] = state
        changed['current_model'] = model
        self.refused(changed)

    def test_mid_epoch_sampler_state_is_refused(self):
        changed = copy.deepcopy(self.payload)
        changed['sampler']['batch_cursor'] = 1
        self.refused(changed)

    def test_changed_identity_is_refused_through_d1(self):
        other = copy.deepcopy(self.identity)
        other['seed'] = 43
        changed = copy.deepcopy(self.payload)
        changed['identity'] = other
        self.refused(changed)

    def test_oversized_file_is_refused_before_decoding(self):
        self.publish(self.payload, self.expectations)
        small = {**self.expectations['io_policy'], 'max_file_bytes': 64}
        with self.assertRaises(Exception) as caught:
            self.retrieve(self.expectations, io_policy=small)
        self.assertIsInstance(caught.exception, (CheckpointValidationError,
                                                 checkpoint_storage.CheckpointSecurityError))

    def test_publishing_to_an_existing_name_is_refused(self):
        self.publish(self.payload, self.expectations)
        with self.assertRaises(checkpoint_storage.CheckpointSecurityError):
            self.publish(self.payload, self.expectations)


class RestoreRefusalTests(unittest.TestCase):
    def setUp(self):
        trainer, expectations, identity = trained('LightGCN')
        self.owned = validate_cuda_checkpoint_payload(
            trainer.cuda_payload(expectations, identity, BINDING),
            expected_device=expected_device(), **d2(expectations))

    def fresh(self, fingerprint=CPU_FP):
        return checkpoint_fixture('LightGCN', CPU, fingerprint, None)[0]

    def test_restore_refuses_another_environment(self):
        with self.assertRaisesRegex(RuntimeRefusal, 'environment'):
            self.fresh({**CPU_FP, 'torch_version': 'other'}).restore_cuda(self.owned)

    def test_restore_refuses_changed_process_generators_without_setting_them(self):
        state = torch.get_rng_state()
        try:
            torch.rand(1)
            moved = torch.get_rng_state()
            with self.assertRaisesRegex(RuntimeRefusal, 'global generators'):
                self.fresh().restore_cuda(self.owned)
            self.assertTrue(torch.equal(torch.get_rng_state(), moved))
        finally:
            torch.set_rng_state(state)


class FixtureRowOrderTests(unittest.TestCase):
    def test_accepted_fixture_row_order_is_the_order_the_accepted_graph_builds(self):
        import sys
        sys.path.insert(0, str(ROOT / 'tests'))
        from test_checkpoint_payload import _make_valid_hetero_payload
        from scripts.differentiable_models import build_hetero_graph
        from scripts.training_data import TrainingData
        _, identity, _, model, _ = _make_valid_hetero_payload()
        data = TrainingData(copy.deepcopy(identity), expected_identity=identity,
                            resource_policy=_policy())
        graph = build_hetero_graph(data.context, data.product_categories, data.product_brands)
        self.assertEqual(model['row_order'], {k: list(v) for k, v in graph.node_ids.items()})


# Scoped user-approved correction (2026-10-24): CategoryPath row order in
# _make_valid_hetero_payload now equals the accepted graph order. The recorded O-OFFLINE
# hash is historical; this file is pinned to its corrected bytes instead.
AMENDED = {'tests/test_checkpoint_payload.py': '1c28cbdd0801c42bdf423650ce0ddf8ecd0fd5fe72e9c22ddad57547f214de97'}


class FrozenFileTests(unittest.TestCase):
    def test_d1_d2_d3_and_h_files_keep_their_recorded_hashes(self):
        recorded = json.loads((ROOT / 'configs/o-offline-implementation.json').read_text())

        def find(node):
            if isinstance(node, dict):
                if 'scripts/checkpoint_storage.py' in node:
                    return node
                for value in node.values():
                    hit = find(value)
                    if hit:
                        return hit
            return None
        hashes = find(recorded)
        for path in ('scripts/checkpoint_identity.py', 'scripts/checkpoint_payload.py',
                     'scripts/checkpoint_storage.py', 'scripts/training_harness.py',
                     'scripts/training_data.py', 'scripts/differentiable_models.py',
                     'tests/test_checkpoint_payload.py', 'tests/test_checkpoint_storage.py'):
            actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            self.assertEqual(actual, AMENDED.get(path, hashes[path]), path)


if __name__ == '__main__':
    unittest.main()
