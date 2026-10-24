"""GPU-READY parity runner and cuda-v1 result schema, exercised on CPU."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import gpu_ready_parity
from scripts.gpu_ready_parity import GPU_READINESS_GATES, main, run, validate_result
from scripts.gpu_runtime import RuntimeRefusal, require_device
from scripts.training_harness import READINESS_GATES

PLAN = json.loads((Path(__file__).resolve().parents[1]
                   / 'configs/gpu-ready-phase2-run-plan.json').read_text())


class ParityRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run('cpu', 'none', PLAN)

    def test_cpu_run_passes_every_check_and_validates(self):
        self.assertTrue(self.result['all_checks_passed'], json.dumps(self.result, default=str)[:2000])
        self.assertEqual(len(self.result['parity']), 12)
        self.assertEqual([r.get('mode', 'same_process') for r in self.result['restart']],
                         ['same_process'] * 2 + ['fresh_process'] * 2)
        self.assertTrue(all(r['distinct_processes'] for r in self.result['restart'][2:]))
        self.assertEqual([p['probe'] for p in self.result['determinism_probes']], ['max_unpool1d'])
        validate_result(copy.deepcopy(self.result))

    def test_gate_list_keeps_frozen_v1_list_untouched(self):
        self.assertEqual(READINESS_GATES[0], 'GPU-READY separately authorized and verified')
        self.assertEqual(GPU_READINESS_GATES[1:], READINESS_GATES[1:])
        self.assertNotIn(READINESS_GATES[0], GPU_READINESS_GATES)

    def test_tampered_or_v1_results_are_refused(self):
        tampered = [
            {**self.result, 'schema_version': 'amazon4u-H-result-v1'},
            {**self.result, 'remaining_readiness_gates': GPU_READINESS_GATES[1:]},
            {**self.result, 'remaining_readiness_gates': list(READINESS_GATES)},
            {**self.result, 'deployment_decision': 'AUTHORIZED'},
            {**self.result, 'extra': 1},
        ]
        failed = copy.deepcopy(self.result)
        failed['parity'][0]['passed'] = False
        tampered.append(failed)
        failed_probe = copy.deepcopy(self.result)
        failed_probe['determinism_probes'][0]['passed'] = False
        tampered.append(failed_probe)
        for bad in tampered:
            with self.assertRaises(ValueError):
                validate_result(bad)

    def test_a_failed_check_turns_the_verdict(self):
        with patch.object(gpu_ready_parity, 'compare',
                          return_value={'max_abs_diff': 1.0, 'passed': False}), \
                patch.object(gpu_ready_parity, 'fresh_process_restart_case',
                             return_value={'passed': True}), \
                patch.object(gpu_ready_parity, 'determinism_probes', return_value=[]):
            result = run('cpu', 'none', {**PLAN, 'fixtures': {**PLAN['fixtures'], 'seeds': [42]}})
        self.assertFalse(result['all_checks_passed'])
        self.assertEqual(result['status'], 'parity_failed')
        validate_result(result)


class FreshProcessTests(unittest.TestCase):
    def test_new_processes_differ_until_the_global_generators_are_pinned(self):
        """The GR06 finding: PyTorch seeds the global CPU generator randomly per process."""
        import subprocess
        import sys
        code = ('import hashlib, torch{}; '
                'print(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())')
        def states(pin):
            src = code.format('; from scripts.gpu_runtime import pin_global_generators; '
                              'pin_global_generators()' if pin else '')
            return {subprocess.run([sys.executable, '-c', src], capture_output=True, text=True,
                                   check=True, cwd=gpu_ready_parity.ROOT).stdout for _ in range(2)}
        self.assertEqual(len(states(pin=False)), 2)
        self.assertEqual(len(states(pin=True)), 1)


class DeterminismProbeTests(unittest.TestCase):
    def verdict(self, policy, control):
        def child(*args, env=None):
            return policy if args[args.index('--policy') + 1] == 'on' and env is None else control
        with patch.object(gpu_ready_parity, '_child', side_effect=child):
            return gpu_ready_parity.determinism_probes('cpu')[0]['passed']

    def test_only_the_documented_refusal_with_a_working_control_passes(self):
        raised = {'raised': True, 'type': 'RuntimeError',
                  'message': 'op does not have a deterministic implementation'}
        self.assertTrue(self.verdict(raised, {'raised': False}))
        self.assertFalse(self.verdict({'raised': False}, {'raised': False}))
        self.assertFalse(self.verdict(raised, raised))
        self.assertFalse(self.verdict({**raised, 'message': 'out of memory'}, {'raised': False}))
        self.assertFalse(self.verdict({'child_error': 'exit 1'}, {'raised': False}))


class RefusalTests(unittest.TestCase):
    def test_unsupported_or_unavailable_devices_are_refused(self):
        for device, accelerator in (('cuda:1', 'gpu'), ('cpu', 'gpu'), ('mps', 'none')):
            with self.assertRaises(RuntimeRefusal):
                require_device(device, accelerator)
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeRefusal, 'no CPU fallback'):
                require_device('cuda:0', 'gpu')

    def test_main_requires_cublas_setting_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'result.json')
            Path(out).write_text('{}')
            with self.assertRaises(SystemExit):
                main(['--device', 'cpu', '--accelerator', 'none', '--plan', 'x', '--out', out])
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ''}):
                with self.assertRaisesRegex(RuntimeRefusal, 'CUBLAS'):
                    main(['--device', 'cpu', '--accelerator', 'none', '--out',
                          os.path.join(tmp, 'new.json'), '--plan',
                          str(Path(__file__).resolve().parents[1] / 'configs/gpu-ready-phase2-run-plan.json')])


if __name__ == '__main__':
    unittest.main()
