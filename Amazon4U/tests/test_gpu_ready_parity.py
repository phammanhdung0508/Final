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
        self.assertEqual(len(self.result['restart']), 6)
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
        for bad in tampered:
            with self.assertRaises(ValueError):
                validate_result(bad)

    def test_a_failed_check_turns_the_verdict(self):
        with patch.object(gpu_ready_parity, 'compare',
                          return_value={'max_abs_diff': 1.0, 'passed': False}):
            result = run('cpu', 'none', {**PLAN, 'fixtures': {**PLAN['fixtures'], 'seeds': [42]}})
        self.assertFalse(result['all_checks_passed'])
        self.assertEqual(result['status'], 'parity_failed')
        validate_result(result)


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
