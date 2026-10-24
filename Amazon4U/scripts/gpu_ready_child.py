"""GPU-READY child processes: fresh-process restart roles (GR06) and determinism probes (GR04).

Each invocation is a new Python process, started by `scripts/gpu_ready_parity.py`, and
writes one JSON object to `--out`. Nothing here is run directly by a user.

Restart roles share nothing but the checkpoint bytes in `--dir`:
  straight  trains all epochs and reports the final state;
  first     trains to the split epoch and publishes a cuda-v1 checkpoint (D3);
  resume    retrieves it in a new process, restores, finishes and reports.

A probe runs one operation PyTorch documents as throwing under
`use_deterministic_algorithms(True)`, with the policy on (`--policy on`, expected to raise)
or off (`--policy off`, the control, expected to succeed).
"""

import argparse
import json
import os
import sys

import torch

PROBES = {
    'max_unpool1d': 'any',
    'histc_cuda': 'cuda',
    'adaptive_avg_pool2d_backward_cuda': 'cuda',
    'put_accumulate_cuda': 'cuda',
    'cublas_matmul_without_workspace_config': 'cuda',
}


def probe(name, device):
    if name == 'max_unpool1d':
        torch.nn.MaxUnpool1d(2)(torch.tensor([[[1., 2.]]], device=device),
                                torch.tensor([[[1, 3]]], device=device))
    elif name == 'histc_cuda':
        torch.histc(torch.arange(8., device=device), bins=4)
    elif name == 'adaptive_avg_pool2d_backward_cuda':
        x = torch.ones(1, 1, 5, 5, device=device, requires_grad=True)
        torch.nn.functional.adaptive_avg_pool2d(x, 2).sum().backward()
    elif name == 'put_accumulate_cuda':
        torch.zeros(3, device=device).put_(torch.tensor([0, 0], device=device),
                                           torch.ones(2, device=device), accumulate=True)
    elif name == 'cublas_matmul_without_workspace_config':
        a = torch.ones(16, 16, device=device)
        a @ a
    if device.type == 'cuda':
        torch.cuda.synchronize()


def run_probe(name, device, policy):
    torch.use_deterministic_algorithms(policy == 'on')
    try:
        probe(name, device)
    except RuntimeError as exc:
        return {'raised': True, 'type': type(exc).__name__, 'message': str(exc)[:300]}
    return {'raised': False}


def run_role(role, method, device, directory, run_binding, determinism_on, epochs, split):
    from scripts.checkpoint_storage_cuda import publish_cuda_checkpoint, retrieve_cuda_checkpoint
    from scripts.gpu_ready_parity import checkpoint_fixture
    from scripts.gpu_runtime import configure_determinism, environment_fingerprint, pin_global_generators
    from scripts.training_harness import D2_KEYS, owned_digest

    pin_global_generators()
    determinism = configure_determinism() if determinism_on else None
    fingerprint = environment_fingerprint(device)
    trainer, expectations, identity = checkpoint_fixture(method, device, fingerprint, determinism)
    d2 = {k: expectations[k] for k in D2_KEYS}
    expected_device = {'fingerprint': fingerprint, 'determinism': determinism,
                       'run_binding': run_binding}
    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        if role == 'resume':
            owned = retrieve_cuda_checkpoint(root, 'restart.a4ucp', expected_device=expected_device,
                                             **d2, io_policy=expectations['io_policy'])
            trainer.restore_cuda(owned)
        todo = {'straight': epochs, 'first': split, 'resume': epochs - split}[role]
        logs = [trainer.train_epoch() for _ in range(todo)]
        payload = trainer.cuda_payload(expectations, identity, run_binding)
        if role == 'first':
            publish_cuda_checkpoint(root, 'restart.a4ucp', payload, expected_device=expected_device,
                                    **d2, io_policy=expectations['io_policy'])
    finally:
        os.close(root)
    return {'pid': os.getpid(), 'logs': logs, 'state_digest': owned_digest(payload),
            'validation_report': json.dumps(trainer.validation_report, sort_keys=True),
            'on_device': all(p.device == device for p in trainer.model.parameters())}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--probe', choices=sorted(PROBES))
    parser.add_argument('--policy', choices=('on', 'off'))
    parser.add_argument('--role', choices=('straight', 'first', 'resume'))
    parser.add_argument('--method')
    parser.add_argument('--dir')
    parser.add_argument('--run-binding')
    parser.add_argument('--determinism', choices=('on', 'off'))
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--split', type=int, default=2)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.probe:
        out = run_probe(args.probe, device, args.policy)
    else:
        out = run_role(args.role, args.method, device, args.dir, json.loads(args.run_binding),
                       args.determinism == 'on', args.epochs, args.split)
    with open(args.out, 'x') as f:
        json.dump(out, f, sort_keys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
