"""Time one real-data checkpoint publish or retrieve (readiness plan, steps 2 to 5).

`publish` builds the expectations and the epoch-boundary payload after `--epochs` training
epochs, then publishes it through the GR06 cuda-v1 path into `--dir`. `retrieve` runs in
a new process, as a resuming run would: it rebuilds the expectations and retrieves the
file. Each prints one JSON record with phase times, the process peak RSS after each phase
and, for publish, the file size and SHA-256 (two publishes of one payload must agree).
No weights are kept: the published file lives in a caller-owned scratch directory.

Usage: python -m scripts.checkpoint_timing publish|retrieve --data <dir> --manifest-sha256 <hex>
           --method LightGCN --condition all_ratings --dir <private dir> [--epochs 0]
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import time

import torch

from scripts.checkpoint_real_data import METHODS, build
from scripts.checkpoint_storage_cuda import publish_cuda_checkpoint, retrieve_cuda_checkpoint
from scripts.gpu_runtime import environment_fingerprint, pin_global_generators
from scripts.profiling_data import CONDITIONS, ProfilingData
from scripts.training_harness import D2_KEYS

FILE_NAME = 'readiness.a4ucp'
RUN_BINDING = {'bundle_sha256': '0' * 64, 'run_spec_sha256': '0' * 64}


def _peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('publish', 'retrieve'))
    ap.add_argument('--data', required=True)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--method', choices=METHODS, required=True)
    ap.add_argument('--condition', choices=CONDITIONS, required=True)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--epochs', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args(argv)

    record = {'action': args.action, 'method': args.method, 'condition': args.condition,
              'epochs': args.epochs, 'phases': []}

    def phase(name, start):
        record['phases'].append({'name': name, 'seconds': round(time.perf_counter() - start, 2),
                                 'peak_rss_mib': _peak_mib()})
        print(json.dumps(record['phases'][-1]), file=sys.stderr, flush=True)

    pin_global_generators()
    device = torch.device(args.device)
    start = time.perf_counter()
    data = ProfilingData(args.data, args.manifest_sha256)
    phase('load', start)
    start = time.perf_counter()
    trainer, expectations, identity = build(data, args.method, args.condition, device=device,
                                            fingerprint=environment_fingerprint(device))
    del data
    phase('build_expectations', start)
    d2 = {k: expectations[k] for k in D2_KEYS}
    expected_device = {'fingerprint': trainer.fingerprint, 'determinism': None,
                       'run_binding': RUN_BINDING}
    root = os.open(args.dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        if args.action == 'publish':
            for _ in range(args.epochs):
                start = time.perf_counter()
                trainer.train_epoch()
                phase('train_epoch', start)
            start = time.perf_counter()
            payload = trainer.cuda_payload(expectations, identity, RUN_BINDING)
            phase('build_payload', start)
            start = time.perf_counter()
            publish_cuda_checkpoint(root, FILE_NAME, payload, expected_device=expected_device,
                                    **d2, io_policy=expectations['io_policy'])
            phase('publish', start)
            path = os.path.join(args.dir, FILE_NAME)
            with open(path, 'rb') as f:
                record['file_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
            record['file_bytes'] = os.path.getsize(path)
        else:
            start = time.perf_counter()
            owned = retrieve_cuda_checkpoint(root, FILE_NAME, expected_device=expected_device,
                                             **d2, io_policy=expectations['io_policy'])
            phase('retrieve', start)
            record['retrieved_epoch'] = owned['progress']['epoch']
    finally:
        os.close(root)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
