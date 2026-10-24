"""GR08: Kaggle CLI transport for GPU-READY runs.

Provides the calls the GPU-READY path makes: `push`, `status`, `open_log_reader` (read
by O-OFFLINE's `consume_kernel_logs`) and `download_outputs`. Each call is one subprocess
with an argv list, no shell and a timeout. Output that does not parse is an error, never a
retry. The CLI reads its own credentials; nothing here reads, prints or stores them.

Kaggle CLI 2.2.5 addresses a kernel version in `status` and `output` as
`<owner>/<slug>/<n>`. This transport always queries the version its own push
acknowledged, and answers only for kernels it pushed itself. Every GPU-READY run also
pushes to a fresh slug, so that version is 1. See docs/gpu-ready-kaggle-execution.md (GR08).
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import uuid
from pathlib import Path

from multiautoresearch.adapters.amazon4u import (
    MAX_CHUNK_BYTES, Amazon4UError, BoundedLogReader, SubmissionUnknownError,
)

CLI_VERSION = 'Kaggle CLI 2.2.5'
# Printed by `kernels status` as str(KernelWorkerStatus.<NAME>). CANCEL_REQUESTED and
# NEW_SCRIPT have no O-OFFLINE token, so they are errors rather than guesses.
STATUS_TOKENS = {
    'QUEUED': 'queued', 'RUNNING': 'running', 'COMPLETE': 'complete', 'ERROR': 'error',
    'CANCEL_ACKNOWLEDGED': 'cancelacknowledged',
}
MAX_LOG_BYTES = 4 * 1024 * 1024
MAX_MESSAGE_CHARS = 500
_PUSHED = re.compile(r'Kernel version (\d+) successfully pushed\.  Please check progress at (\S+)')
_REFUSED_PREFIXES = ('Kernel push error', 'The following are not valid')


def _message(out):
    return (out.stdout + out.stderr).strip()[:MAX_MESSAGE_CHARS]


class KaggleCliTransport:
    def __init__(self, executable, *, command_timeout=300, run=subprocess.run):
        self.executable = str(executable)
        self.command_timeout = command_timeout
        self._run = run
        self._versions: dict[str, int] = {}
        self._outputs: dict[str, Path] = {}

    def _call(self, *args):
        return self._run([self.executable, *args], capture_output=True, text=True,
                         timeout=self.command_timeout, stdin=subprocess.DEVNULL, check=False)

    def require_cli_version(self):
        out = self._call('--version')
        if out.returncode != 0 or out.stdout.strip() != CLI_VERSION:
            raise Amazon4UError(f'Kaggle CLI must be exactly {CLI_VERSION!r}; got {_message(out)!r}')

    def _known(self, kernel_ref):
        if kernel_ref not in self._versions:
            raise Amazon4UError(f'{kernel_ref} was not pushed by this transport')
        return self._versions[kernel_ref]

    def push(self, bundle_dir, *, timeout, accelerator, plan_digest):
        meta = json.loads((Path(bundle_dir) / 'kernel-metadata.json').read_text(encoding='utf-8'))
        owner, slug = meta['id'].split('/')
        try:
            out = self._call('kernels', 'push', '-p', str(bundle_dir), '-t', str(timeout),
                             '--accelerator', accelerator)
        except subprocess.TimeoutExpired as exc:
            raise SubmissionUnknownError('kernels push timed out after the request may have been sent') from exc
        lines = [line.strip() for line in out.stdout.splitlines()]
        if out.returncode != 0 or any(line.startswith(_REFUSED_PREFIXES) for line in lines):
            return {'returncode': out.returncode or 1, 'error': _message(out), 'acknowledged_version': None}
        pushed = [m for m in map(_PUSHED.fullmatch, lines) if m]
        if len(pushed) != 1:
            raise SubmissionUnknownError(f'push output has no single acknowledged version: {_message(out)!r}')
        version = int(pushed[0].group(1))
        ref = f'{owner}/{slug}'
        self._versions[ref] = version
        return {
            'returncode': 0, 'owner': owner, 'slug': slug, 'acknowledged_version': version,
            'version_ref': f'{ref}/v{version}', 'timeout': timeout, 'accelerator': accelerator,
            'plan_digest': plan_digest, 'url': pushed[0].group(2),
        }

    def status(self, kernel_ref, version=None):
        try:
            acknowledged = self._known(kernel_ref)
            versioned = f'{kernel_ref}/{acknowledged}'
            out = self._call('kernels', 'status', versioned)
        except (Amazon4UError, subprocess.TimeoutExpired) as exc:
            return {'returncode': 1, 'error': str(exc)[:MAX_MESSAGE_CHARS]}
        line = re.compile(rf'{re.escape(versioned)} has status "KernelWorkerStatus\.([A-Z_]+)"')
        found = [m.group(1) for m in map(line.fullmatch, map(str.strip, out.stdout.splitlines())) if m]
        if out.returncode != 0 or len(found) != 1 or found[0] not in STATUS_TOKENS:
            return {'returncode': 1, 'error': _message(out)}
        return {'returncode': 0, 'status': STATUS_TOKENS[found[0]], 'version': acknowledged,
                'kernel_ref': kernel_ref}

    def download_outputs(self, kernel_ref, parent):
        """Download into a new private directory under `parent`; return its path."""
        acknowledged = self._known(kernel_ref)
        if kernel_ref in self._outputs:
            raise Amazon4UError(f'outputs of {kernel_ref} were already downloaded')
        target = Path(parent) / f'.download-{uuid.uuid4().hex[:12]}'
        target.mkdir(mode=0o700)
        self._outputs[kernel_ref] = target
        try:
            out = self._call('kernels', 'output', f'{kernel_ref}/{acknowledged}', '-p', str(target), '-q')
        except subprocess.TimeoutExpired as exc:
            raise Amazon4UError(f'kernels output timed out; partial download kept at {target}') from exc
        if out.returncode != 0:
            raise Amazon4UError(f'kernels output failed ({_message(out)!r}); kept at {target}')
        return target

    def open_log_reader(self, kernel_ref, version=None, deadline=None, cancellation=None):
        try:
            acknowledged = self._known(kernel_ref)
            if kernel_ref not in self._outputs:
                raise Amazon4UError('the log is read from downloaded outputs; download first')
            path = self._outputs[kernel_ref] / f'{kernel_ref.split("/")[1]}.log'
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except (Amazon4UError, OSError) as exc:
            return {'returncode': 1, 'error': str(exc)[:MAX_MESSAGE_CHARS], 'reader': None}
        with os.fdopen(fd, 'rb') as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                return {'returncode': 1, 'error': 'log is not a regular file', 'reader': None}
            data = f.read(MAX_LOG_BYTES)
        chunks = [data[i:i + MAX_CHUNK_BYTES] for i in range(0, len(data), MAX_CHUNK_BYTES)]
        return {'returncode': 0, 'reader': BoundedLogReader(chunks), 'version': acknowledged,
                'kernel_ref': kernel_ref}
