"""Readiness C2/D3: the faster wire encoder, parser and COO check match the accepted D3.

The accepted D3 is the pinned copy in tests/accepted_checkpoint/. Both sides get the
same inputs: random wire values under random tight IO policies (encode), resealed
bodies with random truncations and byte changes (decode), and random COO coordinates
(reconstruction). Each outcome is the bytes or value, or the exception class and message.
"""

import hashlib
import math
import random
import struct
import unittest

import torch

from checkpoint_dual import _canonical, accepted_storage
from scripts import checkpoint_storage as current

BASE = {
    'schema_version': 'amazon4u-checkpoint-io-v1', 'profile_id': 'equivalence',
    'max_file_bytes': 1 << 20, 'max_scalar_bytes': 4096, 'max_nodes': 4096, 'max_depth': 16,
    'max_storage_bytes': 1 << 16, 'max_copy_bytes': 1 << 16, 'max_tensor_elements': 4096,
    'max_dimension': 4096, 'max_rank': 4, 'max_source_bytes': 1 << 16, 'max_source_files': 4,
}


def _outcome(fn, *args):
    try:
        return ('ok', _canonical(fn(*args)))
    except Exception as exc:
        return ('raised', type(exc).__name__, str(exc))


def _seal(module, body):
    return module._MAGIC + struct.pack('>Q', len(body)) + hashlib.sha256(body).digest() + body


class _Text(str):
    pass


def _leaf(rng):
    return rng.choice([
        None, True, False, 0, -1, 7, -(2 ** 70), 2 ** 64, 2 ** 40000, 0.0, -0.0, 1.5, float('nan'),
        float('inf'), '', 'a', 'naïve', 'x' * rng.choice([10, 5000]), 'bad\ud800', b'', b'\x00\xff',
        b'z' * 5000, _Text('sub'), {1, 2},
        torch.arange(rng.randint(0, 6)).to(rng.choice([torch.float32, torch.int64, torch.uint8, torch.bool])),
        torch.ones(2, 3)[:, 1], torch.ones(2, requires_grad=True), torch.ones(3, dtype=torch.complex64),
        torch.sparse_coo_tensor(torch.tensor([[0, 1], [2, 0]]), torch.tensor([1.0, 2.0]), (2, 3)).coalesce(),
        torch.sparse_coo_tensor(torch.tensor([[1, 0]]), torch.tensor([1.0, 2.0]), (3,)),
    ])


def _value(rng, depth=0):
    if depth > rng.randint(1, 5) or rng.random() < 0.3:
        return _leaf(rng)
    kind = rng.choice(['dict', 'list', 'tuple'])
    children = [_value(rng, depth + 1) for _ in range(rng.randint(0, 6))]
    if kind == 'dict':
        keys = [rng.choice(['k', 'user_id', 1, 2.5, b'b', True, ('t', 1), ('t', [1]), None, f'k{i}'])
                for i in range(len(children))]
        try:
            return dict(zip(keys, children))
        except TypeError:
            return {f'k{i}': c for i, c in enumerate(children)}
    return children if kind == 'list' else tuple(children)


def _policy(rng):
    p = dict(BASE)
    for name in ('max_file_bytes', 'max_scalar_bytes', 'max_nodes', 'max_depth', 'max_storage_bytes',
                 'max_copy_bytes', 'max_tensor_elements', 'max_dimension', 'max_rank'):
        if rng.random() < 0.15:
            p[name] = rng.choice([0, 1, 2, 3, 8, 9, 64, 100, 300, 5000])
    return p


class StorageEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accepted = accepted_storage()

    def test_encode_gives_the_accepted_bytes_or_refusal(self):
        rng, accepted_values = random.Random(11), 0
        for trial in range(1500):
            value, policy = _value(rng), _policy(rng)
            if rng.random() < 0.05 and isinstance(value, (list, dict)):
                (value.append if isinstance(value, list) else value.setdefault)(*(
                    [value] if isinstance(value, list) else ['self', value]))
            old = _outcome(self.accepted._encode, value, policy)
            with self.subTest(trial=trial):
                self.assertEqual(_outcome(current._encode, value, policy), old)
            accepted_values += old[0] == 'ok'
        self.assertGreater(accepted_values, 300)
        self.assertGreater(1500 - accepted_values, 300)

    def test_decode_gives_the_accepted_value_or_refusal(self):
        rng, changed, trial = random.Random(12), 0, 0
        while trial < 600:
            try:
                data = self.accepted._encode(_value(rng), BASE)
            except Exception:
                continue
            trial += 1
            body = bytearray(data[48:])
            for _ in range(rng.randint(0, 3)):
                action = rng.random()
                if body and action < 0.4:
                    body[rng.randrange(len(body))] = rng.randrange(256)
                elif body and action < 0.7:
                    del body[rng.randrange(len(body)):]
                else:
                    position = rng.randrange(len(body) + 1)
                    body[position:position] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 9)))
            sealed = _seal(self.accepted, bytes(body))
            policy = _policy(rng) if rng.random() < 0.5 else BASE
            old = _outcome(self.accepted._decode, sealed, policy)
            with self.subTest(trial=trial):
                self.assertEqual(_outcome(current._decode, sealed, policy), old)
            changed += old[0] == 'raised'
        self.assertGreater(changed, 150)

    def test_coo_coordinate_check_gives_the_accepted_outcome(self):
        rng = random.Random(13)
        for trial in range(1500):
            rows, cols = rng.randint(0, 3), rng.randint(0, 6)
            shape = tuple(rng.randint(1, 4) for _ in range(rows)) + tuple(rng.randint(1, 3) for _ in range(rng.randint(0, 1)))
            index = [[rng.randrange(shape[r]) if rng.random() < 0.9 else rng.choice([-1, shape[r], 2 ** 62])
                      for _ in range(cols)] for r in range(rows)]
            if rng.random() < 0.5 and rows:
                index = [list(r) for r in zip(*sorted(zip(*index)))] if cols else index
            raw = struct.pack(f'<{rows * cols}q', *[v for r in index for v in r])
            outcomes = []
            for module in (self.accepted, current):
                value_shape = (cols,) + shape[rows:]
                values = module._Dense(3, value_shape, memoryview(bytes(4 * math.prod(value_shape))))
                sparse = module._Sparse(shape, module._Dense(2, (rows, cols), memoryview(raw)), values)
                outcomes.append(_outcome(module._materialize, sparse))
            with self.subTest(trial=trial, index=index, shape=shape):
                self.assertEqual(outcomes[1], outcomes[0])


    def test_targeted_coo_and_dictionary_cases_give_the_accepted_outcome(self):
        coo = [((2, 6), [[1, 0], [0, 5]]), ((2, 6), [[0, 0], [5, 1]]), ((2, 6), [[0, 0], [1, 1]]),
               ((2, 6), [[0, 2], [7, 0]]), ((2, 6), [[1, 0], [9, 0]]), ((3,), [[2, 0]]), ((), [[], []][:0])]
        for shape, index in coo:
            rows, cols = len(index), len(index[0]) if index else 0
            raw = struct.pack(f'<{rows * cols}q', *[v for r in index for v in r])
            outcomes = [_outcome(m._materialize, m._Sparse(shape, m._Dense(2, (rows, cols), memoryview(raw)),
                                                             m._Dense(3, (cols,) + shape[rows:], memoryview(bytes(4 * cols)))))
                        for m in (self.accepted, current)]
            with self.subTest(index=index):
                self.assertEqual(outcomes[1], outcomes[0])
        self.assertEqual(outcomes[0][0], 'ok')

        def leaf(value):
            return current._encode(value, BASE)[48:]
        for keys in (['a', 'a'], ['a', 'b'], [1, 1], [('t', 1), ('t', 1)], [None, 'a'], [[1], 'a'], [1.5, True]):
            body = b'm' + struct.pack('>I', len(keys)) + b''.join(leaf(k) + leaf(0) for k in keys)
            sealed = _seal(self.accepted, body)
            with self.subTest(keys=keys):
                self.assertEqual(_outcome(current._decode, sealed, BASE), _outcome(self.accepted._decode, sealed, BASE))


if __name__ == '__main__':
    unittest.main()
