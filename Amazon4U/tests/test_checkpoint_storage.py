"""D3-only synthetic IO tests. Execution requires Pi's configured tempfile.tempdir.

No existing TestCase classes are imported, no live sources are captured, and all
sentinels, symlinks and deterministic race targets stay inside owned scratch.
"""

import copy
import errno
import hashlib
import importlib
import os
from pathlib import Path
import random
import stat
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from scripts import checkpoint_storage as storage
from scripts.checkpoint_identity import (
    CheckpointError, CheckpointSecurityError, CheckpointValidationError,
    build_source_provenance,
)
from test_checkpoint_payload import (
    _build_coherent_lightgcn_payload,
    _make_valid_hetero_payload,
    _make_synthetic_resource_policy,
)


def _io(**changes):
    policy = {
        "schema_version": "amazon4u-checkpoint-io-v1",
        "profile_id": "independently-approved-synthetic-io",
        "max_file_bytes": 2097152,
        "max_scalar_bytes": 65536,
        "max_nodes": 16384,
        "max_depth": 64,
        "max_storage_bytes": 2097152,
        "max_copy_bytes": 2097152,
        "max_tensor_elements": 1048576,
        "max_dimension": 1048576,
        "max_rank": 8,
        "max_source_bytes": 2097152,
        "max_source_files": 20,
    }
    policy.update(changes)
    return policy


def _fixture(hetero=False, initial=False):
    values = (_make_valid_hetero_payload() if hetero else
              _build_coherent_lightgcn_payload(epoch=0 if initial else 2))
    payload, identity, provenance, model, runtime = [copy.deepcopy(v) for v in values]
    if hetero and initial:
        payload["current_model"]["epoch"] = 0
        payload["best_model"] = None
        payload["optimizer"]["state_dict"]["state"] = {}
        for binding in payload["optimizer"]["parameter_bindings"]:
            binding["updates"] = 0
        payload["progress"] = {"epoch": 0, "global_step": 0,
                               "epoch_optimizer_steps": [], "empty_pool_skips": []}
        payload["sampler"].update(source_order=None, cursor=0, next_epoch=1)
        payload["selection"].update(history=[], best_score=None, selected_epoch=None,
                                    bad_checks=0, status="initial", stop_reason=None)
    arguments = {
        "expected_identity": identity, "expected_provenance": provenance,
        "expected_model": model, "expected_runtime": runtime,
        "resource_policy": _make_synthetic_resource_policy(), "io_policy": _io(),
    }
    return payload, arguments


def _private_dir(path):
    path.mkdir(mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def _private_bytes(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC |
                 os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        return stream.write(data)


def _seal(body):
    return storage._MAGIC + struct.pack(">Q", len(body)) + hashlib.sha256(body).digest() + body


def _dense(shape, raw, declared=None, code=3):
    return (b"D" + bytes([code]) + struct.pack(">I", len(shape)) +
            b"".join(struct.pack(">Q", d) for d in shape) +
            struct.pack(">Q", len(raw) if declared is None else declared) + raw)


class CheckpointStorageTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(tempfile.tempdir, "Pi must configure the declared scratch root")
        self.temporary = tempfile.TemporaryDirectory(dir=tempfile.tempdir)
        self.addCleanup(self.temporary.cleanup)
        self.scratch = Path(self.temporary.name)
        os.chmod(self.scratch, 0o700)
        self.root = self.scratch / "approved"
        _private_dir(self.root)
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.addCleanup(os.close, self.fd)

    def equal(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        if type(actual) is torch.Tensor:
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.layout, expected.layout)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            if actual.layout == torch.sparse_coo:
                self.assertTrue(actual.is_coalesced())
                self.equal(actual.indices(), expected.indices())
                self.equal(actual.values(), expected.values())
            else:
                self.assertTrue(torch.equal(actual.reshape(-1).contiguous().view(torch.uint8),
                                            expected.reshape(-1).contiguous().view(torch.uint8)))
        elif type(actual) is dict:
            self.assertEqual(list(actual), list(expected))
            for key in actual:
                self.equal(actual[key], expected[key])
        elif type(actual) in (list, tuple):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.equal(left, right)
        elif type(actual) is float:
            self.assertEqual(struct.pack(">d", actual), struct.pack(">d", expected))
        else:
            self.assertEqual(actual, expected)

    def source_path(self, logical):
        prefix, basename = logical.split("/")
        return self.source_paths[prefix] / basename

    def source_tree(self, provenance):
        container = self.scratch / "source-container"
        _private_dir(container)
        self.source_paths = {prefix: container / ("approved-" + prefix)
                             for prefix in ("scripts", "configs")}
        self.source_roots = {}
        for prefix, directory in self.source_paths.items():
            _private_dir(directory)
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            self.source_roots[prefix] = fd
            self.addCleanup(os.close, fd)
        sizes = {}
        for path, entry in provenance["source_files"].items():
            _private_bytes(self.source_path(path), entry["content_bytes"])
            sizes[path] = len(entry["content_bytes"])
        # Bootstrap FDs and establish a genuine valid control before any tripwire.
        self.equal(self.acquire(sizes, provenance), provenance)
        return sizes

    def acquire(self, sizes, provenance, **changes):
        arguments = {
            "git_evidence": copy.deepcopy(provenance["git"]),
            "environment_records": copy.deepcopy(provenance["environment"]),
            "expected_provenance": copy.deepcopy(provenance), "io_policy": _io(),
        }
        arguments.update(changes)
        return storage.acquire_sources(self.source_roots, sizes, **arguments)

    def test_canonical_import_and_mandatory_arguments(self):
        self.assertIs(storage, importlib.import_module("checkpoint_storage"))
        with self.assertRaises(TypeError):
            storage.retrieve_checkpoint(self.fd, "state.a4ucp")
        with self.assertRaises(TypeError):
            storage.publish_checkpoint(self.fd, "state.a4ucp", {})
        with self.assertRaises(TypeError):
            storage.acquire_sources(self.fd, {})

    def test_complete_roundtrips_both_families_initial_and_epoch(self):
        for hetero, initial in ((False, True), (False, False), (True, True), (True, False)):
            with self.subTest(hetero=hetero, initial=initial):
                payload, arguments = _fixture(hetero=hetero, initial=initial)
                self.assertEqual(storage.publish_checkpoint(self.fd, "state.a4ucp", payload,
                                                            **arguments), "state.a4ucp")
                self.assertLessEqual((self.root / "state.a4ucp").stat().st_size, 2097152)
                result = storage.retrieve_checkpoint(self.fd, "state.a4ucp", **arguments)
                self.equal(result, payload)
                if hetero:
                    self.assertEqual(len(result["optimizer"]["parameter_bindings"]), 36)
                (self.root / "state.a4ucp").unlink()
                self.assertEqual(list(self.root.iterdir()), [])

    def test_owned_outputs_and_global_state_unchanged(self):
        payload, arguments = _fixture()
        py = random.getstate()
        numpy = np.random.get_state()
        native = torch.get_rng_state().clone()
        dtype = torch.get_default_dtype()
        storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        result = storage.retrieve_checkpoint(self.fd, "state.a4ucp", **arguments)
        original = result["current_model"]["state_dict"]["user_embeddings"].clone()
        payload["current_model"]["state_dict"]["user_embeddings"][0, 0] += 1
        self.equal(result["current_model"]["state_dict"]["user_embeddings"], original)
        result["current_model"]["state_dict"]["user_embeddings"][0, 0] += 2
        again = storage.retrieve_checkpoint(self.fd, "state.a4ucp", **arguments)
        self.equal(again["current_model"]["state_dict"]["user_embeddings"], original)
        self.assertEqual(random.getstate(), py)
        now = np.random.get_state()
        self.assertEqual(now[0], numpy[0])
        self.assertTrue(np.array_equal(now[1], numpy[1]))
        self.assertEqual(now[2:], numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), native))
        self.assertEqual(torch.get_default_dtype(), dtype)

    def test_independent_expected_contracts_and_corrupted_payload(self):
        payload, arguments = _fixture()
        storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        for key in ("expected_identity", "expected_provenance", "expected_model", "expected_runtime"):
            with self.subTest(key=key):
                bad = copy.deepcopy(arguments)
                bad[key] = {}
                with self.assertRaises(CheckpointError):
                    storage.retrieve_checkpoint(self.fd, "state.a4ucp", **bad)
        invalid = copy.deepcopy(payload)
        invalid["progress"]["epoch"] = 3
        _private_bytes(self.root / "invalid.a4ucp", storage._encode(invalid, _io()))
        with self.assertRaises(CheckpointError):
            storage.retrieve_checkpoint(self.fd, "invalid.a4ucp", **arguments)
        with self.assertRaises(CheckpointError):
            storage.publish_checkpoint(self.fd, "never.a4ucp", invalid, **arguments)
        self.assertFalse((self.root / "never.a4ucp").exists())

    def test_actual_source_pipeline_and_changed_bytes(self):
        payload, arguments = _fixture()
        provenance = arguments["expected_provenance"]
        sizes = self.source_tree(provenance)
        actual = self.acquire(sizes, provenance)
        self.equal(actual, provenance)
        payload["provenance"] = actual
        storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.equal(storage.retrieve_checkpoint(self.fd, "state.a4ucp", **arguments), payload)
        path = next(iter(sizes))
        _private_bytes(self.source_path(path), b"x" * sizes[path])
        with self.assertRaises(CheckpointError):
            self.acquire(sizes, provenance)

    def test_source_requests_preflight_before_io_or_hashing(self):
        _, arguments = _fixture()
        provenance = arguments["expected_provenance"]
        sizes = self.source_tree(provenance)
        bad_requests = [
            {"scripts/checkpoint_storage.py": 1}, {"../scripts/checkpoint_identity.py": 1},
            {"/scripts/checkpoint_identity.py": 1}, {"scripts/checkpoint_identity.py": True},
            {"scripts/checkpoint_identity.py": -1},
            {"scripts/nested/checkpoint_identity.py": 1},
            {"scripts/../scripts/checkpoint_identity.py": 1},
            {"configs/unknown.json": 1}, {1: 1}, [], {},
        ]
        for request in bad_requests:
            with self.subTest(request=request), mock.patch.object(storage.os, "open", side_effect=AssertionError("opened")), \
                    mock.patch.object(storage, "build_source_provenance", side_effect=AssertionError("hashed")):
                with self.assertRaises(CheckpointError):
                    self.acquire(request, provenance)
        with mock.patch.object(storage.os, "open", side_effect=AssertionError("opened")):
            with self.assertRaises(CheckpointError):
                self.acquire(sizes, provenance, io_policy=_io(max_source_bytes=sum(sizes.values()) - 1))
        actual = self.acquire(sizes, provenance,
                              io_policy=_io(max_source_bytes=sum(sizes.values()), max_source_files=len(sizes)))
        self.equal(actual, provenance)

    def test_source_all_physical_sizes_checked_before_reads(self):
        _, arguments = _fixture()
        provenance = arguments["expected_provenance"]
        sizes = self.source_tree(provenance)
        last = list(sizes)[-1]
        _private_bytes(self.source_path(last), b"x" * (sizes[last] + 1))
        with mock.patch.object(storage, "_read_fd", side_effect=AssertionError("read before aggregate physical preflight")):
            with self.assertRaises(CheckpointError):
                self.acquire(sizes, provenance)

    def test_source_missing_and_symlink_refusal(self):
        _, arguments = _fixture()
        provenance = arguments["expected_provenance"]
        sizes = self.source_tree(provenance)
        path = next(iter(sizes))
        target = self.scratch / "synthetic-sentinel"
        sentinel = b"synthetic sentinel only"
        _private_bytes(target, sentinel)
        self.source_path(path).unlink()
        with self.assertRaises(CheckpointError):
            self.acquire(sizes, provenance)
        self.source_path(path).symlink_to(target)
        with self.assertRaises(CheckpointError):
            self.acquire(sizes, provenance)
        self.assertEqual(target.read_bytes(), sentinel)

    def test_source_ancestor_replacement_cannot_redirect_approved_inodes(self):
        _, arguments = _fixture()
        provenance = arguments["expected_provenance"]
        sizes = self.source_tree(provenance)
        real_readable = storage._readable
        swapped = []

        def replace_ancestor(path_fd):
            if not swapped:
                container = self.scratch / "source-container"
                container.rename(self.scratch / "moved-source-container")
                _private_dir(container)
                for directory in self.source_paths.values():
                    _private_dir(directory)
                for logical, size in sizes.items():
                    _private_bytes(self.source_path(logical), b"x" * size)
                swapped.append(True)
            return real_readable(path_fd)

        with mock.patch.object(storage, "_readable", side_effect=replace_ancestor):
            self.equal(self.acquire(sizes, provenance), provenance)
        self.assertEqual(swapped, [True])
        self.equal(self.acquire(sizes, provenance), provenance)
        for logical, size in sizes.items():
            self.assertEqual(self.source_path(logical).read_bytes(), b"x" * size)

    def test_source_leaf_substitution_refused_before_readable_open_both_prefixes(self):
        _, arguments = _fixture()
        provenance = arguments["expected_provenance"]
        sizes = self.source_tree(provenance)
        real_open = os.open
        for logical in sizes:
            with self.subTest(logical=logical):
                entry = provenance["source_files"][logical]
                expected = build_source_provenance(
                    {logical: entry["content_bytes"]}, copy.deepcopy(provenance["git"]),
                    copy.deepcopy(provenance["environment"]))
                request = {logical: sizes[logical]}
                self.equal(self.acquire(request, expected), expected)
                leaf = self.source_path(logical)
                moved = leaf.with_name("moved-synthetic-leaf")
                swapped = []

                def swap(name, flags, *args, **kwargs):
                    fd = real_open(name, flags, *args, **kwargs)
                    if name == leaf.name and flags & os.O_PATH and not swapped:
                        leaf.rename(moved)
                        _private_bytes(leaf, b"x" * sizes[logical])
                        swapped.append(True)
                    return fd

                with mock.patch.object(storage.os, "open", side_effect=swap), \
                        mock.patch.object(storage, "_readable", side_effect=AssertionError("opened substituted leaf")), \
                        mock.patch.object(storage, "_read_fd", side_effect=AssertionError("read substituted leaf")):
                    with self.assertRaises(CheckpointSecurityError):
                        self.acquire(request, expected)
                self.assertEqual(swapped, [True])
                leaf.unlink()
                moved.rename(leaf)
                self.equal(self.acquire(request, expected), expected)

    def test_flat_names_symlinks_directories_and_fifo(self):
        payload, arguments = _fixture()
        for name in ("../escape.a4ucp", "/absolute.a4ucp", "nested/state.a4ucp", "state", " .a4ucp"):
            with self.subTest(name=name), self.assertRaises(CheckpointSecurityError):
                storage.publish_checkpoint(self.fd, name, payload, **arguments)
        sentinel = self.scratch / "synthetic-sentinel"
        _private_bytes(sentinel, b"untouched")
        (self.root / "link.a4ucp").symlink_to(sentinel)
        (self.root / "directory.a4ucp").mkdir()
        os.mkfifo(self.root / "pipe.a4ucp", 0o600)
        with mock.patch.object(storage, "_read_fd", side_effect=AssertionError("read nonregular object")), \
                mock.patch.object(storage, "_readable", side_effect=AssertionError("data-opened nonregular object")):
            for name in ("link.a4ucp", "directory.a4ucp", "pipe.a4ucp"):
                with self.subTest(name=name), self.assertRaises(CheckpointSecurityError):
                    storage.retrieve_checkpoint(self.fd, name, **arguments)
        self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_file_bounds_precede_decode_and_exact_limit(self):
        payload, arguments = _fixture()
        storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        size = (self.root / "state.a4ucp").stat().st_size
        exact = copy.deepcopy(arguments)
        exact["io_policy"]["max_file_bytes"] = size
        self.equal(storage.retrieve_checkpoint(self.fd, "state.a4ucp", **exact), payload)
        exact["io_policy"]["max_file_bytes"] = size - 1
        with mock.patch.object(storage, "_decode", side_effect=AssertionError("decoder reached")):
            with self.assertRaises(CheckpointError):
                storage.retrieve_checkpoint(self.fd, "state.a4ucp", **exact)

    def test_publication_collision_preserves_every_target_type(self):
        payload, arguments = _fixture(initial=True)
        sentinel = self.scratch / "synthetic-sentinel"
        _private_bytes(sentinel, b"sentinel")
        _private_bytes(self.root / "existing.a4ucp", b"original")
        (self.root / "directory.a4ucp").mkdir()
        (self.root / "link.a4ucp").symlink_to(sentinel)
        for name in ("existing.a4ucp", "directory.a4ucp", "link.a4ucp"):
            with self.subTest(name=name), self.assertRaises(CheckpointSecurityError):
                storage.publish_checkpoint(self.fd, name, payload, **arguments)
        self.assertEqual((self.root / "existing.a4ucp").read_bytes(), b"original")
        self.assertTrue((self.root / "directory.a4ucp").is_dir())
        self.assertTrue((self.root / "link.a4ucp").is_symlink())
        self.assertEqual(sentinel.read_bytes(), b"sentinel")
        self.assertFalse(any(p.name.startswith(".a4u-incomplete-") for p in self.root.iterdir()))

    def test_partial_write_failure_cleans_only_owned_temp(self):
        payload, arguments = _fixture(initial=True)
        unrelated = self.root / "unrelated"
        _private_bytes(unrelated, b"retain")
        original = os.write
        calls = []

        def fail_second(fd, data):
            calls.append(True)
            if len(calls) == 2:
                raise OSError(errno.EIO, "synthetic interrupted write")
            return original(fd, data[:17])

        with mock.patch.object(storage.os, "write", side_effect=fail_second):
            with self.assertRaises(CheckpointSecurityError):
                storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.assertEqual(len(calls), 2)
        self.assertEqual(list(self.root.iterdir()), [unrelated])
        self.assertEqual(unrelated.read_bytes(), b"retain")

    def test_competing_publication_no_clobber_without_threads(self):
        payload, arguments = _fixture(initial=True)
        original = storage._link_fd
        competed = []

        def competitor(fd, root_fd, destination):
            _private_bytes(self.root / destination, b"synthetic competing winner")
            competed.append(True)
            return original(fd, root_fd, destination)

        with mock.patch.object(storage, "_link_fd", side_effect=competitor):
            with self.assertRaises(CheckpointSecurityError):
                storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.assertEqual(competed, [True])
        self.assertEqual((self.root / "state.a4ucp").read_bytes(), b"synthetic competing winner")
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_serialization_quota_before_any_output_creation(self):
        payload, arguments = _fixture(initial=True)
        arguments["io_policy"] = _io(max_file_bytes=64)
        with mock.patch.object(storage.os, "open", side_effect=AssertionError("created oversized output")):
            with self.assertRaises(CheckpointValidationError):
                storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_codec_exact_types_integer_bits_and_float_bits(self):
        value = {
            7: (b"\x00\xff", [1 << 127, -(1 << 130), True, None, -0.0]),
            9: torch.tensor([0.0, -0.0, 1.25], dtype=torch.float32),
        }
        with mock.patch.object(torch, "load", side_effect=AssertionError("native loader forbidden")):
            data = storage._encode(value, _io())
            result = storage._decode(data, _io())
        self.equal(result, value)
        self.assertEqual(list(result), [7, 9])
        self.assertIs(type(result[7]), tuple)
        self.assertIs(type(result[7][1]), list)

    def test_codec_storage_copy_file_scalar_node_and_depth_boundaries(self):
        value = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        data = storage._encode(value, _io())
        exact = _io(max_storage_bytes=16, max_copy_bytes=16, max_file_bytes=len(data))
        self.equal(storage._decode(data, exact), value)
        self.assertEqual(storage._encode(value, exact), data)
        for field, limit in (("max_storage_bytes", 15), ("max_copy_bytes", 15),
                             ("max_file_bytes", len(data) - 1)):
            with self.subTest(field=field), mock.patch.object(storage, "_materialize", side_effect=AssertionError("allocated")):
                with self.assertRaises(CheckpointValidationError):
                    storage._decode(data, _io(**{field: limit}))
            with self.assertRaises(CheckpointValidationError):
                storage._encode(value, _io(**{field: limit}))
        for value, exact_policy, small_policy in (
            (b"abcd", _io(max_scalar_bytes=4), _io(max_scalar_bytes=3)),
            ([None, None], _io(max_nodes=3), _io(max_nodes=2)),
            ([[None]], _io(max_depth=3), _io(max_depth=2)),
        ):
            encoded = storage._encode(value, exact_policy)
            self.equal(storage._decode(encoded, exact_policy), value)
            with self.assertRaises(CheckpointValidationError):
                storage._encode(value, small_policy)
            with mock.patch.object(storage, "_materialize", side_effect=AssertionError("allocated")):
                with self.assertRaises(CheckpointValidationError):
                    storage._decode(encoded, small_policy)

    def test_malformed_storage_shape_stride_extent_rejected_before_native(self):
        bodies = [
            _dense((5,), b"\x00" * 4),
            _dense((2,), b"\x00" * 4, declared=8),
            _dense((0, 100), b""),
            _dense((5,), b"\x00" * 20),
            _dense((1, 1, 1), b"\x00" * 4),
            _dense((1,), b"\x00" * 4, code=255),
        ]
        small = _io(max_dimension=4, max_tensor_elements=4, max_rank=2)
        for body in bodies:
            with self.subTest(body=body), mock.patch.object(torch, "frombuffer", side_effect=AssertionError("native allocation")), \
                    mock.patch.object(torch, "empty", side_effect=AssertionError("native allocation")):
                with self.assertRaises(CheckpointValidationError):
                    storage._decode(_seal(body), small)
        control = _seal(_dense((1,), struct.pack("<f", 1.0)))
        self.equal(storage._decode(control, small), torch.tensor([1.0], dtype=torch.float32))

    def test_aggregate_preflight_finishes_before_first_tensor_rebuild(self):
        first = _dense((1,), b"\x00" * 4)
        second = _dense((4,), b"\x00" * 16)
        data = _seal(b"l" + struct.pack(">I", 2) + first + second)
        with mock.patch.object(storage, "_materialize", side_effect=AssertionError("early allocation")):
            with self.assertRaises(CheckpointValidationError):
                storage._decode(data, _io(max_storage_bytes=19))
        result = storage._decode(data, _io(max_storage_bytes=20, max_copy_bytes=20))
        self.assertEqual([tuple(t.shape) for t in result], [(1,), (4,)])

    def test_corruption_truncation_unsupported_native_and_tags(self):
        good = storage._encode({"valid": b"small"}, _io())
        damaged = good[:-1] + bytes([good[-1] ^ 1])
        cases = [b"", good[:-1], good + b"x", damaged, b"PK\x03\x04synthetic-not-an-archive",
                 _seal(b"P"), _seal(b"ntrailing"), _seal(b"s" + struct.pack(">Q", 1) + b"\xff")]
        with mock.patch.object(torch, "load", side_effect=AssertionError("native loader reached")), \
                mock.patch.object(storage, "_materialize", side_effect=AssertionError("native reconstruction reached")):
            for data in cases:
                with self.subTest(data=data), self.assertRaises(CheckpointValidationError):
                    storage._decode(data, _io())
        self.equal(storage._decode(good, _io()), {"valid": b"small"})

    def test_custom_objects_refused_without_hooks_and_no_pickle_fixture(self):
        class Forbidden:
            def __reduce__(self):
                raise AssertionError("reduce hook executed")

            def __repr__(self):
                raise AssertionError("repr hook executed")

        with self.assertRaises(CheckpointValidationError):
            storage._encode(Forbidden(), _io())
        payload, arguments = _fixture(initial=True)
        payload["scheduler"] = Forbidden()
        with self.assertRaises(CheckpointError):
            storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_sparse_descriptor_and_coordinate_refusals(self):
        control = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
            torch.tensor([1.0, 2.0], dtype=torch.float32), (2, 2), is_coalesced=True)
        self.equal(storage._decode(storage._encode(control, _io()), _io()), control)
        shape = struct.pack(">IQQ", 2, 2, 2)
        bad_descriptor = b"C" + shape + _dense((1, 1), struct.pack("<q", 0), code=2) + _dense((1,), b"\x00" * 4)
        with mock.patch.object(storage, "_materialize", side_effect=AssertionError("bad descriptor reached native")):
            with self.assertRaises(CheckpointValidationError):
                storage._decode(_seal(bad_descriptor), _io())
        # Small raw integer records, not a forged native tensor/archive.
        bad_coordinates = (b"C" + shape +
                           _dense((2, 1), struct.pack("<qq", 2, 0), code=2) +
                           _dense((1,), struct.pack("<f", 1.0)))
        with mock.patch.object(torch, "sparse_coo_tensor", side_effect=AssertionError("invalid index reached constructor")):
            with self.assertRaises(CheckpointValidationError):
                storage._decode(_seal(bad_coordinates), _io())

    def test_all_twenty_logical_ids_use_both_independent_prefix_roots(self):
        _, arguments = _fixture()
        base = arguments["expected_provenance"]
        sources = {path: ("synthetic " + path + "\n").encode("utf-8")
                   for path in sorted(storage.ALL_ALLOWLISTED_PATHS)}
        self.assertEqual(len(sources), 20)
        provenance = build_source_provenance(
            sources, copy.deepcopy(base["git"]), copy.deepcopy(base["environment"]))
        sizes = self.source_tree(provenance)
        self.assertEqual({path.split("/")[0] for path in sizes}, {"scripts", "configs"})
        self.equal(self.acquire(sizes, provenance), provenance)
        for logical, entry in provenance["source_files"].items():
            self.assertEqual(entry["path"], logical)
            self.assertEqual(entry["content_bytes"], self.source_path(logical).read_bytes())

    def test_source_root_binding_schema_types_inodes_and_permissions(self):
        _, arguments = _fixture()
        provenance = arguments["expected_provenance"]
        sizes = self.source_tree(provenance)
        evidence = {
            "git_evidence": copy.deepcopy(provenance["git"]),
            "environment_records": copy.deepcopy(provenance["environment"]),
            "expected_provenance": copy.deepcopy(provenance), "io_policy": _io(),
        }
        scripts = self.source_roots["scripts"]
        configs = self.source_roots["configs"]
        invalid = [None, scripts, {}, {"scripts": scripts},
                   {"scripts": scripts, "configs": configs, "other": self.fd},
                   {"scripts": True, "configs": configs},
                   {"scripts": "scripts", "configs": configs},
                   {"scripts": -1, "configs": configs}]
        for bindings in invalid:
            with self.subTest(bindings=bindings), \
                    mock.patch.object(storage.os, "open", side_effect=AssertionError("opened invalid binding")), \
                    mock.patch.object(storage.os, "dup", side_effect=AssertionError("duplicated invalid binding")):
                with self.assertRaises(CheckpointSecurityError):
                    storage.acquire_sources(bindings, sizes, **evidence)
        regular = self.scratch / "not-a-directory"
        _private_bytes(regular, b"synthetic")
        file_fd = os.open(regular, os.O_RDONLY | os.O_CLOEXEC)
        self.addCleanup(os.close, file_fd)
        for bindings in ({"scripts": file_fd, "configs": configs},
                         {"scripts": scripts, "configs": scripts}):
            with self.subTest(bindings=bindings), \
                    mock.patch.object(storage, "_read_fd", side_effect=AssertionError("read invalid binding")):
                with self.assertRaises(CheckpointSecurityError):
                    storage.acquire_sources(bindings, sizes, **evidence)
        os.chmod(self.source_paths["configs"], 0o770)
        try:
            with self.assertRaises(CheckpointSecurityError):
                self.acquire(sizes, provenance)
        finally:
            os.chmod(self.source_paths["configs"], 0o700)
        self.equal(self.acquire(sizes, provenance), provenance)

    def test_source_nonregular_leaves_refused_before_data_open(self):
        _, arguments = _fixture()
        provenance = arguments["expected_provenance"]
        sizes = self.source_tree(provenance)
        for logical in sizes:
            with self.subTest(logical=logical):
                leaf = self.source_path(logical)
                content = leaf.read_bytes()
                expected = build_source_provenance(
                    {logical: b""}, copy.deepcopy(provenance["git"]),
                    copy.deepcopy(provenance["environment"]))
                request = {logical: 0}
                _private_bytes(leaf, b"")
                try:
                    # Same-size positive control excludes unrelated size refusals.
                    self.equal(self.acquire(request, expected), expected)
                    leaf.unlink()
                    os.mkfifo(leaf, 0o600)
                    with mock.patch.object(storage, "_readable", side_effect=AssertionError("data-opened target FIFO")), \
                            mock.patch.object(storage, "_read_fd", side_effect=AssertionError("read target FIFO")):
                        with self.assertRaisesRegex(
                                CheckpointSecurityError, "single-link caller-owned regular file"):
                            self.acquire(request, expected)
                finally:
                    leaf.unlink(missing_ok=True)
                    _private_bytes(leaf, content)
                self.equal(self.acquire(sizes, provenance), provenance)

    def test_classified_device_mode_refused_without_actual_device_fixture(self):
        payload, arguments = _fixture(initial=True)
        storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.equal(storage.retrieve_checkpoint(self.fd, "state.a4ucp", **arguments), payload)
        real_open, real_fstat = os.open, os.fstat
        classified = []

        def opened(name, flags, *args, **kwargs):
            fd = real_open(name, flags, *args, **kwargs)
            if name == "state.a4ucp":
                self.assertTrue(flags & os.O_PATH)
                classified.append(fd)
            return fd

        def classified_stat(fd):
            original = real_fstat(fd)
            if fd in classified:
                fields = list(original)
                fields[0] = stat.S_IFCHR | 0o600
                return os.stat_result(fields)
            return original

        with mock.patch.object(storage.os, "open", side_effect=opened), \
                mock.patch.object(storage.os, "fstat", side_effect=classified_stat), \
                mock.patch.object(storage, "_readable", side_effect=AssertionError("device data open")), \
                mock.patch.object(storage, "_read_fd", side_effect=AssertionError("device read")):
            with self.assertRaises(CheckpointSecurityError):
                storage.retrieve_checkpoint(self.fd, "state.a4ucp", **arguments)
        self.assertEqual(len(classified), 1)

    def test_anonymous_publication_uses_held_inode_not_named_decoy(self):
        payload, arguments = _fixture(initial=True)
        decoy = self.root / ".a4u-incomplete-decoy"
        original = storage._link_fd
        linked = []

        def checked_link(fd, root_fd, name):
            before = os.fstat(fd)
            self.assertEqual(before.st_nlink, 0)
            self.assertEqual(list(self.root.iterdir()), [])
            _private_bytes(decoy, b"unrelated decoy")
            original(fd, root_fd, name)
            actual = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            self.assertEqual((actual.st_dev, actual.st_ino), (before.st_dev, before.st_ino))
            self.assertNotEqual(actual.st_ino, decoy.stat().st_ino)
            linked.append(True)

        with mock.patch.object(storage, "_link_fd", side_effect=checked_link), \
                mock.patch.object(storage.os, "unlink", side_effect=AssertionError("publication unlinked a name")):
            storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.assertEqual(linked, [True])
        self.assertEqual(decoy.read_bytes(), b"unrelated decoy")
        self.equal(storage.retrieve_checkpoint(self.fd, "state.a4ucp", **arguments), payload)
        self.assertEqual({p.name for p in self.root.iterdir()}, {decoy.name, "state.a4ucp"})

    def test_link_failure_and_post_link_error_preserve_unrelated_and_completed(self):
        payload, arguments = _fixture(initial=True)
        decoy = self.root / ".a4u-incomplete-decoy"
        _private_bytes(decoy, b"retain")
        with mock.patch.object(storage, "_link_fd", side_effect=OSError(errno.EIO, "synthetic link failure")), \
                mock.patch.object(storage.os, "unlink", side_effect=AssertionError("failure unlinked a name")):
            with self.assertRaises(CheckpointSecurityError):
                storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.assertEqual(list(self.root.iterdir()), [decoy])
        self.assertEqual(decoy.read_bytes(), b"retain")
        original = storage._link_fd
        completed = []

        def linked_then_failed(fd, root_fd, name):
            original(fd, root_fd, name)
            completed.append(True)
            raise OSError(errno.EIO, "synthetic error after completed publication")

        with mock.patch.object(storage, "_link_fd", side_effect=linked_then_failed), \
                mock.patch.object(storage.os, "unlink", side_effect=AssertionError("completed rollback")):
            with self.assertRaises(CheckpointSecurityError):
                storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.assertEqual(completed, [True])
        self.assertEqual(decoy.read_bytes(), b"retain")
        self.equal(storage.retrieve_checkpoint(self.fd, "state.a4ucp", **arguments), payload)
        self.assertEqual({p.name for p in self.root.iterdir()}, {decoy.name, "state.a4ucp"})

    def test_required_anonymous_temporary_failure_has_no_named_fallback(self):
        payload, arguments = _fixture(initial=True)
        attempted = []

        def unavailable(name, flags, *args, **kwargs):
            self.assertEqual(name, ".")
            self.assertEqual(flags & os.O_TMPFILE, os.O_TMPFILE)
            attempted.append(True)
            raise OSError(errno.EOPNOTSUPP, "synthetic unsupported anonymous temporary")

        with mock.patch.object(storage.os, "open", side_effect=unavailable), \
                mock.patch.object(storage, "_link_fd", side_effect=AssertionError("linked absent temporary")):
            with self.assertRaises(CheckpointSecurityError):
                storage.publish_checkpoint(self.fd, "state.a4ucp", payload, **arguments)
        self.assertEqual(attempted, [True])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_native_abi_and_procfs_classification_fail_closed(self):
        with mock.patch.object(storage.os, "confstr", return_value="unsupported-libc"), \
                mock.patch.object(storage.ctypes, "CDLL", side_effect=AssertionError("unchecked native ABI")):
            with self.assertRaises(CheckpointSecurityError):
                with storage._Root(self.fd):
                    self.fail("unsupported ABI accepted")
        self.assertEqual(storage.ctypes.sizeof(storage._StatFS), 120)
        self.assertEqual(storage._StatFS.f_fsid.offset, 56)
        self.assertEqual(storage._StatFS.f_spare.offset, 88)

        def not_procfs(fd, information):
            information._obj.f_type = 0
            return 0

        with mock.patch.object(storage, "_native_function", return_value=not_procfs):
            with self.assertRaises(CheckpointSecurityError):
                with storage._proc_fds():
                    self.fail("non-procfs transport accepted")
        with mock.patch.object(storage, "_native_function", return_value=lambda *args: -1):
            with self.assertRaises(OSError):
                with storage._proc_fds():
                    self.fail("failed procfs classification accepted")

    def test_explicit_little_endian_wire_profile_and_opaque_uint8(self):
        self.assertEqual(storage._MAGIC, b"A4UCP\x00\x02L")
        raw_float = b"\x00\x00\x80\x3f\x00\x00\x00\x80"
        float_record = _dense((2,), raw_float)
        floats = storage._decode(_seal(float_record), _io())
        self.equal(floats, torch.tensor([1.0, -0.0], dtype=torch.float32))
        self.assertEqual(storage._encode(floats, _io()), _seal(float_record))
        raw_int = b"\x08\x07\x06\x05\x04\x03\x02\x01"
        integer_record = _dense((1,), raw_int, code=2)
        integers = storage._decode(_seal(integer_record), _io())
        self.equal(integers, torch.tensor([0x0102030405060708], dtype=torch.int64))
        self.assertEqual(storage._encode(integers, _io()), _seal(integer_record))
        opaque = b"\x00\xff\x01\x80\x55"
        byte_record = _dense((5,), opaque, code=1)
        bytes_tensor = storage._decode(_seal(byte_record), _io())
        self.equal(bytes_tensor, torch.tensor([0, 255, 1, 128, 85], dtype=torch.uint8))
        self.assertEqual(storage._encode(bytes_tensor, _io()), _seal(byte_record))
        good = _seal(float_record)
        with mock.patch.object(storage, "_Parser", side_effect=AssertionError("parsed unsupported profile")), \
                mock.patch.object(storage, "_materialize", side_effect=AssertionError("rebuilt unsupported profile")):
            for magic in (b"A4UCP\x00\x01\n", b"A4UCP\x00\x02B", b"A4UCP\x00\x03L"):
                with self.subTest(magic=magic), self.assertRaises(CheckpointValidationError):
                    storage._decode(magic + good[8:], _io())
        with mock.patch.object(storage.sys, "byteorder", "big"), \
                mock.patch.object(storage, "_Encoder", side_effect=AssertionError("encoded unsupported platform")), \
                mock.patch.object(storage, "_Parser", side_effect=AssertionError("parsed unsupported platform")), \
                mock.patch.object(storage, "_materialize", side_effect=AssertionError("rebuilt unsupported platform")):
            with self.assertRaises(CheckpointValidationError):
                storage._encode(None, _io())
            with self.assertRaises(CheckpointValidationError):
                storage._decode(good, _io())

    def test_policy_strictness_and_unapproved_directory_permissions(self):
        for bad in (_io(max_nodes=True), _io(max_depth=129), _io(max_rank=33),
                    _io(max_source_bytes=2097153), {"schema_version": "wrong"}):
            with self.subTest(bad=bad), self.assertRaises(CheckpointError):
                storage.retrieve_checkpoint(self.fd, "state.a4ucp", expected_identity={},
                                            expected_provenance={}, expected_model={}, expected_runtime={},
                                            resource_policy=_make_synthetic_resource_policy(), io_policy=bad)
        os.chmod(self.root, 0o770)
        self.addCleanup(os.chmod, self.root, 0o700)
        with self.assertRaises(CheckpointSecurityError):
            with storage._Root(self.fd):
                self.fail("unapproved writable directory accepted")

    def test_supplied_git_environment_are_not_replaced_by_host_defaults(self):
        _, arguments = _fixture()
        provenance = arguments["expected_provenance"]
        sizes = self.source_tree(provenance)
        with self.assertRaises(CheckpointError):
            self.acquire(sizes, provenance, git_evidence={})
        with self.assertRaises(CheckpointError):
            self.acquire(sizes, provenance, environment_records={})
        changed_sources = {p: e["content_bytes"] for p, e in provenance["source_files"].items()}
        different = copy.deepcopy(provenance["git"])
        different["commit"] = "b" * 40
        independently_different = build_source_provenance(changed_sources, different, provenance["environment"])
        with self.assertRaises(CheckpointError):
            self.acquire(sizes, provenance, expected_provenance=independently_different)
