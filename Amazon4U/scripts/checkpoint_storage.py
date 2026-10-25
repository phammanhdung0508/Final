"""D3 checkpoint IO: descriptor roots, bounded non-pickle format, D1/D2 integration.

Public operations require independently approved directory descriptors and policies.
See docs/checkpoint-storage.md for platform, concurrency, format and durability limits.
No native Torch archive or pickle is accepted by this module.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
import re
import ctypes
import stat
import struct
import sys

import torch

from scripts.checkpoint_identity import (
    CheckpointError,
    CheckpointSecurityError,
    CheckpointValidationError,
    MAX_PROVENANCE_SOURCE_BYTES,
    ALL_ALLOWLISTED_PATHS,
    validate_allowlisted_relative_path,
    build_source_provenance,
    validate_source_provenance,
)
from scripts.checkpoint_payload import validate_checkpoint_payload

if __name__ == "scripts.checkpoint_storage":
    sys.modules.setdefault("checkpoint_storage", sys.modules[__name__])
elif __name__ == "checkpoint_storage":
    sys.modules.setdefault("scripts.checkpoint_storage", sys.modules[__name__])
    if "scripts" in sys.modules:
        setattr(sys.modules["scripts"], "checkpoint_storage", sys.modules[__name__])

__all__ = ["acquire_sources", "publish_checkpoint", "retrieve_checkpoint"]

_MAGIC = b"A4UCP\x00\x02L"
_HEADER_SIZE = 48
_POLICY_SCHEMA = "amazon4u-checkpoint-io-v1"
_LIMITS = (
    "max_file_bytes", "max_scalar_bytes", "max_nodes", "max_depth",
    "max_storage_bytes", "max_copy_bytes", "max_tensor_elements",
    "max_dimension", "max_rank", "max_source_bytes", "max_source_files",
)
_DTYPES = {
    1: torch.uint8, 2: torch.int64, 3: torch.float32, 4: torch.float64,
    5: torch.int32, 6: torch.int16, 7: torch.int8, 8: torch.bool,
    9: torch.float16, 10: torch.bfloat16,
}
_SIZES = {1: 1, 2: 8, 3: 4, 4: 8, 5: 4, 6: 2, 7: 1, 8: 1, 9: 2, 10: 2}
_CODES = {dtype: code for code, dtype in _DTYPES.items()}
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\.a4ucp\Z")


def _fail(message):
    raise CheckpointValidationError(message)


def _policy(policy):
    if type(policy) is not dict:
        _fail("io_policy must be an exact dict")
    if any(type(k) is not str for k in policy):
        _fail("io_policy keys must be exact strings")
    if set(policy) != {"schema_version", "profile_id", *_LIMITS}:
        _fail("io_policy keys mismatch")
    if type(policy["schema_version"]) is not str or policy["schema_version"] != _POLICY_SCHEMA:
        _fail("unsupported IO policy schema")
    profile = policy["profile_id"]
    if type(profile) is not str or not profile or len(profile) > 256 or profile.strip() != profile:
        _fail("invalid IO profile_id")
    try:
        profile.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckpointValidationError("invalid IO profile text") from exc
    if any(ord(c) < 32 or ord(c) == 127 for c in profile):
        _fail("invalid IO profile text")
    for key in _LIMITS:
        value = policy[key]
        if type(value) is not int or not 1 <= value <= (1 << 63) - 1:
            _fail("IO limits must be positive exact signed-64-bit integers")
    if policy["max_depth"] > 128 or policy["max_rank"] > 32:
        _fail("v1 supports wire depth <=128 and tensor rank <=32")
    if policy["max_nodes"] > (1 << 32) - 1:
        _fail("v1 node count exceeds its representation")
    if policy["max_source_bytes"] > MAX_PROVENANCE_SOURCE_BYTES:
        _fail("source policy exceeds the accepted D1 source bound")
    if policy["max_source_files"] > len(ALL_ALLOWLISTED_PATHS):
        _fail("source file policy exceeds the registered D1 set")
    return dict(policy)


def _name(name):
    if type(name) is not str or not _NAME.fullmatch(name):
        raise CheckpointSecurityError("checkpoint name must be one flat approved .a4ucp basename")
    return name


@contextmanager
def _os_errors():
    try:
        yield
    except OSError as exc:
        raise CheckpointSecurityError("checkpoint filesystem operation refused or failed") from exc


def _signature(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_gid,
            st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _directory(st):
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o022:
        raise CheckpointSecurityError("directory must be caller-owned and not group/world writable")


def _regular(st):
    if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1
            or st.st_uid != os.geteuid() or st.st_mode & 0o022):
        raise CheckpointSecurityError("expected a single-link caller-owned regular file")


class _Root:
    """The approved object is the directory inode, not its ancestor pathname."""

    def __init__(self, fd):
        self.fd = None
        _native_platform()
        if not all(hasattr(os, flag) for flag in
                   ("O_PATH", "O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK", "O_CLOEXEC")):
            raise CheckpointSecurityError("required Linux descriptor IO flags unavailable")
        if type(fd) is not int or fd < 0:
            raise CheckpointSecurityError("root_fd must be an independently approved directory descriptor")
        self.fd = os.dup(fd)
        try:
            _directory(os.fstat(self.fd))
        except (CheckpointError, OSError):
            os.close(self.fd)
            self.fd = None
            raise

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        os.close(self.fd)


class _Anchor:
    """Track direct children of one independently approved directory inode."""

    def __init__(self, root):
        self.root = root
        st = os.fstat(root.fd)
        _directory(st)
        self.start = _signature(st)
        self.files = []

    def file(self, parent_fd, name, limit, exact=None):
        if (parent_fd != self.root.fd or type(name) is not str
                or name in ("", ".", "..") or "/" in name or "\\" in name):
            raise CheckpointSecurityError("physical acquisition requires a direct-child basename")
        self.check()
        # O_PATH classifies without invoking a device/FIFO data open.
        path_fd = os.open(name, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
                          dir_fd=self.root.fd)
        fd = None
        try:
            st = os.fstat(path_fd)
            _regular(st)
            if st.st_dev != os.fstat(self.root.fd).st_dev:
                raise CheckpointSecurityError("cross-device file refused")
            if st.st_size > limit or (exact is not None and st.st_size != exact):
                raise CheckpointSecurityError("physical file size violates the approved request")
            before = _signature(st)
            current = os.stat(name, dir_fd=self.root.fd, follow_symlinks=False)
            if _signature(current) != before:
                raise CheckpointSecurityError("classified leaf was replaced")
            self.check()
            # Trusted procfs reopens this held inode, never the source filename.
            fd = _readable(path_fd)
            if _signature(os.fstat(fd)) != before:
                raise CheckpointSecurityError("classified inode changed before readable acquisition")
            self.files.append((self.root.fd, name, fd, before))
            self.check()
            return fd, st.st_size
        except (CheckpointError, OSError):
            if fd is not None:
                if self.files and self.files[-1][2] == fd:
                    self.files.pop()
                os.close(fd)
            raise
        finally:
            os.close(path_fd)

    def check(self):
        if _signature(os.fstat(self.root.fd)) != self.start:
            raise CheckpointSecurityError("approved directory changed during acquisition")
        for parent, name, fd, before in self.files:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if _signature(current) != before or _signature(os.fstat(fd)) != before:
                raise CheckpointSecurityError("file replacement or mutation refused")

    def close(self):
        for _, _, fd, _ in reversed(self.files):
            os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def _read_fd(fd, size):
    parts = []
    remaining = size + 1
    while remaining:
        part = os.read(fd, min(65536, remaining))
        if not part:
            break
        parts.append(part)
        remaining -= len(part)
    data = b"".join(parts)
    if len(data) != size:
        raise CheckpointSecurityError("file grew, shrank or was truncated during read")
    return data


def _shape(shape, p):
    if len(shape) > p["max_rank"]:
        _fail("tensor rank exceeds IO policy")
    projected = 1
    for dim in shape:
        if type(dim) is not int or not 0 <= dim <= p["max_dimension"]:
            _fail("tensor dimension exceeds IO policy")
        # Also bound dimensions following a zero dimension; zero cannot hide an extent.
        projected *= max(1, dim)
        if projected > p["max_tensor_elements"]:
            _fail("tensor shape/canonical stride extent exceeds IO policy")
    return 0 if 0 in shape else projected


class _Meter:
    def __init__(self, p):
        self.p = p
        self.nodes = 0
        self.storage = 0
        self.copies = 0

    def node(self, depth):
        self.nodes += 1
        if self.nodes > self.p["max_nodes"] or depth > self.p["max_depth"]:
            _fail("wire node/depth limit exceeded")

    def tensor(self, size):
        self.storage += size
        self.copies += size
        if self.storage > self.p["max_storage_bytes"]:
            _fail("full reconstructed storage limit exceeded")
        if self.copies > self.p["max_copy_bytes"]:
            _fail("projected independent tensor copy limit exceeded")


_PACK_Q = struct.Struct(">Q").pack
_PACK_I = struct.Struct(">I").pack
_PACK_D = struct.Struct(">d").pack
_CONTAINER_TAGS = {dict: b"m", list: b"l", tuple: b"u"}
_UNPACK_Q = struct.Struct(">Q").unpack_from
_UNPACK_I = struct.Struct(">I").unpack_from
_UNPACK_D = struct.Struct(">d").unpack_from
_TAGS = tuple(bytes([b]) for b in range(256))


def _key(value):
    t = type(value)
    return t in (str, bytes, int, float, bool) or (t is tuple and all(_key(v) for v in value))


class _Encoder:
    def __init__(self, p, materialize):
        self.p = p
        self.meter = _Meter(p)
        self.buffer = bytearray() if materialize else None
        self.size = _HEADER_SIZE
        self.active = set()

    def reserve(self, size):
        self.size += size
        if self.size > self.p["max_file_bytes"]:
            _fail("serialized file exceeds IO policy")

    def put(self, data):
        self.reserve(len(data))
        if self.buffer is not None:
            self.buffer.extend(data)

    def scalar(self, data):
        if len(data) > self.p["max_scalar_bytes"]:
            _fail("scalar byte limit exceeded")
        self.put(struct.pack(">Q", len(data)))
        self.put(data)

    def _tagged_scalar(self, tag, data):
        """put(tag) then scalar(data), with the same checks in the same order."""
        p = self.p
        self.size += len(tag)
        if self.size > p["max_file_bytes"]:
            _fail("serialized file exceeds IO policy")
        if len(data) > p["max_scalar_bytes"]:
            _fail("scalar byte limit exceeded")
        self.size += 8 + len(data)
        if self.size > p["max_file_bytes"]:
            _fail("serialized file exceeds IO policy")
        if self.buffer is not None:
            self.buffer += tag
            self.buffer += _PACK_Q(len(data))
            self.buffer += data

    def shape(self, shape):
        _shape(shape, self.p)
        self.put(struct.pack(">I", len(shape)))
        for dim in shape:
            self.put(struct.pack(">Q", dim))

    def node(self, value, depth=1):
        # Readiness C2/D3: the meter, reserve and scalar checks are inlined, in their
        # original order and with their messages. Consecutive reservations with no check
        # between them are added at once; the size only grows, so the same one fails.
        p = self.p
        meter = self.meter
        meter.nodes += 1
        if meter.nodes > p["max_nodes"] or depth > p["max_depth"]:
            _fail("wire node/depth limit exceeded")
        t = type(value)
        buffer = self.buffer
        if value is None or t is bool:
            self.size += 1
            if self.size > p["max_file_bytes"]:
                _fail("serialized file exceeds IO policy")
            if buffer is not None:
                buffer += b"n" if value is None else (b"t" if value else b"f")
        elif t is str:
            if len(value) > p["max_scalar_bytes"]:
                _fail("text character limit exceeded")
            try:
                encoded = value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise CheckpointValidationError("invalid UTF8 text") from exc
            self._tagged_scalar(b"s", encoded)
        elif t is float:
            if not math.isfinite(value):
                _fail("nonfinite scalar float")
            self.size += 9
            if self.size > p["max_file_bytes"]:
                _fail("serialized file exceeds IO policy")
            if buffer is not None:
                buffer += b"r"
                buffer += _PACK_D(value)
        elif t is int:
            count = max(1, (value.bit_length() + 7) // 8)
            if count > p["max_scalar_bytes"]:
                _fail("integer magnitude limit exceeded")
            self._tagged_scalar(b"i\x01" if value < 0 else b"i\x00", abs(value).to_bytes(count, "big"))
        elif t is bytes:
            self._tagged_scalar(b"b", value)
        elif t is dict or t is list or t is tuple:
            if id(value) in self.active or len(value) > p["max_nodes"]:
                _fail("cycle or oversized wire container")
            self.active.add(id(value))
            try:
                self.size += 5
                if self.size > p["max_file_bytes"]:
                    _fail("serialized file exceeds IO policy")
                if buffer is not None:
                    buffer += _CONTAINER_TAGS[t]
                    buffer += _PACK_I(len(value))
                node = self.node
                child = depth + 1
                if t is dict:
                    for k, v in value.items():
                        if type(k) is not str and not _key(k):
                            _fail("unsupported dictionary key")
                        node(k, child)
                        node(v, child)
                else:
                    for item in value:
                        node(item, child)
            finally:
                self.active.remove(id(value))
        elif t is torch.Tensor:
            if value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None:
                _fail("wire tensors must be detached CPU tensors")
            if value.layout == torch.sparse_coo:
                if not value.is_coalesced():
                    _fail("wire COO must be coalesced")
                self.put(b"C")
                self.shape(tuple(value.shape))
                self.node(value.indices(), depth + 1)
                self.node(value.values(), depth + 1)
            elif value.layout == torch.strided:
                if value.dtype not in _CODES or not value.is_contiguous() or value.storage_offset() != 0:
                    _fail("wire dense tensors require supported dtype and canonical compact layout")
                count = _shape(tuple(value.shape), self.p)
                size = count * value.element_size()
                if value.untyped_storage().nbytes() != size:
                    _fail("wire dense tensor must carry its full compact backing storage")
                self.meter.tensor(size)
                self.put(b"D" + bytes([_CODES[value.dtype]]))
                self.shape(tuple(value.shape))
                self.put(struct.pack(">Q", size))
                if self.buffer is None:
                    self.reserve(size)
                else:
                    # The complete counting pass has already bounded this allocation.
                    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
                    self.put(raw)
            else:
                _fail("unsupported wire tensor layout")
        else:
            _fail("unsupported wire object or subclass")


def _encode(value, p):
    _wire_platform()
    counter = _Encoder(p, False)
    counter.node(value)
    writer = _Encoder(p, True)
    writer.node(value)
    if writer.size != counter.size:
        _fail("input changed during serialization")
    body = bytes(writer.buffer)
    return _MAGIC + struct.pack(">Q", len(body)) + hashlib.sha256(body).digest() + body


@dataclass(frozen=True)
class _Dense:
    code: int
    shape: tuple
    data: memoryview


@dataclass(frozen=True)
class _Sparse:
    shape: tuple
    indices: _Dense
    values: _Dense


class _Parser:
    def __init__(self, body, p):
        self.body = memoryview(body)
        self.position = 0
        self.p = p
        self.meter = _Meter(p)

    def take(self, count):
        if count < 0 or count > len(self.body) - self.position:
            _fail("truncated wire record")
        result = self.body[self.position:self.position + count]
        self.position += count
        return result

    def number(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def scalar(self):
        count = self.number(">Q")
        if count > self.p["max_scalar_bytes"]:
            _fail("declared scalar bytes exceed IO policy")
        return self.take(count)

    def shape(self):
        rank = self.number(">I")
        if rank > self.p["max_rank"]:
            _fail("declared rank exceeds IO policy")
        result = tuple(self.number(">Q") for _ in range(rank))
        _shape(result, self.p)
        return result

    def node(self, depth=1):
        # Readiness C2/D3: take, number and scalar are inlined for the scalar and container
        # tags, with the original bounds checks and messages; tensors use the methods.
        p = self.p
        meter = self.meter
        meter.nodes += 1
        if meter.nodes > p["max_nodes"] or depth > p["max_depth"]:
            _fail("wire node/depth limit exceeded")
        body = self.body
        end = len(body)
        position = self.position
        if position + 1 > end:
            _fail("truncated wire record")
        tag = _TAGS[body[position]]
        position += 1
        if tag == b"n":
            self.position = position
            return None
        if tag == b"t" or tag == b"f":
            self.position = position
            return tag == b"t"
        if tag == b"i":
            if position + 1 > end:
                _fail("truncated wire record")
            sign = body[position]
            self.position = position + 1
            raw = self.scalar()
            if sign > 1 or not raw or (len(raw) > 1 and raw[0] == 0):
                _fail("noncanonical integer")
            value = int.from_bytes(raw, "big")
            if sign and value == 0:
                _fail("negative zero integer")
            return -value if sign else value
        if tag == b"r":
            if position + 8 > end:
                _fail("truncated wire record")
            value = _UNPACK_D(body, position)[0]
            self.position = position + 8
            if not math.isfinite(value):
                _fail("nonfinite wire float")
            return value
        if tag == b"s" or tag == b"b":
            if position + 8 > end:
                _fail("truncated wire record")
            count = _UNPACK_Q(body, position)[0]
            position += 8
            if count > p["max_scalar_bytes"]:
                _fail("declared scalar bytes exceed IO policy")
            if count > end - position:
                _fail("truncated wire record")
            self.position = position + count
            raw = body[position:position + count]
            if tag == b"b":
                return bytes(raw)
            try:
                return str(raw, "utf-8")
            except UnicodeDecodeError as exc:
                raise CheckpointValidationError("invalid wire UTF8") from exc
        if tag == b"m" or tag == b"l" or tag == b"u":
            if position + 4 > end:
                _fail("truncated wire record")
            count = _UNPACK_I(body, position)[0]
            self.position = position + 4
            if count > p["max_nodes"]:
                _fail("declared container count exceeds IO policy")
            node = self.node
            child = depth + 1
            if tag == b"m":
                result = {}
                for _ in range(count):
                    key = node(child)
                    if (type(key) is not str and not _key(key)) or key in result:
                        _fail("unsupported or duplicate wire dictionary key")
                    result[key] = node(child)
                return result
            result = [node(child) for _ in range(count)]
            return tuple(result) if tag == b"u" else result
        self.position = position
        if tag == b"D":
            code = self.number(">B")
            if code not in _DTYPES:
                _fail("unsupported wire dtype")
            shape = self.shape()
            count = _shape(shape, self.p)
            size = self.number(">Q")
            if size != count * _SIZES[code]:
                _fail("raw storage bytes do not exactly cover canonical tensor extent")
            self.meter.tensor(size)
            return _Dense(code, shape, self.take(size))
        if tag == b"C":
            shape = self.shape()
            indices = self.node(depth + 1)
            values = self.node(depth + 1)
            if type(indices) is not _Dense or type(values) is not _Dense:
                _fail("COO components must be dense wire tensors")
            if indices.code != 2 or len(indices.shape) != 2 or not values.shape:
                _fail("invalid COO component descriptors")
            sparse_dim, nnz = indices.shape
            if sparse_dim > len(shape) or values.shape != (nnz,) + shape[sparse_dim:]:
                _fail("COO shape/component extent mismatch")
            return _Sparse(shape, indices, values)
        _fail("unsupported wire tag; native archives/pickles are not accepted")


def _check_coo_coordinates(indices, shape):
    """Each column in bounds and strictly after the previous one, lexicographically.

    Readiness C2/D3: the accepted per-column Python loop, vectorized. The first column
    that fails either test decides the message, bounds before order, as in the loop.
    """
    rows, cols = indices.shape
    if cols == 0:
        return
    limits = torch.tensor(shape[:rows], dtype=torch.int64).reshape(rows, 1)
    out_of_bounds = ((indices < 0) | (indices >= limits)).any(dim=0)
    bad = out_of_bounds.clone()
    if cols > 1:
        later, earlier = indices[:, 1:], indices[:, :-1]
        decided = torch.zeros(cols - 1, dtype=torch.bool)
        increasing = torch.zeros(cols - 1, dtype=torch.bool)
        for row in range(rows):
            greater, less = later[row] > earlier[row], later[row] < earlier[row]
            increasing |= greater & ~decided
            decided |= greater | less
        bad[1:] |= ~increasing
    failing = torch.nonzero(bad)
    if failing.numel():
        if out_of_bounds[int(failing[0, 0])]:
            _fail("COO index out of bounds")
        _fail("COO coordinates are not strictly canonical/coalesced")


def _materialize(value):
    t = type(value)
    if t is _Dense:
        if not value.data:
            return torch.empty(value.shape, dtype=_DTYPES[value.code], device="cpu")
        # Only reached after the ENTIRE file's descriptor and aggregate preflight.
        backing = bytearray(value.data)
        return torch.frombuffer(backing, dtype=_DTYPES[value.code]).reshape(value.shape)
    if t is _Sparse:
        indices = _materialize(value.indices)
        values = _materialize(value.values)
        _check_coo_coordinates(indices, value.shape)
        return torch.sparse_coo_tensor(indices, values, value.shape, device="cpu",
                                       is_coalesced=True, check_invariants=True)
    if t is dict:
        return {_materialize(k): _materialize(v) for k, v in value.items()}
    if t is list:
        return [_materialize(v) for v in value]
    if t is tuple:
        return tuple(_materialize(v) for v in value)
    return value


def _decode(data, p):
    if type(data) is not bytes or len(data) > p["max_file_bytes"]:
        _fail("file bytes exceed IO policy")
    if len(data) < _HEADER_SIZE or data[:8] != _MAGIC:
        _fail("unsupported checkpoint format or byte profile")
    _wire_platform()
    length = struct.unpack(">Q", data[8:16])[0]
    if length != len(data) - _HEADER_SIZE:
        _fail("checkpoint envelope length mismatch")
    body = memoryview(data)[_HEADER_SIZE:]
    if hashlib.sha256(body).digest() != data[16:48]:
        _fail("checkpoint integrity digest mismatch")
    parser = _Parser(body, p)
    tree = parser.node()
    if parser.position != len(body):
        _fail("trailing wire bytes")
    try:
        return _materialize(tree)
    except (RuntimeError, ValueError, TypeError, OverflowError) as exc:
        raise CheckpointValidationError("bounded CPU tensor reconstruction failed") from exc


def _write_all(fd, data):
    view = memoryview(data)
    position = 0
    while position < len(view):
        written = os.write(fd, view[position:position + 65536])
        if written <= 0:
            raise CheckpointSecurityError("checkpoint write made no progress")
        position += written


def _wire_platform():
    if sys.byteorder != "little":
        _fail("A4UCP v2 L requires a little-endian native tensor platform")


class _StatFS(ctypes.Structure):
    # Only the explicitly checked Linux x86_64 glibc ABI is supported.
    _fields_ = [
        ("f_type", ctypes.c_long), ("f_bsize", ctypes.c_long),
        ("f_blocks", ctypes.c_ulong), ("f_bfree", ctypes.c_ulong),
        ("f_bavail", ctypes.c_ulong), ("f_files", ctypes.c_ulong),
        ("f_ffree", ctypes.c_ulong), ("f_fsid", ctypes.c_int * 2),
        ("f_namelen", ctypes.c_long), ("f_frsize", ctypes.c_long),
        ("f_flags", ctypes.c_long), ("f_spare", ctypes.c_long * 4),
    ]


def _native_platform():
    try:
        supported = (
            sys.platform == "linux" and os.uname().machine == "x86_64"
            and os.confstr("CS_GNU_LIBC_VERSION").startswith("glibc ")
            and ctypes.sizeof(ctypes.c_int) == 4
            and ctypes.sizeof(ctypes.c_long) == 8
            and ctypes.sizeof(ctypes.c_ulong) == 8
            and ctypes.sizeof(ctypes.c_void_p) == 8
            and ctypes.sizeof(_StatFS) == 120
            and ctypes.alignment(_StatFS) == 8
            and _StatFS.f_fsid.offset == 56
            and _StatFS.f_namelen.offset == 64
            and _StatFS.f_flags.offset == 80
            and _StatFS.f_spare.offset == 88
        )
    except (AttributeError, OSError, ValueError):
        supported = False
    if not supported:
        raise CheckpointSecurityError("requires the Linux x86_64 glibc statfs ABI")


def _native_function(name, arguments):
    _native_platform()
    try:
        function = getattr(ctypes.CDLL(None, use_errno=True), name)
    except (AttributeError, OSError) as exc:
        raise CheckpointSecurityError("required Linux libc facility unavailable") from exc
    function.argtypes = arguments
    function.restype = ctypes.c_int
    return function


@contextmanager
def _proc_fds():
    # Internal transport only; requires a trusted genuine /proc mount namespace.
    # No caller pathname or checkpoint value participates in this lookup.
    _native_platform()
    fd = os.open("/proc/self/fd", os.O_RDONLY | os.O_DIRECTORY |
                 os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        information = _StatFS()
        fstatfs = _native_function("fstatfs", (ctypes.c_int, ctypes.POINTER(_StatFS)))
        if fstatfs(fd, ctypes.byref(information)) != 0:
            raise OSError(ctypes.get_errno(), "procfs classification failed")
        if information.f_type != 0x9FA0:
            raise CheckpointSecurityError("descriptor transport requires genuine procfs")
        yield fd
    finally:
        os.close(fd)


def _readable(path_fd):
    with _proc_fds() as proc_fd:
        return os.open(str(path_fd), os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC,
                       dir_fd=proc_fd)


def _anonymous_temp(root_fd):
    if not hasattr(os, "O_TMPFILE"):
        raise CheckpointSecurityError("anonymous checkpoint temporary unavailable")
    # O_TMPFILE|O_EXCL would prohibit subsequent publication.
    fd = os.open(".", os.O_TMPFILE | os.O_RDWR | os.O_CLOEXEC,
                 0o600, dir_fd=root_fd)
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 0 or st.st_size != 0
                or st.st_uid != os.geteuid() or st.st_mode & 0o022
                or st.st_dev != os.fstat(root_fd).st_dev):
            raise CheckpointSecurityError("anonymous temporary inode refused")
        return fd
    except (CheckpointError, OSError):
        os.close(fd)
        raise


def _link_fd(fd, root_fd, name):
    # Documented unprivileged Linux O_TMPFILE publication through a held fd.
    # No mutable temporary name, AT_EMPTY_PATH capability, or named fallback.
    linkat = _native_function("linkat", (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int))
    with _proc_fds() as proc_fd:
        if linkat(proc_fd, str(fd).encode("ascii"), root_fd,
                  os.fsencode(name), 0x400) != 0:  # AT_SYMLINK_FOLLOW
            raise OSError(ctypes.get_errno(), "descriptor-bound exclusive link failed")


def acquire_sources(source_roots, source_sizes, *, git_evidence, environment_records,
                    expected_provenance, io_policy):
    """Read direct children of independently approved prefix directory descriptors.

    source_roots is an exact dict with distinct directory inode bindings for both
    'scripts' and 'configs'. source_sizes retains the full registered D1 logical IDs
    and independently approved exact byte lengths. No project-root traversal occurs.
    """
    p = _policy(io_policy)
    if type(source_sizes) is not dict or not source_sizes or len(source_sizes) > p["max_source_files"]:
        raise CheckpointSecurityError("source request must be a nonempty bounded exact dict")
    requests = []
    total = 0
    for path, size in source_sizes.items():
        if type(path) is not str or type(size) is not int or size < 0:
            raise CheckpointSecurityError("source requests require exact string paths and nonnegative integer sizes")
        normalized = validate_allowlisted_relative_path(path)
        if normalized != path or "\\" in path or len(path.split("/")) != 2:
            raise CheckpointSecurityError("source acquisition requires registered two-component POSIX paths")
        prefix, basename = path.split("/")
        total += size
        if total > p["max_source_bytes"]:
            raise CheckpointSecurityError("aggregate source request exceeds IO policy")
        requests.append((path, prefix, basename, size))
    if (type(source_roots) is not dict
            or any(type(prefix) is not str for prefix in source_roots)
            or set(source_roots) != {"scripts", "configs"}):
        raise CheckpointSecurityError("source_roots requires exactly scripts and configs directory bindings")
    if any(type(fd) is not int or fd < 0 for fd in source_roots.values()):
        raise CheckpointSecurityError("source roots must be independently approved directory descriptors")
    bindings = dict(source_roots)
    with _os_errors(), ExitStack() as stack:
        roots = {prefix: stack.enter_context(_Root(bindings[prefix]))
                 for prefix in sorted(bindings)}
        identities = [(os.fstat(root.fd).st_dev, os.fstat(root.fd).st_ino)
                      for root in roots.values()]
        if len(set(identities)) != len(identities):
            raise CheckpointSecurityError("source prefixes require distinct approved directory inodes")
        anchors = {prefix: stack.enter_context(_Anchor(root))
                   for prefix, root in roots.items()}

        def check_all():
            for anchor in anchors.values():
                anchor.check()

        opened = []
        # Complete syntax, root-binding and aggregate request preflight precedes IO.
        # Every physical size is checked before the first source read or hash.
        for path, prefix, basename, size in requests:
            anchor = anchors[prefix]
            fd, _ = anchor.file(roots[prefix].fd, basename, size, exact=size)
            opened.append((path, fd, size))
        check_all()
        sources = {path: _read_fd(fd, size) for path, fd, size in opened}
        check_all()
        result = build_source_provenance(sources, git_evidence, environment_records)
        validate_source_provenance(result, expected_provenance)
        check_all()
        return result


def publish_checkpoint(root_fd, name, payload, *, expected_identity, expected_provenance,
                       expected_model, expected_runtime, resource_policy, io_policy):
    """Validate and exclusively publish a flat basename; never replace a destination.

    Returns the basename. An error after the atomic link may leave a completed
    checkpoint: completed artifacts are never automatically rolled back.
    """
    p = _policy(io_policy)
    name = _name(name)
    with _os_errors(), _Root(root_fd) as root:
        owned = validate_checkpoint_payload(
            payload, expected_identity=expected_identity, expected_provenance=expected_provenance,
            expected_model=expected_model, expected_runtime=expected_runtime,
            resource_policy=resource_policy)
        data = _encode(owned, p)
        fd = _anonymous_temp(root.fd)
        try:
            _write_all(fd, data)
            if os.fstat(fd).st_size != len(data):
                raise CheckpointSecurityError("incomplete checkpoint write")
            os.fsync(fd)
            _directory(os.fstat(root.fd))
            # Atomic no-clobber publication of this held inode, not a mutable name.
            _link_fd(fd, root.fd, name)
            return name
        finally:
            # Incomplete anonymous inodes disappear on close. Completed names stay.
            os.close(fd)


def retrieve_checkpoint(root_fd, name, *, expected_identity, expected_provenance,
                        expected_model, expected_runtime, resource_policy, io_policy):
    """Bound physical bytes and every tensor extent before reconstruction, then D1/D2."""
    p = _policy(io_policy)
    name = _name(name)
    with _os_errors(), _Root(root_fd) as root, _Anchor(root) as anchor:
        fd, size = anchor.file(root.fd, name, p["max_file_bytes"])
        anchor.check()
        data = _read_fd(fd, size)
        anchor.check()
        decoded = _decode(data, p)
        result = validate_checkpoint_payload(
            decoded, expected_identity=expected_identity, expected_provenance=expected_provenance,
            expected_model=expected_model, expected_runtime=expected_runtime,
            resource_policy=resource_policy)
        anchor.check()
        return result
