"""GR06: publish and retrieve cuda-v1 checkpoints with the accepted D3 mechanism.

D3's public functions call the v1 validator directly, so a cuda-v1 payload cannot pass
through them. These two functions are D3's `publish_checkpoint` and
`retrieve_checkpoint` with only the validator swapped. Root anchoring, bounded
encoding and decoding, the no-clobber link and the refusal rules are D3's own private
helpers, unchanged. `tests/test_checkpoint_cuda.py` pins `checkpoint_storage.py` to its
recorded SHA-256, so a D3 change fails a test instead of drifting under this module.
"""

import os

from scripts import checkpoint_storage as d3
from scripts.checkpoint_payload_cuda import validate_cuda_checkpoint_payload


def publish_cuda_checkpoint(root_fd, name, payload, *, expected_device, expected_identity,
                            expected_provenance, expected_model, expected_runtime,
                            resource_policy, io_policy):
    """Validate and exclusively publish; never replace a destination (as D3)."""
    p = d3._policy(io_policy)
    name = d3._name(name)
    with d3._os_errors(), d3._Root(root_fd) as root:
        owned = validate_cuda_checkpoint_payload(
            payload, expected_device=expected_device, expected_identity=expected_identity,
            expected_provenance=expected_provenance, expected_model=expected_model,
            expected_runtime=expected_runtime, resource_policy=resource_policy)
        data = d3._encode(owned, p)
        fd = d3._anonymous_temp(root.fd)
        try:
            d3._write_all(fd, data)
            if os.fstat(fd).st_size != len(data):
                raise d3.CheckpointSecurityError("incomplete checkpoint write")
            os.fsync(fd)
            d3._directory(os.fstat(root.fd))
            d3._link_fd(fd, root.fd, name)
            return name
        finally:
            os.close(fd)


def retrieve_cuda_checkpoint(root_fd, name, *, expected_device, expected_identity,
                             expected_provenance, expected_model, expected_runtime,
                             resource_policy, io_policy):
    """Bound bytes and tensor extents before reconstruction, then validate (as D3)."""
    p = d3._policy(io_policy)
    name = d3._name(name)
    with d3._os_errors(), d3._Root(root_fd) as root, d3._Anchor(root) as anchor:
        fd, size = anchor.file(root.fd, name, p["max_file_bytes"])
        anchor.check()
        data = d3._read_fd(fd, size)
        anchor.check()
        decoded = d3._decode(data, p)
        result = validate_cuda_checkpoint_payload(
            decoded, expected_device=expected_device, expected_identity=expected_identity,
            expected_provenance=expected_provenance, expected_model=expected_model,
            expected_runtime=expected_runtime, resource_policy=resource_policy)
        anchor.check()
        return result
