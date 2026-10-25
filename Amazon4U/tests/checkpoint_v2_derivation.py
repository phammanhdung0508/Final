"""How identity v2's semantic payload checks are derived from D2 v1's.

`derive(v1_source)` cuts v1's `_validate_semantic_payload` out of the D2 source and makes
exactly the substitutions below. `scripts/checkpoint_v2.py` holds the result verbatim
between its DERIVED markers; `tests/test_checkpoint_v2.py` checks that it still does, so
a later change to either side fails a test instead of drifting.
"""

START = 'def _validate_semantic_payload('
END = '# ---------------------------------------------------------------------------\n# Public Entry Point'
V1_IDENTITY_BLOCK = '''    # Reconstruct and validate Identity & Provenance through accepted D1 APIs
    raw_actual_id = payload["identity"]
    if type(raw_actual_id) is not dict:
        raise CheckpointValidationError("payload.identity must be a dict")
    actual_identity = CanonicalResumeIdentity.from_dict(raw_actual_id)
    if built_expected is not None and "identity" in built_expected:
        exp_identity = built_expected["identity"]
    else:
        exp_identity = CanonicalResumeIdentity.from_dict(expected_identity)
        if built_expected is not None:
            built_expected["identity"] = exp_identity

    # Both identities were just returned by from_dict and are unchanged (readiness C2).
    _validate_identity_pair(actual_identity, exp_identity, content_verified=True)
'''
V2_IDENTITY_BLOCK = '''    # Identity v2: the stored reference must match the caller's trusted identity. The
    # trusted identity's content serves every content check below.
    actual_identity = validate_identity_reference(payload["identity"], expected_identity)
'''
SUBSTITUTIONS = (
    ('def _validate_semantic_payload(', 'def _validate_semantic_payload_v2('),
    ('    expected_identity: Dict[str, Any],\n', '    expected_identity: CanonicalResumeIdentity,\n'),
    ('    built_expected: Optional[Dict[str, CanonicalResumeIdentity]] = None,\n', ''),
    ('''

    Readiness C2: `built_expected`, when given, keeps the expected identity from_dict built
    in the first pass, so the second pass over the owned copy reuses it unchanged.
    """''', '"""'),
    ('PAYLOAD_SCHEMA_VERSION', 'PAYLOAD_SCHEMA_VERSION_V2'),
    (V1_IDENTITY_BLOCK, V2_IDENTITY_BLOCK),
)


def derive(v1_source):
    body = v1_source[v1_source.index(START):v1_source.index(END)].rstrip() + '\n'
    for old, new in SUBSTITUTIONS:
        if old not in body:
            raise AssertionError(f'v1 source no longer contains {old[:60]!r}')
        body = body.replace(old, new)
    return body
