"""Checkpoint readiness C1: D1 to D3 expectations built from the real profiling dataset.

D2 and D3 validate a checkpoint against caller expectations: the D1 identity, the model's
row order and fixed buffers, the runtime, and resource and IO policies. GPU-READY took all
of them from the accepted D2/D3 test fixtures. Here the identity and model expectations
come from `ProfilingData` instead, and the policies are sized for Musical_Instruments.
Provenance, the runtime profile, RNG bindings and the remaining resolved settings still
come from the accepted fixture (imported unchanged, as `gpu_ready_parity` does); freezing
them for training is a later gate. Seeds and batch size are set in every place D1 and D2
read them. See docs/checkpoint-real-data-readiness-plan.md.
"""

import copy
import sys
from pathlib import Path

import torch

from scripts.checkpoint_identity import RELATION_SCHEMAS, build_canonical_identity
from scripts.differentiable_models import build_hetero_graph
from scripts.learned_models import LightGCNGraph
from scripts.training_harness import D2_KEYS, PrivateRNG
from scripts.vector_trainer import VectorTrainer
from scripts.vectorized_models import VectorHeteroGNN, VectorLightGCN

LIGHTGCN = 'LightGCN'
HETERO = 'Heterogeneous KG+GNN'
METHODS = (LIGHTGCN, HETERO)
# LightGCN's sparse adjacency counts as users x items cells (57,439 x 24,556 here), so the
# tensor-element bound must exceed 2^30. The rest leave several times the measured need.
RESOURCE_POLICY = {
    'schema_version': 'amazon4u-payload-resources-v1',
    'profile_id': 'amazon4u-payload-musical-instruments-v1',
    'kind': 'runtime',
    'max_input_bytes': 1 << 31,
    'max_output_bytes': 1 << 31,
    'max_container_depth': 32,
    'max_visited_values': 1 << 26,
}
IO_CHANGES = {
    'profile_id': 'amazon4u-io-musical-instruments-v1',
    'max_file_bytes': 1 << 31,
    'max_nodes': 1 << 26,
    'max_storage_bytes': 1 << 30,
    'max_copy_bytes': 1 << 30,
    'max_tensor_elements': 1 << 31,
    'max_dimension': 1 << 24,
}


def _fixture(method):
    tests_dir = str(Path(__file__).resolve().parents[1] / 'tests')
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_checkpoint_storage import _fixture as accepted
    _, expectations = accepted(hetero=method != LIGHTGCN, initial=True)
    return copy.deepcopy(expectations)


def _records(rows):
    return [{'user_id': u, 'parent_asin': p, 'rating': r, 'timestamp': ts} for u, p, r, ts in rows]


def identity_fields(data, method, condition, context, kg, settings):
    """Keyword arguments for D1 `build_canonical_identity`, from the loaded dataset."""
    edges = [{'src_type': RELATION_SCHEMAS[r][0], 'src_id': s, 'relation': r,
              'dst_type': RELATION_SCHEMAS[r][1], 'dst_id': d, 'weight': 1.0}
             for r, pairs in kg.edges.items() for s, d in sorted(pairs)]
    return dict(
        method=method, category=data.manifest['category'], feedback_condition=condition,
        seed=settings['determinism']['seed'],
        training_records=_records(data.train),
        # D1 keys both histories by every training user; >=4 users without positives map to [].
        positive_history={u: sorted(context.positive_history.get(u, ())) for u in context.all_rated_history},
        all_rated_history={u: sorted(v) for u, v in context.all_rated_history.items()},
        warm_items=list(context.sorted_warm_items), graph_nodes=kg.node_ids, graph_edges=edges,
        product_metadata={p: {'categories': data.product_categories.get(p, []),
                              'brand': data.product_brands.get(p)} for p in kg.products},
        validation_inputs={'targets': {u: [p] for u, p, _, _ in data.validation},
                           'records': _records(data.validation)},
        resolved_settings=settings)


def _settings(expectations, *, seed, batch_size, device):
    settings = expectations['expected_identity']['resolved_settings']
    for stream in settings['determinism']['rng_streams']:
        if stream['seed'] is not None:
            stream['seed'] = seed
    settings['determinism']['seed'] = seed
    settings['training']['batch_size'] = batch_size
    settings['training']['validation_frequency'] = 1
    settings['execution']['execution_device'] = str(device)
    for generator in expectations['expected_runtime']['rng_generator_bindings'].values():
        generator['seed_material'] = seed
    return settings


def lightgcn_adjacency(graph):
    """The fixed user-item buffer D2 expects: symmetric degree weights, coalesced."""
    u_map = {u: i for i, u in enumerate(graph.users)}
    i_map = {p: i for i, p in enumerate(graph.warm_items)}
    pairs = [(u_map[u], i_map[p], float(w)) for u in graph.users for p, w in graph.user_neighbors[u]]
    rows, cols, weights = zip(*pairs)
    return torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(weights, dtype=torch.float32),
                                   (len(graph.users), len(graph.warm_items))).coalesce()


def build(data, method, condition, *, seed=42, batch_size=2048, device=torch.device('cpu'),
          fingerprint, determinism=None):
    """(trainer, expectations, identity dict) for one method and condition on real data.

    `expectations` holds every D2/D3 keyword (`D2_KEYS` plus `io_policy`). The trainer is a
    fresh EX01 `VectorTrainer` whose model rows follow `expected_model['row_order']`.
    """
    if method not in METHODS:
        raise ValueError(f'unknown method {method!r}')
    expectations = _fixture(method)
    settings = _settings(expectations, seed=seed, batch_size=batch_size, device=device)
    context = data.context(condition)
    kg = build_hetero_graph(context, data.product_categories, data.product_brands)
    identity_obj = build_canonical_identity(**identity_fields(data, method, condition, context, kg, settings))
    identity = identity_obj.to_dict()
    if method == LIGHTGCN:
        graph = LightGCNGraph(context)
        gen = torch.Generator().manual_seed(seed)
        model = VectorLightGCN(graph, torch.normal(0.0, 0.1, (len(graph.users), 64), generator=gen),
                               torch.normal(0.0, 0.1, (len(graph.warm_items), 64), generator=gen),
                               device)
        rows = {'User': list(graph.users), 'Product': list(graph.warm_items)}
        buffers = {'adj_user_item': lightgcn_adjacency(graph)}
    else:
        model = VectorHeteroGNN(kg, seed, device)
        rows = {k: list(v) for k, v in kg.node_ids.items()}
        buffers = {}
    expectations.update(
        expected_identity=identity,
        expected_model={'method': method, 'row_order': rows, 'buffer_state': buffers},
        resource_policy=copy.deepcopy(RESOURCE_POLICY))
    expectations['io_policy'].update(IO_CHANGES)
    if set(expectations) != {*D2_KEYS, 'io_policy'}:
        raise ValueError('expectations must hold exactly the D2 keywords and io_policy')
    trainer = VectorTrainer(model, data.validation_targets(),
                            rng=PrivateRNG(identity_obj, expectations['expected_runtime']),
                            batch_size=batch_size, validation_frequency=1,
                            fingerprint=fingerprint, determinism=determinism)
    return trainer, expectations, identity
