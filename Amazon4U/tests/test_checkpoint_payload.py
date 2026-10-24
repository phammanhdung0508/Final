"""Unit, adversarial, and regression tests for pure checkpoint payload and state validation contracts.

Contains:
- All 27 retained recovery test methods covering dual import, positive controls,
  resource accounting boundaries, custom object/hook safety, fail-closed handling,
  model/tensor mutations, Adam validation, RNG/codecs, sampler/progress, selection
  history, best model snapshots, raw detachment/forged sparse checks, output oracle,
  visits limits, mixed lazy Adam, W=1/W=2 staged workers, batch_then_filter, valid empty
  hetero tables, storage deduplication, and strict schemas.
- All 7 ported diagnostic/oracle test methods:
  - test_r1_metaclass_hash_hook (C8)
  - test_r2_huge_update_domain_error (C8)
  - test_r3_torch_worker_seed_integer (C2, C5)
  - test_r4_python_worker_seed_integer (C2, C5)
  - test_r5a_old_empty_pool_fixture_independent_derivation (C9)
  - test_r5b_genuine_untrainable_t0_domain_refusal (C9)
  - test_r6_btf_fixture_adjacency_mismatch_oracle (C3, C9)
The retained inventory above describes the original 34 payload methods, not the
current additive suite. Pi reported 329 methods in the current pre-revision full
discovery, including all 320 preceding methods. This revision adds further methods;
its final discovered count, results, and coverage remain pending Pi verification.
"""

from collections import defaultdict
import copy
import math
from pathlib import Path
import random
import sys
import unittest
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import checkpoint_payload
import scripts.checkpoint_payload

from checkpoint_identity import (
    CHECKPOINT_SCHEMA_VERSION,
    CONFIGURED_RNG_COMPONENTS,
    NODE_TYPES,
    RELATION_SCHEMAS,
    CanonicalResumeIdentity,
    CheckpointError,
    CheckpointIdentityMismatchError,
    CheckpointProvenanceError,
    CheckpointSecurityError,
    CheckpointValidationError,
    build_canonical_identity,
    build_source_provenance,
)
from checkpoint_payload import (
    PAYLOAD_CAPABILITY,
    PAYLOAD_REQUIRED_KEYS,
    PAYLOAD_SCHEMA_VERSION,
    RESOURCE_POLICY_SCHEMA_VERSION,
    SUPPORTED_CODECS,
    SYNTHETIC_RESOURCE_PROFILE_ID,
    _ceil_div,
    _check_strict_int,
    preflight_resource_accounting,
    validate_checkpoint_payload,
    validate_resource_policy,
)
from differentiable_models import (
    DifferentiableHeteroGNN,
    DifferentiableLightGCN,
    HeteroTrainingGraph,
    LightGCNGraph,
    TrainingContext,
)


# ---------------------------------------------------------------------------
# Synthetic Test Fixture Builders
# ---------------------------------------------------------------------------

def _make_synthetic_resource_policy(
    max_input_bytes: int = 2097152,
    max_output_bytes: int = 2097152,
    max_container_depth: int = 32,
    max_visited_values: int = 16384,
    kind: str = "synthetic",
    profile_id: str = SYNTHETIC_RESOURCE_PROFILE_ID,
) -> Dict[str, Any]:
    """Create a valid synthetic or runtime resource policy."""
    return {
        "schema_version": RESOURCE_POLICY_SCHEMA_VERSION,
        "profile_id": profile_id,
        "kind": kind,
        "max_input_bytes": max_input_bytes,
        "max_output_bytes": max_output_bytes,
        "max_container_depth": max_container_depth,
        "max_visited_values": max_visited_values,
    }


def _make_synthetic_identity_and_provenance(
    method: str = "LightGCN",
    runtime_workers: int = 0,
    empty_metadata: bool = False,
    with_empty_pool: bool = False,
) -> Tuple[CanonicalResumeIdentity, Dict[str, Any]]:
    """Build small valid canonical identity and source provenance fixtures with complete training graph."""
    if with_empty_pool:
        training_records = [
            {"user_id": "u1", "parent_asin": "p1", "rating": 5.0, "timestamp": 1000},
            {"user_id": "u1", "parent_asin": "p2", "rating": 5.0, "timestamp": 2000},
            {"user_id": "u2", "parent_asin": "p3", "rating": 5.0, "timestamp": 3000},
            {"user_id": "u3", "parent_asin": "p1", "rating": 5.0, "timestamp": 4000},
            {"user_id": "u3", "parent_asin": "p2", "rating": 5.0, "timestamp": 5000},
            {"user_id": "u3", "parent_asin": "p3", "rating": 2.0, "timestamp": 6000},
        ]
        positive_history = {
            "u1": ["p1", "p2"],
            "u2": ["p3"],
            "u3": ["p1", "p2"],
        }
        all_rated_history = {
            "u1": ["p1", "p2"],
            "u2": ["p3"],
            "u3": ["p1", "p2", "p3"],
        }
        warm_items = ["p1", "p2", "p3"]
        graph_nodes = {
            "User": ["u1", "u2", "u3"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": ['c:["Electronics"]', 'c:["Electronics","Audio"]'],
            "Brand": ["b:Sony"],
        }
        graph_edges = [
            ("User", "u1", "rated", "Product", "p1", 1.0),
            ("User", "u1", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p3", 1.0),
            ("User", "u3", "rated", "Product", "p1", 1.0),
            ("User", "u3", "rated", "Product", "p2", 1.0),
            ("User", "u3", "rated", "Product", "p3", 1.0),
            ("Product", "p1", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u1", 1.0),
            ("Product", "p3", "rated_by", "User", "u2", 1.0),
            ("Product", "p1", "rated_by", "User", "u3", 1.0),
            ("Product", "p2", "rated_by", "User", "u3", 1.0),
            ("Product", "p3", "rated_by", "User", "u3", 1.0),
            ("Product", "p1", "belongs_to", "CategoryPath", 'c:["Electronics","Audio"]', 1.0),
            ("Product", "p1", "belongs_to", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("Product", "p2", "belongs_to", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("CategoryPath", 'c:["Electronics","Audio"]', "contains_product", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics"]', "contains_product", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics"]', "contains_product", "Product", "p2", 1.0),
            ("Product", "p1", "has_brand", "Brand", "b:Sony", 1.0),
            ("Brand", "b:Sony", "brand_of", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics","Audio"]', "category_child_of", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("CategoryPath", 'c:["Electronics"]', "category_parent_of", "CategoryPath", 'c:["Electronics","Audio"]', 1.0),
        ]
        product_metadata = {
            "p1": {"categories": ['c:["Electronics","Audio"]'], "brand": "b:Sony"},
            "p2": {"categories": ['c:["Electronics"]'], "brand": None},
            "p3": {"categories": [], "brand": None},
        }
        validation_inputs = {
            "targets": {"u1": ["p3"], "u2": ["p1"]},
            "records": [("u1", "p3", 5.0, 5000), ("u2", "p1", 4.0, 6000)],
        }
    elif empty_metadata:
        training_records = [
            {"user_id": "u1", "parent_asin": "p1", "rating": 4.0, "timestamp": 1000},
            {"user_id": "u1", "parent_asin": "p2", "rating": 5.0, "timestamp": 2000},
            {"user_id": "u2", "parent_asin": "p2", "rating": 4.0, "timestamp": 3000},
            {"user_id": "u2", "parent_asin": "p3", "rating": 4.0, "timestamp": 4000},
        ]
        positive_history = {
            "u1": ["p1", "p2"],
            "u2": ["p2", "p3"],
        }
        all_rated_history = {
            "u1": ["p1", "p2"],
            "u2": ["p2", "p3"],
        }
        warm_items = ["p1", "p2", "p3"]
        graph_nodes = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": [],
            "Brand": [],
        }
        graph_edges = [
            ("User", "u1", "rated", "Product", "p1", 1.0),
            ("User", "u1", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p3", 1.0),
            ("Product", "p1", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u2", 1.0),
            ("Product", "p3", "rated_by", "User", "u2", 1.0),
        ]
        product_metadata = {
            "p1": {"categories": [], "brand": None},
            "p2": {"categories": [], "brand": None},
            "p3": {"categories": [], "brand": None},
        }
        validation_inputs = {
            "targets": {"u1": ["p3"], "u2": ["p1"]},
            "records": [("u1", "p3", 5.0, 5000), ("u2", "p1", 4.0, 6000)],
        }
    else:
        training_records = [
            {"user_id": "u1", "parent_asin": "p1", "rating": 4.0, "timestamp": 1000},
            {"user_id": "u1", "parent_asin": "p2", "rating": 5.0, "timestamp": 2000},
            {"user_id": "u2", "parent_asin": "p2", "rating": 4.0, "timestamp": 3000},
            {"user_id": "u2", "parent_asin": "p3", "rating": 4.0, "timestamp": 4000},
        ]
        positive_history = {
            "u1": ["p1", "p2"],
            "u2": ["p2", "p3"],
        }
        all_rated_history = {
            "u1": ["p1", "p2"],
            "u2": ["p2", "p3"],
        }
        warm_items = ["p1", "p2", "p3"]
        graph_nodes = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": ['c:["Electronics"]', 'c:["Electronics","Audio"]'],
            "Brand": ["b:Sony"],
        }
        graph_edges = [
            ("User", "u1", "rated", "Product", "p1", 1.0),
            ("User", "u1", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p3", 1.0),
            ("Product", "p1", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u2", 1.0),
            ("Product", "p3", "rated_by", "User", "u2", 1.0),
            ("Product", "p1", "belongs_to", "CategoryPath", 'c:["Electronics","Audio"]', 1.0),
            ("Product", "p1", "belongs_to", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("Product", "p2", "belongs_to", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("CategoryPath", 'c:["Electronics","Audio"]', "contains_product", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics"]', "contains_product", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics"]', "contains_product", "Product", "p2", 1.0),
            ("Product", "p1", "has_brand", "Brand", "b:Sony", 1.0),
            ("Brand", "b:Sony", "brand_of", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics","Audio"]', "category_child_of", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("CategoryPath", 'c:["Electronics"]', "category_parent_of", "CategoryPath", 'c:["Electronics","Audio"]', 1.0),
        ]
        product_metadata = {
            "p1": {"categories": ['c:["Electronics","Audio"]'], "brand": "b:Sony"},
            "p2": {"categories": ['c:["Electronics"]'], "brand": None},
            "p3": {"categories": [], "brand": None},
        }
        validation_inputs = {
            "targets": {"u1": ["p3"], "u2": ["p1"]},
            "records": [("u1", "p3", 5.0, 5000), ("u2", "p1", 4.0, 6000)],
        }
    if runtime_workers > 0:
        worker_settings = {
            "multiprocessing_context": "spawn",
            "persistent_workers": False,
            "prefetch_factor": 2,
            "worker_init_function": "worker_init_fn",
        }
        worker_derivation = {
            "applicable": True,
            "algorithm": "torch_worker_init_v1",
            "base_seed": 42,
            "derivation": "seed_sequence",
            "reason": None,
        }
    else:
        worker_settings = {
            "multiprocessing_context": None,
            "persistent_workers": False,
            "prefetch_factor": None,
            "worker_init_function": None,
        }
        worker_derivation = {
            "applicable": False,
            "algorithm": None,
            "base_seed": None,
            "derivation": None,
            "reason": "zero_workers",
        }

    resolved_settings = {
        "protocol": {
            "candidate_policy": "full_candidate_ranking",
            "lifecycle_state": "IMPLEMENTATION_REGRESSION_VERIFICATION",
            "protocol_version": "amazon4u-protocol-v2.0.0",
        },
        "execution": {
            "cpu_threads": 4,
            "execution_device": "cpu",
            "execution_label": "synthetic_contract_verification",
            "runtime_mode": "benchmark_verification",
            "runtime_workers": runtime_workers,
            "worker_settings": worker_settings,
        },
        "model": {
            "architecture": method,
            "embedding_dim": 64,
            "initialization": "normal_0_0.1_xavier_uniform_gain_1",
            "num_layers": 3 if method == "LightGCN" else 2,
        },
        "training": {
            "adam_betas": [0.9, 0.999],
            "adam_epsilon": 1e-08,
            "batch_size": 2,
            "endpoint_regularization": 1e-05,
            "learning_rate": 0.001,
            "loss_function": "BPR",
            "max_epochs": 100,
            "optimizer": "Adam",
            "patience": 10,
            "validation_frequency": 2,
            "weight_decay": 0.0,
        },
        "sampling": {
            "negative_strategy": "uniform_eligible",
            "negatives_per_positive": 1,
            "neighbor_fanouts": [15, 10],
            "pool_policy": "condition_warm_excluding_rated",
        },
        "determinism": {
            "deterministic_algorithms": True,
            "reproducibility_version": "amazon4u-reproducibility-v1",
            "rng_streams": [
                {"name": "Python", "applicable": True, "algorithm": "stdlib_random", "seed": 42, "derivation": "fixed_seed", "reason": None},
                {"name": "NumPy", "applicable": True, "algorithm": "numpy_pcg64", "seed": 42, "derivation": "fixed_seed", "reason": None},
                {"name": "PyTorch_CPU", "applicable": True, "algorithm": "torch_cpu", "seed": 42, "derivation": "fixed_seed", "reason": None},
                {"name": "PyTorch_CUDA_if_applicable", "applicable": False, "algorithm": None, "seed": None, "derivation": None, "reason": "not_applicable_cpu"},
                {"name": "initialization", "applicable": True, "algorithm": "model_init", "seed": 42, "derivation": "fixed_seed", "reason": None},
                {"name": "epoch_shuffle", "applicable": True, "algorithm": "epoch_shuffle", "seed": 42, "derivation": "fixed_seed", "reason": None},
                {"name": "negative_sampler", "applicable": True, "algorithm": "neg_sampler", "seed": 42, "derivation": "fixed_seed", "reason": None},
                {"name": "neighbor_sampler", "applicable": True, "algorithm": "neighbor_sampler", "seed": 42, "derivation": "fixed_seed", "reason": None},
                {"name": "DataLoader_generator", "applicable": True, "algorithm": "dl_gen", "seed": 42, "derivation": "fixed_seed", "reason": None},
                {
                    "name": "DataLoader_workers",
                    "applicable": True if runtime_workers > 0 else False,
                    "algorithm": "worker_stream" if runtime_workers > 0 else None,
                    "seed": 42 if runtime_workers > 0 else None,
                    "derivation": "worker_seed" if runtime_workers > 0 else None,
                    "reason": None if runtime_workers > 0 else "zero_workers",
                },
            ],
            "seed": 42,
            "worker_derivation": worker_derivation,
        },
    }

    identity = build_canonical_identity(
        method=method,
        category="Musical_Instruments",
        feedback_condition="rating_ge_4" if with_empty_pool else "all_ratings",
        seed=42,
        training_records=training_records,
        positive_history=positive_history,
        all_rated_history=all_rated_history,
        warm_items=warm_items,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        product_metadata=product_metadata,
        validation_inputs=validation_inputs,
        resolved_settings=resolved_settings,
    )

    source_files = {
        "scripts/checkpoint_identity.py": b"# checkpoint identity stub\n",
        "configs/compute-plan.json": b"{}\n",
    }
    git_evidence = {
        "commit": "a" * 40,
        "dirty": False,
        "tracked_diff": "",
        "untracked_sources": [],
    }
    environment = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform_system": "Linux",
        "platform_release": "6.0",
        "platform_machine": "x86_64",
        "cpu_count": 4,
        "torch_version": str(torch.__version__),
        "library_versions": {"numpy": np.__version__},
    }
    provenance = build_source_provenance(
        source_files=source_files,
        git_evidence=git_evidence,
        environment_records=environment,
    )

    return identity, provenance


def _make_lightgcn_sparse_buffer(num_users: int = 2, num_items: int = 3) -> torch.Tensor:
    """Build coalesced LightGCN sparse COO adjacency buffer with exact symmetric degree normalization."""
    # Pairs: (u1, p1), (u1, p2), (u2, p2), (u2, p3)
    # deg(u1)=2, deg(u2)=2; deg(p1)=1, deg(p2)=2, deg(p3)=1.
    # W(u1, p1) = 1/sqrt(2*1) = 1/sqrt(2)
    # W(u1, p2) = 1/sqrt(2*2) = 0.5
    # W(u2, p2) = 1/sqrt(2*2) = 0.5
    # W(u2, p3) = 1/sqrt(2*1) = 1/sqrt(2)
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.int64)
    w_outer = 1.0 / math.sqrt(2.0)
    values = torch.tensor([w_outer, 0.5, 0.5, w_outer], dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (num_users, num_items)).coalesce()


def _make_valid_lightgcn_payload(
    epoch: int = 2,
    global_step: int = 4,
    bad_checks: int = 0,
    score: float = 0.5,
    runtime_workers: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build a complete, valid LightGCN test payload and corresponding expected descriptors."""
    identity, provenance = _make_synthetic_identity_and_provenance(method="LightGCN", runtime_workers=runtime_workers)

    u_list = ["u1", "u2"]
    p_list = ["p1", "p2", "p3"]
    adj_buffer = _make_lightgcn_sparse_buffer(len(u_list), len(p_list))

    expected_model = {
        "method": "LightGCN",
        "row_order": {"User": u_list, "Product": p_list},
        "buffer_state": {"adj_user_item": adj_buffer.clone()},
    }

    # Generators
    py_state = random.Random(42).getstate()
    np_raw_state = np.random.RandomState(42).get_state()
    np_mt_state = ("MT19937", list(map(int, np_raw_state[1])), int(np_raw_state[2]), int(np_raw_state[3]), float(np_raw_state[4]))

    pcg = np.random.PCG64(42)
    pcg_state = copy.deepcopy(pcg.state)

    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(42)
    torch_native_state = torch_gen.get_state()

    generators = [
        {"id": "gen_py", "codec": "python.random.MT19937.v3", "worker_id": None, "seed_material": 42, "state": py_state},
        {"id": "gen_np_mt", "codec": "numpy.random.RandomState.MT19937.v1", "worker_id": None, "seed_material": 42, "state": np_mt_state},
        {"id": "gen_np_pcg", "codec": "numpy.random.PCG64.v1", "worker_id": None, "seed_material": 42, "state": pcg_state},
        {"id": "gen_torch", "codec": "torch.cpu.Generator.native.v1", "worker_id": None, "seed_material": 42, "state": torch_native_state},
    ]

    expected_generator_bindings = {
        "gen_py": {"codec": "python.random.MT19937.v3", "worker_id": None, "seed_material": 42},
        "gen_np_mt": {"codec": "numpy.random.RandomState.MT19937.v1", "worker_id": None, "seed_material": 42},
        "gen_np_pcg": {"codec": "numpy.random.PCG64.v1", "worker_id": None, "seed_material": 42},
        "gen_torch": {"codec": "torch.cpu.Generator.native.v1", "worker_id": None, "seed_material": 42},
    }

    worker_gen_ids: List[str] = []
    for i in range(runtime_workers):
        g_w_py_id = f"gen_worker_{i}_py"
        g_w_np_id = f"gen_worker_{i}_np"
        g_w_torch_id = f"gen_worker_{i}_torch"

        generators.append({
            "id": g_w_py_id,
            "codec": "python.random.MT19937.v3",
            "worker_id": i,
            "seed_material": 42000 + i,
            "state": random.Random(42000 + i).getstate(),
        })
        generators.append({
            "id": g_w_np_id,
            "codec": "numpy.random.PCG64.v1",
            "worker_id": i,
            "seed_material": 42000 + i,
            "state": copy.deepcopy(np.random.PCG64(42000 + i).state),
        })
        t_gen_w = torch.Generator(device="cpu")
        t_gen_w.manual_seed(42000 + i)
        generators.append({
            "id": g_w_torch_id,
            "codec": "torch.cpu.Generator.native.v1",
            "worker_id": i,
            "seed_material": 42000 + i,
            "state": t_gen_w.get_state(),
        })

        expected_generator_bindings[g_w_py_id] = {"codec": "python.random.MT19937.v3", "worker_id": i, "seed_material": 42000 + i}
        expected_generator_bindings[g_w_np_id] = {"codec": "numpy.random.PCG64.v1", "worker_id": i, "seed_material": 42000 + i}
        expected_generator_bindings[g_w_torch_id] = {"codec": "torch.cpu.Generator.native.v1", "worker_id": i, "seed_material": 42000 + i}

        worker_gen_ids.extend([g_w_py_id, g_w_np_id, g_w_torch_id])

    expected_runtime = {
        "runtime_profile": {
            "profile_id": "synthetic_profile_v1",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "adam_step_dtype": "torch.float32",
            "batch_policy": "filter_before_batch",
        },
        "optimizer_groups": [
            {
                "parameter_names": ["user_embeddings", "item_embeddings"],
                "options": {
                    "lr": 0.001,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                    "decoupled_weight_decay": False,
                },
            }
        ],
        "rng_role_bindings": {
            "Python": ["gen_py"],
            "NumPy": ["gen_np_pcg"],
            "PyTorch_CPU": ["gen_torch"],
            "PyTorch_CUDA_if_applicable": [],
            "initialization": ["gen_torch"],
            "epoch_shuffle": ["gen_py"],
            "negative_sampler": ["gen_np_mt"],
            "neighbor_sampler": ["gen_torch"],
            "DataLoader_generator": ["gen_torch"],
            "DataLoader_workers": worker_gen_ids,
        },
        "rng_generator_bindings": expected_generator_bindings,
        "sampler_algorithm": "uniform_eligible_negative_epoch_sampler_v1",
    }

    # Model parameters
    g = torch.Generator(device="cpu").manual_seed(101)
    user_emb = torch.randn(len(u_list), 64, dtype=torch.float32, generator=g)
    item_emb = torch.randn(len(p_list), 64, dtype=torch.float32, generator=g)

    current_model = {
        "epoch": epoch,
        "row_order": {"User": u_list, "Product": p_list},
        "state_dict": {
            "user_embeddings": user_emb.clone(),
            "item_embeddings": item_emb.clone(),
            "adj_user_item": adj_buffer.clone(),
        },
    }

    # Optimizer
    optimizer = {
        "algorithm": "Adam",
        "state_dict": {
            "state": {
                0: {
                    "step": torch.tensor(float(global_step), dtype=torch.float32),
                    "exp_avg": torch.zeros(len(u_list), 64, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros(len(u_list), 64, dtype=torch.float32),
                },
                1: {
                    "step": torch.tensor(float(global_step), dtype=torch.float32),
                    "exp_avg": torch.zeros(len(p_list), 64, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros(len(p_list), 64, dtype=torch.float32),
                },
            },
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": 0.001,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                    "decoupled_weight_decay": False,
                }
            ],
        },
        "parameter_bindings": [
            {"id": 0, "name": "user_embeddings", "updates": global_step},
            {"id": 1, "name": "item_embeddings", "updates": global_step},
        ],
    }

    # RNG
    roles = []
    for s in identity.resolved_settings["determinism"]["rng_streams"]:
        r_rec = dict(s)
        if s["applicable"]:
            r_rec["generator_ids"] = expected_runtime["rng_role_bindings"][s["name"]]
        else:
            r_rec["generator_ids"] = []
        roles.append(r_rec)

    rng_dict = {"roles": roles, "generators": generators}

    if epoch == 0:
        best_model = None
        sampler = {
            "algorithm": "uniform_eligible_negative_epoch_sampler_v1",
            "identity_hash": identity.identity_hash,
            "source_order": None,
            "cursor": 0,
            "next_epoch": 1,
            "batch_cursor": 0,
            "pending_batches": 0,
            "prefetch_pending": 0,
            "inflight_workers": 0,
            "worker_ids": list(range(runtime_workers)),
        }
        progress = {
            "epoch": 0,
            "global_step": 0,
            "epoch_optimizer_steps": [],
            "empty_pool_skips": [],
        }
        selection = {
            "metric": "validation_NDCG@10",
            "history": [],
            "best_score": None,
            "selected_epoch": None,
            "bad_checks": 0,
            "status": "initial",
            "stop_reason": None,
        }
        optimizer["state_dict"]["state"] = {}
        for b in optimizer["parameter_bindings"]:
            b["updates"] = 0
    else:
        best_model = {
            "epoch": epoch,
            "row_order": {"User": u_list, "Product": p_list},
            "state_dict": {
                "user_embeddings": user_emb.clone(),
                "item_embeddings": item_emb.clone(),
                "adj_user_item": adj_buffer.clone(),
            },
        } if epoch >= 2 else None

        source_order = [("u1", "p1"), ("u1", "p2"), ("u2", "p2"), ("u2", "p3")]
        sampler = {
            "algorithm": "uniform_eligible_negative_epoch_sampler_v1",
            "identity_hash": identity.identity_hash,
            "source_order": source_order,
            "cursor": 4,
            "next_epoch": epoch + 1,
            "batch_cursor": 0,
            "pending_batches": 0,
            "prefetch_pending": 0,
            "inflight_workers": 0,
            "worker_ids": list(range(runtime_workers)),
        }

        progress = {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_optimizer_steps": [2] * epoch,
            "empty_pool_skips": [0] * epoch,
        }

        sched_eps = [e for e in range(2, epoch + 1, 2)]
        history = [{"epoch": e, "score": score} for e in sched_eps]
        selection = {
            "metric": "validation_NDCG@10",
            "history": history,
            "best_score": score if sched_eps else None,
            "selected_epoch": sched_eps[0] if sched_eps else None,
            "bad_checks": bad_checks,
            "status": "ready",
            "stop_reason": None,
        }

    scheduler = {"applicable": False, "state": None, "reason": "not_used_constant_learning_rate"}
    amp = {"applicable": False, "state": None, "reason": "not_used_cpu_float32"}

    payload = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "capability": PAYLOAD_CAPABILITY,
        "identity": identity.to_dict(),
        "provenance": provenance,
        "runtime_profile": copy.deepcopy(expected_runtime["runtime_profile"]),
        "current_model": current_model,
        "best_model": best_model,
        "optimizer": optimizer,
        "rng": rng_dict,
        "sampler": sampler,
        "progress": progress,
        "selection": selection,
        "scheduler": scheduler,
        "amp": amp,
    }

    return payload, identity.to_dict(), provenance, expected_model, expected_runtime


def _make_valid_hetero_payload(empty_metadata: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build a complete, valid HeteroGNN test payload and corresponding expected descriptors."""
    identity, provenance = _make_synthetic_identity_and_provenance(method="Heterogeneous KG+GNN", empty_metadata=empty_metadata)

    if empty_metadata:
        row_order = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": [],
            "Brand": [],
        }
    else:
        row_order = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": ['c:["Electronics","Audio"]', 'c:["Electronics"]'],
            "Brand": ["b:Sony"],
        }
    expected_model = {
        "method": "Heterogeneous KG+GNN",
        "row_order": row_order,
        "buffer_state": {},
    }

    # Generators
    py_state = random.Random(42).getstate()
    np_raw_state = np.random.RandomState(42).get_state()
    np_mt_state = ("MT19937", list(map(int, np_raw_state[1])), int(np_raw_state[2]), int(np_raw_state[3]), float(np_raw_state[4]))

    pcg = np.random.PCG64(42)
    pcg_state = copy.deepcopy(pcg.state)

    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(42)
    torch_native_state = torch_gen.get_state()

    generators = [
        {"id": "gen_py", "codec": "python.random.MT19937.v3", "worker_id": None, "seed_material": 42, "state": py_state},
        {"id": "gen_np_mt", "codec": "numpy.random.RandomState.MT19937.v1", "worker_id": None, "seed_material": 42, "state": np_mt_state},
        {"id": "gen_np_pcg", "codec": "numpy.random.PCG64.v1", "worker_id": None, "seed_material": 42, "state": pcg_state},
        {"id": "gen_torch", "codec": "torch.cpu.Generator.native.v1", "worker_id": None, "seed_material": 42, "state": torch_native_state},
    ]

    expected_generator_bindings = {
        "gen_py": {"codec": "python.random.MT19937.v3", "worker_id": None, "seed_material": 42},
        "gen_np_mt": {"codec": "numpy.random.RandomState.MT19937.v1", "worker_id": None, "seed_material": 42},
        "gen_np_pcg": {"codec": "numpy.random.PCG64.v1", "worker_id": None, "seed_material": 42},
        "gen_torch": {"codec": "torch.cpu.Generator.native.v1", "worker_id": None, "seed_material": 42},
    }

    param_names = [
        "user_embeddings",
        "product_embeddings",
        "category_embeddings",
        "brand_embeddings",
    ]
    for rel in RELATION_SCHEMAS:
        param_names.append(f"relation_weights_l0.{rel}")
    for rel in RELATION_SCHEMAS:
        param_names.append(f"relation_weights_l1.{rel}")
    for ntype in NODE_TYPES:
        param_names.append(f"self_weights_l0.{ntype}")
    for ntype in NODE_TYPES:
        param_names.append(f"self_biases_l0.{ntype}")
    for ntype in NODE_TYPES:
        param_names.append(f"self_weights_l1.{ntype}")
    for ntype in NODE_TYPES:
        param_names.append(f"self_biases_l1.{ntype}")

    expected_runtime = {
        "runtime_profile": {
            "profile_id": "synthetic_profile_v1",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "adam_step_dtype": "torch.float32",
            "batch_policy": "filter_before_batch",
        },
        "optimizer_groups": [
            {
                "parameter_names": param_names,
                "options": {
                    "lr": 0.001,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                    "decoupled_weight_decay": False,
                },
            }
        ],
        "rng_role_bindings": {
            "Python": ["gen_py"],
            "NumPy": ["gen_np_pcg"],
            "PyTorch_CPU": ["gen_torch"],
            "PyTorch_CUDA_if_applicable": [],
            "initialization": ["gen_torch"],
            "epoch_shuffle": ["gen_py"],
            "negative_sampler": ["gen_np_mt"],
            "neighbor_sampler": ["gen_torch"],
            "DataLoader_generator": ["gen_torch"],
            "DataLoader_workers": [],
        },
        "rng_generator_bindings": expected_generator_bindings,
        "sampler_algorithm": "uniform_eligible_negative_epoch_sampler_v1",
    }

    # Model parameters (36 tensors)
    g = torch.Generator(device="cpu").manual_seed(102)
    state_dict: Dict[str, torch.Tensor] = {
        "user_embeddings": torch.randn(len(row_order["User"]), 64, dtype=torch.float32, generator=g),
        "product_embeddings": torch.randn(len(row_order["Product"]), 64, dtype=torch.float32, generator=g),
        "category_embeddings": torch.randn(len(row_order["CategoryPath"]), 64, dtype=torch.float32, generator=g),
        "brand_embeddings": torch.randn(len(row_order["Brand"]), 64, dtype=torch.float32, generator=g),
    }
    for rel in RELATION_SCHEMAS:
        state_dict[f"relation_weights_l0.{rel}"] = torch.randn(64, 64, dtype=torch.float32, generator=g)
    for rel in RELATION_SCHEMAS:
        state_dict[f"relation_weights_l1.{rel}"] = torch.randn(64, 64, dtype=torch.float32, generator=g)
    for ntype in NODE_TYPES:
        state_dict[f"self_weights_l0.{ntype}"] = torch.randn(64, 64, dtype=torch.float32, generator=g)
    for ntype in NODE_TYPES:
        state_dict[f"self_biases_l0.{ntype}"] = torch.zeros(64, dtype=torch.float32)
    for ntype in NODE_TYPES:
        state_dict[f"self_weights_l1.{ntype}"] = torch.randn(64, 64, dtype=torch.float32, generator=g)
    for ntype in NODE_TYPES:
        state_dict[f"self_biases_l1.{ntype}"] = torch.zeros(64, dtype=torch.float32)

    current_model = {
        "epoch": 2,
        "row_order": row_order,
        "state_dict": {k: v.clone() for k, v in state_dict.items()},
    }
    best_model = {
        "epoch": 2,
        "row_order": row_order,
        "state_dict": {k: v.clone() for k, v in state_dict.items()},
    }

    # Optimizer with 36 parameter bindings
    param_bindings = []
    opt_state: Dict[int, Dict[str, Any]] = {}
    for idx, p_name in enumerate(param_names):
        param_bindings.append({"id": idx, "name": p_name, "updates": 4})
        p_shape = state_dict[p_name].shape
        opt_state[idx] = {
            "step": torch.tensor(4.0, dtype=torch.float32),
            "exp_avg": torch.zeros(p_shape, dtype=torch.float32),
            "exp_avg_sq": torch.zeros(p_shape, dtype=torch.float32),
        }

    optimizer = {
        "algorithm": "Adam",
        "state_dict": {
            "state": opt_state,
            "param_groups": [
                {
                    "params": list(range(36)),
                    "lr": 0.001,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                    "decoupled_weight_decay": False,
                }
            ],
        },
        "parameter_bindings": param_bindings,
    }

    # RNG
    roles = []
    for s in identity.resolved_settings["determinism"]["rng_streams"]:
        r_rec = dict(s)
        if s["applicable"]:
            r_rec["generator_ids"] = expected_runtime["rng_role_bindings"][s["name"]]
        else:
            r_rec["generator_ids"] = []
        roles.append(r_rec)

    rng_dict = {"roles": roles, "generators": generators}

    source_order = [("u1", "p1"), ("u1", "p2"), ("u2", "p2"), ("u2", "p3")]
    sampler = {
        "algorithm": "uniform_eligible_negative_epoch_sampler_v1",
        "identity_hash": identity.identity_hash,
        "source_order": source_order,
        "cursor": 4,
        "next_epoch": 3,
        "batch_cursor": 0,
        "pending_batches": 0,
        "prefetch_pending": 0,
        "inflight_workers": 0,
        "worker_ids": [],
    }

    progress = {
        "epoch": 2,
        "global_step": 4,
        "epoch_optimizer_steps": [2, 2],
        "empty_pool_skips": [0, 0],
    }

    selection = {
        "metric": "validation_NDCG@10",
        "history": [{"epoch": 2, "score": 0.65}],
        "best_score": 0.65,
        "selected_epoch": 2,
        "bad_checks": 0,
        "status": "ready",
        "stop_reason": None,
    }

    scheduler = {"applicable": False, "state": None, "reason": "not_used_constant_learning_rate"}
    amp = {"applicable": False, "state": None, "reason": "not_used_cpu_float32"}

    payload = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "capability": PAYLOAD_CAPABILITY,
        "identity": identity.to_dict(),
        "provenance": provenance,
        "runtime_profile": copy.deepcopy(expected_runtime["runtime_profile"]),
        "current_model": current_model,
        "best_model": best_model,
        "optimizer": optimizer,
        "rng": rng_dict,
        "sampler": sampler,
        "progress": progress,
        "selection": selection,
        "scheduler": scheduler,
        "amp": amp,
    }

    return payload, identity.to_dict(), provenance, expected_model, expected_runtime


# ---------------------------------------------------------------------------
# Test Cases (27 Retained Recovery Methods + 7 Ported Diagnostic Methods)
# ---------------------------------------------------------------------------

class TestCheckpointPayloadValidation(unittest.TestCase):
    """Exhaustive test suite for checkpoint payload and state validation."""

    def test_dual_import_canonicalization(self) -> None:
        """Verify checkpoint_payload and scripts.checkpoint_payload share identical objects."""
        self.assertIs(checkpoint_payload.validate_checkpoint_payload, scripts.checkpoint_payload.validate_checkpoint_payload)
        self.assertIs(checkpoint_payload.preflight_resource_accounting, scripts.checkpoint_payload.preflight_resource_accounting)
        self.assertIs(checkpoint_payload.validate_resource_policy, scripts.checkpoint_payload.validate_resource_policy)

    def test_positive_lightgcn_payload(self) -> None:
        """Positive control: complete valid LightGCN payload passes and returns owned snapshot."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        res = validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )

        self.assertIsInstance(res, dict)
        self.assertEqual(res["schema_version"], PAYLOAD_SCHEMA_VERSION)
        self.assertEqual(res["progress"]["epoch"], 2)
        self.assertIsNot(res, payload)
        self.assertIsNot(res["current_model"]["state_dict"]["user_embeddings"], payload["current_model"]["state_dict"]["user_embeddings"])

    def test_positive_heterognn_payload(self) -> None:
        """Positive control: complete valid HeteroGNN payload passes and returns owned snapshot."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_hetero_payload()
        policy = _make_synthetic_resource_policy()

        res = validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )

        self.assertIsInstance(res, dict)
        self.assertEqual(len(res["current_model"]["state_dict"]), 36)
        self.assertIsNot(res, payload)

    def test_fixture_alignment_and_normalization(self) -> None:
        """Truthful fixture alignment: LightGCN sparse buffer matches DifferentiableLightGCN and symmetric formula."""
        # 1. Check literal normalized weights: outer edges 1/sqrt(2), middle edges 0.5
        adj = _make_lightgcn_sparse_buffer(2, 3)
        expected_outer = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(float(adj.values()[0].item()), expected_outer, places=6)
        self.assertAlmostEqual(float(adj.values()[1].item()), 0.5, places=6)
        self.assertAlmostEqual(float(adj.values()[2].item()), 0.5, places=6)
        self.assertAlmostEqual(float(adj.values()[3].item()), expected_outer, places=6)

        # 2. Build graph and DifferentiableLightGCN directly from TrainingContext
        interactions = [
            ("u1", "p1", 5.0),
            ("u1", "p2", 5.0),
            ("u2", "p2", 5.0),
            ("u2", "p3", 5.0),
        ]
        ctx = TrainingContext(interactions, condition="all_ratings")
        l_graph = LightGCNGraph(ctx)
        u_base = torch.zeros(2, 64)
        i_base = torch.zeros(3, 64)
        model = DifferentiableLightGCN(l_graph, u_base, i_base)
        native_adj = getattr(model, "adj_user_item")

        # Compare indices and bitwise values
        self.assertTrue(torch.equal(adj.indices(), native_adj.indices()))
        self.assertTrue((adj.values().view(torch.int32) == native_adj.values().view(torch.int32)).all())

    def test_ownership_and_no_mutation(self) -> None:
        """Input payload and expected descriptors are never mutated; returned snapshot is completely independent."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        # Capture original values
        orig_user_val = float(payload["current_model"]["state_dict"]["user_embeddings"][0, 0].item())

        res = validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )

        # Mutate input payload tensor and container
        payload["current_model"]["state_dict"]["user_embeddings"][0, 0] += 10.0
        payload["current_model"]["epoch"] = 999
        self.assertEqual(float(res["current_model"]["state_dict"]["user_embeddings"][0, 0].item()), orig_user_val)
        self.assertEqual(res["current_model"]["epoch"], 2)

        # Mutate returned snapshot tensor
        res["current_model"]["state_dict"]["user_embeddings"][0, 0] = -999.0
        self.assertNotEqual(float(payload["current_model"]["state_dict"]["user_embeddings"][0, 0].item()), -999.0)

        # Late failure no-mutation: corrupt selection history at step 10
        corrupt_payload = copy.deepcopy(payload)
        corrupt_payload["current_model"]["epoch"] = 2
        corrupt_payload["current_model"]["state_dict"]["user_embeddings"][0, 0] = orig_user_val
        corrupt_payload["selection"]["history"][0]["score"] = 99.0  # Invalid score > 1.0

        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(
                corrupt_payload,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )

        # Ensure corrupt_payload is unchanged
        self.assertEqual(corrupt_payload["selection"]["history"][0]["score"], 99.0)
        self.assertEqual(float(corrupt_payload["current_model"]["state_dict"]["user_embeddings"][0, 0].item()), orig_user_val)

    def test_resource_policy_validation(self) -> None:
        """Resource policy schema, profile IDs, exact synthetic constants, and runtime policy."""
        policy = _make_synthetic_resource_policy()
        self.assertEqual(validate_resource_policy(policy)["max_input_bytes"], 2097152)

        # Synthetic kind with altered constants fails closed
        bad_synth = copy.deepcopy(policy)
        bad_synth["max_input_bytes"] = 2097151
        with self.assertRaises(CheckpointValidationError):
            validate_resource_policy(bad_synth)

        bad_synth2 = copy.deepcopy(policy)
        bad_synth2["profile_id"] = "custom-profile"
        with self.assertRaises(CheckpointValidationError):
            validate_resource_policy(bad_synth2)

        # Runtime kind with custom limits succeeds
        runtime_policy = _make_synthetic_resource_policy(
            max_input_bytes=1000,
            max_output_bytes=1000,
            max_container_depth=10,
            max_visited_values=50,
            kind="runtime",
            profile_id="custom-runtime-profile",
        )
        parsed = validate_resource_policy(runtime_policy)
        self.assertEqual(parsed["kind"], "runtime")
        self.assertEqual(parsed["max_input_bytes"], 1000)

        # Missing / extra keys / wrong types fail closed
        bad = copy.deepcopy(policy)
        del bad["max_input_bytes"]
        with self.assertRaises(CheckpointValidationError):
            validate_resource_policy(bad)

        bad = copy.deepcopy(policy)
        bad["extra_key"] = 123
        with self.assertRaises(CheckpointValidationError):
            validate_resource_policy(bad)

        bad = copy.deepcopy(policy)
        bad["max_input_bytes"] = True
        with self.assertRaises(CheckpointValidationError):
            validate_resource_policy(bad)

    def test_exact_resource_boundaries_and_accounting(self) -> None:
        """Exact B and B+1 input/output boundaries, storage deduplication, backing storage views, depth, and visits."""
        # 1. Test distinct storages over same buffer address
        buf = bytearray(100)
        t1 = torch.frombuffer(buf, dtype=torch.uint8)
        t2 = torch.frombuffer(buf, dtype=torch.uint8)
        self.assertEqual(t1.data_ptr(), t2.data_ptr())
        policy_500 = _make_synthetic_resource_policy(max_input_bytes=500, max_output_bytes=500, kind="runtime", profile_id="r1")
        m, s, c, v = preflight_resource_accounting([t1, t2], policy_500)
        self.assertEqual(s, 200)

        # 2. View of large storage behind tiny view
        large_st = torch.empty(100000, dtype=torch.float32).untyped_storage()
        tiny_view = torch.tensor([], dtype=torch.float32).set_(large_st, 0, (1,))
        tight_in = _make_synthetic_resource_policy(max_input_bytes=1000, kind="runtime", profile_id="r2")
        with self.assertRaises(CheckpointValidationError) as ctx:
            preflight_resource_accounting(tiny_view, tight_in)
        self.assertIn("exceeded limit", str(ctx.exception))

        # 3. True shared input storage / expanded output
        shared_t1 = torch.ones(500, dtype=torch.float32)
        shared_t2 = shared_t1[0:250]
        tight_out = _make_synthetic_resource_policy(max_input_bytes=3000, max_output_bytes=2500, kind="runtime", profile_id="r3")
        with self.assertRaises(CheckpointValidationError) as ctx:
            preflight_resource_accounting([shared_t1, shared_t2], tight_out)
        self.assertIn("projected output bytes", str(ctx.exception))

        # 4. Exact container depth 32 passes; depth 33 fails
        depth_32: Any = 42
        for _ in range(32):
            depth_32 = [depth_32]
        p_depth = _make_synthetic_resource_policy(max_container_depth=32, kind="runtime", profile_id="r4")
        preflight_resource_accounting(depth_32, p_depth)

        depth_33 = [depth_32]
        with self.assertRaises(CheckpointValidationError) as ctx:
            preflight_resource_accounting(depth_33, p_depth)
        self.assertIn("max_container_depth", str(ctx.exception))

        # 5. Exact visits: 10 passes, 11 fails
        p_visits = _make_synthetic_resource_policy(max_visited_values=10, kind="runtime", profile_id="r5")
        preflight_resource_accounting([1, 2, 3, 4, 5, 6, 7, 8, 9], p_visits)
        with self.assertRaises(CheckpointValidationError) as ctx:
            preflight_resource_accounting([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p_visits)
        self.assertIn("max_visited_values", str(ctx.exception))

        # 6. Exact B and B+1 input boundary test under synthetic profile (B = 2,097,152)
        synth_policy = _make_synthetic_resource_policy()
        exact_b_obj = b"x" * (2097152 - 8)
        m, s, c, v = preflight_resource_accounting(exact_b_obj, synth_policy)
        self.assertEqual(m + s, 2097152)

        over_b_obj = b"x" * (2097152 - 8 + 1)
        with self.assertRaises(CheckpointValidationError) as ctx:
            preflight_resource_accounting(over_b_obj, synth_policy)
        self.assertIn("exceeded limit", str(ctx.exception))

    def test_custom_objects_and_type_safety_without_hooks(self) -> None:
        """Custom objects, subclasses, and hook sentinels are rejected without executing hooks."""
        policy = _make_synthetic_resource_policy()

        # Attacker metaclass that raises AssertionError on any hook
        class MetaclassAttacker(type):
            def __getattribute__(cls, name: str) -> Any:
                raise AssertionError(f"Attacker metaclass hook __getattribute__({name}) called!")
            def __getattr__(cls, name: str) -> Any:
                raise AssertionError(f"Attacker metaclass hook __getattr__({name}) called!")

        class AttackObject(metaclass=MetaclassAttacker):
            def __repr__(self) -> str:
                raise AssertionError("Custom __repr__ hook was executed!")
            def __str__(self) -> str:
                raise AssertionError("Custom __str__ hook was executed!")
            def __eq__(self, other: Any) -> bool:
                raise AssertionError("Custom __eq__ hook was executed!")
            def __hash__(self) -> int:
                raise AssertionError("Custom __hash__ hook was executed!")

        # Verify AttackObject rejected without executing any hooks
        with self.assertRaises(CheckpointValidationError):
            preflight_resource_accounting(AttackObject(), policy)

        with self.assertRaises(CheckpointValidationError):
            preflight_resource_accounting({"key": AttackObject()}, policy)

        # Custom dict/list/int/str/Tensor subclasses
        class CustomDict(dict):
            pass
        class CustomList(list):
            pass
        class CustomInt(int):
            pass
        class CustomStr(str):
            pass

        with self.assertRaises(CheckpointValidationError):
            preflight_resource_accounting(CustomDict({"a": 1}), policy)
        with self.assertRaises(CheckpointValidationError):
            preflight_resource_accounting(CustomList([1, 2]), policy)
        with self.assertRaises(CheckpointValidationError):
            preflight_resource_accounting(CustomInt(42), policy)
        with self.assertRaises(CheckpointValidationError):
            preflight_resource_accounting(CustomStr("test"), policy)
        with self.assertRaises(CheckpointValidationError):
            preflight_resource_accounting(torch.nn.Parameter(torch.zeros(4)), policy)

        # Reject CanonicalResumeIdentity object passed directly as expected_identity
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        cid = CanonicalResumeIdentity.from_dict(exp_id)
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(
                payload,
                expected_identity=cid,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )

    def test_fail_closed_error_handling(self) -> None:
        """Invalid Unicode, huge invalid integers, and mixed key types fail with CheckpointValidationError."""
        policy = _make_synthetic_resource_policy()

        # Lone surrogate invalid Unicode
        invalid_unicode = "\ud800"
        with self.assertRaises(CheckpointValidationError):
            preflight_resource_accounting(invalid_unicode, policy)

        # Huge negative integer (-(10**5000)) does not crash with Python digit conversion ValueError
        huge_int = -(10 ** 5000)
        with self.assertRaises(CheckpointValidationError):
            _check_strict_int(huge_int, "test_int", min_val=0)

        # Huge positive integer (10**5000) passed to numeric validator does not raise OverflowError
        from scripts.checkpoint_payload import _check_strict_number
        res_num = _check_strict_number(10 ** 5000, "test_huge_pos", min_val=0)
        self.assertEqual(res_num, 10 ** 5000)

        # Mixed primitive key types in dictionary
        mixed_dict: Dict[Any, Any] = {1: "a", "b": "c"}
        with self.assertRaises(CheckpointValidationError):
            preflight_resource_accounting(mixed_dict, policy)

    def test_envelope_and_inapplicable_components(self) -> None:
        """Envelope keys, schema version, capability, active scheduler, and active AMP."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        # Missing key
        bad = copy.deepcopy(payload)
        del bad["progress"]
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Extra key
        bad = copy.deepcopy(payload)
        bad["extra_key"] = 123
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Wrong schema version
        bad = copy.deepcopy(payload)
        bad["schema_version"] = "amazon4u-wrong-version"
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Wrong capability
        bad = copy.deepcopy(payload)
        bad["capability"] = "mid_epoch_restoration"
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Active scheduler or AMP
        bad = copy.deepcopy(payload)
        bad["scheduler"]["applicable"] = True
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        bad = copy.deepcopy(payload)
        bad["amp"]["applicable"] = True
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

    def test_lightgcn_model_parameter_and_buffer_mutations(self) -> None:
        """LightGCN parameter and sparse COO buffer negative matrix."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        # Shape mismatch
        bad = copy.deepcopy(payload)
        bad["current_model"]["state_dict"]["user_embeddings"] = torch.randn(2, 32, dtype=torch.float32)
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Non-finite values
        bad = copy.deepcopy(payload)
        bad["current_model"]["state_dict"]["user_embeddings"][0, 0] = float("nan")
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Sparse buffer uncoalesced
        bad = copy.deepcopy(payload)
        indices = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0], dtype=torch.float32)
        bad["current_model"]["state_dict"]["adj_user_item"] = torch.sparse_coo_tensor(indices, vals, (2, 3))
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Sparse buffer indices out of lexicographic order
        bad = copy.deepcopy(payload)
        indices = torch.tensor([[1, 0], [0, 0]], dtype=torch.int64)
        vals = torch.tensor([0.5, 0.5], dtype=torch.float32)
        bad["current_model"]["state_dict"]["adj_user_item"] = torch.sparse_coo_tensor(indices, vals, (2, 3)).coalesce()
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Sparse buffer duplicate indices
        bad = copy.deepcopy(payload)
        indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
        vals = torch.tensor([0.5, 0.5], dtype=torch.float32)
        bad["current_model"]["state_dict"]["adj_user_item"] = torch.sparse_coo_tensor(indices, vals, (2, 3)).coalesce()
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Sparse buffer graph content mutation (weight changed)
        bad = copy.deepcopy(payload)
        mutated_adj = _make_lightgcn_sparse_buffer(2, 3)
        mutated_vals = mutated_adj.values().clone()
        mutated_vals[0] = 0.99
        bad["current_model"]["state_dict"]["adj_user_item"] = torch.sparse_coo_tensor(mutated_adj.indices(), mutated_vals, (2, 3)).coalesce()
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Meta device tensor rejection
        bad = copy.deepcopy(payload)
        bad["current_model"]["state_dict"]["user_embeddings"] = torch.empty(2, 64, device="meta")
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

    def test_heterognn_model_parameter_mutations(self) -> None:
        """HeteroGNN 36 parameter table-driven negative matrix and zero-buffer enforcement."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_hetero_payload()
        policy = _make_synthetic_resource_policy()

        # Missing parameter
        bad = copy.deepcopy(payload)
        del bad["current_model"]["state_dict"]["brand_embeddings"]
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Registered buffer presence (HeteroGNN must have 0 buffers)
        bad = copy.deepcopy(payload)
        bad["current_model"]["state_dict"]["forbidden_buffer"] = torch.zeros(64)
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Table-driven test covering multiple parameter categories
        test_cases = [
            ("user_embeddings", (2, 32)),
            ("relation_weights_l0.rated", (32, 64)),
            ("self_weights_l0.User", (64, 32)),
            ("self_biases_l0.User", (32,)),
            ("self_biases_l1.Product", (128,)),
        ]
        for p_name, bad_shape in test_cases:
            bad = copy.deepcopy(payload)
            bad["current_model"]["state_dict"][p_name] = torch.zeros(bad_shape, dtype=torch.float32)
            with self.assertRaises(CheckpointValidationError):
                validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

    def test_adam_optimizer_validation(self) -> None:
        """Adam optimizer: groups, options, moment shapes, lazy initialization, step precision."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        # Wrong moment shape
        bad = copy.deepcopy(payload)
        bad["optimizer"]["state_dict"]["state"][0]["exp_avg"] = torch.zeros(2, 32, dtype=torch.float32)
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Negative second moment
        bad = copy.deepcopy(payload)
        bad["optimizer"]["state_dict"]["state"][0]["exp_avg_sq"][0, 0] = -1.0
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Negative first moment is valid (e.g. gradient direction)
        neg_first = copy.deepcopy(payload)
        neg_first["optimizer"]["state_dict"]["state"][0]["exp_avg"][0, 0] = -0.5
        res = validate_checkpoint_payload(neg_first, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)
        self.assertIsNotNone(res)

        # Lazy initialization: updates=0 with complete zero state passes
        zero_state_payload = copy.deepcopy(payload)
        zero_state_payload["progress"]["epoch"] = 0
        zero_state_payload["progress"]["global_step"] = 0
        zero_state_payload["progress"]["epoch_optimizer_steps"] = []
        zero_state_payload["progress"]["empty_pool_skips"] = []
        zero_state_payload["sampler"]["cursor"] = 0
        zero_state_payload["sampler"]["source_order"] = None
        zero_state_payload["sampler"]["next_epoch"] = 1
        zero_state_payload["current_model"]["epoch"] = 0
        zero_state_payload["best_model"] = None
        zero_state_payload["selection"]["history"] = []
        zero_state_payload["selection"]["best_score"] = None
        zero_state_payload["selection"]["selected_epoch"] = None
        zero_state_payload["selection"]["status"] = "initial"
        zero_state_payload["optimizer"]["parameter_bindings"][0]["updates"] = 0
        zero_state_payload["optimizer"]["parameter_bindings"][1]["updates"] = 0
        zero_state_payload["optimizer"]["state_dict"]["state"] = {
            0: {
                "step": torch.tensor(0.0, dtype=torch.float32),
                "exp_avg": torch.zeros(2, 64, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(2, 64, dtype=torch.float32),
            },
            1: {
                "step": torch.tensor(0.0, dtype=torch.float32),
                "exp_avg": torch.zeros(3, 64, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(3, 64, dtype=torch.float32),
            },
        }
        res = validate_checkpoint_payload(zero_state_payload, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)
        self.assertIsNotNone(res)

        # updates=0 with explicit state[id]=None fails closed
        bad = copy.deepcopy(zero_state_payload)
        bad["optimizer"]["state_dict"]["state"][0] = None
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Orphan state ID in optimizer state
        bad = copy.deepcopy(payload)
        bad["optimizer"]["state_dict"]["state"][99] = {
            "step": torch.tensor(4.0, dtype=torch.float32),
            "exp_avg": torch.zeros(2, 64),
            "exp_avg_sq": torch.zeros(2, 64),
        }
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Adam step precision: representable 2**24 + 2 and 2**25 pass, unrepresentable 2**25 + 1 fails
        exact_24_plus_2 = (2 ** 24) + 2
        exact_25 = 2 ** 25
        inexact_step = (2 ** 25) + 1

        shapes = {"user_embeddings": (2, 64), "item_embeddings": (3, 64)}

        # 2**24 + 2 passes validator
        opt_exact1 = copy.deepcopy(payload["optimizer"])
        opt_exact1["parameter_bindings"][0]["updates"] = exact_24_plus_2
        opt_exact1["state_dict"]["state"][0]["step"] = torch.tensor(float(exact_24_plus_2), dtype=torch.float32)
        scripts.checkpoint_payload._validate_adam_optimizer(
            opt_exact1,
            exp_rt["optimizer_groups"],
            "torch.float32",
            exact_24_plus_2,
            "LightGCN",
            validated_param_shapes=shapes,
        )

        # 2**25 passes validator
        opt_exact2 = copy.deepcopy(payload["optimizer"])
        opt_exact2["parameter_bindings"][0]["updates"] = exact_25
        opt_exact2["state_dict"]["state"][0]["step"] = torch.tensor(float(exact_25), dtype=torch.float32)
        scripts.checkpoint_payload._validate_adam_optimizer(
            opt_exact2,
            exp_rt["optimizer_groups"],
            "torch.float32",
            exact_25,
            "LightGCN",
            validated_param_shapes=shapes,
        )

        # 2**25 + 1 fails validator
        opt_bad = copy.deepcopy(payload["optimizer"])
        opt_bad["parameter_bindings"][0]["updates"] = inexact_step
        opt_bad["state_dict"]["state"][0]["step"] = torch.tensor(float(inexact_step), dtype=torch.float32)
        with self.assertRaises(CheckpointValidationError) as ctx:
            scripts.checkpoint_payload._validate_adam_optimizer(
                opt_bad,
                exp_rt["optimizer_groups"],
                "torch.float32",
                inexact_step * 2,
                "LightGCN",
                validated_param_shapes=shapes,
            )
        self.assertIn("cannot be represented exactly", str(ctx.exception))

    def test_rng_and_codecs_validation(self) -> None:
        """RNG roles, 4 codecs, seed matching against bindings and identity, global RNG preservation."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        # Capture global RNG state before
        py_rnd_before = random.getstate()
        np_rnd_before = np.random.get_state()
        torch_rnd_before = torch.get_rng_state()

        validate_checkpoint_payload(payload, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Global RNG preserved on success
        self.assertEqual(random.getstate(), py_rnd_before)
        self.assertEqual(np.random.get_state()[0], np_rnd_before[0])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rnd_before))

        # Generator seed material mismatch against expected generator bindings
        bad = copy.deepcopy(payload)
        bad["rng"]["generators"][0]["seed_material"] = 999
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # PCG64 even inc fails
        bad = copy.deepcopy(payload)
        bad["rng"]["generators"][2]["state"]["state"]["inc"] = 2  # Even inc invalid
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Python MT19937 wrong version fails
        bad = copy.deepcopy(payload)
        state_list = list(bad["rng"]["generators"][0]["state"])
        state_list[0] = 2  # version 2 invalid
        bad["rng"]["generators"][0]["state"] = tuple(state_list)
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # RandomState wrong name fails
        bad = copy.deepcopy(payload)
        rs_list = list(bad["rng"]["generators"][1]["state"])
        rs_list[0] = "OTHER"
        bad["rng"]["generators"][1]["state"] = tuple(rs_list)
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Global RNG preserved on failure
        self.assertEqual(random.getstate(), py_rnd_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rnd_before))

    def test_sampler_and_progress_validation(self) -> None:
        """Sampler domain, quiescent boundary, batch policies, and exact integer ceil."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        # Non-quiescent sampler (active batch cursor)
        bad = copy.deepcopy(payload)
        bad["sampler"]["batch_cursor"] = 1
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Incomplete source_order permutation
        bad = copy.deepcopy(payload)
        bad["sampler"]["source_order"] = [("u1", "p1"), ("u1", "p1"), ("u2", "p2"), ("u2", "p3")]
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Exact integer ceil division test for large batch size
        self.assertEqual(_ceil_div(4, 2**14000), 1)

    def test_selection_history_and_patience_boundary(self) -> None:
        """Selection strict improvement, exact-tie earliest retention, terminal patience stop epoch."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        # First score 0.0 passes and selects epoch 2
        p_zero = copy.deepcopy(payload)
        p_zero["selection"]["history"] = [{"epoch": 2, "score": 0.0}]
        p_zero["selection"]["best_score"] = 0.0
        p_zero["selection"]["selected_epoch"] = 2
        res = validate_checkpoint_payload(p_zero, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)
        self.assertEqual(res["selection"]["best_score"], 0.0)

        # Patience stop test (10 bad checks at epoch 22)
        tie_payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload(epoch=4, global_step=8, bad_checks=1, score=0.5)
        pat_payload = copy.deepcopy(tie_payload)
        pat_payload["progress"]["epoch"] = 22
        pat_payload["progress"]["global_step"] = 44
        pat_payload["progress"]["epoch_optimizer_steps"] = [2] * 22
        pat_payload["progress"]["empty_pool_skips"] = [0] * 22
        pat_payload["sampler"]["next_epoch"] = 23
        pat_payload["current_model"]["epoch"] = 22
        pat_payload["best_model"]["epoch"] = 2
        for b in pat_payload["optimizer"]["parameter_bindings"]:
            b["updates"] = 44
        for p_id in (0, 1):
            pat_payload["optimizer"]["state_dict"]["state"][p_id]["step"] = torch.tensor(44.0, dtype=torch.float32)

        history = [{"epoch": 2, "score": 0.9}]
        for ep in range(4, 24, 2):
            history.append({"epoch": ep, "score": 0.1})
        pat_payload["selection"]["history"] = history
        pat_payload["selection"]["best_score"] = 0.9
        pat_payload["selection"]["selected_epoch"] = 2
        pat_payload["selection"]["bad_checks"] = 10
        pat_payload["selection"]["status"] = "stopped"
        pat_payload["selection"]["stop_reason"] = "patience"

        res = validate_checkpoint_payload(pat_payload, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)
        self.assertEqual(res["selection"]["status"], "stopped")
        self.assertEqual(res["selection"]["stop_reason"], "patience")

        # Extending epoch to 23 after patience stop at 22 fails closed
        over_epoch_payload = copy.deepcopy(pat_payload)
        over_epoch_payload["progress"]["epoch"] = 23
        over_epoch_payload["current_model"]["epoch"] = 23
        over_epoch_payload["progress"]["epoch_optimizer_steps"] = [2] * 23
        over_epoch_payload["progress"]["empty_pool_skips"] = [0] * 23
        over_epoch_payload["progress"]["global_step"] = 46
        over_epoch_payload["sampler"]["next_epoch"] = 24
        for b in over_epoch_payload["optimizer"]["parameter_bindings"]:
            b["updates"] = 46
        for p_id in (0, 1):
            over_epoch_payload["optimizer"]["state_dict"]["state"][p_id]["step"] = torch.tensor(46.0, dtype=torch.float32)

        with self.assertRaises(CheckpointValidationError) as ctx:
            validate_checkpoint_payload(over_epoch_payload, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)
        self.assertIn("cannot exceed it", str(ctx.exception))

    def test_best_model_snapshot_semantics(self) -> None:
        """best_model must be None before check; when epochs coincide tensor contents must match bitwise including signed zero."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload(epoch=0, global_step=0)
        policy = _make_synthetic_resource_policy()

        # At epoch 0 before check, best_model must be None
        payload["progress"]["epoch"] = 0
        payload["progress"]["global_step"] = 0
        payload["progress"]["epoch_optimizer_steps"] = []
        payload["progress"]["empty_pool_skips"] = []
        payload["sampler"]["source_order"] = None
        payload["sampler"]["cursor"] = 0
        payload["sampler"]["next_epoch"] = 1
        payload["current_model"]["epoch"] = 0
        payload["best_model"] = None
        payload["selection"]["history"] = []
        payload["selection"]["best_score"] = None
        payload["selection"]["selected_epoch"] = None
        payload["selection"]["bad_checks"] = 0
        payload["selection"]["status"] = "initial"
        payload["selection"]["stop_reason"] = None
        payload["optimizer"]["parameter_bindings"][0]["updates"] = 0
        payload["optimizer"]["parameter_bindings"][1]["updates"] = 0
        payload["optimizer"]["state_dict"]["state"] = {}

        res = validate_checkpoint_payload(payload, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)
        self.assertIsNone(res["best_model"])

        # Signed zero bitwise test: +0.0 vs -0.0
        payload2, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload(epoch=2)
        payload2["current_model"]["state_dict"]["user_embeddings"][0, 0] = 0.0
        payload2["best_model"]["state_dict"]["user_embeddings"][0, 0] = -0.0
        # torch.equal treats +0.0 and -0.0 as equal, but bitwise comparison detects the difference
        with self.assertRaises(CheckpointValidationError) as ctx:
            validate_checkpoint_payload(payload2, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)
        self.assertIn("contents differ", str(ctx.exception))

    def test_raw_tensor_detachment_and_sparse_forgery_rejection(self) -> None:
        """Detachment and canonical coalescing must be validated on raw input before copying."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        # 1. Raw tensor with requires_grad=True must fail closed on raw inspection before output copy
        bad_grad = copy.deepcopy(payload)
        bad_grad["current_model"]["state_dict"]["user_embeddings"].requires_grad_(True)
        with self.assertRaises(CheckpointValidationError) as ctx:
            validate_checkpoint_payload(
                bad_grad,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )
        self.assertIn("must be detached", str(ctx.exception))

        # 2. Raw uncoalesced sparse COO tensor must fail closed before output copy can repair it
        bad_sparse = copy.deepcopy(payload)
        uncoalesced_indices = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
        uncoalesced_values = torch.tensor([0.5, 0.5], dtype=torch.float32)
        bad_sparse["current_model"]["state_dict"]["adj_user_item"] = torch.sparse_coo_tensor(
            uncoalesced_indices, uncoalesced_values, (2, 3)
        )
        self.assertFalse(bad_sparse["current_model"]["state_dict"]["adj_user_item"].is_coalesced())
        with self.assertRaises(CheckpointValidationError) as ctx:
            validate_checkpoint_payload(
                bad_sparse,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )
        self.assertIn("must be coalesced", str(ctx.exception))

        # 3. Forged coalesced=True with out-of-order indices fails closed on lexicographic check
        bad_forged_order = copy.deepcopy(payload)
        forged_indices = torch.tensor([[1, 0], [0, 0]], dtype=torch.int64)
        forged_values = torch.tensor([0.5, 0.5], dtype=torch.float32)
        forged_tensor = torch.sparse_coo_tensor(forged_indices, forged_values, (2, 3))
        forged_tensor._coalesced_(True)
        self.assertTrue(forged_tensor.is_coalesced())
        bad_forged_order["current_model"]["state_dict"]["adj_user_item"] = forged_tensor
        with self.assertRaises(CheckpointValidationError) as ctx:
            validate_checkpoint_payload(
                bad_forged_order,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )
        self.assertIn("canonical strictly ascending order", str(ctx.exception))

        # 4. Forged coalesced=True with duplicate indices fails closed
        bad_forged_dup = copy.deepcopy(payload)
        dup_indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
        dup_values = torch.tensor([0.5, 0.5], dtype=torch.float32)
        dup_tensor = torch.sparse_coo_tensor(dup_indices, dup_values, (2, 3))
        dup_tensor._coalesced_(True)
        self.assertTrue(dup_tensor.is_coalesced())
        bad_forged_dup["current_model"]["state_dict"]["adj_user_item"] = dup_tensor
        with self.assertRaises(CheckpointValidationError) as ctx:
            validate_checkpoint_payload(
                bad_forged_dup,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )
        self.assertIn("canonical strictly ascending order", str(ctx.exception))

    def test_exact_independent_projected_output_oracle(self) -> None:
        """Exact projected output oracle: base uint8 of 1,048,576 bytes with two views."""
        synth_policy = _make_synthetic_resource_policy()
        base = torch.zeros(1048576, dtype=torch.uint8)
        view1 = base
        view2_pass = base[:1048402]

        # M = 174, S = 1,048,576, C = 2,096,978, M+C = 2,097,152 (exact B, passes)
        m, s, c, v = preflight_resource_accounting([view1, view2_pass], synth_policy)
        self.assertEqual(m, 174)
        self.assertEqual(s, 1048576)
        self.assertEqual(c, 2096978)
        self.assertEqual(m + c, 2097152)

        # Extend view 2 by 1 byte to base[:1048403]: M = 174, M+C = 2,097,153 (exact B+1, fails)
        view2_fail = base[:1048403]
        with self.assertRaises(CheckpointValidationError) as ctx:
            preflight_resource_accounting([view1, view2_fail], synth_policy)
        self.assertIn("projected output bytes", str(ctx.exception))

    def test_exact_visit_limits_under_synthetic_profile(self) -> None:
        """Exact 16,384 visits passes; exact 16,385 visits fails under synthetic resource policy."""
        synth_policy = _make_synthetic_resource_policy()
        # 1 container list + 16383 integers = 16384 visited values (exact limit passes)
        exact_16384_list = [1] * 16383
        m, s, c, v = preflight_resource_accounting(exact_16384_list, synth_policy)
        self.assertEqual(v, 16384)

        # 1 container list + 16384 integers = 16385 visited values (exact limit + 1 fails)
        over_16385_list = [1] * 16384
        with self.assertRaises(CheckpointValidationError) as ctx:
            preflight_resource_accounting(over_16385_list, synth_policy)
        self.assertIn("exceeded max_visited_values", str(ctx.exception))

    def test_adam_mixed_lazy_and_updated_parameters(self) -> None:
        """Adam optimizer with mixed lazy (updates=0) and updated (updates>0) parameters."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload(epoch=2, global_step=4)
        policy = _make_synthetic_resource_policy()

        # Parameter 0: never updated (updates=0), omitted from state
        # Parameter 1: updated with global_step=4
        mixed_payload = copy.deepcopy(payload)
        mixed_payload["optimizer"]["parameter_bindings"][0]["updates"] = 0
        del mixed_payload["optimizer"]["state_dict"]["state"][0]
        mixed_payload["optimizer"]["parameter_bindings"][1]["updates"] = 4

        res = validate_checkpoint_payload(
            mixed_payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )
        self.assertIsNotNone(res)
        self.assertNotIn(0, res["optimizer"]["state_dict"]["state"])
        self.assertIn(1, res["optimizer"]["state_dict"]["state"])

    def test_staged_worker_context_w1(self) -> None:
        """Staged W=1 worker context with 3 worker generators (Python, NumPy, TorchCPU) and sampler worker_ids=[0]."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload(runtime_workers=1)
        policy = _make_synthetic_resource_policy()

        res = validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )
        self.assertEqual(res["sampler"]["worker_ids"], [0])
        # 4 parent generators + 3 worker generators = 7 generators total
        self.assertEqual(len(res["rng"]["generators"]), 7)
        self.assertEqual(len(res["rng"]["roles"][9]["generator_ids"]), 3)

    def test_batch_then_filter_empty_chunk_and_partial_batch(self) -> None:
        """Batch policy batch_then_filter with genuine non-empty K, empty chunk, and partial batch."""
        identity, provenance = _make_synthetic_identity_and_provenance(method="LightGCN", with_empty_pool=True)

        u_list = ["u1", "u2", "u3"]
        p_list = ["p1", "p2", "p3"]
        # Valid 5 retained positive edges: (u1, p1), (u1, p2), (u2, p3), (u3, p1), (u3, p2)
        # Degree weights: u1=2, u2=1, u3=2; p1=2, p2=2, p3=1
        # (u1, p1): 0.5, (u1, p2): 0.5, (u2, p3): 1.0, (u3, p1): 0.5, (u3, p2): 0.5
        indices = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 2, 0, 1]], dtype=torch.int64)
        vals = torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5], dtype=torch.float32)
        adj_buffer = torch.sparse_coo_tensor(indices, vals, (3, 3)).coalesce()

        expected_model = {
            "method": "LightGCN",
            "row_order": {"User": u_list, "Product": p_list},
            "buffer_state": {"adj_user_item": adj_buffer.clone()},
        }

        g = torch.Generator(device="cpu").manual_seed(101)
        user_emb = torch.randn(len(u_list), 64, dtype=torch.float32, generator=g)
        item_emb = torch.randn(len(p_list), 64, dtype=torch.float32, generator=g)

        current_model = {
            "epoch": 2,
            "row_order": {"User": u_list, "Product": p_list},
            "state_dict": {
                "user_embeddings": user_emb.clone(),
                "item_embeddings": item_emb.clone(),
                "adj_user_item": adj_buffer.clone(),
            },
        }
        best_model = {
            "epoch": 2,
            "row_order": {"User": u_list, "Product": p_list},
            "state_dict": {
                "user_embeddings": user_emb.clone(),
                "item_embeddings": item_emb.clone(),
                "adj_user_item": adj_buffer.clone(),
            },
        }

        py_state = random.Random(42).getstate()
        np_raw_state = np.random.RandomState(42).get_state()
        np_mt_state = ("MT19937", list(map(int, np_raw_state[1])), int(np_raw_state[2]), int(np_raw_state[3]), float(np_raw_state[4]))
        pcg_state = copy.deepcopy(np.random.PCG64(42).state)
        torch_native_state = torch.Generator(device="cpu").manual_seed(42).get_state()

        generators = [
            {"id": "gen_py", "codec": "python.random.MT19937.v3", "worker_id": None, "seed_material": 42, "state": py_state},
            {"id": "gen_np_mt", "codec": "numpy.random.RandomState.MT19937.v1", "worker_id": None, "seed_material": 42, "state": np_mt_state},
            {"id": "gen_np_pcg", "codec": "numpy.random.PCG64.v1", "worker_id": None, "seed_material": 42, "state": pcg_state},
            {"id": "gen_torch", "codec": "torch.cpu.Generator.native.v1", "worker_id": None, "seed_material": 42, "state": torch_native_state},
        ]

        expected_runtime = {
            "runtime_profile": {
                "profile_id": "synthetic_profile_v1",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "numpy_version": np.__version__,
                "torch_version": str(torch.__version__),
                "adam_step_dtype": "torch.float32",
                "batch_policy": "batch_then_filter",
            },
            "optimizer_groups": [
                {
                    "parameter_names": ["user_embeddings", "item_embeddings"],
                    "options": {
                        "lr": 0.001,
                        "betas": (0.9, 0.999),
                        "eps": 1e-08,
                        "weight_decay": 0.0,
                        "amsgrad": False,
                        "maximize": False,
                        "foreach": None,
                        "capturable": False,
                        "differentiable": False,
                        "fused": None,
                        "decoupled_weight_decay": False,
                    },
                }
            ],
            "rng_role_bindings": {
                "Python": ["gen_py"],
                "NumPy": ["gen_np_pcg"],
                "PyTorch_CPU": ["gen_torch"],
                "PyTorch_CUDA_if_applicable": [],
                "initialization": ["gen_torch"],
                "epoch_shuffle": ["gen_py"],
                "negative_sampler": ["gen_np_mt"],
                "neighbor_sampler": ["gen_torch"],
                "DataLoader_generator": ["gen_torch"],
                "DataLoader_workers": [],
            },
            "rng_generator_bindings": {
                "gen_py": {"codec": "python.random.MT19937.v3", "worker_id": None, "seed_material": 42},
                "gen_np_mt": {"codec": "numpy.random.RandomState.MT19937.v1", "worker_id": None, "seed_material": 42},
                "gen_np_pcg": {"codec": "numpy.random.PCG64.v1", "worker_id": None, "seed_material": 42},
                "gen_torch": {"codec": "torch.cpu.Generator.native.v1", "worker_id": None, "seed_material": 42},
            },
            "sampler_algorithm": "uniform_eligible_negative_epoch_sampler_v1",
        }

        # P has 5 pairs: (u1, p1), (u1, p2), (u2, p3), (u3, p1), (u3, p2)
        # K has 2 pairs: (u3, p1), (u3, p2)
        # Chunk 0: [("u1", "p1"), ("u1", "p2")] -> 2 valid pairs -> 1 step
        # Chunk 1: [("u3", "p1"), ("u3", "p2")] -> 2 pairs in K -> empty after filtering -> 0 steps!
        # Chunk 2: [("u2", "p3")] -> partial batch (size 1) -> 1 step!
        source_order = [("u1", "p1"), ("u1", "p2"), ("u3", "p1"), ("u3", "p2"), ("u2", "p3")]
        sampler = {
            "algorithm": "uniform_eligible_negative_epoch_sampler_v1",
            "identity_hash": identity.identity_hash,
            "source_order": source_order,
            "cursor": 5,
            "next_epoch": 3,
            "batch_cursor": 0,
            "pending_batches": 0,
            "prefetch_pending": 0,
            "inflight_workers": 0,
            "worker_ids": [],
        }

        progress = {
            "epoch": 2,
            "global_step": 4,
            "epoch_optimizer_steps": [2, 2],
            "empty_pool_skips": [2, 2],
        }

        selection = {
            "metric": "validation_NDCG@10",
            "history": [{"epoch": 2, "score": 0.5}],
            "best_score": 0.5,
            "selected_epoch": 2,
            "bad_checks": 0,
            "status": "ready",
            "stop_reason": None,
        }

        optimizer = {
            "algorithm": "Adam",
            "state_dict": {
                "state": {
                    0: {"step": torch.tensor(4.0, dtype=torch.float32), "exp_avg": torch.zeros(3, 64), "exp_avg_sq": torch.zeros(3, 64)},
                    1: {"step": torch.tensor(4.0, dtype=torch.float32), "exp_avg": torch.zeros(3, 64), "exp_avg_sq": torch.zeros(3, 64)},
                },
                "param_groups": [
                    {
                        "params": [0, 1],
                        "lr": 0.001,
                        "betas": (0.9, 0.999),
                        "eps": 1e-08,
                        "weight_decay": 0.0,
                        "amsgrad": False,
                        "maximize": False,
                        "foreach": None,
                        "capturable": False,
                        "differentiable": False,
                        "fused": None,
                        "decoupled_weight_decay": False,
                    }
                ],
            },
            "parameter_bindings": [
                {"id": 0, "name": "user_embeddings", "updates": 4},
                {"id": 1, "name": "item_embeddings", "updates": 4},
            ],
        }

        roles = []
        for s in identity.resolved_settings["determinism"]["rng_streams"]:
            r_rec = dict(s)
            if s["applicable"]:
                r_rec["generator_ids"] = expected_runtime["rng_role_bindings"][s["name"]]
            else:
                r_rec["generator_ids"] = []
            roles.append(r_rec)
        rng_dict = {"roles": roles, "generators": generators}

        btf_payload = {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "capability": PAYLOAD_CAPABILITY,
            "identity": identity.to_dict(),
            "provenance": provenance,
            "runtime_profile": copy.deepcopy(expected_runtime["runtime_profile"]),
            "current_model": current_model,
            "best_model": best_model,
            "optimizer": optimizer,
            "rng": rng_dict,
            "sampler": sampler,
            "progress": progress,
            "selection": selection,
            "scheduler": {"applicable": False, "state": None, "reason": "not_used_constant_learning_rate"},
            "amp": {"applicable": False, "state": None, "reason": "not_used_cpu_float32"},
        }

        policy = _make_synthetic_resource_policy()

        res = validate_checkpoint_payload(
            btf_payload,
            expected_identity=identity.to_dict(),
            expected_provenance=provenance,
            expected_model=expected_model,
            expected_runtime=expected_runtime,
            resource_policy=policy,
        )
        self.assertEqual(res["runtime_profile"]["batch_policy"], "batch_then_filter")
        self.assertEqual(res["progress"]["epoch_optimizer_steps"], [2, 2])
        self.assertEqual(res["progress"]["empty_pool_skips"], [2, 2])

        # Explicit negative witness: old erroneous six-edge graph fails closed
        bad_old_six_edge = copy.deepcopy(btf_payload)
        old_bad_indices = torch.tensor([[0, 0, 1, 2, 2, 2], [0, 1, 1, 0, 1, 2]], dtype=torch.int64)
        old_bad_vals = torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5, 0.5], dtype=torch.float32)
        bad_old_six_edge["current_model"]["state_dict"]["adj_user_item"] = torch.sparse_coo_tensor(
            old_bad_indices, old_bad_vals, (3, 3)
        ).coalesce()
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(
                bad_old_six_edge,
                expected_identity=identity.to_dict(),
                expected_provenance=provenance,
                expected_model=expected_model,
                expected_runtime=expected_runtime,
                resource_policy=policy,
            )

        # Negative: clearing source_order tests order length mismatch
        bad_zero_pairs = copy.deepcopy(btf_payload)
        bad_zero_pairs["sampler"]["source_order"] = []
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(
                bad_zero_pairs,
                expected_identity=identity.to_dict(),
                expected_provenance=provenance,
                expected_model=expected_model,
                expected_runtime=expected_runtime,
                resource_policy=policy,
            )

    def test_valid_empty_hetero_metadata_tables(self) -> None:
        """Valid empty Hetero metadata tables: CategoryPath and Brand (0, 64) with all 36 parameters required."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_hetero_payload(empty_metadata=True)
        policy = _make_synthetic_resource_policy()

        res = validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )
        self.assertEqual(res["current_model"]["row_order"]["CategoryPath"], [])
        self.assertEqual(res["current_model"]["row_order"]["Brand"], [])
        self.assertEqual(res["current_model"]["state_dict"]["category_embeddings"].shape, (0, 64))
        self.assertEqual(res["current_model"]["state_dict"]["brand_embeddings"].shape, (0, 64))

        # Negative: empty User or Product fails closed
        bad_empty_user = copy.deepcopy(payload)
        bad_empty_user["current_model"]["row_order"]["User"] = []
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(
                bad_empty_user,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )

    def test_rng_comprehensive_matrix_and_worker_w2(self) -> None:
        """Comprehensive RNG matrix: W=2 multi-family worker contexts, negatives for roles, bindings, and states."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload(runtime_workers=2)
        policy = _make_synthetic_resource_policy()

        # W=2 passes: 4 parent + 6 worker (2 workers * 3 families) = 10 generators
        res = validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )
        self.assertEqual(len(res["rng"]["generators"]), 10)
        self.assertEqual(len(res["rng"]["roles"][9]["generator_ids"]), 6)
        self.assertEqual(res["sampler"]["worker_ids"], [0, 1])

        # Negative: duplicate generator ID within a role
        bad_dup_role = copy.deepcopy(payload)
        bad_dup_role["rng"]["roles"][0]["generator_ids"] = ["gen_py", "gen_py"]
        bad_rt_dup = copy.deepcopy(exp_rt)
        bad_rt_dup["rng_role_bindings"]["Python"] = ["gen_py", "gen_py"]
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad_dup_role, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=bad_rt_dup, resource_policy=policy)

        # Negative: orphan generator in registry not bound to any role
        bad_orphan_reg = copy.deepcopy(payload)
        bad_orphan_reg["rng"]["generators"].append({
            "id": "gen_orphan",
            "codec": "python.random.MT19937.v3",
            "worker_id": None,
            "seed_material": 999,
            "state": random.Random(999).getstate(),
        })
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad_orphan_reg, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Negative: inactive role with non-empty generator IDs
        bad_inactive = copy.deepcopy(payload)
        for r in bad_inactive["rng"]["roles"]:
            if r["name"] == "PyTorch_CUDA_if_applicable":
                r["generator_ids"] = ["gen_torch"]
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad_inactive, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # Negative: worker generator missing family (e.g. worker 0 has only Python, missing NumPy and TorchCPU)
        bad_missing_fam = copy.deepcopy(payload)
        bad_missing_fam["rng"]["roles"][9]["generator_ids"] = ["gen_worker_0_py", "gen_worker_1_py", "gen_worker_1_np", "gen_worker_1_torch"]
        bad_missing_fam["rng"]["generators"] = [g for g in bad_missing_fam["rng"]["generators"] if g["id"] not in ("gen_worker_0_np", "gen_worker_0_torch")]
        bad_rt_miss = copy.deepcopy(exp_rt)
        bad_rt_miss["rng_role_bindings"]["DataLoader_workers"] = bad_missing_fam["rng"]["roles"][9]["generator_ids"]
        del bad_rt_miss["rng_generator_bindings"]["gen_worker_0_np"]
        del bad_rt_miss["rng_generator_bindings"]["gen_worker_0_torch"]
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad_missing_fam, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=bad_rt_miss, resource_policy=policy)

    def test_storage_deduplication_same_address_different_capacity(self) -> None:
        """Storage deduplication: distinct StorageImpls sharing base address with different capacities are tracked independently."""
        buf = bytearray(256)
        mv = memoryview(buf)
        t1 = torch.frombuffer(mv[:128], dtype=torch.uint8)
        t2 = torch.frombuffer(mv, dtype=torch.uint8)

        self.assertEqual(t1.data_ptr(), t2.data_ptr())
        self.assertNotEqual(t1.untyped_storage().nbytes(), t2.untyped_storage().nbytes())

        policy = _make_synthetic_resource_policy(max_input_bytes=1000, max_output_bytes=1000, kind="runtime", profile_id="r_same_addr")
        m, s, c, v = preflight_resource_accounting([t1, t2], policy)
        # S must account for both storage capacities: 128 + 256 = 384
        self.assertEqual(s, 384)

    def test_strict_schemas_and_types(self) -> None:
        """Strict schema and type enforcement: float for int, int for bool, empty strings, and extra expected fields."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()

        # selected_epoch is float 2.0 instead of int 2
        bad_epoch = copy.deepcopy(payload)
        bad_epoch["selection"]["selected_epoch"] = 2.0
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad_epoch, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # role applicable is int 1 instead of bool True
        bad_app = copy.deepcopy(payload)
        bad_app["rng"]["roles"][0]["applicable"] = 1
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad_app, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # role seed is float 42.0 instead of int 42
        bad_seed = copy.deepcopy(payload)
        bad_seed["rng"]["roles"][0]["seed"] = 42.0
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad_seed, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # runtime_profile string is whitespace-only
        bad_profile_str = copy.deepcopy(payload)
        bad_profile_str["runtime_profile"]["profile_id"] = "   "
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(bad_profile_str, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=exp_rt, resource_policy=policy)

        # extra unknown field in expected_runtime
        bad_exp_rt = copy.deepcopy(exp_rt)
        bad_exp_rt["extra_unexpected_field"] = "bad"
        with self.assertRaises(CheckpointValidationError):
            validate_checkpoint_payload(payload, expected_identity=exp_id, expected_provenance=exp_prov, expected_model=exp_model, expected_runtime=bad_exp_rt, resource_policy=policy)


# ---------------------------------------------------------------------------
# Ported Diagnostic / Oracle Suite (7 Named Methods)
# ---------------------------------------------------------------------------

class TestDiagnosticContractAndOracles(unittest.TestCase):
    """Ported diagnostic reproductions and independent oracles exercising candidate validator."""

    def test_r1_metaclass_hash_hook(self) -> None:
        """R1: unsupported object with attacker metaclass overriding __hash__ inside valid payload."""
        class AttackerMeta(type):
            hook_calls = 0

            def __hash__(cls) -> int:
                AttackerMeta.hook_calls += 1
                return id(cls)

        class AttackerObject(metaclass=AttackerMeta):
            pass

        # 1. First validate complete valid full control via PUBLIC API
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()
        control_res = checkpoint_payload.validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )
        self.assertIsInstance(control_res, dict)

        # 2. Apply single documented corruption: insert unsupported object as VALUE
        AttackerMeta.hook_calls = 0
        attacker_instance = AttackerObject()
        self.assertEqual(AttackerMeta.hook_calls, 0, "Metaclass hook invoked during instance creation")

        payload["custom_value"] = attacker_instance

        # 3. Expected: CheckpointValidationError raised AND exactly 0 hook calls
        refusal_raised = False
        try:
            checkpoint_payload.validate_checkpoint_payload(
                payload,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )
        except CheckpointValidationError:
            refusal_raised = True

        self.assertTrue(refusal_raised, "Expected CheckpointValidationError for unsupported custom object")
        self.assertEqual(
            AttackerMeta.hook_calls,
            0,
            f"Metaclass __hash__ hook was invoked {AttackerMeta.hook_calls} time(s) during type-name lookup; expected 0",
        )

    def test_r2_huge_update_domain_error(self) -> None:
        """R2: parameter 0 updates changed to 10**5000, leaving global_step G=4."""
        # 1. First validate complete valid full control via PUBLIC API
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload()
        policy = _make_synthetic_resource_policy()
        control_res = checkpoint_payload.validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )
        self.assertIsInstance(control_res, dict)

        # 2. Apply single documented corruption: parameter 0 updates = 10**5000, leave G=4
        corrupted_payload = copy.deepcopy(payload)
        corrupted_payload["optimizer"]["parameter_bindings"][0]["updates"] = 10**5000
        self.assertEqual(corrupted_payload["progress"]["global_step"], 4)

        # 3. Catch actual exception safely to record TYPE without triggering conversion errors
        exc_type = None
        try:
            checkpoint_payload.validate_checkpoint_payload(
                corrupted_payload,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )
        except BaseException as exc:
            exc_type = type(exc)

        # 4. Assert required contract outcome: CheckpointValidationError family, not native ValueError
        self.assertIsNotNone(exc_type, "Expected exception for huge update integer")
        self.assertTrue(
            issubclass(exc_type, CheckpointValidationError),
            f"Expected domain CheckpointValidationError family, got native {exc_type.__name__}",
        )

    def test_r3_torch_worker_seed_integer(self) -> None:
        """R3: worker 0 Torch seed_material corrupted from int N to [N] in both actual and expected."""
        # 1. First validate complete valid W=1 full control via PUBLIC API
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload(runtime_workers=1)
        policy = _make_synthetic_resource_policy()
        control_res = checkpoint_payload.validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )
        self.assertIsInstance(control_res, dict)
        self.assertEqual(len(control_res["rng"]["generators"]), 7)

        # 2. Apply single documented corruption: worker 0 Torch seed_material int N -> [N]
        torch_gen = None
        for g in payload["rng"]["generators"]:
            if g.get("worker_id") == 0 and g.get("codec") == "torch.cpu.Generator.native.v1":
                torch_gen = g
                break
        self.assertIsNotNone(torch_gen, "Worker 0 Torch generator not found in W=1 fixture")
        self.assertIsInstance(torch_gen["seed_material"], int)

        torch_gid = torch_gen["id"]
        original_seed = torch_gen["seed_material"]
        torch_gen["seed_material"] = [original_seed]
        exp_rt["rng_generator_bindings"][torch_gid]["seed_material"] = [original_seed]

        # 3. Assert required contract outcome: domain refusal of forbidden list
        with self.assertRaises(
            CheckpointValidationError,
            msg="Validator accepted forbidden list [N] for Torch worker seed_material; expected CheckpointValidationError",
        ):
            checkpoint_payload.validate_checkpoint_payload(
                payload,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )

    def test_r4_python_worker_seed_integer(self) -> None:
        """R4: worker 0 Python seed_material corrupted from int N to [N] in both actual and expected."""
        # 1. First validate complete valid W=1 full control via PUBLIC API
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_lightgcn_payload(runtime_workers=1)
        policy = _make_synthetic_resource_policy()
        control_res = checkpoint_payload.validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )
        self.assertIsInstance(control_res, dict)
        self.assertEqual(len(control_res["rng"]["generators"]), 7)

        # 2. Apply single documented corruption: worker 0 Python seed_material int N -> [N]
        py_gen = None
        for g in payload["rng"]["generators"]:
            if g.get("worker_id") == 0 and g.get("codec") == "python.random.MT19937.v3":
                py_gen = g
                break
        self.assertIsNotNone(py_gen, "Worker 0 Python generator not found in W=1 fixture")
        self.assertIsInstance(py_gen["seed_material"], int)

        py_gid = py_gen["id"]
        original_seed = py_gen["seed_material"]
        py_gen["seed_material"] = [original_seed]
        exp_rt["rng_generator_bindings"][py_gid]["seed_material"] = [original_seed]

        # 3. Assert required contract outcome: domain refusal of invalid codec seed type
        with self.assertRaises(
            CheckpointValidationError,
            msg="Validator accepted forbidden list [N] for Python worker seed_material; expected CheckpointValidationError",
        ):
            checkpoint_payload.validate_checkpoint_payload(
                payload,
                expected_identity=exp_id,
                expected_provenance=exp_prov,
                expected_model=exp_model,
                expected_runtime=exp_rt,
                resource_policy=policy,
            )

    def test_r5a_old_empty_pool_fixture_independent_derivation(self) -> None:
        """R5a: Independently derive P/K/T = 5/2/3 for with_empty_pool fixture and prove clearing source_order leaves T=3."""
        identity, provenance = _make_synthetic_identity_and_provenance(method="LightGCN", with_empty_pool=True)

        # 1. Independent derivation of P, K, T from identity histories
        P: Set[Tuple[str, str]] = set()
        for u, items in identity.positive_history.items():
            for item in items:
                P.add((u, item))

        K: Set[Tuple[str, str]] = set()
        warm_set = set(identity.warm_items)
        for u, pos_items in identity.positive_history.items():
            all_rated = set(identity.all_rated_history.get(u, ()))
            eligible_negatives = warm_set - all_rated
            if len(eligible_negatives) == 0:
                for item in pos_items:
                    K.add((u, item))

        T = len(P - K)

        self.assertEqual(len(P), 5, "Expected |P| = 5")
        self.assertEqual(len(K), 2, "Expected |K| = 2")
        self.assertEqual(T, 3, "Expected T = 3")
        self.assertEqual(sorted(P), [("u1", "p1"), ("u1", "p2"), ("u2", "p3"), ("u3", "p1"), ("u3", "p2")])
        self.assertEqual(sorted(K), [("u3", "p1"), ("u3", "p2")])
        self.assertEqual(sorted(P - K), [("u1", "p1"), ("u1", "p2"), ("u2", "p3")])

        # Also verify validator's internal pair computation matches independent derivation
        p_val, k_val = checkpoint_payload._compute_condition_positive_and_empty_pool_pairs(identity)
        self.assertEqual(p_val, P)
        self.assertEqual(k_val, K)

        # 2. Demonstrate that clearing source_order does NOT alter T
        source_order: List[Tuple[str, str]] = []
        self.assertNotEqual(len(source_order), len(P))
        self.assertEqual(T, 3, "Clearing source_order does not change dataset T=3")

    def test_r5b_genuine_untrainable_t0_domain_refusal(self) -> None:
        """R5b: Build a tiny genuinely untrainable boundary (T=0) via accepted public D1 builder."""
        base_payload, base_exp_id, base_exp_prov, base_exp_model, base_exp_rt = _make_valid_lightgcn_payload(epoch=2, global_step=4)

        u_list = ["u1", "u2"]
        p_list = ["p1", "p2"]
        warm_items = ["p1", "p2"]

        training_records = [
            {"user_id": "u1", "parent_asin": "p1", "rating": 5.0, "timestamp": 1000},
            {"user_id": "u1", "parent_asin": "p2", "rating": 5.0, "timestamp": 2000},
            {"user_id": "u2", "parent_asin": "p1", "rating": 5.0, "timestamp": 3000},
            {"user_id": "u2", "parent_asin": "p2", "rating": 5.0, "timestamp": 4000},
        ]
        positive_history = {
            "u1": ["p1", "p2"],
            "u2": ["p1", "p2"],
        }
        all_rated_history = {
            "u1": ["p1", "p2"],
            "u2": ["p1", "p2"],
        }
        graph_nodes = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2"],
            "CategoryPath": [],
            "Brand": [],
        }
        graph_edges = [
            ("User", "u1", "rated", "Product", "p1", 1.0),
            ("User", "u1", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p1", 1.0),
            ("User", "u2", "rated", "Product", "p2", 1.0),
            ("Product", "p1", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u1", 1.0),
            ("Product", "p1", "rated_by", "User", "u2", 1.0),
            ("Product", "p2", "rated_by", "User", "u2", 1.0),
        ]
        product_metadata = {
            "p1": {"categories": [], "brand": None},
            "p2": {"categories": [], "brand": None},
        }
        validation_inputs = {
            "targets": {"u1": ["p_cold_target"]},
            "records": [("u1", "p_cold_target", 5.0, 5000)],
        }

        identity = build_canonical_identity(
            method="LightGCN",
            category="Musical_Instruments",
            feedback_condition="all_ratings",
            seed=42,
            training_records=training_records,
            positive_history=positive_history,
            all_rated_history=all_rated_history,
            warm_items=warm_items,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            product_metadata=product_metadata,
            validation_inputs=validation_inputs,
            resolved_settings=copy.deepcopy(base_exp_id["resolved_settings"]),
        )

        # Independently assert P == K and T = 0 BEFORE calling validator
        P = set((u, i) for u in identity.positive_history for i in identity.positive_history[u])
        K = set()
        warm_set = set(identity.warm_items)
        for u in identity.positive_history:
            eligible = warm_set - set(identity.all_rated_history[u])
            if len(eligible) == 0:
                for i in identity.positive_history[u]:
                    K.add((u, i))
        T = len(P - K)

        self.assertEqual(len(P), 4)
        self.assertEqual(P, K, "Every positive pair must belong to empty negative pool K")
        self.assertEqual(T, 0, "Trainable pairs T must be exactly 0")

        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.int64)
        vals = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
        adj_buffer = torch.sparse_coo_tensor(indices, vals, (2, 2)).coalesce()

        expected_model = {
            "method": "LightGCN",
            "row_order": {"User": u_list, "Product": p_list},
            "buffer_state": {"adj_user_item": adj_buffer.clone()},
        }

        current_model = {
            "epoch": 0,
            "row_order": {"User": u_list, "Product": p_list},
            "state_dict": {
                "user_embeddings": torch.zeros(2, 64, dtype=torch.float32),
                "item_embeddings": torch.zeros(2, 64, dtype=torch.float32),
                "adj_user_item": adj_buffer.clone(),
            },
        }

        expected_runtime = copy.deepcopy(base_exp_rt)
        expected_runtime["optimizer_groups"][0]["parameter_names"] = ["user_embeddings", "item_embeddings"]

        sampler = {
            "algorithm": "uniform_eligible_negative_epoch_sampler_v1",
            "identity_hash": identity.identity_hash,
            "source_order": None,
            "cursor": 0,
            "next_epoch": 1,
            "batch_cursor": 0,
            "pending_batches": 0,
            "prefetch_pending": 0,
            "inflight_workers": 0,
            "worker_ids": [],
        }

        progress = {
            "epoch": 0,
            "global_step": 0,
            "epoch_optimizer_steps": [],
            "empty_pool_skips": [],
        }

        selection = {
            "metric": "validation_NDCG@10",
            "history": [],
            "best_score": None,
            "selected_epoch": None,
            "bad_checks": 0,
            "status": "initial",
            "stop_reason": None,
        }

        optimizer = {
            "algorithm": "Adam",
            "state_dict": {
                "state": {0: {}, 1: {}},
                "param_groups": [
                    {
                        "params": [0, 1],
                        "lr": 0.001,
                        "betas": (0.9, 0.999),
                        "eps": 1e-08,
                        "weight_decay": 0.0,
                        "amsgrad": False,
                        "maximize": False,
                        "foreach": None,
                        "capturable": False,
                        "differentiable": False,
                        "fused": None,
                        "decoupled_weight_decay": False,
                    }
                ],
            },
            "parameter_bindings": [
                {"id": 0, "name": "user_embeddings", "updates": 0},
                {"id": 1, "name": "item_embeddings", "updates": 0},
            ],
        }

        provenance = copy.deepcopy(base_exp_prov)

        payload = {
            "schema_version": checkpoint_payload.PAYLOAD_SCHEMA_VERSION,
            "capability": checkpoint_payload.PAYLOAD_CAPABILITY,
            "identity": identity.to_dict(),
            "provenance": provenance,
            "runtime_profile": copy.deepcopy(expected_runtime["runtime_profile"]),
            "current_model": current_model,
            "best_model": None,
            "optimizer": optimizer,
            "rng": copy.deepcopy(base_payload["rng"]),
            "sampler": sampler,
            "progress": progress,
            "selection": selection,
            "scheduler": {"applicable": False, "state": None, "reason": "not_used_constant_learning_rate"},
            "amp": {"applicable": False, "state": None, "reason": "not_used_cpu_float32"},
        }

        policy = _make_synthetic_resource_policy()

        with self.assertRaises(CheckpointValidationError) as ctx:
            checkpoint_payload.validate_checkpoint_payload(
                payload,
                expected_identity=identity.to_dict(),
                expected_provenance=provenance,
                expected_model=expected_model,
                expected_runtime=expected_runtime,
                resource_policy=policy,
            )

        self.assertIn(
            "Untrainable run: condition-positive trainable pairs T=0 <= 0",
            str(ctx.exception),
            "Expected domain error specifically from untrainable/T<=0 guard",
        )

    def test_r6_btf_fixture_adjacency_mismatch_oracle(self) -> None:
        """R6: Independently compare old BTF fixture declared adjacency against retained positives and LightGCNGraph."""
        identity, provenance = _make_synthetic_identity_and_provenance(method="LightGCN", with_empty_pool=True)

        # 1. Retained positive pairs in identity
        retained_positives = set()
        for u, items in identity.positive_history.items():
            for item in items:
                retained_positives.add((u, item))
        self.assertEqual(len(retained_positives), 5)
        self.assertEqual(sorted(retained_positives), [("u1", "p1"), ("u1", "p2"), ("u2", "p3"), ("u3", "p1"), ("u3", "p2")])

        # 2. Old BTF fixture declared adjacency indices
        u_list = ["u1", "u2", "u3"]
        p_list = ["p1", "p2", "p3"]
        old_indices = torch.tensor([[0, 0, 1, 2, 2, 2], [0, 1, 1, 0, 1, 2]], dtype=torch.int64)
        old_declared_edges = set((u_list[r], p_list[c]) for r, c in zip(old_indices[0].tolist(), old_indices[1].tolist()))
        self.assertEqual(len(old_declared_edges), 6)

        # 3. Mismatch set differences
        extra_in_old = old_declared_edges - retained_positives
        missing_in_old = retained_positives - old_declared_edges
        self.assertEqual(extra_in_old, {("u2", "p2"), ("u3", "p3")})
        self.assertEqual(missing_in_old, {("u2", "p3")})

        # 4. Accepted LightGCN degree normalization on retained positives
        user_degrees = {u: len(items) for u, items in identity.positive_history.items()}
        item_degrees: Dict[str, int] = {}
        for u, items in identity.positive_history.items():
            for item in items:
                item_degrees[item] = item_degrees.get(item, 0) + 1

        norm_weights = {}
        for u, item in retained_positives:
            norm_weights[(u, item)] = 1.0 / math.sqrt(user_degrees[u] * item_degrees[item])

        self.assertEqual(len(norm_weights), 5)
        self.assertAlmostEqual(norm_weights[("u1", "p1")], 0.5, places=6)
        self.assertAlmostEqual(norm_weights[("u1", "p2")], 0.5, places=6)
        self.assertAlmostEqual(norm_weights[("u2", "p3")], 1.0, places=6)
        self.assertAlmostEqual(norm_weights[("u3", "p1")], 0.5, places=6)
        self.assertAlmostEqual(norm_weights[("u3", "p2")], 0.5, places=6)

        self.assertNotEqual(len(norm_weights), len(old_declared_edges))

        # 5. Negative witness: passing old 6-edge buffer against 5-edge expected buffer fails closed
        old_vals = torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5, 0.5], dtype=torch.float32)
        old_buffer = torch.sparse_coo_tensor(old_indices, old_vals, (3, 3)).coalesce()
        valid_indices = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 2, 0, 1]], dtype=torch.int64)
        valid_vals = torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5], dtype=torch.float32)
        valid_buffer = torch.sparse_coo_tensor(valid_indices, valid_vals, (3, 3)).coalesce()

        with self.assertRaises(CheckpointValidationError) as ctx:
            checkpoint_payload._validate_sparse_adj_buffer(
                old_buffer,
                valid_buffer,
                num_users=3,
                num_items=3,
                role="current_model",
            )
        self.assertIn("indices do not match trusted graph contents", str(ctx.exception))





# ===========================================================================
# 37-Part Blocker Matrix Tests & Coherent Boundary Helpers
# ===========================================================================

EXECUTED_CASES: Set[str] = set()


def _build_coherent_identity_and_provenance(
    method: str = "LightGCN",
    runtime_workers: int = 0,
    empty_metadata: bool = False,
    with_empty_pool: bool = False,
    validation_frequency: int = 2,
    max_epochs: int = 100,
    seed: int = 42,
    active_cuda_role: bool = False,
) -> Tuple[CanonicalResumeIdentity, Dict[str, Any]]:
    """Build coherent canonical resume identity and source provenance with custom training/determinism settings."""
    if with_empty_pool:
        training_records = [
            {"user_id": "u1", "parent_asin": "p1", "rating": 5.0, "timestamp": 1000},
            {"user_id": "u1", "parent_asin": "p2", "rating": 5.0, "timestamp": 2000},
            {"user_id": "u2", "parent_asin": "p3", "rating": 5.0, "timestamp": 3000},
            {"user_id": "u3", "parent_asin": "p1", "rating": 5.0, "timestamp": 4000},
            {"user_id": "u3", "parent_asin": "p2", "rating": 5.0, "timestamp": 5000},
            {"user_id": "u3", "parent_asin": "p3", "rating": 2.0, "timestamp": 6000},
        ]
        positive_history = {
            "u1": ["p1", "p2"],
            "u2": ["p3"],
            "u3": ["p1", "p2"],
        }
        all_rated_history = {
            "u1": ["p1", "p2"],
            "u2": ["p3"],
            "u3": ["p1", "p2", "p3"],
        }
        warm_items = ["p1", "p2", "p3"]
        graph_nodes = {
            "User": ["u1", "u2", "u3"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": ['c:["Electronics"]', 'c:["Electronics","Audio"]'],
            "Brand": ["b:Sony"],
        }
        graph_edges = [
            ("User", "u1", "rated", "Product", "p1", 1.0),
            ("User", "u1", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p3", 1.0),
            ("User", "u3", "rated", "Product", "p1", 1.0),
            ("User", "u3", "rated", "Product", "p2", 1.0),
            ("User", "u3", "rated", "Product", "p3", 1.0),
            ("Product", "p1", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u1", 1.0),
            ("Product", "p3", "rated_by", "User", "u2", 1.0),
            ("Product", "p1", "rated_by", "User", "u3", 1.0),
            ("Product", "p2", "rated_by", "User", "u3", 1.0),
            ("Product", "p3", "rated_by", "User", "u3", 1.0),
            ("Product", "p1", "belongs_to", "CategoryPath", 'c:["Electronics","Audio"]', 1.0),
            ("Product", "p1", "belongs_to", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("Product", "p2", "belongs_to", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("CategoryPath", 'c:["Electronics","Audio"]', "contains_product", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics"]', "contains_product", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics"]', "contains_product", "Product", "p2", 1.0),
            ("Product", "p1", "has_brand", "Brand", "b:Sony", 1.0),
            ("Brand", "b:Sony", "brand_of", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics","Audio"]', "category_child_of", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("CategoryPath", 'c:["Electronics"]', "category_parent_of", "CategoryPath", 'c:["Electronics","Audio"]', 1.0),
        ]
        product_metadata = {
            "p1": {"categories": ['c:["Electronics","Audio"]'], "brand": "b:Sony"},
            "p2": {"categories": ['c:["Electronics"]'], "brand": None},
            "p3": {"categories": [], "brand": None},
        }
        validation_inputs = {
            "targets": {"u1": ["p3"], "u2": ["p1"]},
            "records": [("u1", "p3", 5.0, 5000), ("u2", "p1", 4.0, 6000)],
        }
    elif empty_metadata:
        training_records = [
            {"user_id": "u1", "parent_asin": "p1", "rating": 4.0, "timestamp": 1000},
            {"user_id": "u1", "parent_asin": "p2", "rating": 5.0, "timestamp": 2000},
            {"user_id": "u2", "parent_asin": "p2", "rating": 4.0, "timestamp": 3000},
            {"user_id": "u2", "parent_asin": "p3", "rating": 4.0, "timestamp": 4000},
        ]
        positive_history = {
            "u1": ["p1", "p2"],
            "u2": ["p2", "p3"],
        }
        all_rated_history = {
            "u1": ["p1", "p2"],
            "u2": ["p2", "p3"],
        }
        warm_items = ["p1", "p2", "p3"]
        graph_nodes = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": [],
            "Brand": [],
        }
        graph_edges = [
            ("User", "u1", "rated", "Product", "p1", 1.0),
            ("User", "u1", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p3", 1.0),
            ("Product", "p1", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u2", 1.0),
            ("Product", "p3", "rated_by", "User", "u2", 1.0),
        ]
        product_metadata = {
            "p1": {"categories": [], "brand": None},
            "p2": {"categories": [], "brand": None},
            "p3": {"categories": [], "brand": None},
        }
        validation_inputs = {
            "targets": {"u1": ["p3"], "u2": ["p1"]},
            "records": [("u1", "p3", 5.0, 5000), ("u2", "p1", 4.0, 6000)],
        }
    else:
        training_records = [
            {"user_id": "u1", "parent_asin": "p1", "rating": 4.0, "timestamp": 1000},
            {"user_id": "u1", "parent_asin": "p2", "rating": 5.0, "timestamp": 2000},
            {"user_id": "u2", "parent_asin": "p2", "rating": 4.0, "timestamp": 3000},
            {"user_id": "u2", "parent_asin": "p3", "rating": 4.0, "timestamp": 4000},
        ]
        positive_history = {
            "u1": ["p1", "p2"],
            "u2": ["p2", "p3"],
        }
        all_rated_history = {
            "u1": ["p1", "p2"],
            "u2": ["p2", "p3"],
        }
        warm_items = ["p1", "p2", "p3"]
        graph_nodes = {
            "User": ["u1", "u2"],
            "Product": ["p1", "p2", "p3"],
            "CategoryPath": ['c:["Electronics"]', 'c:["Electronics","Audio"]'],
            "Brand": ["b:Sony"],
        }
        graph_edges = [
            ("User", "u1", "rated", "Product", "p1", 1.0),
            ("User", "u1", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p2", 1.0),
            ("User", "u2", "rated", "Product", "p3", 1.0),
            ("Product", "p1", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u1", 1.0),
            ("Product", "p2", "rated_by", "User", "u2", 1.0),
            ("Product", "p3", "rated_by", "User", "u2", 1.0),
            ("Product", "p1", "belongs_to", "CategoryPath", 'c:["Electronics","Audio"]', 1.0),
            ("Product", "p1", "belongs_to", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("Product", "p2", "belongs_to", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("CategoryPath", 'c:["Electronics","Audio"]', "contains_product", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics"]', "contains_product", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics"]', "contains_product", "Product", "p2", 1.0),
            ("Product", "p1", "has_brand", "Brand", "b:Sony", 1.0),
            ("Brand", "b:Sony", "brand_of", "Product", "p1", 1.0),
            ("CategoryPath", 'c:["Electronics","Audio"]', "category_child_of", "CategoryPath", 'c:["Electronics"]', 1.0),
            ("CategoryPath", 'c:["Electronics"]', "category_parent_of", "CategoryPath", 'c:["Electronics","Audio"]', 1.0),
        ]
        product_metadata = {
            "p1": {"categories": ['c:["Electronics","Audio"]'], "brand": "b:Sony"},
            "p2": {"categories": ['c:["Electronics"]'], "brand": None},
            "p3": {"categories": [], "brand": None},
        }
        validation_inputs = {
            "targets": {"u1": ["p3"], "u2": ["p1"]},
            "records": [("u1", "p3", 5.0, 5000), ("u2", "p1", 4.0, 6000)],
        }

    if runtime_workers > 0:
        worker_settings = {
            "multiprocessing_context": "spawn",
            "persistent_workers": False,
            "prefetch_factor": 2,
            "worker_init_function": "worker_init_fn",
        }
        worker_derivation = {
            "applicable": True,
            "algorithm": "torch_worker_init_v1",
            "base_seed": seed,
            "derivation": "seed_sequence",
            "reason": None,
        }
    else:
        worker_settings = {
            "multiprocessing_context": None,
            "persistent_workers": False,
            "prefetch_factor": None,
            "worker_init_function": None,
        }
        worker_derivation = {
            "applicable": False,
            "algorithm": None,
            "base_seed": None,
            "derivation": None,
            "reason": "zero_workers",
        }

    rng_streams = [
        {"name": "Python", "applicable": True, "algorithm": "stdlib_random", "seed": seed, "derivation": "fixed_seed", "reason": None},
        {"name": "NumPy", "applicable": True, "algorithm": "numpy_pcg64", "seed": seed, "derivation": "fixed_seed", "reason": None},
        {"name": "PyTorch_CPU", "applicable": True, "algorithm": "torch_cpu", "seed": seed, "derivation": "fixed_seed", "reason": None},
        {
            "name": "PyTorch_CUDA_if_applicable",
            "applicable": True if active_cuda_role else False,
            "algorithm": "torch_cuda" if active_cuda_role else None,
            "seed": seed if active_cuda_role else None,
            "derivation": "fixed_seed" if active_cuda_role else None,
            "reason": None if active_cuda_role else "not_applicable_cpu",
        },
        {"name": "initialization", "applicable": True, "algorithm": "model_init", "seed": seed, "derivation": "fixed_seed", "reason": None},
        {"name": "epoch_shuffle", "applicable": True, "algorithm": "epoch_shuffle", "seed": seed, "derivation": "fixed_seed", "reason": None},
        {"name": "negative_sampler", "applicable": True, "algorithm": "neg_sampler", "seed": seed, "derivation": "fixed_seed", "reason": None},
        {"name": "neighbor_sampler", "applicable": True, "algorithm": "neighbor_sampler", "seed": seed, "derivation": "fixed_seed", "reason": None},
        {"name": "DataLoader_generator", "applicable": True, "algorithm": "dl_gen", "seed": seed, "derivation": "fixed_seed", "reason": None},
        {
            "name": "DataLoader_workers",
            "applicable": True if runtime_workers > 0 else False,
            "algorithm": "worker_stream" if runtime_workers > 0 else None,
            "seed": seed if runtime_workers > 0 else None,
            "derivation": "worker_seed" if runtime_workers > 0 else None,
            "reason": None if runtime_workers > 0 else "zero_workers",
        },
    ]

    resolved_settings = {
        "protocol": {
            "candidate_policy": "full_candidate_ranking",
            "lifecycle_state": "IMPLEMENTATION_REGRESSION_VERIFICATION",
            "protocol_version": "amazon4u-protocol-v2.0.0",
        },
        "execution": {
            "cpu_threads": 4,
            "execution_device": "cpu",
            "execution_label": "synthetic_contract_verification",
            "runtime_mode": "benchmark_verification",
            "runtime_workers": runtime_workers,
            "worker_settings": worker_settings,
        },
        "model": {
            "architecture": method,
            "embedding_dim": 64,
            "initialization": "normal_0_0.1_xavier_uniform_gain_1",
            "num_layers": 3 if method == "LightGCN" else 2,
        },
        "training": {
            "adam_betas": [0.9, 0.999],
            "adam_epsilon": 1e-08,
            "batch_size": 2,
            "endpoint_regularization": 1e-05,
            "learning_rate": 0.001,
            "loss_function": "BPR",
            "max_epochs": max_epochs,
            "optimizer": "Adam",
            "patience": 10,
            "validation_frequency": validation_frequency,
            "weight_decay": 0.0,
        },
        "sampling": {
            "negative_strategy": "uniform_eligible",
            "negatives_per_positive": 1,
            "neighbor_fanouts": [15, 10],
            "pool_policy": "condition_warm_excluding_rated",
        },
        "determinism": {
            "deterministic_algorithms": True,
            "reproducibility_version": "amazon4u-reproducibility-v1",
            "rng_streams": rng_streams,
            "seed": seed,
            "worker_derivation": worker_derivation,
        },
    }

    identity = build_canonical_identity(
        method=method,
        category="Musical_Instruments",
        feedback_condition="rating_ge_4" if with_empty_pool else "all_ratings",
        seed=seed,
        training_records=training_records,
        positive_history=positive_history,
        all_rated_history=all_rated_history,
        warm_items=warm_items,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        product_metadata=product_metadata,
        validation_inputs=validation_inputs,
        resolved_settings=resolved_settings,
    )

    source_files = {
        "scripts/checkpoint_identity.py": b"# checkpoint identity stub\n",
        "configs/compute-plan.json": b"{}\n",
    }
    git_evidence = {
        "commit": "a" * 40,
        "dirty": False,
        "tracked_diff": "",
        "untracked_sources": [],
    }
    environment = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform_system": "Linux",
        "platform_release": "6.0",
        "platform_machine": "x86_64",
        "cpu_count": 4,
        "torch_version": str(torch.__version__),
        "library_versions": {"numpy": np.__version__},
    }
    provenance = build_source_provenance(
        source_files=source_files,
        git_evidence=git_evidence,
        environment_records=environment,
    )
    return identity, provenance


def _build_coherent_lightgcn_payload(
    epoch: int = 2,
    validation_frequency: int = 2,
    max_epochs: int = 100,
    scores: Optional[List[float]] = None,
    score_mode: str = "increasing",
    runtime_workers: int = 0,
    with_empty_pool: bool = False,
    active_cuda_role: bool = False,
    adam_options_param_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], CanonicalResumeIdentity, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build coherent LightGCN payload where selection, best_model, Adam updates, and progress exactly align."""
    identity, provenance = _build_coherent_identity_and_provenance(
        method="LightGCN",
        runtime_workers=runtime_workers,
        with_empty_pool=with_empty_pool,
        validation_frequency=validation_frequency,
        max_epochs=max_epochs,
        active_cuda_role=active_cuda_role,
    )

    u_list = ["u1", "u2", "u3"] if with_empty_pool else ["u1", "u2"]
    p_list = ["p1", "p2", "p3"]
    if with_empty_pool:
        # Independent five-edge oracle: user degrees 2,1,2;
        # product degrees 2,2,1. The low-rated u3,p3 edge is absent.
        adj_buffer = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 1, 2, 2], [0, 1, 2, 0, 1]], dtype=torch.int64),
            torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5], dtype=torch.float32),
            (3, 3),
            is_coalesced=True,
        )
    else:
        adj_buffer = _make_lightgcn_sparse_buffer(len(u_list), len(p_list))

    expected_model = {
        "method": "LightGCN",
        "row_order": {"User": list(u_list), "Product": list(p_list)},
        "buffer_state": {"adj_user_item": adj_buffer.clone()},
    }

    # Generators
    py_state = random.Random(42).getstate()
    np_raw_state = np.random.RandomState(42).get_state()
    np_mt_state = ("MT19937", list(map(int, np_raw_state[1])), int(np_raw_state[2]), int(np_raw_state[3]), float(np_raw_state[4]))
    pcg_state = copy.deepcopy(np.random.PCG64(42).state)
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(42)
    torch_native_state = torch_gen.get_state()

    generators = [
        {"id": "gen_py", "codec": "python.random.MT19937.v3", "worker_id": None, "seed_material": 42, "state": py_state},
        {"id": "gen_np_mt", "codec": "numpy.random.RandomState.MT19937.v1", "worker_id": None, "seed_material": 42, "state": np_mt_state},
        {"id": "gen_np_pcg", "codec": "numpy.random.PCG64.v1", "worker_id": None, "seed_material": 42, "state": pcg_state},
        {"id": "gen_torch", "codec": "torch.cpu.Generator.native.v1", "worker_id": None, "seed_material": 42, "state": torch_native_state},
    ]

    expected_generator_bindings = {
        "gen_py": {"codec": "python.random.MT19937.v3", "worker_id": None, "seed_material": 42},
        "gen_np_mt": {"codec": "numpy.random.RandomState.MT19937.v1", "worker_id": None, "seed_material": 42},
        "gen_np_pcg": {"codec": "numpy.random.PCG64.v1", "worker_id": None, "seed_material": 42},
        "gen_torch": {"codec": "torch.cpu.Generator.native.v1", "worker_id": None, "seed_material": 42},
    }

    worker_gen_ids: List[str] = []
    for i in range(runtime_workers):
        g_w_py_id = f"gen_worker_{i}_py"
        g_w_np_id = f"gen_worker_{i}_np"
        g_w_torch_id = f"gen_worker_{i}_torch"
        generators.append({
            "id": g_w_py_id,
            "codec": "python.random.MT19937.v3",
            "worker_id": i,
            "seed_material": 42000 + i,
            "state": random.Random(42000 + i).getstate(),
        })
        generators.append({
            "id": g_w_np_id,
            "codec": "numpy.random.PCG64.v1",
            "worker_id": i,
            "seed_material": 42000 + i,
            "state": copy.deepcopy(np.random.PCG64(42000 + i).state),
        })
        t_gen_w = torch.Generator(device="cpu")
        t_gen_w.manual_seed(42000 + i)
        generators.append({
            "id": g_w_torch_id,
            "codec": "torch.cpu.Generator.native.v1",
            "worker_id": i,
            "seed_material": 42000 + i,
            "state": t_gen_w.get_state(),
        })
        expected_generator_bindings[g_w_py_id] = {"codec": "python.random.MT19937.v3", "worker_id": i, "seed_material": 42000 + i}
        expected_generator_bindings[g_w_np_id] = {"codec": "numpy.random.PCG64.v1", "worker_id": i, "seed_material": 42000 + i}
        expected_generator_bindings[g_w_torch_id] = {"codec": "torch.cpu.Generator.native.v1", "worker_id": i, "seed_material": 42000 + i}
        worker_gen_ids.extend([g_w_py_id, g_w_np_id, g_w_torch_id])

    opt_options: Dict[str, Any] = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }
    if adam_options_param_names is not None:
        opt_options["param_names"] = copy.deepcopy(adam_options_param_names)

    expected_runtime = {
        "runtime_profile": {
            "profile_id": "synthetic_profile_v1",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "adam_step_dtype": "torch.float32",
            "batch_policy": "filter_before_batch",
        },
        "sampler_algorithm": "uniform_eligible_negative_epoch_sampler_v1",
        "optimizer_groups": [
            {
                "parameter_names": ["user_embeddings", "item_embeddings"],
                "options": copy.deepcopy(opt_options),
            }
        ],
        "rng_role_bindings": {
            "Python": ["gen_py"],
            "NumPy": ["gen_np_pcg"],
            "PyTorch_CPU": ["gen_torch"],
            "PyTorch_CUDA_if_applicable": ["gen_torch"] if active_cuda_role else [],
            "initialization": ["gen_torch"],
            "epoch_shuffle": ["gen_py"],
            "negative_sampler": ["gen_np_mt"],
            "neighbor_sampler": ["gen_torch"],
            "DataLoader_generator": ["gen_torch"],
            "DataLoader_workers": list(worker_gen_ids),
        },
        "rng_generator_bindings": expected_generator_bindings,
    }

    embedding_generator = torch.Generator(device="cpu").manual_seed(101)
    user_emb = torch.randn(len(u_list), 64, dtype=torch.float32, generator=embedding_generator)
    item_emb = torch.randn(len(p_list), 64, dtype=torch.float32, generator=embedding_generator)

    current_model = {
        "epoch": epoch,
        "row_order": {"User": list(u_list), "Product": list(p_list)},
        "state_dict": {
            "user_embeddings": user_emb.clone(),
            "item_embeddings": item_emb.clone(),
            "adj_user_item": adj_buffer.clone(),
        },
    }

    global_step = 2 * epoch
    optimizer_pg: Dict[str, Any] = {
        "params": [0, 1],
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }
    if adam_options_param_names is not None:
        optimizer_pg["param_names"] = copy.deepcopy(adam_options_param_names)

    if epoch == 0:
        opt_state = {}
        bindings = [
            {"name": "user_embeddings", "id": 0, "updates": 0},
            {"name": "item_embeddings", "id": 1, "updates": 0},
        ]
    else:
        opt_state = {
            0: {
                "step": torch.tensor(float(global_step), dtype=torch.float32),
                "exp_avg": torch.zeros_like(user_emb),
                "exp_avg_sq": torch.zeros_like(user_emb),
            },
            1: {
                "step": torch.tensor(float(global_step), dtype=torch.float32),
                "exp_avg": torch.zeros_like(item_emb),
                "exp_avg_sq": torch.zeros_like(item_emb),
            },
        }
        bindings = [
            {"name": "user_embeddings", "id": 0, "updates": global_step},
            {"name": "item_embeddings", "id": 1, "updates": global_step},
        ]

    optimizer = {
        "algorithm": "Adam",
        "state_dict": {
            "state": opt_state,
            "param_groups": [optimizer_pg],
        },
        "parameter_bindings": bindings,
    }

    # Selection history recomputation
    scheduled_epochs: List[int] = []
    ep_cursor = validation_frequency
    while ep_cursor <= epoch and ep_cursor <= max_epochs:
        scheduled_epochs.append(ep_cursor)
        ep_cursor += validation_frequency

    if scores is not None:
        score_list = list(scores)
    elif score_mode == "increasing":
        score_list = [round(0.1 + 0.005 * i, 6) for i in range(len(scheduled_epochs))]
    elif score_mode == "ties":
        score_list = [0.5] * len(scheduled_epochs)
    else:
        score_list = [0.5] * len(scheduled_epochs)

    history: List[Dict[str, Any]] = []
    recomputed_best_score: Optional[float] = None
    recomputed_selected_epoch: Optional[int] = None
    recomputed_bad_checks = 0

    for idx, (sched_ep, sc) in enumerate(zip(scheduled_epochs, score_list)):
        history.append({"epoch": sched_ep, "score": sc})
        if idx == 0:
            recomputed_best_score = sc
            recomputed_selected_epoch = sched_ep
            recomputed_bad_checks = 0
        else:
            assert recomputed_best_score is not None
            if sc > recomputed_best_score:
                recomputed_best_score = sc
                recomputed_selected_epoch = sched_ep
                recomputed_bad_checks = 0
            else:
                recomputed_bad_checks += 1
                if recomputed_bad_checks == 10:
                    history = history[:idx + 1]
                    break

    is_patience_stopped = recomputed_bad_checks == 10
    is_max_epochs_stopped = epoch == max_epochs

    if is_patience_stopped and is_max_epochs_stopped:
        status = "stopped"
        stop_reason: Optional[str] = "patience_and_max_epochs"
    elif is_patience_stopped:
        status = "stopped"
        stop_reason = "patience"
    elif is_max_epochs_stopped:
        status = "stopped"
        stop_reason = "max_epochs"
    elif epoch == 0:
        status = "initial"
        stop_reason = None
    else:
        status = "ready"
        stop_reason = None

    selection = {
        "metric": "validation_NDCG@10",
        "history": history,
        "best_score": recomputed_best_score,
        "selected_epoch": recomputed_selected_epoch,
        "bad_checks": recomputed_bad_checks,
        "status": status,
        "stop_reason": stop_reason,
    }

    if recomputed_selected_epoch is None:
        best_model = None
    else:
        if recomputed_selected_epoch == epoch:
            best_sd = {
                "user_embeddings": user_emb.clone(),
                "item_embeddings": item_emb.clone(),
                "adj_user_item": adj_buffer.clone(),
            }
        else:
            best_generator = torch.Generator(device="cpu").manual_seed(202)
            best_sd = {
                "user_embeddings": torch.randn(len(u_list), 64, dtype=torch.float32, generator=best_generator),
                "item_embeddings": torch.randn(len(p_list), 64, dtype=torch.float32, generator=best_generator),
                "adj_user_item": adj_buffer.clone(),
            }
        best_model = {
            "epoch": recomputed_selected_epoch,
            "row_order": {"User": list(u_list), "Product": list(p_list)},
            "state_dict": best_sd,
        }

    # Sampler & Progress
    if with_empty_pool:
        source_order = [
            ("u1", "p1"), ("u1", "p2"), ("u2", "p3"),
            ("u3", "p1"), ("u3", "p2"),
        ]
    else:
        source_order = [("u1", "p1"), ("u1", "p2"), ("u2", "p2"), ("u2", "p3")]
    sampler = {
        "algorithm": "uniform_eligible_negative_epoch_sampler_v1",
        "identity_hash": identity.identity_hash,
        "source_order": list(source_order) if epoch > 0 else None,
        "cursor": len(source_order) if epoch > 0 else 0,
        "next_epoch": epoch + 1,
        "batch_cursor": 0,
        "pending_batches": 0,
        "prefetch_pending": 0,
        "inflight_workers": 0,
        "worker_ids": list(range(runtime_workers)),
    }

    progress = {
        "epoch": epoch,
        "global_step": global_step,
        "epoch_optimizer_steps": [2] * epoch,
        "empty_pool_skips": [2 if with_empty_pool else 0] * epoch,
    }

    roles = []
    for s in identity.resolved_settings["determinism"]["rng_streams"]:
        r_rec = dict(s)
        if s["applicable"]:
            r_rec["generator_ids"] = list(expected_runtime["rng_role_bindings"][s["name"]])
        else:
            r_rec["generator_ids"] = []
        roles.append(r_rec)

    payload = {
        "schema_version": "amazon4u-checkpoint-payload-v1",
        "capability": "epoch_boundary_validation_only",
        "identity": identity.to_dict(),
        "provenance": provenance,
        "runtime_profile": copy.deepcopy(expected_runtime["runtime_profile"]),
        "current_model": current_model,
        "best_model": best_model,
        "optimizer": optimizer,
        "scheduler": {"applicable": False, "state": None, "reason": "not_used_constant_learning_rate"},
        "amp": {"applicable": False, "state": None, "reason": "not_used_cpu_float32"},
        "rng": {"roles": roles, "generators": generators},
        "sampler": sampler,
        "progress": progress,
        "selection": selection,
    }

    return payload, identity.to_dict(), provenance, expected_model, expected_runtime


class TestCheckpointPayloadMatrix(unittest.TestCase):
    """Part-row smoke tests; method and prefix counts do not establish closure."""

    def setUp(self) -> None:
        self.policy = _make_synthetic_resource_policy()

    def _validate(self, payload, exp_id, exp_prov, exp_model, exp_rt, policy=None):
        if policy is None:
            policy = self.policy
        return validate_checkpoint_payload(
            payload,
            expected_identity=exp_id,
            expected_provenance=exp_prov,
            expected_model=exp_model,
            expected_runtime=exp_rt,
            resource_policy=policy,
        )

    def test_row_P01_dependency_baseline_and_imports(self) -> None:
        """P01: Verify dual import canonicalization and accepted dependency baseline symbols."""
        case_id = "P01/import/alias/canonical"
        with self.subTest(case_id=case_id):
            self.assertIs(checkpoint_payload.validate_checkpoint_payload, scripts.checkpoint_payload.validate_checkpoint_payload)
            self.assertIs(checkpoint_payload.preflight_resource_accounting, scripts.checkpoint_payload.preflight_resource_accounting)
            self.assertIs(checkpoint_payload.validate_resource_policy, scripts.checkpoint_payload.validate_resource_policy)
            EXECUTED_CASES.add(case_id)

    def test_row_F01_fixture_construction_and_guard_reachability(self) -> None:
        """F01: Verify coherent boundary fixtures pass, and old E4 tie defect fails selection guard."""
        for ep, mode, cid in [
            (0, "increasing", "F01/fixture/E0_initial/valid"),
            (1, "increasing", "F01/fixture/E1_precheck/valid"),
            (2, "increasing", "F01/fixture/E2_first/valid"),
            (4, "ties", "F01/fixture/E4_tie/valid"),
            (4, "increasing", "F01/fixture/E4_improve/valid"),
        ]:
            with self.subTest(case_id=cid):
                payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=ep, score_mode=mode)
                res = self._validate(payload, exp_id, exp_prov, exp_model, exp_rt)
                self.assertIsInstance(res, dict)
                EXECUTED_CASES.add(cid)

        case_id = "F01/fixture/E4_tie/mismatched_epoch_fails"
        with self.subTest(case_id=case_id):
            payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=4, score_mode="ties")
            payload["best_model"]["epoch"] = 4  # defect: selection has 2, best_model has 4
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(payload, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIn("does not match recomputed selected epoch", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

    def test_row_E01_public_api_envelope_and_applicability(self) -> None:
        """E01: Exact 14-root envelope and applicability records."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "E01/root/selection/missing"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            del p["selection"]
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "E01/root/optimizer/wrong_type"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"] = [1, 2, 3]
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "E01/amp/applicable/true_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["amp"]["applicable"] = True
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "E01/scheduler/applicable/true_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["scheduler"]["applicable"] = True
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_E02_identity_and_provenance_binding(self) -> None:
        """E02: Actual and expected identity and provenance binding."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "E02/identity/seed/mismatch"
        with self.subTest(case_id=case_id):
            alt_id, _ = _build_coherent_identity_and_provenance(method="LightGCN", seed=999)
            with self.assertRaises(CheckpointIdentityMismatchError):
                self._validate(payload, alt_id.to_dict(), exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "E02/provenance/commit/mismatch"
        with self.subTest(case_id=case_id):
            alt_prov = copy.deepcopy(exp_prov)
            alt_prov["git"]["commit"] = "b" * 40
            with self.assertRaises(CheckpointProvenanceError):
                self._validate(payload, exp_id, alt_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_E03_trusted_descriptor_list_schemas(self) -> None:
        """E03: Enforce strict list schema for original declared list fields in expected_runtime."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "E03/runtime/optimizer_groups/tuple_rejected"
        with self.subTest(case_id=case_id):
            bad_rt = copy.deepcopy(exp_rt)
            bad_rt["optimizer_groups"] = tuple(bad_rt["optimizer_groups"])
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(payload, exp_id, exp_prov, exp_model, bad_rt)
            self.assertIn("optimizer_groups must be a non-empty list", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

        case_id = "E03/runtime/rng_role_bindings/tuple_rejected"
        with self.subTest(case_id=case_id):
            bad_rt = copy.deepcopy(exp_rt)
            bad_rt["rng_role_bindings"]["PyTorch_CPU"] = tuple(bad_rt["rng_role_bindings"]["PyTorch_CPU"])
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(payload, exp_id, exp_prov, exp_model, bad_rt)
            self.assertIn("must be a list of IDs", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

    def test_row_E04_runtime_step_policy_coherence(self) -> None:
        """E04: Runtime profile and default step policy coherence."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "E04/profile/batch_policy/unsupported"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["runtime_profile"]["batch_policy"] = "unsupported_batch_policy"
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "E04/profile/profile_id/mismatch"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["runtime_profile"]["profile_id"] = "different_profile"
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_R01_resource_policy_schema_and_configurability(self) -> None:
        """R01: Resource policy schema and bounds enforcement."""
        case_id = "R01/policy/max_total_bytes/negative"
        with self.subTest(case_id=case_id):
            bad_pol = dict(self.policy)
            bad_pol["max_input_bytes"] = -1
            with self.assertRaises(CheckpointValidationError):
                checkpoint_payload.validate_resource_policy(bad_pol)
            EXECUTED_CASES.add(case_id)

    def test_row_R02_resource_accounting_oracles(self) -> None:
        """R02: Exact resource accounting and bounds violation."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "R02/preflight/total_bytes/exceeded"
        with self.subTest(case_id=case_id):
            tight_pol = _make_synthetic_resource_policy(max_input_bytes=100, kind="runtime", profile_id="runtime_tight_in")
            with self.assertRaises(CheckpointValidationError):
                self._validate(payload, exp_id, exp_prov, exp_model, exp_rt, policy=tight_pol)
            EXECUTED_CASES.add(case_id)

    def test_row_R03_preflight_validation_copy_revalidation_order(self) -> None:
        """R03: Preflight resource checks precede model/tensor parsing and copy."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "R03/order/preflight_before_copy"
        with self.subTest(case_id=case_id):
            tight_pol = _make_synthetic_resource_policy(max_input_bytes=50, kind="runtime", profile_id="runtime_tight_in_50")
            with self.assertRaises(CheckpointValidationError):
                self._validate(payload, exp_id, exp_prov, exp_model, exp_rt, policy=tight_pol)
            EXECUTED_CASES.add(case_id)

    def test_row_M01_lightgcn_recipe_and_tensor_properties(self) -> None:
        """M01: Complete LightGCN recipe parameter categories and tensor properties."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "M01/current_model/user_embeddings/nan"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["current_model"]["state_dict"]["user_embeddings"][0, 0] = float("nan")
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "M01/current_model/item_embeddings/inf"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["current_model"]["state_dict"]["item_embeddings"][0, 0] = float("inf")
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "M01/best_model/user_embeddings/shape"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["best_model"]["state_dict"]["user_embeddings"] = torch.zeros(1, 64)
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "M01/current_model/user_embeddings/requires_grad"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            t = p["current_model"]["state_dict"]["user_embeddings"].clone()
            t.requires_grad = True
            p["current_model"]["state_dict"]["user_embeddings"] = t
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_M02_hetero_recipe_and_36_parameters(self) -> None:
        """M02: Complete HeteroGNN recipe and 36 parameters."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _make_valid_hetero_payload()
        case_id = "M02/current_model/base_embeddings/nan"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["current_model"]["state_dict"]["user_embeddings"][0, 0] = float("nan")
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "M02/current_model/param/missing"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            del p["current_model"]["state_dict"]["self_weights_l0.User"]
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_M03_sparse_coo_canonicality_and_trusted_contents(self) -> None:
        """M03: Sparse COO canonicality and degree normalization validation."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "M03/buffer/adj_user_item/uncoalesced"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            raw = p["current_model"]["state_dict"]["adj_user_item"]
            uncoal = torch.sparse_coo_tensor(raw.indices(), raw.values(), raw.shape, is_coalesced=False)
            p["current_model"]["state_dict"]["adj_user_item"] = uncoal
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "M03/buffer/adj_user_item/wrong_weights"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            raw = p["current_model"]["state_dict"]["adj_user_item"]
            bad_w = torch.sparse_coo_tensor(raw.indices(), raw.values() * 2.0, raw.shape).coalesce()
            p["current_model"]["state_dict"]["adj_user_item"] = bad_w
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_A01_adam_optional_param_names_binding(self) -> None:
        """A01: Adam optional param_names concrete sequence type and value binding."""
        names = ["user_embeddings", "item_embeddings"]
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2, adam_options_param_names=list(names))

        case_id = "A01/adam/param_names/list_control_valid"
        with self.subTest(case_id=case_id):
            res = self._validate(payload, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIsInstance(res, dict)
            EXECUTED_CASES.add(case_id)

        payload_t, exp_id_t, exp_prov_t, exp_model_t, exp_rt_t = _build_coherent_lightgcn_payload(epoch=2, adam_options_param_names=tuple(names))
        case_id = "A01/adam/param_names/tuple_control_valid"
        with self.subTest(case_id=case_id):
            res = self._validate(payload_t, exp_id_t, exp_prov_t, exp_model_t, exp_rt_t)
            self.assertIsInstance(res, dict)
            EXECUTED_CASES.add(case_id)

        case_id = "A01/adam/param_names/dict_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["param_groups"][0]["param_names"] = dict.fromkeys(names, None)
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "A01/adam/param_names/scalar_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["param_groups"][0]["param_names"] = 7
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "A01/adam/param_names/cross_type_mismatch"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["param_groups"][0]["param_names"] = tuple(names)
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_A02_adam_packed_ids_groups_and_bindings(self) -> None:
        """A02: Adam packed parameter IDs, parameter order, and hyperparameters."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "A02/adam/params/id_mismatch"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["param_groups"][0]["params"] = [1, 2]
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "A02/adam/amsgrad/true_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["param_groups"][0]["amsgrad"] = True
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_A03_adam_lazy_states_and_aggregate_bound(self) -> None:
        """A03: Adam lazy parameter states and aggregate update lower bound."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "A03/adam/updates/all_zero_when_g_positive"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["state"][0]["step"] = torch.tensor(0.0)
            p["optimizer"]["state_dict"]["state"][1]["step"] = torch.tensor(0.0)
            p["optimizer"]["parameter_bindings"][0]["updates"] = 0
            p["optimizer"]["parameter_bindings"][1]["updates"] = 0
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIn("global_step", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

        case_id = "A03/adam/updates/sum_less_than_g"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["parameter_bindings"][0]["updates"] = 2
            p["optimizer"]["parameter_bindings"][1]["updates"] = 1
            p["optimizer"]["state_dict"]["state"][0]["step"] = torch.tensor(2.0)
            p["optimizer"]["state_dict"]["state"][1]["step"] = torch.tensor(1.0)
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIn("global_step", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

    def test_row_A04_adam_moment_and_native_scalar_step_precision(self) -> None:
        """A04: Adam moments and scalar step dtype/precision."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "A04/adam/moments/nan"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["state"][0]["exp_avg"][0, 0] = float("nan")
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "A04/adam/step/python_int_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["state"][0]["step"] = 4
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_A05_adam_foreach_fused_type_before_equality(self) -> None:
        """A05: Type checks for foreach and fused options must precede equality comparison."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        payload["optimizer"]["state_dict"]["param_groups"][0]["foreach"] = False
        payload["optimizer"]["state_dict"]["param_groups"][0]["fused"] = False
        exp_rt["optimizer_groups"][0]["options"]["foreach"] = False
        exp_rt["optimizer_groups"][0]["options"]["fused"] = False

        case_id = "A05/adam/flags_false/valid"
        with self.subTest(case_id=case_id):
            res = self._validate(payload, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIsInstance(res, dict)
            EXECUTED_CASES.add(case_id)

        case_id = "A05/adam/foreach/tensor_rejected_before_equality"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["param_groups"][0]["foreach"] = torch.tensor([0, 1])
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIn("foreach", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

        case_id = "A05/adam/fused/tensor_rejected_before_equality"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["param_groups"][0]["fused"] = torch.tensor([0, 1])
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIn("fused", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

    def test_row_G01_canonical_cuda_role_cpu_inactivity(self) -> None:
        """G01: Canonical PyTorch_CUDA_if_applicable role must be inactive in CPU profile."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "G01/rng/role_name/noncanonical_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["rng"]["roles"][3]["name"] = "PyTorch_CUDA"
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

        case_id = "G01/rng/cuda_role/active_rejected_on_cpu"
        with self.subTest(case_id=case_id):
            p_cuda, id_cuda, prov_cuda, mod_cuda, rt_cuda = _build_coherent_lightgcn_payload(epoch=2, active_cuda_role=True)
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p_cuda, id_cuda, prov_cuda, mod_cuda, rt_cuda)
            self.assertIn("PyTorch_CUDA_if_applicable role cannot be active in this CPU profile", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

    def test_row_G02_rng_registry_and_role_sharing(self) -> None:
        """G02: RNG registry, exact 10 roles, and shared generator IDs."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "G02/rng/roles/count_mismatch"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["rng"]["roles"] = p["rng"]["roles"][:-1]
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_G03_python_mt19937_v3_codec(self) -> None:
        """G03: Python MT19937 v3 codec validation."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "G03/codec/python/version_mismatch"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            st = list(p["rng"]["generators"][0]["state"])
            st[0] = 2
            p["rng"]["generators"][0]["state"] = tuple(st)
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_G04_numpy_randomstate_mt19937_codec(self) -> None:
        """G04: NumPy RandomState MT19937 codec validation."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "G04/codec/numpy_mt/name_mismatch"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            st = list(p["rng"]["generators"][1]["state"])
            st[0] = "WrongName"
            p["rng"]["generators"][1]["state"] = tuple(st)
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_G05_numpy_pcg64_codec(self) -> None:
        """G05: NumPy PCG64 codec validation."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "G05/codec/numpy_pcg/bit_gen_mismatch"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["rng"]["generators"][2]["state"]["bit_generator"] = "WrongBitGen"
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_G06_torch_cpu_generator_codec_and_native_binding(self) -> None:
        """G06: Native Torch CPU generator codec and state validation."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "G06/codec/torch/dtype_not_uint8"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["rng"]["generators"][3]["state"] = p["rng"]["generators"][3]["state"].to(torch.int32)
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_G07_seed_materials_actual_expected_and_worker_lists(self) -> None:
        """G07: Seed materials types, intrinsic bounds, and worker word lists."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "G07/seed/torch/exceeds_uint64"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["rng"]["generators"][3]["seed_material"] = 0xFFFFFFFFFFFFFFFF + 1
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_G08_complete_live_rng_preservation(self) -> None:
        """G08: Pure validator execution leaves global live RNG state bit-identical."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "G08/live_rng/success_preserved"
        with self.subTest(case_id=case_id):
            py_before = random.getstate()
            np_before = copy.deepcopy(np.random.get_state())
            torch_before = torch.get_rng_state()

            res = self._validate(payload, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIsInstance(res, dict)

            self.assertEqual(random.getstate(), py_before)
            self.assertEqual(np.random.get_state()[0], np_before[0])
            self.assertTrue(np.array_equal(np.random.get_state()[1], np_before[1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
            EXECUTED_CASES.add(case_id)

    def test_row_G09_parent_seed_type_validation_before_equality(self) -> None:
        """G09: Codec-specific parent seed type validation must precede bound role equality."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "G09/seed/torch_parent/tensor_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["rng"]["generators"][3]["seed_material"] = torch.tensor([42, 43])
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIn("seed_material", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

        case_id = "G09/seed/python_parent/tensor_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["rng"]["generators"][0]["seed_material"] = torch.tensor([42, 43])
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIn("seed_material", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

    def test_row_S01_positive_empty_negative_domain_oracles(self) -> None:
        """S01: Sampler positive and empty negative pool domain oracles."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2, with_empty_pool=True)
        case_id = "S01/sampler/source_order/clearing_mismatch"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["sampler"]["source_order"] = []
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_S02_sampler_orders_quiescence_and_progress_counters(self) -> None:
        """S02: Sampler quiescent epoch boundaries and progress counters."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "S02/sampler/quiescent/batch_cursor_nonzero"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["sampler"]["batch_cursor"] = 1
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIn("Quiescent epoch boundary violated", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

        case_id = "S02/progress/steps/sum_mismatch"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["progress"]["epoch_optimizer_steps"] = [1, 1]
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_T01_selection_history_ties_patience(self) -> None:
        """T01: Validation selection history, ties, and patience boundary."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "T01/selection/score/nan_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["selection"]["history"][0]["score"] = float("nan")
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            EXECUTED_CASES.add(case_id)

    def test_row_T02_selection_max_and_combined_stop(self) -> None:
        """T02: Selection stopping conditions for max_epochs alone and simultaneous patience+max."""
        case_id = "T02/selection/E100_F3_max_only/status_stopped"
        with self.subTest(case_id=case_id):
            payload_f3, exp_id_f3, exp_prov_f3, exp_mod_f3, exp_rt_f3 = _build_coherent_lightgcn_payload(
                epoch=100, validation_frequency=3, max_epochs=100
            )
            res = self._validate(payload_f3, exp_id_f3, exp_prov_f3, exp_mod_f3, exp_rt_f3)
            self.assertEqual(res["selection"]["status"], "stopped")
            self.assertEqual(res["selection"]["stop_reason"], "max_epochs")
            self.assertEqual(res["selection"]["selected_epoch"], 99)
            EXECUTED_CASES.add(case_id)

        case_id = "T02/selection/E100_F2_combined/stop_reason_combined"
        with self.subTest(case_id=case_id):
            scores_50 = [round(0.01 * (i + 1), 4) for i in range(40)] + [round(0.01 * 40, 4)] * 10
            payload_f2, exp_id_f2, exp_prov_f2, exp_mod_f2, exp_rt_f2 = _build_coherent_lightgcn_payload(
                epoch=100, validation_frequency=2, max_epochs=100, scores=scores_50
            )
            res = self._validate(payload_f2, exp_id_f2, exp_prov_f2, exp_mod_f2, exp_rt_f2)
            self.assertEqual(res["selection"]["status"], "stopped")
            self.assertEqual(res["selection"]["stop_reason"], "patience_and_max_epochs")
            self.assertEqual(res["selection"]["selected_epoch"], 80)
            self.assertEqual(res["selection"]["bad_checks"], 10)
            EXECUTED_CASES.add(case_id)

    def test_row_T03_current_best_identity(self) -> None:
        """T03: Current and best model binding and bitwise identical tensors for shared epoch."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "T03/best_model/equal_epoch_content_difference"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["best_model"]["state_dict"]["user_embeddings"] = p["best_model"]["state_dict"]["user_embeddings"] + 0.01
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertIn("contents differ", str(ctx.exception))
            EXECUTED_CASES.add(case_id)

    def test_row_O01_owned_output_isolation(self) -> None:
        """O01: Owned output snapshot and complete alias isolation."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        res = self._validate(payload, exp_id, exp_prov, exp_model, exp_rt)

        case_id = "O01/ownership/output_mutation_isolation"
        with self.subTest(case_id=case_id):
            orig_val = payload["current_model"]["state_dict"]["user_embeddings"][0, 0].item()
            res["current_model"]["state_dict"]["user_embeddings"][0, 0] += 10.0
            self.assertEqual(payload["current_model"]["state_dict"]["user_embeddings"][0, 0].item(), orig_val)
            EXECUTED_CASES.add(case_id)

    def test_row_O02_no_mutation_on_failure(self) -> None:
        """O02: Complete no-mutation on early and late validation failures."""
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "O02/no_mutation/late_selection_failure"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["selection"]["best_score"] = 0.999
            user_orig = p["current_model"]["state_dict"]["user_embeddings"].clone()
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertTrue(torch.equal(p["current_model"]["state_dict"]["user_embeddings"], user_orig))
            EXECUTED_CASES.add(case_id)

    def test_row_D01_bounded_diagnostics(self) -> None:
        """D01: Bounded domain diagnostics: H=10**5000 values must CVE with length <= 4096 and no ValueError/TypeError."""
        H = 10 ** 5000
        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)

        case_id = "D01/diagnostic/lightgcn_keys_H"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            t_user = p["current_model"]["state_dict"]["user_embeddings"]
            t_item = p["current_model"]["state_dict"]["item_embeddings"]
            t_adj = p["current_model"]["state_dict"]["adj_user_item"]
            p["current_model"]["state_dict"] = {H: t_user, H + 1: t_item, H + 2: t_adj}
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            msg = str(ctx.exception)
            self.assertLessEqual(len(msg), 4096)
            self.assertIn("LightGCN", msg)
            EXECUTED_CASES.add(case_id)

        case_id = "D01/diagnostic/params_H"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["param_groups"][0]["params"] = [H, 1]
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            msg = str(ctx.exception)
            self.assertLessEqual(len(msg), 4096)
            EXECUTED_CASES.add(case_id)

        case_id = "D01/diagnostic/betas_H"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["optimizer"]["state_dict"]["param_groups"][0]["betas"] = (H, 0.999)
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            msg = str(ctx.exception)
            self.assertLessEqual(len(msg), 4096)
            EXECUTED_CASES.add(case_id)

        case_id = "D01/diagnostic/role_name_H"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["rng"]["roles"][0]["name"] = H
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            msg = str(ctx.exception)
            self.assertLessEqual(len(msg), 4096)
            EXECUTED_CASES.add(case_id)

        case_id = "D01/diagnostic/sampler_alg_H"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["sampler"]["algorithm"] = H
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            msg = str(ctx.exception)
            self.assertLessEqual(len(msg), 4096)
            EXECUTED_CASES.add(case_id)

        case_id = "D01/diagnostic/codec_H"
        with self.subTest(case_id=case_id):
            bad_rt = copy.deepcopy(exp_rt)
            bad_rt["rng_generator_bindings"]["gen_py"]["codec"] = H
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(payload, exp_id, exp_prov, exp_model, bad_rt)
            msg = str(ctx.exception)
            self.assertLessEqual(len(msg), 4096)
            EXECUTED_CASES.add(case_id)

        case_id = "D01/diagnostic/profile_id_H"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            p["runtime_profile"]["profile_id"] = H
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            msg = str(ctx.exception)
            self.assertLessEqual(len(msg), 4096)
            EXECUTED_CASES.add(case_id)

        case_id = "D01/diagnostic/tuple_keys_TypeError"
        with self.subTest(case_id=case_id):
            bad_rt = copy.deepcopy(exp_rt)
            bad_rt["runtime_profile"] = {(0,): 1, ("x",): 2}
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(payload, exp_id, exp_prov, exp_model, bad_rt)
            msg = str(ctx.exception)
            self.assertLessEqual(len(msg), 4096)
            EXECUTED_CASES.add(case_id)

        case_id = "D01/diagnostic/role_keys_H"
        with self.subTest(case_id=case_id):
            bad_rt = copy.deepcopy(exp_rt)
            bad_rt["rng_role_bindings"] = {H + i: ["gen_py"] for i in range(10)}
            with self.assertRaises(CheckpointValidationError) as ctx:
                self._validate(payload, exp_id, exp_prov, exp_model, bad_rt)
            msg = str(ctx.exception)
            self.assertLessEqual(len(msg), 4096)
            EXECUTED_CASES.add(case_id)

    def test_row_D02_reject_custom_objects(self) -> None:
        """D02: Reject custom objects/subclasses without executing caller hooks."""
        class HostileDict(dict):
            hook_called = False
            def __hash__(self):
                HostileDict.hook_called = True
                return super().__hash__()
            def __eq__(self, other):
                HostileDict.hook_called = True
                return super().__eq__(other)

        payload, exp_id, exp_prov, exp_model, exp_rt = _build_coherent_lightgcn_payload(epoch=2)
        case_id = "D02/safety/hostile_dict_rejected"
        with self.subTest(case_id=case_id):
            p = copy.deepcopy(payload)
            HostileDict.hook_called = False
            p["selection"] = HostileDict(p["selection"])
            with self.assertRaises(CheckpointValidationError):
                self._validate(p, exp_id, exp_prov, exp_model, exp_rt)
            self.assertFalse(HostileDict.hook_called)
            EXECUTED_CASES.add(case_id)

    def test_row_V01_case_coverage_ledger(self) -> None:
        """V01 smoke only: observed prefixes are not an exhaustive case ledger."""
        case_id = "V01/coverage/all_37_rows_executed"
        with self.subTest(case_id=case_id):
            expected_row_prefixes = [
                "P01", "F01", "E01", "E02", "E03", "E04", "R01", "R02", "R03",
                "M01", "M02", "M03", "A01", "A02", "A03", "A04", "A05",
                "G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09",
                "S01", "S02", "T01", "T02", "T03", "O01", "O02", "D01", "D02",
            ]
            for prefix in expected_row_prefixes:
                matched = [c for c in EXECUTED_CASES if c.startswith(prefix + "/")]
                self.assertGreater(len(matched), 0, f"Missing execution cases for row {prefix}")
            EXECUTED_CASES.add(case_id)

    def test_row_V02_disclosure_docs(self) -> None:
        """V02: Verify documentation exact original specification prefix hash preserved."""
        case_id = "V02/docs/exact_spec_prefix_preserved"
        with self.subTest(case_id=case_id):
            import hashlib
            doc_path = ROOT / "docs" / "checkpoint-payload.md"
            with open(doc_path, "rb") as df:
                content = df.read()
            prefix = content[:59605]
            digest = hashlib.sha256(prefix).hexdigest()
            self.assertEqual(digest, "a648d97f73e1213eb37f1a87fe56e76f1171f59baedcce6cd62b2c66f7dd64c4")
            EXECUTED_CASES.add(case_id)


class TestCheckpointPayloadExpandedMatrices(unittest.TestCase):
    """Additional deterministic cases, not a declaration of whole-D2 closure.

    Each test owns its ledger. No ledger depends on another class running first.
    Declared and successful IDs are emitted separately, including on failure.
    """

    def setUp(self):
        self.declared = set()
        self.successful = set()

    def tearDown(self):
        import json
        print("D2_EXPANDED_LEDGER " + json.dumps({
            "test": self._testMethodName,
            "declared": sorted(self.declared),
            "successful": sorted(self.successful),
            "unresolved": sorted(self.declared - self.successful),
        }, separators=(",", ":")))

    def _case(self, case_id, action):
        with self.subTest(case_id=case_id):
            self.assertNotIn(case_id, self.declared)
            self.declared.add(case_id)
            action()
            self.successful.add(case_id)

    def _control(self, family="light", **kwargs):
        if family == "light":
            raw = _build_coherent_lightgcn_payload(**kwargs)
        else:
            raw = _make_valid_hetero_payload(**kwargs)
        # Copy each argument separately: fixtures must not lend mutable
        # expectations to the actual payload, even when builders share them.
        return tuple(copy.deepcopy(value) for value in raw) + (_make_synthetic_resource_policy(),)

    def _call(self, bundle):
        p, identity, provenance, model, runtime, policy = bundle
        return validate_checkpoint_payload(
            p, expected_identity=identity, expected_provenance=provenance,
            expected_model=model, expected_runtime=runtime, resource_policy=policy,
        )

    def _positive(self, bundle):
        self.assertIs(type(self._call(bundle)), dict)

    def _negative(self, bundle, mutation, pattern):
        changed = list(copy.deepcopy(bundle))
        mutation(changed)
        with self.assertRaisesRegex(CheckpointValidationError, pattern):
            self._call(changed)

    def _global_snapshot(self):
        ns = np.random.get_state()
        return (
            random.getstate(),
            (ns[0], ns[1].copy(), ns[2], ns[3], ns[4]),
            torch.get_rng_state().clone(),
            torch.get_default_dtype(),
        )

    def _assert_globals(self, before):
        self.assertEqual(random.getstate(), before[0])
        ns = np.random.get_state()
        self.assertEqual(ns[0], before[1][0])
        self.assertTrue(np.array_equal(ns[1], before[1][1]))
        self.assertEqual(ns[2], before[1][2])
        self.assertEqual(ns[3], before[1][3])
        self.assertEqual(float(ns[4]).hex(), float(before[1][4]).hex())
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(torch.get_default_dtype(), before[3])

    def _same(self, left, right):
        self.assertIs(type(left), type(right))
        if type(left) is torch.Tensor:
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.layout, right.layout)
            self.assertEqual(left.device, right.device)
            self.assertEqual(tuple(left.shape), tuple(right.shape))
            self.assertEqual(left.requires_grad, right.requires_grad)
            if left.layout == torch.sparse_coo:
                self.assertEqual(left.is_coalesced(), right.is_coalesced())
                self._same(left._indices(), right._indices())
                self._same(left._values(), right._values())
            else:
                self.assertTrue(torch.equal(
                    left.detach().contiguous().reshape(-1).view(torch.uint8),
                    right.detach().contiguous().reshape(-1).view(torch.uint8),
                ))
        elif type(left) is dict:
            self.assertEqual(list(left), list(right))
            for key in left:
                self._same(left[key], right[key])
        elif type(left) is list or type(left) is tuple:
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self._same(a, b)
        elif type(left) is float:
            self.assertEqual(left.hex(), right.hex())
        else:
            self.assertEqual(left, right)

    def _tensor_leaves(self, value):
        if type(value) is torch.Tensor:
            if value.layout == torch.sparse_coo:
                yield value._indices()
                yield value._values()
            else:
                yield value
        elif type(value) is dict:
            for child in value.values():
                yield from self._tensor_leaves(child)
        elif type(value) is list or type(value) is tuple:
            for child in value:
                yield from self._tensor_leaves(child)

    def _wreck(self, value, seen=None):
        if seen is None:
            seen = set()
        if type(value) is torch.Tensor:
            if id(value) in seen:
                return
            seen.add(id(value))
            if value.layout == torch.sparse_coo:
                value._indices().add_(1)
                value._values().add_(1)
            else:
                value.add_(1)
        elif type(value) is dict or type(value) is list:
            if id(value) in seen:
                return
            seen.add(id(value))
            children = list(value.values()) if type(value) is dict else list(value)
            for child in children:
                self._wreck(child, seen)
            value.clear()
        elif type(value) is tuple:
            for child in value:
                self._wreck(child, seen)

    def test_dense_parameter_slots(self):
        mutations = (
            "missing", "extra", "shape", "dtype", "layout", "meta",
            "nan", "positive_inf", "negative_inf", "requires_grad",
        )
        for family in ("light", "hetero", "hetero_empty"):
            bundle = self._control(
                "light" if family == "light" else "hetero",
                **({"empty_metadata": True} if family == "hetero_empty" else {}),
            )
            self._case("F01/" + family + "/complete/control", lambda b=bundle: self._positive(b))
            names = bundle[4]["optimizer_groups"][0]["parameter_names"]
            self.assertEqual(len(names), 2 if family == "light" else 36)
            for role in ("current_model", "best_model"):
                for name in names:
                    original = bundle[0][role]["state_dict"][name]
                    for mutation in mutations:
                        if original.numel() == 0 and mutation in ("nan", "positive_inf", "negative_inf"):
                            # No element exists; these cases are represented by
                            # the nonempty metadata control, not by a skip.
                            continue
                        row = "M01" if family == "light" else "M02"
                        cid = "/".join((row, family + "_" + role, name, mutation))
                        def check(b=bundle, r=role, n=name, m=mutation):
                            def change(args):
                                sd = args[0][r]["state_dict"]
                                old = sd[n]
                                if m == "missing":
                                    del sd[n]
                                elif m == "extra":
                                    sd["unexpected_parameter"] = old.clone()
                                elif m == "shape":
                                    sd[n] = torch.zeros((old.numel() + 1,), dtype=torch.float32)
                                elif m == "dtype":
                                    sd[n] = torch.zeros(tuple(old.shape), dtype=torch.float64)
                                elif m == "layout":
                                    sd[n] = torch.sparse_coo_tensor(
                                        torch.empty((old.dim(), 0), dtype=torch.int64),
                                        torch.empty((0,), dtype=torch.float32),
                                        tuple(old.shape), is_coalesced=True,
                                    )
                                elif m == "meta":
                                    sd[n] = torch.empty(tuple(old.shape), device="meta", dtype=torch.float32)
                                elif m == "requires_grad":
                                    sd[n] = old.clone().requires_grad_(True)
                                else:
                                    number = {"nan": float("nan"), "positive_inf": float("inf"), "negative_inf": -float("inf")}[m]
                                    sd[n] = torch.full(tuple(old.shape), number, dtype=torch.float32)
                            patterns = {
                                "missing": r"state_dict.*keys", "extra": r"state_dict.*keys",
                                "shape": r"shape", "dtype": r"float32|dtype",
                                "layout": r"layout|strided|sparse", "meta": r"CPU|cpu|device",
                                "nan": r"finite|NaN", "positive_inf": r"finite|Inf",
                                "negative_inf": r"finite|Inf", "requires_grad": r"detach|grad",
                            }
                            self._negative(b, change, patterns[m])
                        self._case(cid, check)

    def test_optional_names_two_group_controls(self):
        for family in ("light", "hetero"):
            for partition in (1, 2):
                for sequence_name, sequence_type in (("list", list), ("tuple", tuple)):
                    bundle = list(self._control(family))
                    names = list(bundle[4]["optimizer_groups"][0]["parameter_names"])
                    cuts = [names] if partition == 1 else [names[:1], names[1:]]
                    original_options = copy.deepcopy(bundle[4]["optimizer_groups"][0]["options"])
                    expected_groups = []
                    actual_groups = []
                    offset = 0
                    for group_names in cuts:
                        options = copy.deepcopy(original_options)
                        options["param_names"] = sequence_type(group_names)
                        expected_groups.append({"parameter_names": list(group_names), "options": options})
                        actual = copy.deepcopy(options)
                        actual["params"] = list(range(offset, offset + len(group_names)))
                        actual_groups.append(actual)
                        offset += len(group_names)
                    bundle[4]["optimizer_groups"] = expected_groups
                    bundle[0]["optimizer"]["state_dict"]["param_groups"] = actual_groups
                    label = family + "_g" + str(partition) + "_" + sequence_name
                    self._case("A01/" + label + "/param_names/control", lambda b=bundle: self._positive(b))
                    for group_index, group_names in enumerate(cuts):
                        bad_values = (
                            ("mapping", dict.fromkeys(group_names, None)), ("none", None),
                            ("bool", True), ("int", 7), ("float", 1.0),
                            ("str", "names"), ("bytes", b"names"),
                            ("scalar_tensor", torch.tensor(0)),
                            ("multi_tensor", torch.tensor([0, 1])),
                            ("cross_container", tuple(group_names) if sequence_type is list else list(group_names)),
                            ("duplicate", sequence_type([group_names[0]] * (len(group_names) + 1))),
                            ("wrong_element", sequence_type([None] + group_names[1:])),
                        )
                        for bad_label, bad_value in bad_values:
                            for side in ("actual", "expected", "both"):
                                # Cross-container mismatch is an actual/trusted
                                # binding error; either sequence type alone is valid.
                                if bad_label == "cross_container" and side != "actual":
                                    continue
                                cid = "/".join(("A01", label + "_" + side, "group" + str(group_index), bad_label))
                                def check(b=bundle, gi=group_index, v=bad_value, s=side):
                                    def change(args):
                                        if s in ("actual", "both"):
                                            args[0]["optimizer"]["state_dict"]["param_groups"][gi]["param_names"] = copy.deepcopy(v)
                                        if s in ("expected", "both"):
                                            args[4]["optimizer_groups"][gi]["options"]["param_names"] = copy.deepcopy(v)
                                    self._negative(b, change, r"param_names")
                                self._case(cid, check)

    def test_declared_lists_and_flag_types(self):
        bundle = self._control()
        self._case("E03/light/complete/control", lambda: self._positive(bundle))
        for role in CONFIGURED_RNG_COMPONENTS:
            for label, value in (("tuple", tuple(bundle[4]["rng_role_bindings"][role])), ("str", "x"), ("int", 0)):
                def check(r=role, v=value):
                    self._negative(bundle, lambda a: a[4]["rng_role_bindings"].__setitem__(r, v), r"rng_role_bindings.*list of IDs")
                self._case("E03/expected/" + role + "/" + label, check)
        for label, value in (("tuple", tuple(bundle[4]["optimizer_groups"])), ("empty", []), ("dict", {})):
            self._case("E03/expected/optimizer_groups/" + label,
                       lambda v=value: self._negative(bundle, lambda a: a[4].__setitem__("optimizer_groups", v), r"optimizer_groups.*non-empty list"))
        wrong = (("int", 0), ("float", 0.0), ("str", "False"), ("bytes", b"False"),
                 ("dict", {}), ("list", []), ("tuple", ()),
                 ("scalar_tensor", torch.tensor(0)), ("multi_tensor", torch.tensor([0, 1])))
        for family in ("light", "hetero"):
            for flag, supported in (("foreach", (None, False, True)), ("fused", (None, False))):
                for value in supported:
                    control = list(self._control(family))
                    control[0]["optimizer"]["state_dict"]["param_groups"][0][flag] = value
                    control[4]["optimizer_groups"][0]["options"][flag] = value
                    value_label = "none" if value is None else ("true" if value else "false")
                    self._case("A05/" + family + "/" + flag + "/control_" + value_label,
                               lambda b=control: self._positive(b))
                control = list(self._control(family))
                control[0]["optimizer"]["state_dict"]["param_groups"][0][flag] = False
                control[4]["optimizer_groups"][0]["options"][flag] = False
                for label, bad in wrong:
                    for side in ("actual", "expected", "both"):
                        def check(b=control, f=flag, v=bad, s=side):
                            def change(args):
                                if s in ("actual", "both"):
                                    args[0]["optimizer"]["state_dict"]["param_groups"][0][f] = copy.deepcopy(v)
                                if s in ("expected", "both"):
                                    args[4]["optimizer_groups"][0]["options"][f] = copy.deepcopy(v)
                            self._negative(b, change, f)
                        self._case("A05/" + family + "_" + side + "/" + flag + "/" + label, check)

    def test_parent_seed_type_matrix(self):
        bundle = self._control()
        self._case("G09/parent/complete/control", lambda: self._positive(bundle))
        wrong = (("none", None), ("bool", True), ("negative", -1), ("float", 42.0),
                 ("str", "42"), ("bytes", b"42"), ("dict", {}), ("tuple", (42,)),
                 ("list", [42]), ("scalar_tensor", torch.tensor(42)),
                 ("multi_tensor", torch.tensor([42, 43])))
        for index, generator in enumerate(bundle[0]["rng"]["generators"]):
            gid = generator["id"]
            for label, value in wrong:
                for side in ("actual", "expected", "both"):
                    def check(i=index, g=gid, v=value, s=side):
                        def change(args):
                            if s in ("actual", "both"):
                                args[0]["rng"]["generators"][i]["seed_material"] = copy.deepcopy(v)
                            if s in ("expected", "both"):
                                args[4]["rng_generator_bindings"][g]["seed_material"] = copy.deepcopy(v)
                        self._negative(bundle, change, r"seed|word list")
                    self._case("G09/" + side + "/" + gid + "/" + label, check)

    def test_empty_pool_control_before_order_negative(self):
        bundle = self._control(with_empty_pool=True)
        def control():
            result = self._call(bundle)
            self.assertEqual(result["sampler"]["cursor"], 5)
            self.assertEqual(result["progress"]["empty_pool_skips"], [2, 2])
            adj = result["current_model"]["state_dict"]["adj_user_item"]
            self.assertTrue(torch.equal(adj.indices(), torch.tensor([[0, 0, 1, 2, 2], [0, 1, 2, 0, 1]])))
            self.assertTrue(torch.equal(adj.values(), torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5])))
        self._case("S01/P5_K2_T3/complete/control", control)
        self._case("S01/P5_K2_T3/source_order/cleared",
                   lambda: self._negative(bundle, lambda a: a[0]["sampler"].__setitem__("source_order", []), r"source_order"))

    def test_complete_globals_and_failure_contents(self):
        from unittest.mock import patch
        for family in ("light", "hetero"):
            control = self._control(family)
            self._case("F01/" + family + "/preservation/control", lambda b=control: self._positive(b))
            fixtures = []
            for failure in ("success", "selection", "adam", "native_rng", "owned_rng"):
                changed = list(copy.deepcopy(control))
                if failure == "selection":
                    changed[0]["selection"]["best_score"] = 0.999
                elif failure == "adam":
                    changed[0]["optimizer"]["state_dict"]["state"][0]["exp_avg"].fill_(float("nan"))
                elif failure == "native_rng":
                    changed[0]["rng"]["generators"][3]["state"].zero_()
                fixtures.append((failure, changed, copy.deepcopy(changed)))
            # Every fixture and input snapshot exists before live snapshots.
            for failure, changed, saved in fixtures:
                def check(mode=failure, b=changed, snapshot=saved):
                    if mode == "owned_rng":
                        original_copy = checkpoint_payload._make_owned_output_copy
                        def poison(value):
                            owned = original_copy(value)
                            if type(owned) is dict and set(PAYLOAD_REQUIRED_KEYS).issubset(owned):
                                owned["rng"]["generators"][3]["state"].zero_()
                            return owned
                        with patch.object(checkpoint_payload, "_make_owned_output_copy", side_effect=poison):
                            before = self._global_snapshot()
                            with self.assertRaisesRegex(CheckpointValidationError, r"torch|native|generator"):
                                self._call(b)
                            self._assert_globals(before)
                    else:
                        before = self._global_snapshot()
                        if mode == "success":
                            self._positive(b)
                        else:
                            pattern = {"selection": r"best_score|selection", "adam": r"exp_avg|finite", "native_rng": r"torch|native|generator"}[mode]
                            with self.assertRaisesRegex(CheckpointValidationError, pattern):
                                self._call(b)
                        self._assert_globals(before)
                    self._same(b, snapshot)
                self._case("G08/" + family + "/all_globals/" + failure, check)

    def test_preflight_tripwires(self):
        from unittest.mock import patch
        bundle = self._control()
        self._case("R03/light/complete/control", lambda: self._positive(bundle))
        # One shared boundary-negative allocation, just above the logical
        # input bound once its tensor descriptor is charged.
        oversized = torch.zeros(2097152, dtype=torch.uint8)
        for argument_index, label in enumerate(("payload", "expected_identity", "expected_provenance", "expected_model", "expected_runtime")):
            def check(index=argument_index, role=label):
                changed = list(copy.deepcopy(bundle))
                saved = copy.deepcopy(changed)
                changed[index] = oversized
                version = oversized._version
                before = self._global_snapshot()
                with patch.object(checkpoint_payload, "_make_owned_output_copy", side_effect=AssertionError("copy reached")), \
                     patch.object(checkpoint_payload.CanonicalResumeIdentity, "from_dict", side_effect=AssertionError("D1 reached")), \
                     patch.object(checkpoint_payload, "_validate_single_codec_state", side_effect=AssertionError("native parser reached")):
                    pattern = r"^" + role + r" input accounted bytes \(2097179\) exceeded limit \(2097152\)$"
                    with self.assertRaisesRegex(CheckpointValidationError, pattern):
                        self._call(changed)
                self._assert_globals(before)
                self.assertIs(changed[index], oversized)
                self.assertEqual(oversized._version, version)
                self.assertEqual(int(torch.count_nonzero(oversized)), 0)
                for other_index in range(len(changed)):
                    if other_index != index:
                        self._same(changed[other_index], saved[other_index])
            self._case("R03/" + label + "/preflight/before_tripwires", check)

    def test_recursive_storage_and_container_mutations(self):
        for family in ("light", "hetero"):
            bundle = self._control(family)
            self._case("F01/" + family + "/ownership/control", lambda b=bundle: self._positive(b))
            def input_direction(b=copy.deepcopy(bundle)):
                result = self._call(b)
                saved = copy.deepcopy(result)
                # Includes every tensor leaf, every dict/list and all expected
                # arguments, not just a sampled model scalar.
                for argument in b:
                    self._wreck(argument)
                self._same(result, saved)
            self._case("O01/" + family + "/all_inputs/mutate_after_return", input_direction)
            def output_direction(b=copy.deepcopy(bundle)):
                saved_inputs = copy.deepcopy(b)
                result = self._call(b)
                leaves = list(self._tensor_leaves(result))
                for index, leaf in enumerate(leaves):
                    saved_others = [other.clone() for other in leaves]
                    leaf.add_(1)
                    for other_index, other in enumerate(leaves):
                        if other_index != index:
                            self._same(other, saved_others[other_index])
                self._wreck(result)
                self._same(b, saved_inputs)
            self._case("O01/" + family + "/all_output_storages/mutate_after_return", output_direction)

    def test_aggregate_diagnostic_rendering(self):
        bundle = self._control()
        self._case("D01/light/complete/control", lambda: self._positive(bundle))
        for label, keys in (
            ("many_long_keys", ["key_" + str(i) + "_" + "x" * 1000 for i in range(100)]),
            ("many_huge_keys", [10 ** 5000 + i for i in range(100)]),
        ):
            def check(ks=keys):
                changed = list(copy.deepcopy(bundle))
                old = changed[0]["current_model"]["state_dict"]["user_embeddings"]
                changed[0]["current_model"]["state_dict"] = {key: old for key in ks}
                with self.assertRaisesRegex(CheckpointValidationError, r"LightGCN.*keys|keys.*LightGCN") as caught:
                    self._call(changed)
                self.assertLessEqual(len(str(caught.exception)), 4096)
            self._case("D01/current_model/state_dict/" + label, check)

    def _position(self, arguments, index, path):
        value = arguments[index]
        for part in path:
            value = value[part]
        return value

    def _replace_position(self, arguments, index, path, value):
        if not path:
            arguments[index] = value
        else:
            self._position(arguments, index, path[:-1])[path[-1]] = value

    def test_additional_exact_record_schemas(self):
        bundle = self._control()
        self._case("F01/schema_matrix/complete/control", lambda: self._positive(bundle))
        records = [
            ("E01", 0, (), r"payload"),
            ("E04", 0, ("runtime_profile",), r"runtime_profile"),
            ("M01", 0, ("current_model",), r"current_model"),
            ("T03", 0, ("best_model",), r"best_model"),
            ("E01", 0, ("scheduler",), r"scheduler"),
            ("E01", 0, ("amp",), r"amp"),
            ("S02", 0, ("sampler",), r"sampler"),
            ("S02", 0, ("progress",), r"progress"),
            ("T01", 0, ("selection",), r"selection"),
            ("T01", 0, ("selection", "history", 0), r"history"),
            ("A02", 0, ("optimizer",), r"optimizer"),
            ("A02", 0, ("optimizer", "state_dict"), r"state_dict"),
            ("A02", 0, ("optimizer", "state_dict", "param_groups", 0), r"param_groups"),
            ("G02", 0, ("rng",), r"rng"),
            ("E03", 3, (), r"expected_model"),
            ("E03", 3, ("row_order",), r"row_order"),
            ("E03", 3, ("buffer_state",), r"buffer_state"),
            ("E03", 4, (), r"expected_runtime"),
            ("E04", 4, ("runtime_profile",), r"runtime_profile"),
            ("E03", 4, ("optimizer_groups", 0), r"optimizer_groups"),
            ("A02", 4, ("optimizer_groups", 0, "options"), r"options"),
            ("G02", 4, ("rng_role_bindings",), r"rng_role_bindings"),
        ]
        for index in range(len(bundle[0]["optimizer"]["parameter_bindings"])):
            records.append(("A02", 0, ("optimizer", "parameter_bindings", index), r"parameter_bindings"))
        for index in range(len(bundle[0]["rng"]["roles"])):
            records.append(("G02", 0, ("rng", "roles", index), r"rng.roles"))
        for index in range(len(bundle[0]["rng"]["generators"])):
            records.append(("G02", 0, ("rng", "generators", index), r"rng.generators"))
        for gid in bundle[4]["rng_generator_bindings"]:
            records.append(("G02", 4, ("rng_generator_bindings", gid), r"rng_generator_bindings"))
        wrong_records = (
            ("none", None), ("bool", True), ("int", 7), ("float", 1.5),
            ("str", "record"), ("bytes", b"record"), ("list", []),
            ("tuple", ()), ("scalar_tensor", torch.tensor(0)),
            ("multi_tensor", torch.tensor([0, 1])),
        )
        for row, argument, path, pattern in records:
            record = self._position(bundle, argument, path)
            field = "root" if not path else ".".join(str(part) for part in path)
            role = ("actual" if argument == 0 else "expected") + str(argument)
            for key in record:
                def missing(a=argument, p=path, k=key, rx=pattern):
                    self._negative(bundle, lambda args: self._position(args, a, p).__delitem__(k), rx)
                self._case("/".join((row, role, field, "missing_" + str(key))), missing)
            def extra(a=argument, p=path, rx=pattern):
                self._negative(bundle, lambda args: self._position(args, a, p).__setitem__("unexpected_field", None), rx)
            self._case("/".join((row, role, field, "extra")), extra)
            for label, bad in wrong_records:
                # best_model=None is a valid representation before a check,
                # but at this checked boundary it is independently invalid.
                def wrong(a=argument, p=path, value=bad, rx=pattern):
                    self._negative(bundle, lambda args: self._replace_position(args, a, p, copy.deepcopy(value)), rx)
                self._case("/".join((row, role, field, "record_type_" + label)), wrong)

    def test_additional_resource_policy_fields(self):
        bundle = self._control()
        self._case("R01/policy/complete/control", lambda: self._positive(bundle))
        for key, original in bundle[5].items():
            self._case("R01/policy/" + key + "/missing",
                       lambda k=key: self._negative(bundle, lambda a: a[5].__delitem__(k), r"policy|resource"))
            representatives = (
                ("none", None), ("bool", True), ("int", 7), ("float", 1.5),
                ("str", "invalid"), ("bytes", b"invalid"), ("dict", {}),
                ("list", []), ("tuple", ()), ("scalar_tensor", torch.tensor(0)),
                ("multi_tensor", torch.tensor([0, 1])),
            )
            for label, bad in representatives:
                if type(bad) is type(original):
                    continue
                self._case("R01/policy/" + key + "/type_" + label,
                           lambda k=key, v=bad: self._negative(bundle, lambda a: a[5].__setitem__(k, v), r"policy|resource|max_|profile_id|kind|schema_version"))
            if type(original) is int:
                for label, bad in (("zero", 0), ("negative", -1), ("changed_constant", original - 1)):
                    self._case("R01/policy/" + key + "/" + label,
                               lambda k=key, v=bad: self._negative(bundle, lambda a: a[5].__setitem__(k, v), r"policy|resource|max_|synthetic|limit"))
            else:
                self._case("R01/policy/" + key + "/unsupported_value",
                           lambda k=key: self._negative(bundle, lambda a: a[5].__setitem__(k, "unsupported"), r"policy|resource|profile|kind|schema"))
        self._case("R01/policy/root/extra",
                   lambda: self._negative(bundle, lambda a: a[5].__setitem__("unexpected", None), r"policy|resource"))
        smaller = list(copy.deepcopy(bundle))
        smaller[5] = _make_synthetic_resource_policy(
            kind="runtime", profile_id="explicit_smaller_test_profile",
            max_input_bytes=131072, max_output_bytes=131072,
        )
        self._case("R01/runtime/explicit_smaller/control", lambda: self._positive(smaller))
        self._case("R01/actual/resource_policy/payload_override",
                   lambda: self._negative(bundle, lambda a: a[0].__setitem__("resource_policy", copy.deepcopy(a[5])), r"payload.*keys"))

    def test_additional_public_identity_bindings(self):
        bundle = self._control()
        self._case("E02/public/complete/control", lambda: self._positive(bundle))
        alternate, unused_provenance = _build_coherent_identity_and_provenance(seed=43)
        alternate_record = alternate.to_dict()
        for side, argument, path in (("actual", 0, ("identity",)), ("expected", 1, ())):
            def valid_mismatch(a=argument, p=path):
                changed = list(copy.deepcopy(bundle))
                self._replace_position(changed, a, p, copy.deepcopy(alternate_record))
                with self.assertRaises(CheckpointIdentityMismatchError):
                    self._call(changed)
            self._case("E02/" + side + "/identity/rebuilt_seed_mismatch", valid_mismatch)
            for field in ("schema_version", "data_fingerprint", "graph_fingerprint", "validation_fingerprint", "settings_fingerprint", "identity_hash"):
                def malformed(a=argument, p=path, f=field):
                    changed = list(copy.deepcopy(bundle))
                    self._position(changed, a, p)[f] = "invalid"
                    with self.assertRaises(CheckpointError):
                        self._call(changed)
                self._case("E02/" + side + "/identity/invalid_" + field, malformed)

    def test_additional_adam_moments_and_steps(self):
        for family in ("light", "hetero"):
            bundle = self._control(family)
            self._case("A04/" + family + "/complete/control", lambda b=bundle: self._positive(b))
            for moment in ("exp_avg", "exp_avg_sq"):
                for mutation in ("shape", "dtype", "layout", "meta", "nan", "inf", "grad"):
                    def check(b=bundle, name=moment, mode=mutation):
                        def change(args):
                            state = args[0]["optimizer"]["state_dict"]["state"][0]
                            old = state[name]
                            if mode == "shape":
                                value = torch.zeros((old.numel() + 1,), dtype=torch.float32)
                            elif mode == "dtype":
                                value = old.to(dtype=torch.float64)
                            elif mode == "layout":
                                value = old.to_sparse().coalesce()
                            elif mode == "meta":
                                value = torch.empty(tuple(old.shape), device="meta")
                            elif mode == "grad":
                                value = old.clone().requires_grad_(True)
                            else:
                                value = torch.full(tuple(old.shape), float(mode))
                            state[name] = value
                        self._negative(b, change, r"exp_avg|CPU|cpu|device|detach|grad|finite|layout|strided|dtype|float32")
                    self._case("A04/" + family + "/" + moment + "/" + mutation, check)
            self._case("A04/" + family + "/exp_avg_sq/negative",
                       lambda b=bundle: self._negative(b, lambda a: a[0]["optimizer"]["state_dict"]["state"][0]["exp_avg_sq"].fill_(-1), r"exp_avg_sq.*negative"))
            steps = (
                ("int", 4), ("bool", True), ("vector", torch.tensor([4.0])),
                ("dtype", torch.tensor(4.0, dtype=torch.float64)),
                ("meta", torch.empty((), device="meta")),
                ("nan", torch.tensor(float("nan"))),
                ("inf", torch.tensor(float("inf"))),
                ("negative", torch.tensor(-1.0)),
                ("fraction", torch.tensor(4.5)),
                ("update_disagreement", torch.tensor(3.0)),
                ("grad", torch.tensor(4.0, requires_grad=True)),
            )
            for label, value in steps:
                self._case("A04/" + family + "/step/" + label,
                           lambda b=bundle, v=value: self._negative(b, lambda a: a[0]["optimizer"]["state_dict"]["state"][0].__setitem__("step", v), r"step|CPU|cpu|device|detach|grad"))

    def test_additional_raw_copy_ordering(self):
        from unittest.mock import patch
        for family in ("light", "hetero"):
            bundle = self._control(family)
            self._case("R03/" + family + "/raw/control", lambda b=bundle: self._positive(b))
            for role in ("current_model", "best_model"):
                def check(b=bundle, model_role=role):
                    changed = list(copy.deepcopy(b))
                    changed[0][model_role]["state_dict"]["user_embeddings"].requires_grad_(True)
                    saved = copy.deepcopy(changed)
                    before = self._global_snapshot()
                    with patch.object(checkpoint_payload, "_make_owned_output_copy", side_effect=AssertionError("copy reached")):
                        with self.assertRaisesRegex(CheckpointValidationError, r"detach|grad"):
                            self._call(changed)
                    self._same(changed, saved)
                    self._assert_globals(before)
                self._case("R03/" + family + "_" + role + "/raw_grad/before_copy", check)

    def test_additional_bounded_diagnostic_recipes(self):
        bundle = self._control()
        self._case("D01/recipes/complete/control", lambda: self._positive(bundle))
        huge = 10 ** 5000
        long_text = "x" * 20000
        def replace(index, path, value):
            return lambda args: self._replace_position(args, index, path, copy.deepcopy(value))
        recipes = [
            ("homogeneous_huge_state_keys", r"state_dict.*keys", lambda a: a[0]["current_model"].__setitem__("state_dict", {huge + i: a[0]["current_model"]["state_dict"]["user_embeddings"] for i in range(3)})),
            ("expected_codec", r"codec", replace(4, ("rng_generator_bindings", "gen_py", "codec"), huge)),
            ("actual_role_name", r"rng.roles.*name", replace(0, ("rng", "roles", 0, "name"), huge)),
            ("sampler_algorithm", r"sampler.algorithm", replace(0, ("sampler", "algorithm"), huge)),
            ("runtime_profile_id", r"runtime_profile.profile_id", replace(0, ("runtime_profile", "profile_id"), huge)),
            ("packed_parameter_id", r"params.*IDs", replace(0, ("optimizer", "state_dict", "param_groups", 0, "params", 0), huge)),
            ("huge_beta", r"betas", replace(0, ("optimizer", "state_dict", "param_groups", 0, "betas"), (huge, 0.999))),
            ("role_binding_keys", r"rng_role_bindings.*keys", replace(4, ("rng_role_bindings",), {huge + i: [] for i in range(10)})),
            ("heterogeneous_tuple_keys", r"runtime_profile.*keys", replace(4, ("runtime_profile",), {(0,): None, ("x",): None})),
            ("current_epoch", r"current_model.epoch", replace(0, ("current_model", "epoch"), huge)),
            ("long_step_dtype", r"adam_step_dtype", replace(4, ("runtime_profile", "adam_step_dtype"), long_text)),
            ("long_batch_policy", r"batch_policy", replace(4, ("runtime_profile", "batch_policy"), long_text)),
            ("duplicate_parameter_name", r"duplicated.*optimizer groups", replace(4, ("optimizer_groups", 0, "parameter_names"), [long_text, long_text])),
            ("duplicate_row_id", r"Duplicate ID.*row_order", replace(3, ("row_order", "User"), [long_text, long_text])),
            ("invalid_unicode", r"Unicode|UTF|surrogate", replace(0, ("runtime_profile", "profile_id"), "\ud800")),
        ]
        def historical_steps(args):
            args[0]["progress"]["epoch_optimizer_steps"] = [huge, huge]
            args[0]["progress"]["global_step"] = 2 * huge
        recipes.append(("huge_historical_steps", r"epoch_optimizer_steps.*filter_before_batch", historical_steps))
        for field in ("python_version", "numpy_version", "torch_version"):
            def version_mismatch(args, key=field):
                args[0]["runtime_profile"][key] = long_text
                args[4]["runtime_profile"][key] = long_text
            recipes.append(("long_" + field, "runtime_profile.*" + field + ".*provenance", version_mismatch))
        for label, pattern, mutation in recipes:
            def check(change=mutation, rx=pattern):
                changed = list(copy.deepcopy(bundle))
                change(changed)
                with self.assertRaisesRegex(CheckpointValidationError, rx) as caught:
                    self._call(changed)
                self.assertLessEqual(len(str(caught.exception)), 4096)
            self._case("D01/public/diagnostic/" + label, check)
        escaped = "\x01" * 20000
        def escaped_output():
            rendered = checkpoint_payload._safe_repr(escaped)
            self.assertLessEqual(len(rendered), 160)
        self._case("D01/helper/escaped_string/output_bound", escaped_output)

    def test_additional_codec_long_identifier_context(self):
        bundle = list(self._control())
        long_gid = "generator_" + "\x01" * 12000
        old_gid = "gen_py"
        for generator in bundle[0]["rng"]["generators"]:
            if generator["id"] == old_gid:
                generator["id"] = long_gid
        descriptor = bundle[4]["rng_generator_bindings"].pop(old_gid)
        bundle[4]["rng_generator_bindings"][long_gid] = descriptor
        for role in bundle[0]["rng"]["roles"]:
            role["generator_ids"] = [long_gid if gid == old_gid else gid for gid in role["generator_ids"]]
        for name, gids in bundle[4]["rng_role_bindings"].items():
            bundle[4]["rng_role_bindings"][name] = [long_gid if gid == old_gid else gid for gid in gids]
        self._case("D01/long_generator_id/complete/control", lambda: self._positive(bundle))
        for label, state in (("arity", ()), ("gauss_type", (3, random.Random(42).getstate()[1], 1))):
            def check(value=state):
                changed = list(copy.deepcopy(bundle))
                changed[0]["rng"]["generators"][0]["state"] = value
                with self.assertRaisesRegex(CheckpointValidationError, r"Generator.*MT19937|Generator.*gauss") as caught:
                    self._call(changed)
                self.assertLessEqual(len(str(caught.exception)), 4096)
            self._case("D01/long_generator_id/python_codec/" + label, check)

    def test_additional_hostile_positions_without_snapshots(self):
        bundle = self._control()
        self._case("D02/positions/complete/control", lambda: self._positive(bundle))
        events = []
        class Hostile:
            def __getattribute__(self, name):
                events.append("getattribute")
                raise AssertionError("hostile getattribute")
            def __getattr__(self, name):
                events.append("getattr")
                raise AssertionError("hostile getattr")
            def __hash__(self):
                events.append("hash")
                raise AssertionError("hostile hash")
            def __eq__(self, other):
                events.append("eq")
                raise AssertionError("hostile eq")
            def __str__(self):
                events.append("str")
                raise AssertionError("hostile str")
            def __repr__(self):
                events.append("repr")
                raise AssertionError("hostile repr")
        class DictSubclass(dict):
            pass
        class ListSubclass(list):
            pass
        class IntSubclass(int):
            pass
        class StrSubclass(str):
            pass
        class TensorSubclass(torch.Tensor):
            pass
        cycle_list = []
        cycle_list.append(cycle_list)
        cycle_dict = {}
        cycle_dict["self"] = cycle_dict
        candidates = (
            ("hostile", Hostile()), ("dict_subclass", DictSubclass()),
            ("list_subclass", ListSubclass()), ("int_subclass", IntSubclass(1)),
            ("str_subclass", StrSubclass("x")),
            ("tensor_subclass", torch.zeros(1).as_subclass(TensorSubclass)),
            ("parameter", torch.nn.Parameter(torch.zeros(1), requires_grad=False)),
            ("ndarray", np.zeros(1)), ("list_cycle", cycle_list),
            ("dict_cycle", cycle_dict),
        )
        for argument, role in enumerate(("payload", "expected_identity", "expected_provenance", "expected_model", "expected_runtime", "resource_policy")):
            for label, candidate in candidates:
                def check(index=argument, sentinel=candidate):
                    changed = list(bundle)
                    changed[index] = sentinel
                    events.clear()
                    with self.assertRaises(CheckpointValidationError):
                        self._call(changed)
                    self.assertEqual(events, [])
                    self.assertIs(changed[index], sentinel)
                self._case("D02/" + role + "/root/" + label, check)


    def test_additional_rng_registry_long_identifier_diagnostics(self):
        """C8: bound registry/binding identifier contexts without restricting IDs."""
        for family, workers in (("light", 0), ("hetero", 0), ("light", 1)):
            bundle = list(self._control(family, **({"runtime_workers": workers} if workers else {})))
            long_gid = "generator_" + "x" * 12000
            for generator in bundle[0]["rng"]["generators"]:
                if generator["id"] == "gen_py":
                    generator["id"] = long_gid
            descriptor = bundle[4]["rng_generator_bindings"].pop("gen_py")
            bundle[4]["rng_generator_bindings"][long_gid] = descriptor
            for role in bundle[0]["rng"]["roles"]:
                role["generator_ids"] = [long_gid if gid == "gen_py" else gid for gid in role["generator_ids"]]
            for name, gids in bundle[4]["rng_role_bindings"].items():
                bundle[4]["rng_role_bindings"][name] = [long_gid if gid == "gen_py" else gid for gid in gids]
            prefix = "D01/registry_context/" + family + "/W" + str(workers)

            def control(b=bundle):
                before = self._global_snapshot()
                self._positive(b)
                self._assert_globals(before)
            self._case(prefix + "/complete/control", control)

            recipes = (
                ("actual_role_duplicate", r"Duplicate generator ID.*within role.*Python"),
                ("actual_registry_duplicate", r"Duplicate generator ID.*rng\.generators"),
                ("actual_registry_orphan", r"Orphan generator ID.*expected_runtime\.rng_generator_bindings"),
                ("actual_codec_unsupported", r"rng\.generators.*unsupported codec"),
                ("actual_codec_mismatch", r"rng\.generators.*codec.*expected"),
                ("actual_worker_type", r"rng\.generators.*worker_id.*integer"),
                ("actual_worker_range", r"Generator.*worker_id.*runtime_workers"),
                ("actual_seed_type", r"Generator.*Python seed_material.*exact non-negative integer"),
                ("actual_seed_negative", r"Generator.*Python seed_material.*exact non-negative integer"),
                ("expected_role_duplicate", r"Duplicate generator ID.*expected_runtime\.rng_role_bindings.*Python"),
                ("expected_generator_schema", r"expected_runtime\.rng_generator_bindings.*malformed schema"),
                ("expected_codec_unsupported", r"expected_runtime\.rng_generator_bindings.*unsupported codec"),
                ("expected_worker_type", r"expected_runtime\.rng_generator_bindings.*worker_id.*integer"),
                ("expected_worker_negative", r"expected_runtime\.rng_generator_bindings.*worker_id.*>= 0"),
                ("expected_seed_type", r"expected_runtime\.rng_generator_bindings.*seed_material.*exact non-negative int"),
                ("expected_seed_negative", r"expected_runtime\.rng_generator_bindings.*seed_material.*exact non-negative int"),
            )
            if workers:
                recipes += (("actual_worker_binding", r"rng\.generators.*worker_id.*expected"),)
            for label, pattern in recipes:
                # Complete every fixture mutation before taking global snapshots.
                changed = list(copy.deepcopy(bundle))
                generator = next(g for g in changed[0]["rng"]["generators"] if g["id"] == long_gid)
                expected = changed[4]["rng_generator_bindings"][long_gid]
                if label == "actual_role_duplicate":
                    role = next(r for r in changed[0]["rng"]["roles"] if r["name"] == "Python")
                    role["generator_ids"].append(long_gid)
                elif label == "actual_registry_duplicate":
                    changed[0]["rng"]["generators"].append(copy.deepcopy(generator))
                elif label == "actual_registry_orphan":
                    generator["id"] = long_gid + "_orphan"
                elif label == "actual_codec_unsupported":
                    generator["codec"] = "unsupported.codec"
                elif label == "actual_codec_mismatch":
                    generator["codec"] = "numpy.random.RandomState.MT19937.v1"
                elif label == "actual_worker_type":
                    generator["worker_id"] = True
                elif label == "actual_worker_range":
                    generator["worker_id"] = workers
                elif label == "actual_worker_binding":
                    generator["worker_id"] = 0
                elif label == "actual_seed_type":
                    generator["seed_material"] = True
                elif label == "actual_seed_negative":
                    generator["seed_material"] = -1
                elif label == "expected_role_duplicate":
                    changed[4]["rng_role_bindings"]["Python"].append(long_gid)
                elif label == "expected_generator_schema":
                    del expected["seed_material"]
                elif label == "expected_codec_unsupported":
                    expected["codec"] = "unsupported.codec"
                elif label == "expected_worker_type":
                    expected["worker_id"] = True
                elif label == "expected_worker_negative":
                    expected["worker_id"] = -1
                elif label == "expected_seed_type":
                    expected["seed_material"] = True
                elif label == "expected_seed_negative":
                    expected["seed_material"] = -1
                else:
                    self.fail("Unknown synthetic diagnostic recipe")

                def check(b=changed, expected_pattern=pattern):
                    before = self._global_snapshot()
                    with self.assertRaisesRegex(CheckpointValidationError, expected_pattern) as caught:
                        self._call(b)
                    self.assertLessEqual(len(str(caught.exception)), 4096)
                    self._assert_globals(before)
                self._case(prefix + "/" + label, check)


    # Consolidated revision: additive evidence for the original C2-C8 clauses.
    # Existing helpers deliberately detach the five caller arguments separately.
    # No test below restores global state, draws RNG values, or runs an optimizer.

    def _revision_steps(self, bundle, steps):
        """Construct coherent synthetic counters, not a claimed update trajectory."""
        p = bundle[0]
        total = sum(steps)
        p["progress"]["epoch_optimizer_steps"] = list(steps)
        p["progress"]["global_step"] = total
        for binding in p["optimizer"]["parameter_bindings"]:
            binding["updates"] = total
            state = p["optimizer"]["state_dict"]["state"][binding["id"]]
            state["step"] = torch.tensor(float(total), dtype=torch.float32)

    def _revision_generator(self, bundle, gid):
        return next(g for g in bundle[0]["rng"]["generators"] if g["id"] == gid)

    def _revision_bind_role(self, bundle, index, gids):
        role = bundle[0]["rng"]["roles"][index]
        role["generator_ids"] = list(gids)
        bundle[4]["rng_role_bindings"][role["name"]] = list(gids)

    def _revision_private_initialization_bundle(self):
        """Give initialization its own staged archive without claiming capture."""
        bundle = list(self._control("hetero"))
        generator = copy.deepcopy(self._revision_generator(bundle, "gen_torch"))
        generator["id"] = "revision_init_archive"
        bundle[0]["rng"]["generators"].append(generator)
        bundle[4]["rng_generator_bindings"][generator["id"]] = {
            key: copy.deepcopy(generator[key])
            for key in ("codec", "worker_id", "seed_material")
        }
        self._revision_bind_role(bundle, 4, [generator["id"]])
        return bundle

    def test_revision_sparse_roles_and_independent_graph(self):
        """C3: raw COO defects in current, best and independent expectation."""
        bundle = self._control()
        self._case("REV/C3/sparse/complete", lambda: self._positive(bundle))
        original_indices = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.int64)
        self.assertTrue(torch.equal(bundle[3]["buffer_state"]["adj_user_item"].indices(), original_indices))
        paths = (
            ("current", 0, ("current_model", "state_dict")),
            ("best", 0, ("best_model", "state_dict")),
            ("expected", 3, ("buffer_state",)),
        )
        recipes = (
            "type", "dense", "dtype", "shape", "nan", "inf", "grad",
            "uncoalesced", "order", "duplicate", "row_low", "row_high",
            "column_low", "column_high", "weight", "meta",
        )
        for role, argument, path in paths:
            for mode in recipes:
                def check(a=argument, location=path, recipe=mode, name=role):
                    def change(args):
                        owner = self._position(args, a, location)
                        old = owner["adj_user_item"]
                        indices = old.indices().clone()
                        values = old.values().clone()
                        shape = tuple(old.shape)
                        if recipe == "type":
                            value = []
                        elif recipe == "dense":
                            value = torch.zeros(shape, dtype=torch.float32)
                        elif recipe == "dtype":
                            value = old.to(dtype=torch.float64)
                        elif recipe == "shape":
                            value = torch.sparse_coo_tensor(indices, values, (shape[0], shape[1] + 1), is_coalesced=True, check_invariants=False)
                        elif recipe == "grad":
                            value = old.clone().requires_grad_(True)
                        elif recipe == "uncoalesced":
                            value = old.clone()
                            value._coalesced_(False)
                        elif recipe == "meta":
                            value = torch.sparse_coo_tensor(
                                torch.empty((2, values.numel()), dtype=torch.int64, device="meta"),
                                torch.empty((values.numel(),), dtype=torch.float32, device="meta"),
                                shape, is_coalesced=True, check_invariants=False,
                            )
                        else:
                            if recipe == "order":
                                indices[:, [0, 1]] = indices[:, [1, 0]].clone()
                            elif recipe == "duplicate":
                                indices[:, 1] = indices[:, 0]
                            elif recipe == "row_low":
                                indices[0, 0] = -1
                            elif recipe == "row_high":
                                indices[0, -1] = shape[0]
                            elif recipe == "column_low":
                                indices[1, 0] = -1
                            elif recipe == "column_high":
                                indices[1, -1] = shape[1]
                            elif recipe == "weight":
                                values[0] += 0.125
                            else:
                                values[0] = float(recipe)
                            # Never coalesce: that would repair order or duplicates.
                            value = torch.sparse_coo_tensor(indices, values, shape, is_coalesced=True, check_invariants=False)
                        owner["adj_user_item"] = value
                    if recipe == "meta":
                        pattern = r"CPU|cpu|device|meta"
                    elif recipe in ("type", "dense"):
                        pattern = r"adj_user_item.*(Tensor|tensor|COO)"
                    elif recipe == "dtype":
                        pattern = r"adj_user_item.*float32"
                    elif recipe == "shape":
                        pattern = r"adj_user_item.*shape"
                    elif recipe == "grad":
                        pattern = r"detach|grad"
                    elif recipe == "uncoalesced":
                        root_role = "expected_model" if name == "expected" else "payload"
                        pattern = "^" + root_role + r" sparse COO tensor must be coalesced before preflight$"
                    elif name == "expected":
                        # Expected indices/values are compared with independently
                        # valid actual contents; no new expected canonical guard.
                        pattern = r"adj_user_item.*trusted graph contents"
                    elif recipe in ("nan", "inf"):
                        pattern = r"adj_user_item.*non-finite"
                    elif recipe in ("order", "duplicate"):
                        pattern = r"adj_user_item.*canonical strictly ascending"
                    elif recipe.startswith("row"):
                        pattern = r"adj_user_item.*row indices out of range"
                    elif recipe.startswith("column"):
                        pattern = r"adj_user_item.*col indices out of range"
                    else:
                        pattern = r"adj_user_item.*trusted graph contents"
                    self._negative(bundle, change, pattern)
                self._case("REV/C3/sparse/" + role + "/" + mode, check)

        # Independent graph oracle: replace (1,2) by (1,0), then normalize
        # from the replacement graph's degrees, not from trusted values.
        edges = ((0, 0), (0, 1), (1, 0), (1, 1))
        user_degrees = [sum(u == row for u, col in edges) for row in range(2)]
        item_degrees = [sum(i == col for row, col in edges) for i in range(3)]
        self.assertEqual(user_degrees, [2, 2])
        self.assertEqual(item_degrees, [2, 2, 0])
        weights = [1.0 / math.sqrt(user_degrees[u] * item_degrees[i]) for u, i in edges]
        self.assertEqual(weights, [0.5] * 4)
        replacement = torch.sparse_coo_tensor(
            torch.tensor([[u for u, i in edges], [i for u, i in edges]], dtype=torch.int64),
            torch.tensor(weights, dtype=torch.float32), (2, 3),
            is_coalesced=True, check_invariants=False,
        )
        for role in ("current_model", "best_model"):
            self._case("REV/C3/independent_graph/" + role,
                       lambda r=role: self._negative(
                           bundle,
                           lambda args: args[0][r]["state_dict"].__setitem__("adj_user_item", replacement.clone()),
                           r"adj_user_item.*indices do not match trusted graph contents"))
        self.assertTrue(torch.equal(bundle[3]["buffer_state"]["adj_user_item"].indices(), original_indices))

    def test_revision_sparse_literal_accounting_and_view_backing(self):
        """C3/C8: supplied literal COO oracle, aliases and full view backing."""
        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.int64)
        values = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
        adj = torch.sparse_coo_tensor(indices, values, (2, 3), is_coalesced=True, check_invariants=False)
        self.assertEqual(tuple(adj.indices().stride()), (4, 1))
        self.assertEqual(str(adj.dtype), "torch.float32")
        self.assertEqual(str(adj.layout), "torch.sparse_coo")
        self.assertEqual(adj.indices().untyped_storage().nbytes(), 64)
        self.assertEqual(adj.values().untyped_storage().nbytes(), 16)
        # Literal proposed frozen-formula targets; do not derive these by
        # calling production accounting. A mismatch requires source/formula
        # review, never modification of protected accounting to fit an oracle.
        rows = (
            ("single", adj, 245, 80, 80, 28),
            ("alias", [adj, adj], 498, 80, 160, 57),
        )
        for label, value, metadata, storage, copies, visits in rows:
            def check(v=value, m=metadata, s=storage, c=copies, n=visits):
                exact = _make_synthetic_resource_policy(
                    kind="runtime", profile_id="revision_literal_accounting",
                    max_input_bytes=m + s, max_output_bytes=m + c,
                    max_visited_values=n,
                )
                preflight_resource_accounting(v, exact, role="revision_oracle")
                for field, pattern in (
                    ("max_input_bytes", r"input|Input"),
                    ("max_output_bytes", r"output|Output"),
                    ("max_visited_values", r"visit|Visit"),
                ):
                    smaller = dict(exact)
                    smaller[field] -= 1
                    with self.assertRaisesRegex(CheckpointValidationError, pattern):
                        preflight_resource_accounting(v, smaller, role="revision_oracle")
            self._case("REV/C3/accounting/" + label, check)
        self.assertEqual(245 + 80, 325)
        self.assertEqual(498 + 80, 578)
        self.assertEqual(498 + 160, 658)
        self.assertEqual(1 + 2 * 28, 57)

        policy = _make_synthetic_resource_policy(
            kind="runtime", profile_id="revision_small_backing",
            max_input_bytes=512, max_output_bytes=512,
        )
        base = torch.zeros(256, dtype=torch.float32)
        for label, view in (("empty", base[:0]), ("noncontiguous", base[:4:2])):
            def check(v=view):
                compact = v.clone()
                preflight_resource_accounting(compact, policy, role="compact_control")
                self.assertEqual(v.untyped_storage().nbytes(), 1024)
                self.assertLessEqual(v.numel() * v.element_size(), 8)
                with self.assertRaisesRegex(CheckpointValidationError, r"input|Input"):
                    preflight_resource_accounting(v, policy, role="backed_view")
            self._case("REV/C3/accounting/backing/" + label, check)

    def test_revision_four_codec_fields(self):
        """C5: every codec component, using a complete detached payload."""
        bundle = self._control()
        self._case("REV/C5/codecs/complete", lambda: self._positive(bundle))
        py = copy.deepcopy(self._revision_generator(bundle, "gen_py")["state"])
        mt = copy.deepcopy(self._revision_generator(bundle, "gen_np_mt")["state"])
        pcg = copy.deepcopy(self._revision_generator(bundle, "gen_np_pcg")["state"])
        native = self._revision_generator(bundle, "gen_torch")["state"].clone()

        def python_internal(index, value):
            words = list(py[1])
            words[index] = value
            return (3, tuple(words), py[2])

        positive_states = [
            ("python_gauss", "gen_py", (3, py[1], 1.25)),
            ("python_index0", "gen_py", python_internal(624, 0)),
            ("python_index624", "gen_py", python_internal(624, 624)),
            ("mt_position0", "gen_np_mt", ("MT19937", list(mt[1]), 0, 0, 2.5)),
            ("mt_position624", "gen_np_mt", ("MT19937", list(mt[1]), 624, 1, -2.5)),
        ]
        inactive_cache = copy.deepcopy(pcg)
        inactive_cache["has_uint32"] = 0
        inactive_cache["uinteger"] = 0xFFFFFFFF
        positive_states.append(("pcg_inactive_cache", "gen_np_pcg", inactive_cache))
        for label, gid, state in positive_states:
            changed = list(copy.deepcopy(bundle))
            self._revision_generator(changed, gid)["state"] = state
            self._case("REV/C5/codecs/positive/" + label, lambda b=changed: self._positive(b))

        cases = [
            ("py_tuple", "gen_py", list(py), r"MT19937.v3 state must be 3-tuple"),
            ("py_arity", "gen_py", py[:2], r"MT19937.v3 state must be 3-tuple"),
            ("py_version", "gen_py", (2, py[1], None), r"state\[0\].*int 3"),
            ("py_version_bool", "gen_py", (True, py[1], None), r"state\[0\].*int 3"),
            ("py_internal_list", "gen_py", (3, list(py[1]), None), r"internal state must be 625-tuple"),
            ("py_internal_count", "gen_py", (3, py[1][:-1], None), r"internal state must be 625-tuple"),
            ("py_word_bool", "gen_py", python_internal(0, True), r"internal word\[0\].*uint32"),
            ("py_word_low", "gen_py", python_internal(0, -1), r"internal word\[0\].*uint32"),
            ("py_last_word_high", "gen_py", python_internal(623, 1 << 32), r"internal word\[623\].*uint32"),
            ("py_index_low", "gen_py", python_internal(624, -1), r"index out of range"),
            ("py_index_high", "gen_py", python_internal(624, 625), r"index out of range"),
            ("py_index_bool", "gen_py", python_internal(624, True), r"index out of range"),
            ("py_gauss_type", "gen_py", (3, py[1], 1), r"gauss"),
            ("py_gauss_nonfinite", "gen_py", (3, py[1], float("nan")), r"[Ff]inite"),
            ("mt_tuple", "gen_np_mt", list(mt), r"RandomState state must be 5-tuple"),
            ("mt_arity", "gen_np_mt", mt[:4], r"RandomState state must be 5-tuple"),
            ("mt_engine", "gen_np_mt", ("MT19938",) + mt[1:], r"RandomState\[0\].*MT19937"),
            ("mt_keys_type", "gen_np_mt", (mt[0], tuple(mt[1])) + mt[2:], r"keys must be list of 624"),
            ("mt_keys_count", "gen_np_mt", (mt[0], mt[1][:-1]) + mt[2:], r"keys must be list of 624"),
            ("mt_position_bool", "gen_np_mt", (mt[0], list(mt[1]), True, mt[3], mt[4]), r"position"),
            ("mt_position_low", "gen_np_mt", (mt[0], list(mt[1]), -1, mt[3], mt[4]), r"position"),
            ("mt_position_high", "gen_np_mt", (mt[0], list(mt[1]), 625, mt[3], mt[4]), r"position"),
            ("mt_flag_bool", "gen_np_mt", (mt[0], list(mt[1]), mt[2], True, mt[4]), r"has_gauss"),
            ("mt_flag_range", "gen_np_mt", (mt[0], list(mt[1]), mt[2], 2, mt[4]), r"has_gauss"),
            ("mt_cache_type", "gen_np_mt", (mt[0], list(mt[1]), mt[2], 0, 0), r"cached_gauss"),
            ("mt_cache_nonfinite", "gen_np_mt", (mt[0], list(mt[1]), mt[2], 0, float("inf")), r"[Ff]inite"),
        ]
        for label, index, word in (("bool", 0, True), ("low", 0, -1), ("last_high", 623, 1 << 32)):
            keys = list(mt[1])
            keys[index] = word
            cases.append(("mt_word_" + label, "gen_np_mt", (mt[0], keys) + mt[2:], r"RandomState key\[" + str(index) + r"\].*uint32"))
        for label, path, value, pattern in (
            ("engine", ("bit_generator",), "PCG64DXSM", r"bit_generator.*PCG64"),
            ("inner_type", ("state",), [], r"PCG64 inner state"),
            ("state_bool", ("state", "state"), True, r"PCG64 state"),
            ("state_low", ("state", "state"), -1, r"PCG64 state"),
            ("state_high", ("state", "state"), 1 << 128, r"PCG64 state"),
            ("inc_bool", ("state", "inc"), True, r"PCG64 inc"),
            ("inc_low", ("state", "inc"), -1, r"PCG64 inc"),
            ("inc_high", ("state", "inc"), 1 << 128, r"PCG64 inc"),
            ("inc_even", ("state", "inc"), 2, r"PCG64 inc must be odd"),
            ("flag_bool", ("has_uint32",), True, r"has_uint32"),
            ("flag_range", ("has_uint32",), 2, r"has_uint32"),
            ("cache_bool", ("uinteger",), True, r"uinteger"),
            ("cache_low", ("uinteger",), -1, r"uinteger"),
            ("cache_high", ("uinteger",), 1 << 32, r"uinteger"),
        ):
            state = copy.deepcopy(pcg)
            owner = state
            for key in path[:-1]:
                owner = owner[key]
            owner[path[-1]] = value
            cases.append(("pcg_" + label, "gen_np_pcg", state, pattern))
        for label, inner, extra in (("outer_missing", False, False), ("outer_extra", False, True), ("inner_missing", True, False), ("inner_extra", True, True)):
            state = copy.deepcopy(pcg)
            owner = state["state"] if inner else state
            if extra:
                owner["unexpected"] = 0
            else:
                del owner["inc" if inner else "uinteger"]
            cases.append(("pcg_" + label, "gen_np_pcg", state, r"PCG64 (state dict keys|inner state)"))

        backing = torch.empty(native.numel() * 2, dtype=torch.uint8)
        noncontiguous = backing[::2]
        noncontiguous.copy_(native)
        cases.extend([
            ("torch_rank", "gen_torch", native.reshape(1, -1), r"native state must be 1-D contiguous CPU uint8"),
            ("torch_length", "gen_torch", native[:-1].clone(), r"native state length"),
            ("torch_noncontiguous", "gen_torch", noncontiguous, r"native state must be 1-D contiguous CPU uint8"),
            ("torch_meta", "gen_torch", torch.empty(native.shape, dtype=torch.uint8, device="meta"), r"CPU|cpu|device|meta"),
            ("torch_allzero", "gen_torch", torch.zeros_like(native), r"Invalid torch CPU generator native state"),
            # uint8 cannot require gradients. This is the representable malformed alternative.
            ("torch_float_grad", "gen_torch", native.to(dtype=torch.float32).requires_grad_(True), r"native state|detach|grad"),
        ])
        for label, gid, state, pattern in cases:
            self._case("REV/C5/codecs/negative/" + label,
                       lambda g=gid, s=state, rx=pattern: self._negative(
                           bundle, lambda args: self._revision_generator(args, g).__setitem__("state", s), rx))
        # Scalar NaN/Inf fixtures intentionally exercise common preflight before
        # codec parsing; they are not claimed as native-setter evidence.

    def test_revision_ten_roles_and_binding_order(self):
        """C5: representative fields for all roles and ordered binding refusal."""
        bundle = self._control()
        self._case("REV/C5/roles/complete", lambda: self._positive(bundle))
        for index, role in enumerate(bundle[0]["rng"]["roles"]):
            self._case("REV/C5/roles/" + role["name"] + "/algorithm",
                       lambda i=index: self._negative(
                           bundle, lambda args: args[0]["rng"]["roles"][i].__setitem__("algorithm", "revision.invalid.algorithm"),
                           r"rng.roles\[" + str(i) + r"\].*field 'algorithm'"))
        for label, mutation, pattern in (
            ("applicability_type", lambda a: a[0]["rng"]["roles"][0].__setitem__("applicable", 1), r"applicable"),
            ("active_seed_type", lambda a: a[0]["rng"]["roles"][0].__setitem__("seed", True), r"rng.roles\[0\].seed"),
            ("inactive_seed", lambda a: a[0]["rng"]["roles"][3].__setitem__("seed", 42), r"inactive but seed is not None"),
            ("derivation", lambda a: a[0]["rng"]["roles"][0].__setitem__("derivation", "revision.invalid.derivation"), r"field 'derivation'"),
            ("reason", lambda a: a[0]["rng"]["roles"][3].__setitem__("reason", "revision.invalid.reason"), r"field 'reason'"),
            ("order", lambda a: a[0]["rng"]["roles"].reverse(), r"does not match expected order"),
            ("missing", lambda a: a[0]["rng"]["roles"].pop(), r"rng.roles count"),
            ("extra", lambda a: a[0]["rng"]["roles"].append(copy.deepcopy(a[0]["rng"]["roles"][0])), r"rng.roles count"),
        ):
            self._case("REV/C5/roles/" + label, lambda m=mutation, rx=pattern: self._negative(bundle, m, rx))
        workers = self._control(runtime_workers=2)
        self._case("REV/C5/roles/W2/complete", lambda: self._positive(workers))
        self._case("REV/C5/roles/W2/actual_order", lambda: self._negative(
            workers, lambda a: a[0]["rng"]["roles"][9]["generator_ids"].reverse(), r"generator_ids.*expected"))
        self._case("REV/C5/roles/W2/expected_order", lambda: self._negative(
            workers, lambda a: a[4]["rng_role_bindings"]["DataLoader_workers"].reverse(), r"generator_ids.*expected"))

    def test_revision_parent_family_and_cardinality_guards(self):
        """C5: coherent bindings reach family/cardinality, not orphan rejection."""
        for index, family, expected_role in (
            (0, "light", "Python"), (1, "light", "NumPy"),
            (2, "light", "PyTorch_CPU"), (8, "light", "DataLoader_generator"),
            (4, "hetero", "initialization"),
        ):
            bundle = self._control(family)
            self._case("REV/C5/family/" + expected_role + "/complete", lambda b=bundle: self._positive(b))
            def wrong_family(args, role_index=index):
                source_id = "gen_np_pcg" if role_index == 0 else "gen_py"
                target_id = args[0]["rng"]["roles"][role_index]["generator_ids"][0]
                # For roles sharing a generator, a new bound record preserves
                # the other parent families and leaves no unused registry row.
                if role_index in (4, 8):
                    source = copy.deepcopy(self._revision_generator(args, source_id))
                    source["id"] = "revision_wrong_family"
                    args[0]["rng"]["generators"].append(source)
                    args[4]["rng_generator_bindings"][source["id"]] = {
                        k: copy.deepcopy(source[k]) for k in ("codec", "worker_id", "seed_material")
                    }
                    self._revision_bind_role(args, role_index, [source["id"]])
                else:
                    source = self._revision_generator(args, source_id)
                    target = self._revision_generator(args, target_id)
                    target["codec"] = source["codec"]
                    target["state"] = copy.deepcopy(source["state"])
                    args[4]["rng_generator_bindings"][target_id]["codec"] = source["codec"]
            self._case("REV/C5/family/" + expected_role + "/wrong_family",
                       lambda b=bundle, m=wrong_family, name=expected_role: self._negative(
                           b, m, name + r" role must bind exactly one parent"))
            def cardinality(args, role_index=index):
                old = args[0]["rng"]["roles"][role_index]["generator_ids"]
                extra = "gen_np_pcg" if role_index == 0 else "gen_py"
                self._revision_bind_role(args, role_index, list(old) + [extra])
            self._case("REV/C5/family/" + expected_role + "/cardinality",
                       lambda b=bundle, m=cardinality, name=expected_role: self._negative(
                           b, m, name + r" role must bind exactly one parent"))

    def test_revision_staged_workers_seed_words_and_native_seed(self):
        """C5: W0/W1/W2 metadata, worker materials and private native seed."""
        for count in (0, 1, 2):
            bundle = self._control(runtime_workers=count)
            self._case("REV/C5/workers/W" + str(count) + "/complete", lambda b=bundle: self._positive(b))
            if count == 0:
                self._case("REV/C5/workers/W0/worker_scoped", lambda b=bundle: self._negative(
                    b, lambda a: self._revision_generator(a, "gen_py").__setitem__("worker_id", 0),
                    r"worker_id.*runtime_workers is 0"))
            else:
                gid = "gen_worker_0_py"
                for side in ("actual", "expected", "both"):
                    def check(b=bundle, g=gid, which=side, population=count):
                        def change(args):
                            if which in ("actual", "both"):
                                self._revision_generator(args, g)["worker_id"] = population
                            if which in ("expected", "both"):
                                args[4]["rng_generator_bindings"][g]["worker_id"] = population
                        pattern = r"worker_id.*runtime_workers" if which != "expected" else r"worker_id.*expected"
                        self._negative(b, change, pattern)
                    self._case("REV/C5/workers/W" + str(count) + "/id/" + side, check)

        bundle = self._control(runtime_workers=1)
        words = list(copy.deepcopy(bundle))
        gid = "gen_worker_0_np"
        self._revision_generator(words, gid)["seed_material"] = [0, 17, 0xFFFFFFFF]
        words[4]["rng_generator_bindings"][gid]["seed_material"] = [0, 17, 0xFFFFFFFF]
        self._case("REV/C5/workers/wordlist/complete", lambda: self._positive(words))
        for side in ("actual", "expected", "both"):
            for label, bad in (("bool", True), ("low", -1), ("high", 1 << 32)):
                def check(which=side, value=bad):
                    def change(args):
                        if which in ("actual", "both"):
                            self._revision_generator(args, gid)["seed_material"][1] = value
                        if which in ("expected", "both"):
                            args[4]["rng_generator_bindings"][gid]["seed_material"][1] = value
                    self._negative(words, change, r"seed.*word")
                self._case("REV/C5/workers/wordlist/" + side + "/" + label, check)

        huge = list(copy.deepcopy(bundle))
        self._revision_generator(huge, "gen_worker_0_py")["seed_material"] = 10 ** 5000
        huge[4]["rng_generator_bindings"]["gen_worker_0_py"]["seed_material"] = 10 ** 5000
        self._case("REV/C5/workers/python_huge_metadata", lambda: self._positive(huge))
        # This is metadata acceptance, not derivation or seed-reachability proof.
        def native_seed_mismatch(args):
            generator = self._revision_generator(args, "gen_worker_0_torch")
            generator["seed_material"] += 1
            args[4]["rng_generator_bindings"][generator["id"]]["seed_material"] = generator["seed_material"]
            # Keep native state bytes unchanged to reach initial_seed comparison.
        self._case("REV/C5/workers/native_initial_seed", lambda: self._negative(
            bundle, native_seed_mismatch, r"torch initial_seed.*declared seed_material"))

        def worker_family(args):
            source = self._revision_generator(args, "gen_worker_0_py")
            target = self._revision_generator(args, "gen_worker_0_np")
            target["codec"] = source["codec"]
            target["state"] = copy.deepcopy(source["state"])
            args[4]["rng_generator_bindings"][target["id"]]["codec"] = source["codec"]
        self._case("REV/C5/workers/family", lambda: self._negative(
            bundle, worker_family, r"DataLoader_workers for worker 0 must contain exactly one Python, one NumPy, and one TorchCPU"))

    def test_revision_initialization_archive_and_binding_refusals(self):
        """C5: staged archive requirements, with no physical-capture claim."""
        bundle = self._revision_private_initialization_bundle()
        self._case("REV/C5/init/complete", lambda: self._positive(bundle))
        gid = "revision_init_archive"
        def missing_archive(args):
            args[0]["rng"]["generators"] = [g for g in args[0]["rng"]["generators"] if g["id"] != gid]
        self._case("REV/C5/init/missing_archive", lambda: self._negative(
            bundle, missing_archive, r"rng.generators mismatch against active bound roles.*Missing"))
        self._case("REV/C5/init/missing_binding", lambda: self._negative(
            bundle, lambda a: a[4]["rng_generator_bindings"].__delitem__(gid), r"Orphan generator ID.*expected_runtime"))
        self._case("REV/C5/init/absent_native_state", lambda: self._negative(
            bundle, lambda a: self._revision_generator(a, gid).__setitem__("state", None), r"native state must be exact torch.Tensor"))
        def unbound_active(args):
            self._revision_bind_role(args, 4, [])
            args[0]["rng"]["generators"] = [g for g in args[0]["rng"]["generators"] if g["id"] != gid]
            del args[4]["rng_generator_bindings"][gid]
        self._case("REV/C5/init/unbound_active", lambda: self._negative(
            bundle, unbound_active, r"initialization.*active but generator_ids is empty"))

    def test_revision_adam_bindings_groups_options_and_names(self):
        """C4: packed IDs, partition order, numeric options and optional names."""
        for family in ("light", "hetero"):
            bundle = self._control(family)
            self._case("REV/C4/bindings/" + family + "/complete", lambda b=bundle: self._positive(b))
            for label, mutation, pattern in (
                ("packed", lambda a: a[0]["optimizer"]["parameter_bindings"][0].__setitem__("id", 1), r"id must be packed in order"),
                ("binding_duplicate", lambda a: a[0]["optimizer"]["parameter_bindings"].__setitem__(1, copy.deepcopy(a[0]["optimizer"]["parameter_bindings"][0])), r"id must be packed in order"),
                ("params_duplicate", lambda a: a[0]["optimizer"]["state_dict"]["param_groups"][0]["params"].__setitem__(1, 0), r"params IDs.*expected"),
                ("params_order", lambda a: a[0]["optimizer"]["state_dict"]["param_groups"][0]["params"].reverse(), r"params IDs.*expected"),
                ("orphan", lambda a: a[0]["optimizer"]["state_dict"]["state"].__setitem__(999, {}), r"invalid or orphan ID"),
                ("buffer_as_parameter", lambda a: a[0]["optimizer"]["parameter_bindings"][0].__setitem__("name", "adj_user_item"), r"name.*does not match expected"),
            ):
                self._case("REV/C4/bindings/" + family + "/" + label,
                           lambda b=bundle, m=mutation, rx=pattern: self._negative(b, m, rx))

            groups = list(copy.deepcopy(bundle))
            names = list(groups[4]["optimizer_groups"][0]["parameter_names"])
            split = len(names) // 2
            options = copy.deepcopy(groups[4]["optimizer_groups"][0]["options"])
            actual_options = copy.deepcopy(groups[0]["optimizer"]["state_dict"]["param_groups"][0])
            groups[4]["optimizer_groups"] = [
                {"parameter_names": names[:split], "options": copy.deepcopy(options)},
                {"parameter_names": names[split:], "options": copy.deepcopy(options)},
            ]
            actual_groups = []
            for start, end in ((0, split), (split, len(names))):
                group = copy.deepcopy(actual_options)
                group["params"] = list(range(start, end))
                actual_groups.append(group)
            groups[0]["optimizer"]["state_dict"]["param_groups"] = actual_groups
            self._case("REV/C4/groups/" + family + "/complete", lambda b=groups: self._positive(b))
            self._case("REV/C4/groups/" + family + "/actual_order", lambda b=groups: self._negative(
                b, lambda a: a[0]["optimizer"]["state_dict"]["param_groups"].reverse(), r"params IDs.*expected"))
            self._case("REV/C4/groups/" + family + "/expected_order", lambda b=groups: self._negative(
                b, lambda a: a[4]["optimizer_groups"].reverse(), r"name.*does not match expected"))

        bundle = self._control()
        for field, value in (("lr", 0.002), ("betas", (0.8, 0.999)), ("eps", 2e-8), ("weight_decay", 0.1), ("amsgrad", True), ("lr", True)):
            for side in ("actual", "expected", "both"):
                def check(key=field, bad=value, which=side):
                    def change(args):
                        if which in ("actual", "both"):
                            args[0]["optimizer"]["state_dict"]["param_groups"][0][key] = bad
                        if which in ("expected", "both"):
                            args[4]["optimizer_groups"][0]["options"][key] = bad
                    self._negative(bundle, change, key)
                label = field + ("_bool" if type(value) is bool and field == "lr" else "")
                self._case("REV/C4/options/" + label + "/" + side, check)

        named = list(copy.deepcopy(bundle))
        names = list(named[4]["optimizer_groups"][0]["parameter_names"])
        named[4]["optimizer_groups"][0]["options"]["param_names"] = list(names)
        named[0]["optimizer"]["state_dict"]["param_groups"][0]["param_names"] = list(names)
        self._case("REV/C4/names/complete", lambda: self._positive(named))
        self._case("REV/C4/names/missing", lambda: self._negative(
            named, lambda a: a[0]["optimizer"]["state_dict"]["param_groups"][0].__delitem__("param_names"), r"param_groups.*keys mismatch"))
        self._case("REV/C4/names/unexpected", lambda: self._negative(
            bundle, lambda a: a[0]["optimizer"]["state_dict"]["param_groups"][0].__setitem__("param_names", names), r"param_groups.*keys mismatch"))
        for side in ("actual", "expected", "both"):
            def check(which=side):
                def change(args):
                    if which in ("actual", "both"):
                        args[0]["optimizer"]["state_dict"]["param_groups"][0]["param_names"].reverse()
                    if which in ("expected", "both"):
                        args[4]["optimizer_groups"][0]["options"]["param_names"].reverse()
                self._negative(named, change, r"param_names.*(ordered trusted|parameter_names)")
            self._case("REV/C4/names/reordered/" + side, check)

    def test_revision_both_models_lazy_zero_and_aggregate(self):
        """C4: complete zero, mixed never-updated and nonuniform controls."""
        for family in ("light", "hetero"):
            bundle = self._control(family)
            self._case("REV/C4/lazy/" + family + "/complete", lambda b=bundle: self._positive(b))
            mixed = list(copy.deepcopy(bundle))
            mixed[0]["optimizer"]["parameter_bindings"][0]["updates"] = 0
            del mixed[0]["optimizer"]["state_dict"]["state"][0]
            self._case("REV/C4/lazy/" + family + "/mixed_omitted", lambda b=mixed: self._positive(b))
            initialized = list(copy.deepcopy(mixed))
            shape = tuple(initialized[0]["current_model"]["state_dict"][initialized[0]["optimizer"]["parameter_bindings"][0]["name"]].shape)
            initialized[0]["optimizer"]["state_dict"]["state"][0] = {
                "step": torch.tensor(0.0, dtype=torch.float32),
                "exp_avg": torch.zeros(shape, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(shape, dtype=torch.float32),
            }
            self._case("REV/C4/lazy/" + family + "/mixed_initialized_zero", lambda b=initialized: self._positive(b))
            nonuniform = list(copy.deepcopy(bundle))
            nonuniform[0]["optimizer"]["parameter_bindings"][0]["updates"] = 3
            nonuniform[0]["optimizer"]["state_dict"]["state"][0]["step"] = torch.tensor(3.0)
            self._case("REV/C4/lazy/" + family + "/nonuniform_positive", lambda b=nonuniform: self._positive(b))

            initial = list(copy.deepcopy(bundle))
            p = initial[0]
            p["current_model"]["epoch"] = 0
            p["progress"] = {"epoch": 0, "global_step": 0, "epoch_optimizer_steps": [], "empty_pool_skips": []}
            p["sampler"].update(source_order=None, cursor=0, next_epoch=1)
            p["selection"].update(history=[], best_score=None, selected_epoch=None, bad_checks=0, status="initial", stop_reason=None)
            p["best_model"] = None
            p["optimizer"]["state_dict"]["state"] = {}
            for binding in p["optimizer"]["parameter_bindings"]:
                binding["updates"] = 0
            self._case("REV/C4/lazy/" + family + "/G0_omitted", lambda b=initial: self._positive(b))
            zero = list(copy.deepcopy(initial))
            for binding in zero[0]["optimizer"]["parameter_bindings"]:
                shape = tuple(zero[0]["current_model"]["state_dict"][binding["name"]].shape)
                zero[0]["optimizer"]["state_dict"]["state"][binding["id"]] = {
                    "step": torch.tensor(0.0, dtype=torch.float32),
                    "exp_avg": torch.zeros(shape, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros(shape, dtype=torch.float32),
                }
            self._case("REV/C4/lazy/" + family + "/G0_complete_initialized_zero", lambda b=zero: self._positive(b))
            for label, mutation, pattern in (
                ("updated_missing", lambda a: a[0]["optimizer"]["state_dict"]["state"].__delitem__(0), r"updates=.*state is omitted"),
                ("updated_empty", lambda a: a[0]["optimizer"]["state_dict"]["state"].__setitem__(0, {}), r"updates=.*missing or malformed state fields"),
            ):
                self._case("REV/C4/lazy/" + family + "/" + label,
                           lambda b=bundle, m=mutation, rx=pattern: self._negative(b, m, rx))
            for label, total in (("all_zero", 0), ("sum_G_minus_1", 3)):
                def check(b=bundle, amount=total):
                    def change(args):
                        optimizer = args[0]["optimizer"]
                        state0 = copy.deepcopy(optimizer["state_dict"]["state"][0])
                        optimizer["state_dict"]["state"] = {}
                        for binding in optimizer["parameter_bindings"]:
                            binding["updates"] = 0
                        if amount:
                            optimizer["parameter_bindings"][0]["updates"] = amount
                            state0["step"] = torch.tensor(float(amount), dtype=torch.float32)
                            optimizer["state_dict"]["state"][0] = state0
                    self._negative(b, change, r"Sum of parameter updates.*global_step")
                self._case("REV/C4/lazy/" + family + "/" + label, check)

    def test_revision_sampler_counter_and_history_fields(self):
        """C5/C6: quiescent exact integers, domain pairs and progress histories."""
        bundle = self._control()
        self._case("REV/C5/sampler/complete", lambda: self._positive(bundle))
        for field in ("batch_cursor", "pending_batches", "prefetch_pending", "inflight_workers"):
            for label, value in (("bool", False), ("nonzero", 1)):
                self._case("REV/C5/sampler/" + field + "/" + label,
                           lambda key=field, bad=value: self._negative(
                               bundle, lambda a: a[0]["sampler"].__setitem__(key, bad), r"sampler\." + key))
        cases = (
            ("cursor_bool", lambda a: a[0]["sampler"].__setitem__("cursor", True), r"sampler.cursor"),
            ("cursor_incomplete", lambda a: a[0]["sampler"].__setitem__("cursor", 0), r"Completed epoch requires cursor=N"),
            ("next_bool", lambda a: a[0]["sampler"].__setitem__("next_epoch", True), r"sampler.next_epoch"),
            ("next_wrong", lambda a: a[0]["sampler"].__setitem__("next_epoch", 2), r"next_epoch=epoch\+1"),
            ("pair_list", lambda a: a[0]["sampler"]["source_order"].__setitem__(0, list(a[0]["sampler"]["source_order"][0])), r"source_order\[0\].*2-tuple"),
            ("pair_arity", lambda a: a[0]["sampler"]["source_order"].__setitem__(0, ("u1",)), r"source_order\[0\].*2-tuple"),
            ("pair_leaf", lambda a: a[0]["sampler"]["source_order"].__setitem__(0, (True, "p1")), r"source_order\[0\].user_id"),
            ("pair_duplicate", lambda a: a[0]["sampler"]["source_order"].__setitem__(1, a[0]["sampler"]["source_order"][0]), r"valid permutation"),
            ("pair_missing", lambda a: a[0]["sampler"]["source_order"].pop(), r"source_order must be exact list of length N"),
            ("epoch_bool", lambda a: a[0]["progress"].__setitem__("epoch", True), r"progress.epoch|current_model.epoch"),
            ("global_bool", lambda a: a[0]["progress"].__setitem__("global_step", True), r"progress.global_step"),
            ("steps_type", lambda a: a[0]["progress"].__setitem__("epoch_optimizer_steps", (2, 2)), r"epoch_optimizer_steps.*exact list"),
            ("steps_length", lambda a: a[0]["progress"]["epoch_optimizer_steps"].pop(), r"epoch_optimizer_steps.*length"),
            ("steps_bool", lambda a: a[0]["progress"]["epoch_optimizer_steps"].__setitem__(0, True), r"epoch_optimizer_steps\[0\]"),
            ("steps_zero", lambda a: a[0]["progress"]["epoch_optimizer_steps"].__setitem__(0, 0), r"epoch_optimizer_steps\[0\]"),
            ("skips_type", lambda a: a[0]["progress"].__setitem__("empty_pool_skips", (0, 0)), r"empty_pool_skips.*exact list"),
            ("skips_length", lambda a: a[0]["progress"]["empty_pool_skips"].pop(), r"empty_pool_skips.*length"),
            ("skips_bool", lambda a: a[0]["progress"]["empty_pool_skips"].__setitem__(0, False), r"empty_pool_skips\[0\]"),
            ("skips_wrong", lambda a: a[0]["progress"]["empty_pool_skips"].__setitem__(0, 1), r"empty_pool_skips\[0\].*\|K\|"),
            ("sum", lambda a: a[0]["progress"].__setitem__("global_step", 5), r"Sum of epoch_optimizer_steps.*global_step"),
        )
        for label, mutation, pattern in cases:
            self._case("REV/C5/sampler/" + label, lambda m=mutation, rx=pattern: self._negative(bundle, m, rx))
        workers = self._control(runtime_workers=1)
        self._case("REV/C5/sampler/W1/complete", lambda: self._positive(workers))
        for label, value, pattern in (
            ("bool", [False], r"worker_ids\[0\].*exact int 0"),
            ("wrong", [1], r"worker_ids\[0\].*exact int 0"),
            ("missing", [], r"worker_ids length"),
        ):
            self._case("REV/C5/sampler/W1/" + label, lambda v=value, rx=pattern: self._negative(
                workers, lambda a: a[0]["sampler"].__setitem__("worker_ids", v), rx))

    def test_revision_five_pair_batch_policy_bounds(self):
        """C5/C6: unchanged five-edge identity, partial and empty chunks."""
        filtered = self._control(with_empty_pool=True)
        self._case("REV/C5/batching/filter/complete", lambda: self._positive(filtered))
        self.assertEqual(filtered[3]["buffer_state"]["adj_user_item"]._nnz(), 5)
        ordered = [("u1", "p1"), ("u1", "p2"), ("u3", "p1"), ("u3", "p2"), ("u2", "p3")]
        chunked = list(copy.deepcopy(filtered))
        chunked[0]["runtime_profile"]["batch_policy"] = "batch_then_filter"
        chunked[4]["runtime_profile"]["batch_policy"] = "batch_then_filter"
        chunked[0]["sampler"]["source_order"] = list(ordered)
        self._revision_steps(chunked, [2, 2])
        self._case("REV/C5/batching/chunked/empty_partial", lambda: self._positive(chunked))
        older_upper = list(copy.deepcopy(chunked))
        self._revision_steps(older_upper, [3, 2])
        self._case("REV/C5/batching/chunked/older_upper", lambda: self._positive(older_upper))
        latest_three = list(copy.deepcopy(chunked))
        latest_three[0]["sampler"]["source_order"] = [ordered[0], ordered[2], ordered[1], ordered[3], ordered[4]]
        self._revision_steps(latest_three, [2, 3])
        self._case("REV/C5/batching/chunked/latest_three", lambda: self._positive(latest_three))
        self.assertEqual(chunked[0]["progress"]["empty_pool_skips"], [2, 2])
        for label, control, steps, pattern in (
            ("filter_older_exact", filtered, [3, 2], r"epoch_optimizer_steps\[0\].*filter_before_batch"),
            ("filter_latest_exact", filtered, [2, 3], r"epoch_optimizer_steps\[1\].*filter_before_batch"),
            ("chunked_latest", chunked, [2, 3], r"Last epoch optimizer steps.*derived batch_then_filter"),
            ("chunked_older_low", chunked, [1, 2], r"epoch_optimizer_steps\[0\].*out of bounds"),
            ("chunked_older_high", chunked, [4, 2], r"epoch_optimizer_steps\[0\].*out of bounds"),
        ):
            self._case("REV/C5/batching/" + label, lambda b=control, s=steps, rx=pattern: self._negative(
                b, lambda a: self._revision_steps(a, s), rx))

    def test_revision_selection_leaves_and_terminal_corruptions(self):
        """C6: isolated leaf/history/summary and distinct terminal failures."""
        bundle = self._control(epoch=4, scores=[0.5, 0.5])
        self._case("REV/C6/selection/tie_control", lambda: self._positive(bundle))
        for label, value in (("bool", True), ("int", 0), ("low", -0.1), ("high", 1.1), ("nan", float("nan")), ("inf", float("inf"))):
            pattern = r"[Ff]inite" if label in ("nan", "inf") else r"selection.history\[0\].score"
            self._case("REV/C6/selection/score/" + label, lambda v=value, rx=pattern: self._negative(
                bundle, lambda a: a[0]["selection"]["history"][0].__setitem__("score", v), rx))
        cases = (
            ("history_duplicate", lambda a: a[0]["selection"]["history"][1].__setitem__("epoch", 2), r"history\[1\].epoch.*scheduled epoch"),
            ("history_missing", lambda a: a[0]["selection"]["history"].pop(), r"history length.*scheduled"),
            ("history_future", lambda a: a[0]["selection"]["history"][1].__setitem__("epoch", 6), r"history\[1\].epoch.*scheduled epoch"),
            ("history_unscheduled", lambda a: a[0]["selection"]["history"][1].__setitem__("epoch", 3), r"history\[1\].epoch.*scheduled epoch"),
            ("summary_score", lambda a: a[0]["selection"].__setitem__("best_score", 0.6), r"best_score.*recomputed"),
            ("selected_tie", lambda a: a[0]["selection"].__setitem__("selected_epoch", 4), r"selected_epoch.*recomputed"),
            ("selected_future", lambda a: a[0]["selection"].__setitem__("selected_epoch", 6), r"selected_epoch.*recomputed"),
            ("selected_unscheduled", lambda a: a[0]["selection"].__setitem__("selected_epoch", 3), r"selected_epoch.*recomputed"),
            ("bad_checks", lambda a: a[0]["selection"].__setitem__("bad_checks", 0), r"bad_checks.*recomputed"),
            ("bad_checks_bool", lambda a: a[0]["selection"].__setitem__("bad_checks", True), r"selection.bad_checks"),
            ("bad_checks_over", lambda a: a[0]["selection"].__setitem__("bad_checks", 11), r"selection.bad_checks"),
            ("status", lambda a: a[0]["selection"].__setitem__("status", "stopped"), r"selection.status"),
            ("stop_reason", lambda a: a[0]["selection"].__setitem__("stop_reason", "patience"), r"selection.stop_reason"),
        )
        for label, mutation, pattern in cases:
            self._case("REV/C6/selection/" + label, lambda m=mutation, rx=pattern: self._negative(bundle, m, rx))

        maximum = self._control(epoch=100, validation_frequency=3)
        self._case("REV/C6/terminal/F3/complete", lambda: self._positive(maximum))
        self.assertEqual(maximum[0]["selection"]["selected_epoch"], 99)
        for label, mutation, pattern in (
            ("ready", lambda a: a[0]["selection"].__setitem__("status", "ready"), r"selection.status"),
            ("reason", lambda a: a[0]["selection"].__setitem__("stop_reason", "patience"), r"selection.stop_reason"),
            ("best_current_epoch", lambda a: a[0]["best_model"].__setitem__("epoch", 100), r"best_model.epoch.*recomputed selected epoch"),
        ):
            self._case("REV/C6/terminal/F3/" + label, lambda m=mutation, rx=pattern: self._negative(maximum, m, rx))
        scores = [round(0.01 * (i + 1), 4) for i in range(40)] + [0.4] * 10
        combined = self._control(epoch=100, validation_frequency=2, scores=scores)
        self._case("REV/C6/terminal/combined/complete", lambda: self._positive(combined))
        self.assertEqual(combined[0]["selection"]["selected_epoch"], 80)
        self._case("REV/C6/terminal/combined/reason", lambda: self._negative(
            combined, lambda a: a[0]["selection"].__setitem__("stop_reason", "max_epochs"), r"selection.stop_reason"))
        patience = self._control(epoch=22, scores=[0.9] + [0.1] * 10)
        self._case("REV/C6/terminal/patience/complete", lambda: self._positive(patience))
        def post_stop(args):
            p = args[0]
            p["progress"]["epoch"] = 23
            p["progress"]["empty_pool_skips"] = [0] * 23
            p["current_model"]["epoch"] = 23
            p["sampler"]["next_epoch"] = 24
            self._revision_steps(args, [2] * 23)
        self._case("REV/C6/terminal/patience/post_stop", lambda: self._negative(
            patience, post_stop, r"Terminal patience stop occurred.*cannot exceed it"))

    def test_revision_both_models_best_epoch_bit_binding(self):
        """C3/C6: older equal/different finite weights versus same-epoch bits."""
        for family in ("light", "hetero"):
            same = list(self._control(family))
            same[0]["current_model"]["state_dict"]["user_embeddings"][0, 0] = 0.0
            same[0]["best_model"]["state_dict"]["user_embeddings"][0, 0] = 0.0
            self._case("REV/C6/best/" + family + "/same_epoch_control", lambda b=same: self._positive(b))
            self._case("REV/C6/best/" + family + "/signed_zero", lambda b=same: self._negative(
                b, lambda a: a[0]["best_model"]["state_dict"]["user_embeddings"].__setitem__((0, 0), -0.0),
                r"best_model and current_model share epoch.*contents differ"))
            older = list(copy.deepcopy(same))
            older[0]["current_model"]["epoch"] = 4
            older[0]["progress"]["epoch"] = 4
            older[0]["progress"]["empty_pool_skips"] = [0] * 4
            older[0]["sampler"]["next_epoch"] = 5
            self._revision_steps(older, [2] * 4)
            older[0]["selection"].update(
                history=[{"epoch": 2, "score": 0.5}, {"epoch": 4, "score": 0.5}],
                best_score=0.5, selected_epoch=2, bad_checks=1,
            )
            self._case("REV/C6/best/" + family + "/older_equal", lambda b=older: self._positive(b))
            different = list(copy.deepcopy(older))
            different[0]["best_model"]["state_dict"]["user_embeddings"].add_(0.125)
            self._case("REV/C6/best/" + family + "/older_different", lambda b=different: self._positive(b))

    def test_revision_runtime_fields_and_loaded_scalar_policy(self):
        """C2: public actual/expected bindings, including loaded float32 policy."""
        bundle = self._control()
        self._case("REV/C2/runtime/complete", lambda: self._positive(bundle))
        fields = {
            "profile_id": "revision_incompatible_profile",
            "python_version": "0.0.revision",
            "numpy_version": "0.0.revision",
            "torch_version": "0.0.revision",
            "adam_step_dtype": "torch.float64",
            "batch_policy": "unsupported_revision_policy",
        }
        for field, incompatible in fields.items():
            for side in ("actual", "expected", "both"):
                def check(key=field, bad=incompatible, which=side):
                    def change(args):
                        value = "" if key == "profile_id" and which == "both" else bad
                        if which in ("actual", "both"):
                            args[0]["runtime_profile"][key] = value
                        if which in ("expected", "both"):
                            args[4]["runtime_profile"][key] = value
                        if key == "adam_step_dtype" and which == "both":
                            for state in args[0]["optimizer"]["state_dict"]["state"].values():
                                state["step"] = state["step"].to(dtype=torch.float64)
                    self._negative(bundle, change, r"runtime_profile|provenance|adam_step_dtype|scalar|dtype")
                self._case("REV/C2/runtime/" + field + "/" + side, check)
        self.assertEqual(torch.get_default_dtype(), torch.float32)
        # This read-only assertion describes this pinned environment. No setter
        # is used to make the incompatible float64 profile pass or fail.

    def test_revision_public_provenance_routes_and_required_keywords(self):
        """C2: existing public D1 provenance routes and signature-only evidence."""
        bundle = self._control()
        self._case("REV/C2/provenance/complete", lambda: self._positive(bundle))
        original = bundle[2]
        source = {path: record["content_bytes"] for path, record in original["source_files"].items()}
        first = next(iter(source))
        alternatives = []
        changed_source = dict(source)
        changed_source[first] += b"\n# consolidated revision fixture\n"
        alternatives.append(("source", build_source_provenance(changed_source, copy.deepcopy(original["git"]), copy.deepcopy(original["environment"]))))
        environment = copy.deepcopy(original["environment"])
        environment["cpu_count"] += 1
        alternatives.append(("environment", build_source_provenance(source, copy.deepcopy(original["git"]), environment)))
        git = copy.deepcopy(original["git"])
        git["commit"] = ("0" if git["commit"] != "0" * 40 else "1") * 40
        alternatives.append(("git", build_source_provenance(source, git, copy.deepcopy(original["environment"]))))
        for label, alternate in alternatives:
            for side in ("actual", "expected"):
                def check(value=alternate, which=side):
                    changed = list(copy.deepcopy(bundle))
                    if which == "actual":
                        changed[0]["provenance"] = copy.deepcopy(value)
                    else:
                        changed[2] = copy.deepcopy(value)
                    with self.assertRaises((CheckpointValidationError, CheckpointProvenanceError)):
                        self._call(changed)
                self._case("REV/C2/provenance/mismatch/" + label + "/" + side, check)
        for label in ("source_bytes", "source_hash", "environment_type", "git_type"):
            for side in ("actual", "expected", "both"):
                def check(recipe=label, which=side):
                    changed = list(copy.deepcopy(bundle))
                    targets = []
                    if which in ("actual", "both"):
                        targets.append(changed[0]["provenance"])
                    if which in ("expected", "both"):
                        targets.append(changed[2])
                    for provenance in targets:
                        if recipe == "source_bytes":
                            provenance["source_files"][first]["content_bytes"] += b"!"
                        elif recipe == "source_hash":
                            old = provenance["source_files"][first]["sha256"]
                            provenance["source_files"][first]["sha256"] = ("0" if old != "0" * 64 else "1") * 64
                        elif recipe == "environment_type":
                            provenance["environment"]["cpu_count"] = True
                        else:
                            provenance["git"]["dirty"] = 1
                    with self.assertRaises((CheckpointValidationError, CheckpointProvenanceError)):
                        self._call(changed)
                self._case("REV/C2/provenance/malformed/" + label + "/" + side, check)

        kwargs = dict(
            expected_identity=bundle[1], expected_provenance=bundle[2],
            expected_model=bundle[3], expected_runtime=bundle[4], resource_policy=bundle[5],
        )
        for field in kwargs:
            def check(key=field):
                omitted = dict(kwargs)
                del omitted[key]
                with self.assertRaisesRegex(TypeError, key):
                    validate_checkpoint_payload(bundle[0], **omitted)
            # These are Python signature refusals, not successful data validation.
            self._case("REV/C2/signature/omitted/" + field, check)

    def _revision_shared_view_bundle(self):
        bundle = list(self._control())
        base = torch.arange(5 * 128, dtype=torch.float32).reshape(5, 128) / 1024.0
        users = base[:2, ::2]
        products = base[2:, ::2]
        self.assertEqual(tuple(users.shape), (2, 64))
        self.assertEqual(tuple(products.shape), (3, 64))
        self.assertFalse(users.is_contiguous())
        self.assertFalse(products.is_contiguous())
        self.assertEqual(users.untyped_storage().data_ptr(), products.untyped_storage().data_ptr())
        bundle[0]["current_model"]["state_dict"]["user_embeddings"] = users
        bundle[0]["current_model"]["state_dict"]["item_embeddings"] = products
        # The E2 snapshots are bit-identical; input container/tensor aliasing is
        # allowed. Every returned occurrence must nevertheless be independently owned.
        bundle[0]["best_model"] = bundle[0]["current_model"]
        return bundle, base

    def test_revision_genuine_shared_backing_owned_occurrences(self):
        """C7: real shared storage/noncontiguity and content-based separation."""
        for direction in ("actual", "expected", "returned", "current_to_best"):
            def check(which=direction):
                bundle, base = self._revision_shared_view_bundle()
                before_globals = self._global_snapshot()
                result = self._call(bundle)
                self._assert_globals(before_globals)
                self._same(result, bundle[0])
                self.assertTrue(result["current_model"]["state_dict"]["user_embeddings"].is_contiguous())
                input_snapshot = tuple(copy.deepcopy(value) for value in bundle)
                result_snapshot = copy.deepcopy(result)
                if which == "actual":
                    base.add_(1)
                    adj = bundle[0]["current_model"]["state_dict"]["adj_user_item"]
                    adj._indices().add_(1)
                    adj._values().add_(1)
                    bundle[0]["sampler"]["source_order"].clear()
                    bundle[0]["rng"]["roles"][0]["generator_ids"].append("later_actual_mutation")
                    self._same(result, result_snapshot)
                    for index in range(1, len(bundle)):
                        self._same(bundle[index], input_snapshot[index])
                elif which == "expected":
                    adj = bundle[3]["buffer_state"]["adj_user_item"]
                    adj._indices().add_(1)
                    adj._values().add_(1)
                    bundle[3]["row_order"]["User"].append("later_expected_mutation")
                    bundle[4]["rng_role_bindings"]["Python"].clear()
                    self._same(bundle[0], input_snapshot[0])
                    self._same(result, result_snapshot)
                elif which == "returned":
                    result["current_model"]["state_dict"]["user_embeddings"].add_(1)
                    adj = result["current_model"]["state_dict"]["adj_user_item"]
                    adj._indices().add_(1)
                    adj._values().add_(1)
                    result["sampler"]["source_order"].clear()
                    result["rng"]["roles"][0]["generator_ids"].append("later_result_mutation")
                    for actual, snapshot in zip(bundle, input_snapshot):
                        self._same(actual, snapshot)
                else:
                    best_snapshot = copy.deepcopy(result["best_model"])
                    current = result["current_model"]
                    current["state_dict"]["user_embeddings"].add_(1)
                    current["state_dict"]["adj_user_item"]._indices().add_(1)
                    current["state_dict"]["adj_user_item"]._values().add_(1)
                    current["row_order"]["User"].append("later_current_mutation")
                    self._same(result["best_model"], best_snapshot)
                    for actual, snapshot in zip(bundle, input_snapshot):
                        self._same(actual, snapshot)
            self._case("REV/C7/shared_backing/" + direction, check)

    def test_revision_early_failure_preserves_detached_inputs(self):
        """C7: early malformed envelope alongside retained late-failure tests."""
        bundle = list(self._control())
        self._case("REV/C7/early_failure/complete", lambda: self._positive(bundle))
        del bundle[0]["schema_version"]
        snapshot = tuple(copy.deepcopy(value) for value in bundle)
        before = self._global_snapshot()
        with self.assertRaisesRegex(CheckpointValidationError, r"payload.*keys|schema_version"):
            self._call(bundle)
        for actual, saved in zip(bundle, snapshot):
            self._same(actual, saved)
        self._assert_globals(before)


if __name__ == "__main__":
    unittest.main()
