"""EX01: device-aware vectorized execution of the accepted learned models.

The accepted models in `differentiable_models.py` stay the correctness oracle. These
classes compute the same mathematics with edge-list tensors instead of per-node Python
loops, so they can run at full category scale on CPU or CUDA:

- `VectorLightGCN`: Option B exact propagation on the fixed condition-positive graph.
- `VectorHeteroGNN`: the frozen two-layer heterogeneous encoder with full-neighborhood
  masked means. Batch positive-edge masking removes forward and reverse target edges
  before aggregation at both layers.

Parameter names and shapes equal the accepted models, so state dicts move between them.
Aggregation uses `index_select` and `index_add`, which have deterministic CUDA
implementations under `torch.use_deterministic_algorithms(True)`. Sparse matrix
products are avoided for that reason. See docs/gpu-ready-design.md.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.differentiable_models import (
    NODE_TYPES, RELATION_SCHEMAS, DifferentiableHeteroGNN, HeteroTrainingGraph,
)
from scripts.learned_models import LightGCNGraph
from scripts.recommendation_evaluation import TrainingContext

REG_COEFFICIENT = 1e-5
INCOMING = {t: [r for r, (_, dst) in RELATION_SCHEMAS.items() if dst == t] for t in NODE_TYPES}


class SupervisionIndex:
    """Positive and observed (user, item) keys used to fail closed on bad batches."""

    def __init__(self, context: TrainingContext, users, items, device):
        self.n_items = len(items)
        u_idx = {u: i for i, u in enumerate(users)}
        i_idx = {p: i for i, p in enumerate(items)}

        def keys(history):
            flat = sorted(u_idx[u] * self.n_items + i_idx[p]
                          for u, ps in history.items() if u in u_idx for p in ps if p in i_idx)
            return torch.tensor(flat, dtype=torch.long, device=device)
        self.positive = keys(context.positive_history)
        self.observed = keys(context.all_rated_history)

    def check(self, users, positives, negatives):
        if users.ndim != 1 or users.numel() == 0:
            raise ValueError('Cannot score an empty batch')
        if not (users.shape == positives.shape == negatives.shape):
            raise ValueError('Batch tensors must have identical shapes')
        if not torch.isin(users * self.n_items + positives, self.positive).all():
            raise ValueError('Batch positive is not a condition-positive training interaction')
        if torch.isin(users * self.n_items + negatives, self.observed).any():
            raise ValueError('Batch negative is in the user training observed history')


def bpr_with_endpoint_penalty(s_pos, s_neg, user_rows, item_rows, users, positives, negatives):
    """Mean softplus BPR plus 1e-5 times the mean squared norm of unique base endpoints."""
    diff = s_neg - s_pos
    if not torch.isfinite(diff).all():
        raise ValueError('Score differences s_neg - s_pos contain non-finite values (overflow)')
    ranking_loss = F.softplus(diff).mean()
    unique_users = torch.unique(users)
    unique_items = torch.unique(torch.cat([positives, negatives]))
    n_endpoints = unique_users.numel() + unique_items.numel()
    sq_u = (user_rows.index_select(0, unique_users) ** 2).sum()
    sq_i = (item_rows.index_select(0, unique_items) ** 2).sum()
    if not torch.isfinite(sq_u) or not torch.isfinite(sq_i):
        raise ValueError('Squared base norm overflow in endpoint regularization')
    reg_loss = (REG_COEFFICIENT / float(n_endpoints)) * (sq_u + sq_i)
    return {'total_loss': ranking_loss + reg_loss, 'ranking_loss': ranking_loss,
            'reg_loss': reg_loss}


def _check_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{name} contains non-finite values (NaN or Inf)')


class VectorLightGCN(nn.Module):
    """Option B LightGCN: 3 layers, mean of layers 0..3, symmetric degree weights."""

    def __init__(self, graph: LightGCNGraph, user_embeddings: torch.Tensor,
                 item_embeddings: torch.Tensor, device):
        super().__init__()
        if graph.config.embedding_dimension != 64 or graph.config.propagation_layers != 3:
            raise ValueError('LightGCN must be 64d with 3 propagation layers')
        self.graph = graph
        self.layers = graph.config.propagation_layers
        n_users, n_items = len(graph.users), len(graph.warm_items)
        for name, value, rows in (('user_embeddings', user_embeddings, n_users),
                                  ('item_embeddings', item_embeddings, n_items)):
            if value.dtype != torch.float32 or tuple(value.shape) != (rows, 64):
                raise ValueError(f'{name} must be float32 with shape ({rows}, 64)')
            _check_finite(value, name)
        self.user_embeddings = nn.Parameter(user_embeddings.detach().to(device).clone())
        self.item_embeddings = nn.Parameter(item_embeddings.detach().to(device).clone())
        u_map = {u: i for i, u in enumerate(graph.users)}
        i_map = {p: i for i, p in enumerate(graph.warm_items)}
        rows, cols, weights = [], [], []
        for u, nbrs in graph.user_neighbors.items():
            for item, weight in nbrs:
                rows.append(u_map[u])
                cols.append(i_map[item])
                weights.append(float(weight))
        self.register_buffer('edge_users', torch.tensor(rows, dtype=torch.long, device=device))
        self.register_buffer('edge_items', torch.tensor(cols, dtype=torch.long, device=device))
        self.register_buffer('edge_weights',
                             torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1))
        self.supervision = SupervisionIndex(graph.context, graph.users, graph.warm_items, device)

    def accepted_state(self):
        return {'user_embeddings': self.user_embeddings, 'item_embeddings': self.item_embeddings}

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        u, i = self.user_embeddings, self.item_embeddings
        u_layers, i_layers = [u], [i]
        for _ in range(self.layers):
            msg_u = i_layers[-1].index_select(0, self.edge_items) * self.edge_weights
            msg_i = u_layers[-1].index_select(0, self.edge_users) * self.edge_weights
            u_layers.append(torch.zeros_like(u).index_add(0, self.edge_users, msg_u))
            i_layers.append(torch.zeros_like(i).index_add(0, self.edge_items, msg_i))
        prop_u = torch.stack(u_layers).mean(dim=0)
        prop_i = torch.stack(i_layers).mean(dim=0)
        _check_finite(prop_u, 'propagated_users')
        _check_finite(prop_i, 'propagated_items')
        return prop_u, prop_i

    def representations(self):
        return self.forward()

    def compute_batch_loss(self, users, positives, negatives):
        self.supervision.check(users, positives, negatives)
        prop_u, prop_i = self.forward()
        u = prop_u.index_select(0, users)
        s_pos = (u * prop_i.index_select(0, positives)).sum(dim=-1)
        s_neg = (u * prop_i.index_select(0, negatives)).sum(dim=-1)
        return bpr_with_endpoint_penalty(s_pos, s_neg, self.user_embeddings,
                                         self.item_embeddings, users, positives, negatives)


class VectorHeteroGNN(nn.Module):
    """Frozen two-layer heterogeneous encoder with full-neighborhood masked means."""

    def __init__(self, graph: HeteroTrainingGraph, init_seed: int, device,
                 state: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.graph = graph
        self.n_products = len(graph.products)
        # Reuse the accepted constructor so initialization draws are identical.
        reference = DifferentiableHeteroGNN(graph, init_seed=init_seed, embedding_dim=64)
        if state is not None:
            reference.load_state_dict(state, strict=True)
        self.user_embeddings = nn.Parameter(reference.user_embeddings.detach().to(device).clone())
        self.product_embeddings = nn.Parameter(reference.product_embeddings.detach().to(device).clone())
        self.category_embeddings = nn.Parameter(reference.category_embeddings.detach().to(device).clone())
        self.brand_embeddings = nn.Parameter(reference.brand_embeddings.detach().to(device).clone())
        for group in ('relation_weights_l0', 'relation_weights_l1', 'self_weights_l0',
                      'self_biases_l0', 'self_weights_l1', 'self_biases_l1'):
            # Item assignment keeps the accepted order; a dict argument would be sorted,
            # which reorders parameters and breaks Adam state alignment.
            target = nn.ParameterDict()
            for k, v in getattr(reference, group).items():
                target[k] = nn.Parameter(v.detach().to(device).clone())
            setattr(self, group, target)
        for name, value in self.named_parameters():
            _check_finite(value.detach(), name)
        self.counts = {t: len(graph.node_ids[t]) for t in NODE_TYPES}
        self.edges = {}
        for rel, (src_t, dst_t) in RELATION_SCHEMAS.items():
            s_map, d_map = graph.node_to_idx[src_t], graph.node_to_idx[dst_t]
            pairs = sorted((s_map[s], d_map[d]) for s, d in graph.edges[rel])
            src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
            dst = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
            self.edges[rel] = (src, dst)
        u_src, p_dst = self.edges['rated']
        p_src, u_dst = self.edges['rated_by']
        self.rated_keys = u_src * self.n_products + p_dst
        self.rated_by_keys = u_dst * self.n_products + p_src
        self.supervision = SupervisionIndex(graph.context, graph.users, graph.products, device)

    def _embedding(self, ntype):
        return {'User': self.user_embeddings, 'Product': self.product_embeddings,
                'CategoryPath': self.category_embeddings, 'Brand': self.brand_embeddings}[ntype]

    def _masked_edges(self, mask_keys):
        if mask_keys is None:
            return self.edges
        edges = dict(self.edges)
        for rel, keys in (('rated', self.rated_keys), ('rated_by', self.rated_by_keys)):
            keep = ~torch.isin(keys, mask_keys)
            src, dst = self.edges[rel]
            edges[rel] = (src[keep], dst[keep])
        return edges

    def forward(self, mask_keys: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        edges = self._masked_edges(mask_keys)
        device = self.user_embeddings.device
        stats = {}
        for rel, (src, dst) in edges.items():
            n_dst = self.counts[RELATION_SCHEMAS[rel][1]]
            count = torch.bincount(dst, minlength=n_dst).to(torch.float32)
            stats[rel] = (count.clamp(min=1.0).unsqueeze(1), (count > 0).to(torch.float32).unsqueeze(1))
        active = {t: sum((stats[r][1] for r in INCOMING[t]),
                         torch.zeros(self.counts[t], 1, device=device)) for t in NODE_TYPES}
        h = {t: self._embedding(t) for t in NODE_TYPES}
        for layer, relu in ((0, True), (1, False)):
            rel_w = getattr(self, f'relation_weights_l{layer}')
            self_w = getattr(self, f'self_weights_l{layer}')
            self_b = getattr(self, f'self_biases_l{layer}')
            nxt = {}
            for t in NODE_TYPES:
                z = F.linear(h[t], self_w[t], self_b[t])
                messages = torch.zeros_like(z)
                for rel in INCOMING[t]:
                    src, dst = edges[rel]
                    src_t = RELATION_SCHEMAS[rel][0]
                    total = torch.zeros_like(z).index_add(0, dst, h[src_t].index_select(0, src))
                    denom, is_active = stats[rel]
                    messages = messages + F.linear(total / denom, rel_w[rel]) * is_active
                z = z + messages / active[t].clamp(min=1.0)
                nxt[t] = F.relu(z) if relu else z
            h = nxt
        for t in NODE_TYPES:
            _check_finite(h[t], f'forward representation for {t}')
        return h

    def representations(self):
        h = self.forward(None)
        return h['User'], h['Product']

    def compute_batch_loss(self, users, positives, negatives):
        self.supervision.check(users, positives, negatives)
        mask_keys = torch.unique(users * self.n_products + positives)
        h = self.forward(mask_keys)
        u = h['User'].index_select(0, users)
        s_pos = (u * h['Product'].index_select(0, positives)).sum(dim=-1)
        s_neg = (u * h['Product'].index_select(0, negatives)).sum(dim=-1)
        return bpr_with_endpoint_penalty(s_pos, s_neg, self.user_embeddings,
                                         self.product_embeddings, users, positives, negatives)
