"""EX01 trainer: Block 2 recipe on the vectorized models, on CPU or one CUDA device.

Science is the accepted recipe: mean softplus BPR plus the endpoint penalty, uniform
eligible negatives (one per positive, `filter_before_batch`), exact Adam options,
validation NDCG@10 selection with patience 10 counted in checks and 100 maximum epochs
(`training_harness.Selection`).

All randomness comes from private CPU generators. The global CPU and CUDA generators are
checked unchanged across every epoch, so no hidden device randomness can enter. State is
captured only at quiescent epoch boundaries and restores bit for bit in the same
environment.
"""

import copy
import hashlib

import torch

from scripts.gpu_runtime import RuntimeRefusal, global_rng_states, require_rng_unchanged
from scripts.training_harness import ADAM_OPTIONS, Selection, validation_score
from scripts.vectorized_evaluation import evaluate_model

STATE_SCHEMA = 'amazon4u-vector-trainer-state-v1'
RNG_ROLES = ('epoch_shuffle', 'negative_sampler')


def derived_seed(seed, role):
    digest = hashlib.sha256(f'amazon4u-ex01-v1:{seed}:{role}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') >> 1


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


class VectorTrainer:
    def __init__(self, model, targets, *, seed, batch_size, validation_frequency,
                 fingerprint, determinism=None, chunk_users=1024):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.model, self.targets = model, targets
        self.seed, self.batch_size = seed, batch_size
        self.fingerprint, self.determinism = copy.deepcopy(fingerprint), copy.deepcopy(determinism)
        self.chunk_users = chunk_users
        self.device = model.user_embeddings.device
        graph = model.graph
        context = graph.context
        users = graph.users
        items = graph.products if hasattr(graph, 'products') else graph.warm_items
        self.n_items = len(items)
        u_map = {u: i for i, u in enumerate(users)}
        i_map = {p: i for i, p in enumerate(items)}
        pairs = sorted((u_map[u], i_map[p]) for u in users
                       for p in context.positive_history.get(u, ()) if p in i_map)
        self.positives = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
        self.observed = model.supervision.observed.cpu()
        observed_users = self.observed // self.n_items
        self.empty_pool = torch.bincount(observed_users, minlength=len(users)) >= self.n_items
        self.optimizer = torch.optim.Adam(list(model.parameters()), **copy.deepcopy(ADAM_OPTIONS))
        self.generators = {r: torch.Generator(device='cpu').manual_seed(derived_seed(seed, r))
                           for r in RNG_ROLES}
        self.selection = Selection(validation_frequency)
        self.progress = {'epoch': 0, 'global_step': 0, 'epoch_optimizer_steps': [],
                         'empty_pool_skips': []}
        self.best_model = None
        self.validation_report = None
        self.poisoned = False

    def _negatives(self, users):
        gen = self.generators['negative_sampler']
        out = torch.empty_like(users)
        pending = torch.arange(users.numel())
        while pending.numel():
            draw = torch.randint(self.n_items, (pending.numel(),), generator=gen)
            ok = ~torch.isin(users[pending] * self.n_items + draw, self.observed)
            out[pending[ok]] = draw[ok]
            pending = pending[~ok]
        return out

    def sample_epoch(self):
        """Shuffled trainable positives with one eligible negative each, and the skip count."""
        order = self.positives[torch.randperm(len(self.positives),
                                              generator=self.generators['epoch_shuffle'])]
        keep = ~self.empty_pool[order[:, 0]]
        skips = int((~keep).sum())
        order = order[keep]
        if order.numel() == 0:
            raise RuntimeRefusal('no trainable examples remain')
        users, positives = order[:, 0], order[:, 1]
        return users, positives, self._negatives(users), skips

    def step(self, users, positives, negatives):
        """One Adam update on CPU index tensors; refuses nonfinite or missing gradients."""
        batch = [t.to(self.device) for t in (users, positives, negatives)]
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.model.compute_batch_loss(*batch)
        loss['total_loss'].backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeRefusal('missing or nonfinite gradients; optimizer not stepped')
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {k: float(v.detach()) for k, v in loss.items()}

    def _parameters(self):
        return {name: _cpu_copy(p) for name, p in self.model.named_parameters()}

    def train_epoch(self):
        if self.poisoned or self.selection.state['status'] == 'stopped':
            raise RuntimeRefusal('trajectory is failed or scientifically terminal')
        before = global_rng_states(self.device)
        try:
            users, positives, negatives, skips = self.sample_epoch()
            losses = []
            for start in range(0, len(users), self.batch_size):
                sl = slice(start, start + self.batch_size)
                losses.append(self.step(users[sl], positives[sl], negatives[sl]))
            epoch = self.progress['epoch'] + 1
            report = score = None
            if epoch % self.selection.frequency == 0:
                report = evaluate_model(self.model, self.targets, chunk_users=self.chunk_users)
                score = validation_score(report)
            improved = self.selection.advance(epoch, score)
            self.progress['epoch'] = epoch
            self.progress['global_step'] += len(losses)
            self.progress['epoch_optimizer_steps'].append(len(losses))
            self.progress['empty_pool_skips'].append(skips)
            if improved:
                self.best_model = {'epoch': epoch, 'parameters': self._parameters()}
            if report is not None:
                self.validation_report = report
            require_rng_unchanged(before, global_rng_states(self.device), epoch)
            return {'epoch': epoch, 'optimizer_steps': len(losses),
                    'empty_pool_skips': skips, 'losses': losses, 'score': score}
        except Exception:
            self.poisoned = True
            raise

    def capture_state(self):
        if self.poisoned:
            raise RuntimeRefusal('only successful quiescent epoch boundaries are checkpointable')
        return {
            'schema_version': STATE_SCHEMA,
            'seed': self.seed, 'batch_size': self.batch_size,
            'validation_frequency': self.selection.frequency,
            'fingerprint': copy.deepcopy(self.fingerprint),
            'determinism': copy.deepcopy(self.determinism),
            'parameters': self._parameters(),
            'best_model': _cpu_copy(self.best_model),
            'optimizer': _cpu_copy(self.optimizer.state_dict()),
            'generators': {r: g.get_state().clone() for r, g in self.generators.items()},
            'global_rng': _cpu_copy(global_rng_states(self.device)),
            'progress': copy.deepcopy(self.progress),
            'selection': copy.deepcopy(self.selection.state),
        }

    def load_state(self, state):
        """Load into a freshly constructed trainer; any binding mismatch is refused."""
        if self.progress['epoch'] != 0 or self.poisoned:
            raise RuntimeRefusal('restore only into a fresh trainer')
        if state.get('schema_version') != STATE_SCHEMA:
            raise RuntimeRefusal('unsupported trainer state schema')
        for key, mine in (('seed', self.seed), ('batch_size', self.batch_size),
                          ('validation_frequency', self.selection.frequency),
                          ('fingerprint', self.fingerprint), ('determinism', self.determinism)):
            if state[key] != mine:
                raise RuntimeRefusal(f'checkpoint {key} differs from this run; resume refused')
        names = [n for n, _ in self.model.named_parameters()]
        if list(state['parameters']) != names:
            raise RuntimeRefusal('checkpoint parameter inventory differs')
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                value = state['parameters'][name]
                if value.shape != p.shape or value.dtype != p.dtype:
                    raise RuntimeRefusal(f'checkpoint tensor {name} has the wrong shape or dtype')
                p.copy_(value.to(self.device))
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        for r in RNG_ROLES:
            self.generators[r].set_state(state['generators'][r].clone())
        # Never consumed by training; restored so the captured state round-trips exactly.
        if set(state['global_rng']) != set(global_rng_states(self.device)):
            raise RuntimeRefusal('checkpoint global RNG inventory differs from this device')
        torch.set_rng_state(state['global_rng']['torch_cpu'].clone())
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.clone() for s in state['global_rng']['torch_cuda']])
        self.progress = copy.deepcopy(state['progress'])
        self.selection = Selection(state['validation_frequency'], epoch=self.progress['epoch'],
                                   state=state['selection'])
        self.best_model = _cpu_copy(state['best_model'])
