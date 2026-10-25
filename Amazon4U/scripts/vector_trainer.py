"""EX01 trainer: Block 2 recipe on the vectorized models, on CPU or one CUDA device.

Science is the accepted recipe: mean softplus BPR plus the endpoint penalty, uniform
eligible negatives (one per positive, `filter_before_batch`), exact Adam options,
validation NDCG@10 selection with patience 10 counted in checks and 100 maximum epochs
(`training_harness.Selection`).

Randomness goes through H-REF's own `PrivateRNG`, with the generators the run's
`expected_runtime` declares. The epoch order is `PrivateRNG.shuffle` of the sorted
positive pairs, and each negative is the `index(len(pool))`-th item of the accepted
eligible pool, found without building the pool. Given the same generators, the trainer
draws exactly the triples H-REF's `NegativeSampler.prepare_batch` would.

The global CPU and CUDA generators are checked unchanged across every epoch, so no
hidden device randomness can enter. Checkpoints are D2 `amazon4u-checkpoint-payload-v1`
shaped cores (`payload_core`), captured only at quiescent epoch boundaries; see
`checkpoint_payload_cuda.py` for the device profile around them.
"""

import bisect
import copy
from types import SimpleNamespace

import torch

from scripts.checkpoint_payload_cuda import build_cuda_payload
from scripts.checkpoint_v2 import PAYLOAD_SCHEMA_VERSION_V2, build_cuda_payload_v2
from scripts.gpu_runtime import RuntimeRefusal, global_rng_states, require_rng_unchanged
from scripts.training_harness import ADAM_OPTIONS, PrivateRNG, Selection, validation_score
from scripts.vectorized_evaluation import evaluate_model


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def fixture_rng(seed):
    """PrivateRNG for synthetic fixtures without a D1 identity.

    Uses the same codecs as the D2 fixture runtime: Python MT19937 for the epoch shuffle
    and NumPy RandomState for negatives.
    """
    streams = [{'name': role, 'applicable': True, 'algorithm': alg, 'seed': seed,
                'derivation': 'fixed_seed', 'reason': None}
               for role, alg in (('epoch_shuffle', 'stdlib_random'),
                                 ('negative_sampler', 'numpy_mt19937'))]
    runtime = {
        'rng_role_bindings': {'epoch_shuffle': ['shuffle'], 'negative_sampler': ['negatives']},
        'rng_generator_bindings': {
            'shuffle': {'codec': 'python.random.MT19937.v3', 'worker_id': None, 'seed_material': seed},
            'negatives': {'codec': 'numpy.random.RandomState.MT19937.v1', 'worker_id': None,
                          'seed_material': seed},
        },
    }
    identity = SimpleNamespace(resolved_settings={'determinism': {'rng_streams': streams}})
    return PrivateRNG(identity, runtime)


class VectorTrainer:
    def __init__(self, model, targets, *, rng, batch_size, validation_frequency,
                 fingerprint, determinism=None, chunk_users=1024):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.model, self.targets, self.rng = model, targets, rng
        self.batch_size = batch_size
        self.fingerprint, self.determinism = copy.deepcopy(fingerprint), copy.deepcopy(determinism)
        self.chunk_users = chunk_users
        self.device = model.user_embeddings.device
        graph = model.graph
        context = graph.context
        self.users = list(graph.users)
        self.items = list(graph.products if hasattr(graph, 'products') else graph.warm_items)
        u_map = {u: i for i, u in enumerate(self.users)}
        i_map = {p: i for i, p in enumerate(self.items)}
        # Sorted string pairs, as TrainingData.positive_pairs; index order equals string order.
        self.pairs = sorted((u_map[u], i_map[p]) for u in self.users
                            for p in context.positive_history.get(u, ()))
        self.observed = [sorted(i_map[p] for p in context.all_rated_history.get(u, ()) if p in i_map)
                         for u in self.users]
        self.shuffle_id = rng.role_id('epoch_shuffle')
        self.negative_id = rng.role_id('negative_sampler')
        self.optimizer = torch.optim.Adam(list(model.parameters()), **copy.deepcopy(ADAM_OPTIONS))
        self.selection = Selection(validation_frequency)
        self.progress = {'epoch': 0, 'global_step': 0, 'epoch_optimizer_steps': [],
                         'empty_pool_skips': []}
        self.source_order = None
        self.best_model = None
        self.validation_report = None
        self.poisoned = False

    def _negative(self, user):
        """The j-th item of sorted(warm - observed), as NegativeSampler draws pool[j]."""
        observed = self.observed[user]
        size = len(self.items) - len(observed)
        if size == 0:
            return None
        target = self.rng.index(self.negative_id, size)
        lo, hi = target, target + len(observed)
        while lo < hi:
            mid = (lo + hi) // 2
            if mid - bisect.bisect_right(observed, mid) < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def sample_epoch(self):
        """Shuffled positives, one eligible negative each, the skip count and the order."""
        order = list(self.pairs)
        self.rng.shuffle(order)
        triples, skips = [], 0
        for user, positive in order:
            negative = self._negative(user)
            if negative is None:
                skips += 1
            else:
                triples.append((user, positive, negative))
        if not triples:
            raise RuntimeRefusal('no trainable examples remain')
        users, positives, negatives = (torch.tensor(column, dtype=torch.long)
                                       for column in zip(*triples))
        return users, positives, negatives, skips, order

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
            users, positives, negatives, skips, order = self.sample_epoch()
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
            self.source_order = order
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

    def _model_record(self, epoch, parameters, expectations):
        state = dict(parameters)
        state.update(_cpu_copy(expectations['expected_model']['buffer_state']))
        return {'epoch': epoch, 'row_order': copy.deepcopy(expectations['expected_model']['row_order']),
                'state_dict': state}

    def payload_core(self, expectations, identity):
        """The D2 v1-shaped core of this epoch boundary, built as H-REF builds it."""
        if self.poisoned:
            raise RuntimeRefusal('only successful quiescent epoch boundaries are checkpointable')
        runtime = expectations['expected_runtime']
        epoch = self.progress['epoch']
        state = _cpu_copy(self.optimizer.state_dict())
        names = [n for n, _ in self.model.named_parameters()]
        ids = [pid for group in state['param_groups'] for pid in group['params']]
        bindings = [{'id': pid, 'name': name,
                     'updates': int(state['state'][pid]['step'].item()) if pid in state['state'] else 0}
                    for pid, name in zip(ids, names)]
        best = None
        if self.best_model is not None:
            best = self._model_record(self.best_model['epoch'], self.best_model['parameters'],
                                      expectations)
        order = [(self.users[u], self.items[p]) for u, p in self.source_order] if epoch else None
        return {
            'schema_version': 'amazon4u-checkpoint-payload-v1',
            'capability': 'epoch_boundary_validation_only',
            'identity': copy.deepcopy(identity),
            'provenance': copy.deepcopy(expectations['expected_provenance']),
            'runtime_profile': copy.deepcopy(runtime['runtime_profile']),
            'current_model': self._model_record(epoch, self._parameters(), expectations),
            'best_model': best,
            'optimizer': {'algorithm': 'Adam', 'state_dict': state, 'parameter_bindings': bindings},
            'rng': self.rng.capture(),
            'sampler': {
                'algorithm': runtime['sampler_algorithm'],
                'identity_hash': identity['identity_hash'],
                'source_order': order, 'cursor': len(order) if epoch else 0,
                'next_epoch': epoch + 1, 'batch_cursor': 0, 'pending_batches': 0,
                'prefetch_pending': 0, 'inflight_workers': 0, 'worker_ids': [],
            },
            'progress': copy.deepcopy(self.progress),
            'selection': copy.deepcopy(self.selection.state),
            'scheduler': {'applicable': False, 'state': None,
                          'reason': 'not_used_constant_learning_rate'},
            'amp': {'applicable': False, 'state': None, 'reason': 'not_used_cpu_float32'},
        }

    def restore_core(self, core):
        """Load a validated core into this fresh trainer. Validation is the caller's job."""
        if self.progress['epoch'] != 0 or self.poisoned:
            raise RuntimeRefusal('restore only into a fresh trainer')
        state = core['current_model']['state_dict']
        names = [n for n, _ in self.model.named_parameters()]
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(state[name].to(self.device))
        self.optimizer.load_state_dict(copy.deepcopy(core['optimizer']['state_dict']))
        self.rng.load_validated(core['rng'])
        self.progress = copy.deepcopy(core['progress'])
        self.selection = Selection(self.selection.frequency, epoch=self.progress['epoch'],
                                   state=core['selection'])
        u_map = {u: i for i, u in enumerate(self.users)}
        i_map = {p: i for i, p in enumerate(self.items)}
        order = core['sampler']['source_order']
        self.source_order = None if order is None else [(u_map[u], i_map[p]) for u, p in order]
        best = core['best_model']
        self.best_model = None if best is None else {
            'epoch': best['epoch'],
            'parameters': {n: best['state_dict'][n].clone() for n in names}}

    def _device_record(self, run_binding):
        return {'fingerprint': copy.deepcopy(self.fingerprint),
                'determinism': copy.deepcopy(self.determinism),
                'global_rng': _cpu_copy(global_rng_states(self.device)),
                'run_binding': copy.deepcopy(run_binding)}

    def cuda_payload(self, expectations, identity, run_binding):
        """The cuda-v1 checkpoint of this epoch boundary (validate before publishing)."""
        return build_cuda_payload(self.payload_core(expectations, identity),
                                  device_record=self._device_record(run_binding))

    def cuda_payload_v2(self, expectations, reference, run_binding):
        """The cuda-v2 checkpoint: the v1 core with the identity reference in place of the
        identity (docs/checkpoint-identity-v2-plan.md)."""
        core = self.payload_core(expectations, reference)
        core['schema_version'] = PAYLOAD_SCHEMA_VERSION_V2
        return build_cuda_payload_v2(core, device_record=self._device_record(run_binding))

    def restore_cuda(self, owned):
        """Restore a validated cuda-v1 payload into this fresh trainer.

        No global generator is set. Training never consumes them, so a matching process
        must already hold the recorded states; any difference refuses the resume. A new
        process gets them by calling `gpu_runtime.pin_global_generators()` first.
        """
        device = owned['device']
        if device['fingerprint'] != self.fingerprint or device['determinism'] != self.determinism:
            raise RuntimeRefusal('checkpoint environment differs from this run; resume refused')
        try:
            require_rng_unchanged(device['global_rng'], _cpu_copy(global_rng_states(self.device)),
                                  'restore')
        except RuntimeRefusal as exc:
            raise RuntimeRefusal(f'process global generators differ from the checkpoint: {exc}') from exc
        self.restore_core(owned)
