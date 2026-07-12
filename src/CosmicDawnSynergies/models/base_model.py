import json
from os import path as osp

import jax
import optax
import orbax.checkpoint as ocp
from flax import nnx

from CosmicDawnSynergies.utils import get_root_logger


def build_schedule(train_opt):
    """Build an optax learning-rate schedule from the ``train:`` block.

    ``schedule.name`` resolves to ``optax.<name>_schedule``; without a
    ``schedule`` block the optimizer's ``learning_rate`` is a constant.
    ``warmup_iter > 0`` prepends a linear ramp from 0.
    """
    sched_opt = train_opt.get('schedule')
    if sched_opt:
        sched_opt = dict(sched_opt)
        name = sched_opt.pop('name')
        if name == 'piecewise_constant':
            # yaml gives {boundary: scale} with possibly-string keys
            bas = {int(k): float(v) for k, v in sched_opt.pop('boundaries_and_scales').items()}
            schedule = optax.piecewise_constant_schedule(sched_opt.pop('init_value'), bas)
        else:
            schedule = getattr(optax, f'{name}_schedule')(**sched_opt)
    else:
        lr = train_opt['optimizer'].get('learning_rate')
        if lr is None:
            raise ValueError("Provide train.schedule or train.optimizer.learning_rate")
        schedule = optax.constant_schedule(float(lr))

    warmup_iter = train_opt.get('warmup_iter', -1) or -1
    if warmup_iter > 0:
        warmup = optax.linear_schedule(0.0, float(schedule(warmup_iter)), warmup_iter)
        schedule = optax.join_schedules([warmup, schedule], [warmup_iter])
    return schedule


def build_tx(train_opt, schedule):
    """Build the optax gradient transformation chain.

    Order matches the torch loop: clip global grad norm first, then coupled L2
    (torch-Adam ``weight_decay`` semantics) unless the optimizer is decoupled
    (adamw takes its own weight_decay kwarg), then the optimizer itself.
    """
    optim_opt = dict(train_opt['optimizer'])
    name = optim_opt.pop('name')
    optim_opt.pop('learning_rate', None)  # schedule carries the lr

    txs = []
    grad_clip = train_opt.get('grad_clip_norm')
    if grad_clip:
        txs.append(optax.clip_by_global_norm(float(grad_clip)))

    weight_decay = optim_opt.pop('weight_decay', 0.0) or 0.0
    if weight_decay and name != 'adamw':
        txs.append(optax.add_decayed_weights(float(weight_decay)))
    elif weight_decay:
        optim_opt['weight_decay'] = float(weight_decay)

    txs.append(getattr(optax, name)(learning_rate=schedule, **optim_opt))
    return optax.chain(*txs)


class BaseModel:
    """Base model: optax/orbax plumbing shared by all Models."""

    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt.get('is_train', False)
        self.current_iter = 0
        self._ckptr = ocp.StandardCheckpointer()
        self._manager = None

    # ------------------------------------------------------------------ ckpt
    @property
    def ckpt_dir(self):
        return self.opt['path']['checkpoints']

    @property
    def manager(self):
        """Orbax CheckpointManager for step-numbered (resume) checkpoints."""
        if self._manager is None:
            options = ocp.CheckpointManagerOptions(
                max_to_keep=self.opt.get('logger', {}).get('max_checkpoints_to_keep', 3), create=True)
            self._manager = ocp.CheckpointManager(osp.abspath(osp.join(self.ckpt_dir, 'steps')), options=options)
        return self._manager

    def _save_tree(self, name, tree):
        self._ckptr.save(osp.abspath(osp.join(self.ckpt_dir, name)), tree, force=True)

    def _restore_tree(self, name, template_tree):
        template = jax.tree.map(ocp.utils.to_shape_dtype_struct, template_tree)
        return self._ckptr.restore(osp.abspath(osp.join(self.ckpt_dir, name)), template)

    def save_param_stats(self, param_stats, root=None):
        root = root or self.opt['path']['emulators_root']
        with open(osp.join(root, 'param_stats.json'), 'w') as f:
            json.dump(param_stats, f, indent=2)

    @staticmethod
    def load_param_stats(emulator_root):
        with open(osp.join(emulator_root, 'param_stats.json')) as f:
            return json.load(f)

    # ------------------------------------------------------------------ misc
    def print_network(self, net):
        logger = get_root_logger()
        n_params = sum(int(p.size) for p in jax.tree.leaves(nnx.to_pure_dict(nnx.state(net, nnx.Param))))
        logger.info(f'Network: {net.__class__.__name__}, with parameters: {n_params:,d}')
        logger.info(str(net))

    def get_current_log(self):
        return self.log_dict
