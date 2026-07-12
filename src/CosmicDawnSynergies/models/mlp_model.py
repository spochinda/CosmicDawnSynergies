from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from CosmicDawnSynergies.archs import build_arch
from CosmicDawnSynergies.models.base_model import BaseModel, build_schedule, build_tx
from CosmicDawnSynergies.utils import get_root_logger
from CosmicDawnSynergies.utils.registry import MODEL_REGISTRY

_LOSSES = {
    'mse': lambda pred, y: jnp.mean((pred - y) ** 2),
    'mae': lambda pred, y: jnp.mean(jnp.abs(pred - y)),
    'huber': lambda pred, y: jnp.mean(optax.losses.huber_loss(pred, y)),
}


@MODEL_REGISTRY.register()
class MLPModel(BaseModel):
    """MLP emulator Model: jitted optax training, EMA, validation, orbax ckpts."""

    def __init__(self, opt, in_dim=None):
        super().__init__(opt)

        arch_opt = dict(opt['arch'])
        if in_dim is not None:
            arch_opt['in_dim'] = in_dim
        seed = opt.get('manual_seed', 0)
        self.net_g = build_arch(arch_opt, rngs=nnx.Rngs(seed))
        self.print_network(self.net_g)

        loss_name = opt.get('train', {}).get('loss', 'mse')
        if loss_name not in _LOSSES:
            raise NotImplementedError(f"Loss '{loss_name}' not implemented. Use {list(_LOSSES)}.")
        self._loss = _LOSSES[loss_name]

        self.log_dict = OrderedDict()

        if self.is_train:
            self.init_training_settings()
        self._build_jitted_fns()

    # ---------------------------------------------------------------- setup
    def init_training_settings(self):
        self.best_metric_results = {
            'rmse': {'val': float('inf'), 'iter': -1},
            'nrmse': {'val': float('inf'), 'iter': -1},
            'r2': {'val': -float('inf'), 'iter': -1},
        }

        train_opt = self.opt['train']
        self.schedule = build_schedule(train_opt)
        tx = build_tx(train_opt, self.schedule)
        self.optimizer = nnx.Optimizer(self.net_g, tx, wrt=nnx.Param)

        self.ema_decay = train_opt.get('ema_decay', 0) or 0
        if self.ema_decay > 0:
            get_root_logger().info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_arch(dict(self.opt['arch'], in_dim=self.net_g.in_dim),
                                        rngs=nnx.Rngs(self.opt.get('manual_seed', 0)))
            # start EMA from net_g weights
            nnx.update(self.net_g_ema, nnx.state(self.net_g, nnx.Param))

    def _build_jitted_fns(self):
        loss_fn_ = self._loss

        @nnx.jit
        def train_step(net, optimizer, x, y):
            def lf(m):
                return loss_fn_(m(x), y)
            loss, grads = nnx.value_and_grad(lf)(net)
            optimizer.update(net, grads)
            return loss

        @nnx.jit
        def ema_step(net, ema_net, decay):
            params = nnx.state(net, nnx.Param)
            ema_params = nnx.state(ema_net, nnx.Param)
            new_ema = jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params)
            nnx.update(ema_net, new_ema)

        @nnx.jit
        def forward(net, x):
            return net(x)

        self._train_step = train_step
        self._ema_step = ema_step
        self._forward = forward

    # ------------------------------------------------------------- training
    def optimize_parameters(self, batch, current_iter=None):
        x, y = batch
        loss = self._train_step(self.net_g, self.optimizer, x, y)
        if self.ema_decay > 0:
            self._ema_step(self.net_g, self.net_g_ema, self.ema_decay)
        if current_iter is not None:
            self.current_iter = current_iter

        loss = float(jnp.sqrt(loss))
        self.log_dict['loss'] = 10 ** loss if self.opt['dataset']['targets_opt'].get('log', False) else loss

    def get_current_learning_rate(self):
        return [float(self.schedule(self.current_iter))]

    # ----------------------------------------------------------- validation
    def validation(self, val_data, current_iter, tb_logger=None):
        """Full-set validation with RMSE, NRMSE and R² (legacy metric semantics)."""
        all_inputs, all_targets = val_data
        all_outputs = self._forward(self.net_g, all_inputs)

        mse = jnp.mean((all_outputs - all_targets) ** 2)
        rmse = jnp.sqrt(mse)
        nrmse = rmse / jnp.std(all_targets)
        r2 = 1 - (mse / jnp.var(all_targets))

        # rmse on original scale if log transform was applied (not exact due to
        # averaging with offset in logspace)
        rmse_scaled = 10 ** rmse if self.opt['dataset']['targets_opt'].get('log', False) else rmse
        rmse_scaled = float(rmse_scaled - self.opt['dataset']['targets_opt'].get('offset', 0))
        nrmse, r2 = float(nrmse), float(r2)

        is_new_best = self.best_metric_results['rmse']['val'] > rmse_scaled
        if is_new_best:
            self.best_metric_results['rmse'] = {'val': rmse_scaled, 'iter': current_iter}
        if self.best_metric_results['nrmse']['val'] > nrmse:
            self.best_metric_results['nrmse'] = {'val': nrmse, 'iter': current_iter}
        if self.best_metric_results['r2']['val'] < r2:
            self.best_metric_results['r2'] = {'val': r2, 'iter': current_iter}

        best = self.best_metric_results
        get_root_logger().info(
            f'Validation: RMSE={rmse_scaled:.4f} (Best: {best["rmse"]["val"]:.4f}, iter {best["rmse"]["iter"]})'
            f'| NRMSE={nrmse:.4f} (Best: {best["nrmse"]["val"]:.4f}, iter {best["nrmse"]["iter"]})'
            f'| R²={r2:.4f} (Best: {best["r2"]["val"]:.4f}, iter {best["r2"]["iter"]})')

        # best-val checkpoint is the canonical emulator (legacy net_g_latest semantics)
        if is_new_best and self.is_train:
            self.save_best()

        if tb_logger:
            tb_logger.add_scalar('validation/rmse', rmse_scaled, current_iter)
            tb_logger.add_scalar('validation/nrmse', nrmse, current_iter)
            tb_logger.add_scalar('validation/r2', r2, current_iter)

        return {'rmse': rmse_scaled, 'nrmse': nrmse, 'r2': r2}

    # -------------------------------------------------------- save / restore
    def _params_tree(self):
        tree = {'params': nnx.to_pure_dict(nnx.state(self.net_g, nnx.Param))}
        if getattr(self, 'net_g_ema', None) is not None:
            tree['ema'] = nnx.to_pure_dict(nnx.state(self.net_g_ema, nnx.Param))
        return tree

    def save_best(self):
        self._save_tree('best', self._params_tree())

    def save(self, epoch, current_iter):
        """Step-numbered checkpoint (weights + optimizer state) for resuming."""
        tree = self._params_tree()
        tree['opt_state'] = nnx.to_pure_dict(nnx.state(self.optimizer))
        tree['epoch'] = jnp.asarray(epoch)
        tree['iter'] = jnp.asarray(current_iter)
        tree['best'] = {k: {'val': jnp.asarray(v['val']), 'iter': jnp.asarray(v['iter'])}
                        for k, v in self.best_metric_results.items()}
        self.manager.save(max(current_iter, 0), args=ocp.args.StandardSave(tree))
        self.manager.wait_until_finished()

    def resume_training(self):
        """Restore the latest step checkpoint; returns (epoch, iter) or None."""
        step = self.manager.latest_step()
        if step is None:
            return None
        tree = self._params_tree()
        tree['opt_state'] = nnx.to_pure_dict(nnx.state(self.optimizer))
        tree['epoch'] = jnp.asarray(0)
        tree['iter'] = jnp.asarray(0)
        tree['best'] = {k: {'val': jnp.asarray(v['val']), 'iter': jnp.asarray(v['iter'])}
                        for k, v in self.best_metric_results.items()}
        template = jax.tree.map(ocp.utils.to_shape_dtype_struct, tree)
        restored = self.manager.restore(step, args=ocp.args.StandardRestore(template))

        state = nnx.state(self.net_g, nnx.Param)
        nnx.replace_by_pure_dict(state, restored['params'])
        nnx.update(self.net_g, state)
        if 'ema' in restored:
            ema_state = nnx.state(self.net_g_ema, nnx.Param)
            nnx.replace_by_pure_dict(ema_state, restored['ema'])
            nnx.update(self.net_g_ema, ema_state)
        opt_state = nnx.state(self.optimizer)
        nnx.replace_by_pure_dict(opt_state, restored['opt_state'])
        nnx.update(self.optimizer, opt_state)
        self.best_metric_results = {k: {'val': float(v['val']), 'iter': int(v['iter'])}
                                    for k, v in restored['best'].items()}
        return int(restored['epoch']), int(restored['iter'])

    def load_network(self, which='best', use_ema=False):
        """Load weights from ``checkpoints/<which>`` into net_g (inference path)."""
        tree = self._params_tree()
        restored = self._restore_tree(which, tree)
        key = 'ema' if (use_ema and 'ema' in restored) else 'params'
        state = nnx.state(self.net_g, nnx.Param)
        nnx.replace_by_pure_dict(state, restored[key])
        nnx.update(self.net_g, state)
        get_root_logger().info(f"Loaded '{key}' weights from {self.ckpt_dir}/{which}")

    def predict(self, params):
        """Raw forward pass on already-normalized inputs; returns numpy."""
        return np.asarray(self._forward(self.net_g, jnp.asarray(params)))

    def make_predict_fn(self, param_stats=None):
        """Return a pure, jit/vmap-safe function raw physical inputs -> physical
        predictions, encapsulating the emulator's full input/output contract:

        1. log10 on data-dimension columns flagged ``log`` in dataset.data_dims
        2. input normalization (params_opt.normalization) with param_stats
        3. float32 forward pass through net_g
        4. target unscaling (inverse of targets_opt log/offset)

        Args:
            param_stats: override the normalization stats (default: the
                emulator's own). Only for legacy-parity quirks — e.g. the SDC3b
                xHI emulator is normalized with the Pk emulator's stats.
        """
        from CosmicDawnSynergies.models.scaling import NORMALIZATIONS, stats_arrays

        dataset_opt = self.opt['dataset']
        normalize = NORMALIZATIONS[dataset_opt['params_opt'].get('normalization', 'norm_minmax')]
        stats = stats_arrays(param_stats if param_stats is not None else self.param_stats)
        log_flags = [bool(dim.get('log', False)) for dim in dataset_opt.get('data_dims', {}).values()]
        target_log = dataset_opt['targets_opt'].get('log', False)
        target_offset = float(dataset_opt['targets_opt'].get('offset', 0.0) or 0.0)
        net = self.net_g

        def predict_fn(x):
            """x: (batch, in_dim) raw physical columns in emulator input order."""
            cols = [jnp.log10(x[:, i]) if (i < len(log_flags) and log_flags[i]) else x[:, i]
                    for i in range(x.shape[1])]
            x = normalize(jnp.stack(cols, axis=1), stats)
            pred = net(x.astype(jnp.float32))
            if target_log:
                pred = 10 ** pred
            if target_offset > 0:
                pred = pred - target_offset
            return pred

        return predict_fn

    @classmethod
    def from_emulator_dir(cls, emulator_dir, which='best', use_ema=False):
        """Load a trained emulator (converted or JAX-trained) for inference.

        Expects the standard emulator layout: one options .yml, a
        param_stats.json, and checkpoints/<which>.
        """
        import glob
        from os import path as osp

        from CosmicDawnSynergies.utils import yaml_load

        ymls = glob.glob(osp.join(emulator_dir, '*.yml'))
        if len(ymls) != 1:
            raise ValueError(f'Expected exactly one options yml in {emulator_dir}, found {ymls}')
        opt = yaml_load(ymls[0])
        opt['is_train'] = False
        opt.setdefault('path', {})
        opt['path']['emulators_root'] = emulator_dir
        opt['path']['checkpoints'] = osp.join(emulator_dir, 'checkpoints')

        param_stats = cls.load_param_stats(emulator_dir)
        # one input column per param_stats entry (data dims + astro params)
        model = cls(opt, in_dim=opt['arch'].get('in_dim', len(param_stats)))
        model.param_stats = param_stats
        model.load_network(which=which, use_ema=use_ema)
        return model
