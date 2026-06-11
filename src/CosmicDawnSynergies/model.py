import os
import time
from unittest.mock import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
from os import path as osp

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only
from tqdm import tqdm

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass

    def validation(self, dataloader, current_iter, tb_logger):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger)


    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer


    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warm-up.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warm-up iter numbers. -1 for no warm-up.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f'Still cannot save {save_path}. Just ignore it.')
                # raise IOError(f'Cannot save {save_path}.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for emulator training.

    Args:
        in_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        n_hidden (int): Number of hidden layers
        out_dim (int): Output dimension
        activation (str): Activation function ('ReLU', 'LeakyReLU', 'GELU', 'Tanh', etc.)
        out_activation (str): Output activation function ('ReLU', 'Softplus', etc.)
            Use 'Softplus' for non-negative outputs (e.g., power spectra) as it
            avoids the "dying ReLU" problem while guaranteeing positive outputs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 100,
        n_hidden: int = 6,
        out_dim: int = 1,
        activation: str = 'ReLU',
        out_activation: str = None,
    ):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.out_dim = out_dim
        self.activation_name = activation
        self.out_activation_name = out_activation

        # Define activation function
        self.activation_fn = self._get_activation_fn(activation)
        self.out_activation_fn = self._get_activation_fn(out_activation) if out_activation else None

        # Build layers
        layers = []

        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(self._get_activation_fn(activation))

        # Hidden layers
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation_fn(activation))

        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        if self.out_activation_fn is not None:
            layers.append(self._get_activation_fn(out_activation))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation_fn(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        try:
            activation_class = getattr(nn, activation)
            return activation_class()
        except AttributeError:
            raise ValueError(f"Activation '{activation}' not found in torch.nn. "
                           f"Make sure it's a valid PyTorch activation function.")

    def _init_weights(self):
        """Initialize weights using He/Kaiming initialization for better training stability."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming (He) initialization
                # For ReLU-like activations, use 'relu' nonlinearity
                # For other activations, use 'leaky_relu' with a=0.01 as a reasonable default
                if self.activation_name in ['ReLU', 'LeakyReLU', 'PReLU', 'RReLU']:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif self.activation_name in ['Tanh', 'Sigmoid']:
                    # Xavier/Glorot initialization is better for tanh/sigmoid
                    nn.init.xavier_normal_(module.weight)
                else:
                    # Default to Kaiming for other activations (GELU, SiLU, etc.)
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

                # Initialize biases to small positive values to help avoid dead neurons
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim)
        """
        output = self.network(x)
        output = output.squeeze(-1) if self.out_dim == 1 else output

        return output
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPModel(BaseModel):
    """MLP Model that inherits from BaseModel for emulator training."""
    
    def __init__(self, opt):
        super(MLPModel, self).__init__(opt)
        
        # Build network
        self.network_opt = opt['network_opt']
        self.net_g = MLP(**self.network_opt)
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        # Loss function
        loss_type = opt['train'].get('loss_fn', 'MSELoss')
        if loss_type == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif loss_type == 'L1Loss':
            self.criterion = nn.L1Loss()
        elif loss_type == 'SmoothL1Loss':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented.')
        
        self.criterion = self.criterion.to(self.device)
        
        if self.is_train:
            self.init_training_settings()
        
        # For logging
        self.log_dict = OrderedDict()
    
    def init_training_settings(self):
        self.best_metric_results = dict()
        self.best_metric_results['rmse'] = dict()
        self.best_metric_results['rmse']['val'] = float('inf')
        self.best_metric_results['rmse']['iter'] = -1
        self.best_metric_results['nrmse'] = dict()
        self.best_metric_results['nrmse']['val'] = float('inf')
        self.best_metric_results['nrmse']['iter'] = -1
        self.best_metric_results['r2'] = dict()
        self.best_metric_results['r2']['val'] = -float('inf')
        self.best_metric_results['r2']['iter'] = -1

        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = MLP(**self.network_opt).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optimizer_opt'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optimizer_opt'])
        self.optimizers.append(self.optimizer_g)


    
    def optimize_parameters(self, data, current_iter=None):
        """Optimize network parameters."""
        self.optimizer_g.zero_grad()
        
        # Extract data from dictionary
        inputs = data['params'].to(self.device)
        targets = data['target'].to(self.device)
        
        # Forward pass
        outputs = self.net_g(inputs)
        
        # Calculate loss
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer_g.step()

        if hasattr(self, 'net_g_ema'):
            self.model_ema(self.ema_decay)

        # Update log
        self.log_dict['loss'] = torch.pow(10, torch.sqrt(loss)).item() if self.opt['dataset']['targets_opt'].get('log', False) else torch.sqrt(loss).item()
        
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks with param_stats.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # Add param_stats if available
        if hasattr(self, 'param_stats'):
            save_dict['param_stats'] = self.param_stats

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network and param_stats.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_dict = torch.load(load_path, map_location=lambda storage, loc: storage)

        # Load param_stats if available
        if 'param_stats' in load_dict:
            self.param_stats = load_dict['param_stats']
            logger.info(f'Loaded param_stats from {load_path}')

        # Load network weights
        if param_key is not None:
            if param_key not in load_dict and 'params' in load_dict:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_dict[param_key]
        else:
            load_net = load_dict

        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def nondist_validation(self, dataloader, current_iter, tb_logger):
        """Validation function for regression models using RMSE, NRMSE, and R² metrics."""

        # Get the entire validation dataset directly from the dataset tensors
        val_dataset = dataloader.dataset
        all_inputs = val_dataset.params.to(self.device)
        all_targets = val_dataset.targets.to(self.device)

        self.net_g.eval()
        with torch.no_grad():
            # Single forward pass through entire validation set
            all_outputs = self.net_g(all_inputs)

            # Compute MSE and RMSE
            mse = F.mse_loss(all_outputs, all_targets, reduction='mean')
            rmse = torch.sqrt(mse)

            # Normalized RMSE (RMSE / std of targets)
            nrmse = rmse / torch.std(all_targets)

            # R² score: 1 - (MSE / variance of targets)
            r2 = 1 - (mse / torch.var(all_targets))

            # Get rmse on original scale if log transform was applied (not exact due averaging with offset in logspace)
            rmse_scaled = 10**rmse if self.opt['dataset']['targets_opt'].get('log', False) else rmse
            rmse_scaled = rmse_scaled - self.opt['dataset']['targets_opt'].get('offset', 0)

            if self.best_metric_results['rmse']['val'] > rmse_scaled.item():
                self.best_metric_results['rmse']['val'] = rmse_scaled.item()
                self.best_metric_results['rmse']['iter'] = current_iter
            if self.best_metric_results['nrmse']['val'] > nrmse.item():
                self.best_metric_results['nrmse']['val'] = nrmse.item()
                self.best_metric_results['nrmse']['iter'] = current_iter
            if self.best_metric_results['r2']['val'] < r2.item():
                self.best_metric_results['r2']['val'] = r2.item()
                self.best_metric_results['r2']['iter'] = current_iter

            logger = get_root_logger()
            logger.info(f'Validation: RMSE={rmse_scaled:.4f} (Best: {self.best_metric_results["rmse"]["val"]:.4f}, iter {self.best_metric_results["rmse"]["iter"]})| NRMSE={nrmse:.4f} (Best: {self.best_metric_results["nrmse"]["val"]:.4f}, iter {self.best_metric_results["nrmse"]["iter"]})| R²={r2:.4f} (Best: {self.best_metric_results["r2"]["val"]:.4f}, iter {self.best_metric_results["r2"]["iter"]})')

            if tb_logger:
                tb_logger.add_scalar('validation/rmse', rmse_scaled, current_iter)
                tb_logger.add_scalar('validation/nrmse', nrmse, current_iter)
                tb_logger.add_scalar('validation/r2', r2, current_iter)

        self.net_g.train()

    def nondist_validation_old(self, dataloader, current_iter, tb_logger):
        """Validation function for regression models using RMSE metric."""
        use_pbar = self.opt.get('val', {}).get('pbar', False)

        total_mse = 0.0
        num_samples = 0
        
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='batch')

        self.net_g.eval()
        with torch.no_grad():
            for idx, val_data in enumerate(dataloader):
                self.feed_data(val_data)
                
                # Forward pass
                self.output = self.net_g(self.input)
                
                # Calculate MSE for RMSE computation
                if hasattr(self, 'target'):
                    mse = F.mse_loss(self.output, self.target, reduction='sum')
                    total_mse += mse.item()
                    num_samples += self.input.size(0)
                
                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f'Validation batch {idx}')
        
        if use_pbar:
            pbar.close()

        # Calculate RMSE
        if num_samples > 0:
            rmse = torch.sqrt(torch.tensor(total_mse / num_samples))
            logger = get_root_logger()
            logger.info(f'Validation RMSE: {rmse:.6f}')
            
            if tb_logger:
                tb_logger.add_scalar('validation/rmse', rmse, current_iter)
        
        self.net_g.train()
    
    def test(self):
        """Test function for inference."""
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.input)
        self.net_g.train()
    
    def get_current_visuals(self):
        """Return current visual results for logging."""
        out_dict = OrderedDict()
        out_dict['input'] = self.input.detach().cpu()
        if hasattr(self, 'output'):
            out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'target'):
            out_dict['target'] = self.target.detach().cpu()
        return out_dict

    def dist_validation(self, dataloader, current_iter, tb_logger):
        """Distributed validation."""
        # For now, use the same logic as non-distributed
        # Could be enhanced to properly aggregate across GPUs
        self.nondist_validation(dataloader, current_iter, tb_logger)


class BaseModelSklearn:
    """Base model for sklearn models."""

    def __init__(self, opt):
        self.opt = opt
        self.network_opt = opt.get('network_opt', {}) or {}
        self.net_g = None  # To be set by subclass

    def save(self, epoch, current_iter):
        """Save the sklearn model using joblib."""
        if current_iter == -1:
            save_filename = "net_g_latest.joblib"
        else:
            save_filename = f"net_g_{current_iter}.joblib"
        save_path = osp.join(self.opt['path']['models'], save_filename)
        joblib.dump(self.net_g, save_path)
        logger = get_root_logger()
        logger.info(f'Saved sklearn model to {save_path}')

    def validation(self, dataloader, current_iter, tb_logger):
        """Validation function for sklearn regression models using RMSE metric."""
        import numpy as np

        # Get the entire validation dataset directly from the dataset tensors
        val_dataset = dataloader.dataset
        params = val_dataset.params.numpy() if hasattr(val_dataset.params, 'numpy') else val_dataset.params
        targets = val_dataset.targets.numpy() if hasattr(val_dataset.targets, 'numpy') else val_dataset.targets

        # Predict using sklearn model
        predictions = self.net_g.predict(params)

        # Compute metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)

        # Normalized RMSE (RMSE / std of targets)
        nrmse = rmse / np.std(targets)

        # R² score: 1 - (MSE / variance of targets)
        r2 = 1 - (mse / np.var(targets))

        # Get rmse on original scale if log transform was applied
        if self.opt['dataset']['targets_opt'].get('log', False):
            rmse = 10 ** rmse
        rmse = rmse - self.opt['dataset']['targets_opt'].get('offset', 0)

        logger = get_root_logger()
        logger.info(f'Validation RMSE: {rmse:.6f} | NRMSE: {nrmse:.6f} | R²: {r2:.6f}')

        if tb_logger:
            tb_logger.add_scalar('validation/rmse', rmse, current_iter)
            tb_logger.add_scalar('validation/nrmse', nrmse, current_iter)
            tb_logger.add_scalar('validation/r2', r2, current_iter)


class RandomForestModel(BaseModelSklearn):
    """Random Forest Model using scikit-learn for emulator training."""

    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = RandomForestRegressor(**self.network_opt)

    def get_feature_importance(self, feature_names=None):
        """Get feature importances from the trained Random Forest model.

        Args:
            feature_names (list, optional): List of feature names. If None, uses feature_0, feature_1, etc.

        Returns:
            dict: Dictionary mapping feature names to importance scores, sorted by importance.
        """
        importances = self.net_g.feature_importances_

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        # Create dict sorted by importance (descending)
        importance_dict = dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))

        # Log feature importances
        logger = get_root_logger()
        logger.info('Feature Importances:')
        for name, importance in importance_dict.items():
            logger.info(f'  {name}: {importance:.4f}')

        return importance_dict


class GradientBoostingModel(BaseModelSklearn):
    """Gradient Boosting Model using scikit-learn for emulator training."""

    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = GradientBoostingRegressor(**self.network_opt)

    def get_feature_importance(self, feature_names=None):
        """Get feature importances from the trained Gradient Boosting model.

        Args:
            feature_names (list, optional): List of feature names. If None, uses feature_0, feature_1, etc.

        Returns:
            dict: Dictionary mapping feature names to importance scores, sorted by importance.
        """
        importances = self.net_g.feature_importances_

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        # Create dict sorted by importance (descending)
        importance_dict = dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))

        # Log feature importances
        logger = get_root_logger()
        logger.info('Feature Importances:')
        for name, importance in importance_dict.items():
            logger.info(f'  {name}: {importance:.4f}')

        return importance_dict