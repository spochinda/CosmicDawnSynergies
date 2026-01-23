import datetime
import logging
from copy import deepcopy
import math
import time
import sklearn
import torch
from os import path as osp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.CosmicDawnSynergies.dataset import BaseDataset
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_root_logger,
                           init_tb_logger, init_wandb_logger, scandir)
from src.CosmicDawnSynergies.utils import parse_emu_options, copy_file, dict2str, make_emu_dirs, mkdir_and_rename, get_time_str
import src.CosmicDawnSynergies.model as models


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    """Create simplified train and validation dataloaders from single dataset configuration"""

    # Build dataset from the single 'dataset' configuration
    dataset_config = opt.get('dataset', {})
    if not dataset_config:
        raise ValueError("No 'dataset' configuration found in options")

    logger.info("Building BaseDataset")

    # Create the main dataset which contains both train and val splits
    base_dataset = BaseDataset(dataset_config)

    # Check if we should preload data to GPU
    preload_to_gpu = dataset_config.get('preload_to_gpu', False)
    data_on_gpu = preload_to_gpu and opt.get('num_gpu', 0) > 0 and torch.cuda.is_available()

    # Move datasets to GPU if requested and available to eliminate host-to-device transfers
    if data_on_gpu:
        device = torch.device('cuda')
        logger.info("Preloading datasets to GPU (preload_to_gpu=True)")
        base_dataset.train_dataset.params = base_dataset.train_dataset.params.to(device)
        base_dataset.train_dataset.targets = base_dataset.train_dataset.targets.to(device)
        base_dataset.val_dataset.params = base_dataset.val_dataset.params.to(device)
        base_dataset.val_dataset.targets = base_dataset.val_dataset.targets.to(device)

    # Get PyTorch datasets directly from BaseDataset
    train_set = base_dataset.train_dataset
    val_set = base_dataset.val_dataset
    
    # Get batch size from dataset options (or fallback to train options for backward compatibility)
    batch_size = opt['dataset'].get('batch_size_per_gpu', opt['train'].get('batch_size', 32))
    # When data is on GPU, must use num_workers=0 (can't pickle CUDA tensors across processes)
    num_workers = 0 if data_on_gpu else opt['dataset'].get('num_worker_per_gpu', opt.get('num_worker_per_gpu', 0))

    if data_on_gpu and num_workers > 0:
        logger.info("Setting num_workers=0 because data is on GPU (CUDA tensors cannot be shared across processes)")

    # Setup distributed training samplers if enabled
    train_sampler = None
    val_sampler = None
    
    if opt.get('dist', False) and opt.get('world_size', 1) > 1:
        train_sampler = DistributedSampler(
            train_set, 
            num_replicas=opt['world_size'], 
            rank=opt['rank'],
            shuffle=True,
            seed=opt.get('manual_seed', 0)
        )
        val_sampler = DistributedSampler(
            val_set, 
            num_replicas=opt['world_size'], 
            rank=opt['rank'],
            shuffle=False
        )
        shuffle = False  # DistributedSampler handles shuffling
    else:
        shuffle = True
    
    # Create DataLoaders
    # Set pin_memory=False if data is already on GPU (no need to pin CPU memory)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=not data_on_gpu,
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=not data_on_gpu,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    # Calculate training statistics
    batch_size_total = batch_size * opt.get('world_size', 1)
    num_iter_per_epoch = math.ceil(len(train_set) / batch_size_total)
    
    # Use epochs if specified, otherwise calculate from total_iter
    if 'epochs' in opt['train']:
        total_epochs = opt['train']['epochs']
        total_iters = total_epochs * num_iter_per_epoch
    else:
        total_iters = int(opt['train']['total_iter'])
        total_epochs = math.ceil(total_iters / num_iter_per_epoch)
    
    logger.info('Training statistics:'
                f'\n\tNumber of train samples: {len(train_set)}'
                f'\n\tNumber of val samples: {len(val_set)}'
                f'\n\tBatch size per gpu: {batch_size}'
                f'\n\tWorld size (gpu number): {opt.get("world_size", 1)}'
                f'\n\tTotal batch size: {batch_size_total}'
                f'\n\tIterations per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

    # Extract param_stats from dataset if available
    param_stats = getattr(base_dataset, 'param_stats', None)

    return train_loader, train_sampler, val_loader, total_epochs, total_iters, param_stats


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('trained_emulators', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        if opt['num_gpu'] > 0:
            device_id = torch.cuda.current_device()
            resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            resume_state = torch.load(resume_state_path, map_location='cpu')
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_emu_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_emu_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_file(args.opt, opt['path']['emulators_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Version Information: PyTorch: {torch.__version__}')
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters, param_stats = result

    # create model
    if opt['model_type'] == 'MLPModel':
        opt['network_opt']['in_dim'] = train_loader.dataset.params.shape[1]
    model = getattr(models, opt.get('model_type', 'MLPModel'))(opt)

    # Store normalization stats in model if available (before resume, in case resume overwrites)
    if param_stats is not None:
        model.param_stats = param_stats

    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()
    if isinstance(model.net_g, sklearn.base.BaseEstimator) is False:
        for epoch in range(start_epoch, total_epochs + 1):
            # Set epoch for distributed sampler
            if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch)
                
            # Standard PyTorch training loop
            for train_data in train_loader:
                data_timer.record()

                current_iter += 1
                if current_iter > total_iters:
                    break
                    
                # update learning rate
                model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                
                # training
                model.optimize_parameters(train_data, current_iter)
                iter_timer.record()
                
                if current_iter == 1:
                    # reset start time in msg_logger for more accurate eta_time
                    # not work in resume mode
                    msg_logger.reset_start_time()
                    
                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

                # validation
                save_best = opt['val'].get('save_best', False) and (current_iter > opt['val'].get('val_start', 0))
                if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                    logger.info('Running validation...')
                    previous_best = deepcopy(model.best_metric_results)
                    model.validation(val_loader, current_iter, tb_logger)
                    current_best = deepcopy(model.best_metric_results)
                    if save_best:
                        if previous_best != current_best:
                            logger.info('Saving models and training states.')
                            model.save(epoch, current_iter)
                        else:
                            logger.info('No improvement in validation metrics, skipping model save.')
                    
                if (current_iter % opt['logger']['save_checkpoint_freq'] == 0) and not save_best:
                    logger.info('Saving models and training states.')
                    model.save(epoch, current_iter)
                    
                data_timer.start()
                iter_timer.start()
                
            # Check if we've reached the total iterations
            if current_iter >= total_iters:
                break
                
        # end of epoch
    else:
        # sklearn model training
        params = train_loader.dataset.params
        targets = train_loader.dataset.targets
        model.net_g.fit(params, targets)
        current_iter = total_iters

        # Log feature importances if available
        if hasattr(model, 'get_feature_importance'):
            model.get_feature_importance()

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)