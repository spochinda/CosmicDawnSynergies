import datetime
import logging
import math
import time
from os import path as osp

import jax
import jax.numpy as jnp

from src.CosmicDawnSynergies.data import DeviceBatcher, build_dataset
from src.CosmicDawnSynergies.models import build_model
from src.CosmicDawnSynergies.utils import (AvgTimer, MessageLogger, copy_file, dict2str, get_root_logger,
                                           get_time_str, init_tb_logger, make_emu_dirs, mkdir_and_rename,
                                           parse_emu_options)
from src.CosmicDawnSynergies.utils.dist import batch_sharding, get_mesh


def create_train_val_data(opt, logger, mesh=None):
    """Build the dataset and place train/val arrays on device."""
    dataset_config = opt.get('dataset', {})
    if not dataset_config:
        raise ValueError("No 'dataset' configuration found in options")

    logger.info(f"Building {dataset_config.get('type', 'BaseDataset')}")
    dataset = build_dataset(dataset_config)

    batch_size = dataset_config.get('batch_size', opt.get('train', {}).get('batch_size', 32))
    sharding = batch_sharding(mesh)
    train_batcher = DeviceBatcher(dataset.params_train, dataset.targets_train, batch_size,
                                  shuffle=True, drop_last=True, sharding=sharding)
    val_data = (jnp.asarray(dataset.params_val, dtype=jnp.float32),
                jnp.asarray(dataset.targets_val, dtype=jnp.float32))

    num_iter_per_epoch = len(train_batcher)
    if 'epochs' in opt['train']:
        total_epochs = opt['train']['epochs']
        total_iters = total_epochs * num_iter_per_epoch
        opt['train']['total_iter'] = total_iters
    else:
        total_iters = int(opt['train']['total_iter'])
        total_epochs = math.ceil(total_iters / num_iter_per_epoch)

    logger.info('Training statistics:'
                f'\n\tNumber of train samples: {train_batcher.n}'
                f'\n\tNumber of val samples: {len(val_data[0])}'
                f'\n\tBatch size: {batch_size}'
                f'\n\tDevices: {jax.device_count()}'
                f'\n\tIterations per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

    return dataset, train_batcher, val_data, total_epochs, total_iters


def train_pipeline(root_path):
    opt, args = parse_emu_options(root_path, is_train=True)
    opt['root_path'] = root_path

    mesh = get_mesh()

    # mkdir for experiments and logger (auto_resume keeps the existing dir)
    resume_requested = opt['auto_resume'] or opt['path'].get('resume_state')
    if not resume_requested:
        make_emu_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the emulator root
    copy_file(args.opt, opt['path']['emulators_root'])

    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file)
    logger.info(f'Version Information: JAX {jax.__version__}, devices: {jax.devices()}')
    logger.info(dict2str(opt))

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # dataset and device placement
    dataset, train_batcher, val_data, total_epochs, total_iters = create_train_val_data(opt, logger, mesh)

    # model (in_dim inferred from the dataset)
    model = build_model(opt, in_dim=dataset.in_dim)
    model.param_stats = dataset.param_stats
    model.save_param_stats(dataset.param_stats)

    # resume
    start_epoch, current_iter = 0, 0
    if resume_requested:
        resumed = model.resume_training()
        if resumed is not None:
            start_epoch, current_iter = resumed
            logger.info(f'Resuming training from epoch: {start_epoch}, iter: {current_iter}.')
        else:
            logger.info('auto_resume requested but no checkpoint found; training from scratch.')

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # PRNG for batch shuffling: derived from manual_seed, folded per epoch
    shuffle_key = jax.random.key(opt['manual_seed'])

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    epoch = start_epoch
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_key = jax.random.fold_in(shuffle_key, epoch)
        for batch in train_batcher.epoch(epoch_key):
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break

            model.optimize_parameters(batch, current_iter)
            iter_timer.record()

            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                msg_logger.reset_start_time()

            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # validation (saves the best-val checkpoint internally)
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                logger.info('Running validation...')
                model.validation(val_data, current_iter, tb_logger)

            # periodic step checkpoint for resuming
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            data_timer.start()
            iter_timer.start()

        if current_iter >= total_iters:
            break

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    current_iter = min(current_iter, total_iters)
    model.save(epoch, current_iter)
    if opt.get('val') is not None:
        model.validation(val_data, current_iter, tb_logger)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
