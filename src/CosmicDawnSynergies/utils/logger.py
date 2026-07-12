import datetime
import logging
import time


def get_root_logger(logger_name='CosmicDawnSynergies', log_level=logging.INFO, log_file=None):
    """Root logger writing to console and optionally a file.

    Safe to call repeatedly; handlers are added once.
    """
    logger = logging.getLogger(logger_name)
    # check own handlers only: jax/absl put handlers on the root logger at import
    if logger.handlers:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    return logger


def init_tb_logger(log_dir):
    from tensorboardX import SummaryWriter
    return SummaryWriter(log_dir=log_dir)


class AvgTimer:

    def __init__(self, window=200):
        self.window = window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        self.start_time = self.tic = time.time()

    def record(self):
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


class MessageLogger:
    """Format training log messages (basicsr-style).

    Args:
        opt (dict): Full options dict; uses name, train.total_iter, logger.print_freq.
        start_iter (int): Iteration the run started from (for ETA on resume).
        tb_logger: tensorboardX SummaryWriter or None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self):
        self.start_time = time.time()

    def __call__(self, log_vars):
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (f'[{self.exp_name[:31]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            if self.tb_logger:
                label = f'losses/{k}' if k.startswith('l_') or k == 'loss' else k
                self.tb_logger.add_scalar(label, v, current_iter)
        self.logger.info(message)
