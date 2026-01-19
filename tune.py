"""
Optuna hyperparameter tuning for MLP emulators.

This script optimizes the neural network architecture and optimizer
configuration for emulator training. All settings are read from a YAML config file.

Usage:
    python tune.py -opt options/tune/tune_Pk_SDC3b.yml
"""

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import optuna

# Disable output buffering for SLURM compatibility
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from src.CosmicDawnSynergies.dataset import BaseDataset
from src.CosmicDawnSynergies.model import MLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global holders
OPT = None
TRAIN_SET = None
VAL_SET = None
IN_DIM = None


def load_config(opt_path):
    """Load configuration from YAML file."""
    global OPT, TRAIN_SET, VAL_SET, IN_DIM

    with open(opt_path, 'r') as f:
        OPT = yaml.safe_load(f)

    dataset_config = OPT.get('dataset', {})

    print(f"Loading dataset from {opt_path}")
    base_dataset = BaseDataset(dataset_config)

    TRAIN_SET = base_dataset.train_dataset
    VAL_SET = base_dataset.val_dataset

    IN_DIM = TRAIN_SET.params.shape[1]

    print(f"Dataset loaded: {len(TRAIN_SET)} train samples, {len(VAL_SET)} val samples")
    print(f"Input dimension: {IN_DIM}")
    print(f"Device: {DEVICE}")

    return OPT


def create_dataloaders(batch_size):
    """Create dataloaders with specified batch size."""
    use_cuda = DEVICE.type == 'cuda'
    dataset_config = OPT.get('dataset', {})
    num_workers = dataset_config.get('num_worker_per_gpu', 4)

    train_loader = DataLoader(
        TRAIN_SET,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        VAL_SET,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def suggest_param(trial, name, config):
    """Suggest a parameter based on its configuration."""
    param_type = config['type']

    if param_type == 'categorical':
        return trial.suggest_categorical(name, config['choices'])

    # For int/float types, get low and high
    low = config['low']
    high = config['high']

    if param_type == 'int':
        return trial.suggest_int(name, low, high)
    elif param_type == 'float':
        log_scale = config.get('log', False)
        return trial.suggest_float(name, low, high, log=log_scale)
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def define_model(trial):
    """Define MLP model with Optuna-suggested hyperparameters."""
    search_space = OPT['search_space']
    model_config = OPT.get('model', {})

    # Get tunable parameters
    n_hidden = suggest_param(trial, 'n_hidden', search_space['n_hidden'])
    hidden_dim = suggest_param(trial, 'hidden_dim', search_space['hidden_dim'])

    # Get activation - either from search_space (tunable) or model config (fixed)
    if 'activation' in search_space:
        activation = suggest_param(trial, 'activation', search_space['activation'])
    else:
        activation = model_config.get('activation', 'ReLU')

    # Get fixed parameters
    out_activation = model_config.get('out_activation', 'ReLU')
    out_dim = model_config.get('out_dim', 1)

    model = MLP(
        in_dim=IN_DIM,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        out_dim=out_dim,
        activation=activation,
        out_activation=out_activation,
    )

    return model


def compute_r2(predictions, targets):
    """Compute R² score."""
    mse = F.mse_loss(predictions, targets)
    var = targets.var()
    return 1 - (mse / var)


def compute_rmse(predictions, targets):
    """Compute RMSE."""
    mse = F.mse_loss(predictions, targets)
    return torch.sqrt(mse)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Metric configurations: (direction, transform_for_optuna)
# direction: "maximize" or "minimize" for Optuna
# transform: function to transform the value for Optuna (since we always maximize)
METRIC_CONFIG = {
    'r2': {
        'direction': 'maximize',
        'transform': lambda x: x,  # Higher is better, no transform needed
        'format': '.4f',
    },
    'rmse': {
        'direction': 'minimize',
        'transform': lambda x: -x,  # Lower is better, negate for maximization
        'format': '.6f',
    },
    'n_params': {
        'direction': 'minimize',
        'transform': lambda x: -math.log(x),  # Minimize params, use -log for scale
        'format': ',',
    },
}


def get_objective_value(metric_name, r2, rmse, n_params):
    """Get the raw value for a given metric."""
    if metric_name == 'r2':
        return r2
    elif metric_name == 'rmse':
        return rmse
    elif metric_name == 'n_params':
        return n_params
    else:
        raise ValueError(f"Unknown metric: {metric_name}. Choose from: r2, rmse, n_params")


def transform_for_optuna(metric_name, value):
    """Transform a metric value for Optuna optimization (we always maximize)."""
    return METRIC_CONFIG[metric_name]['transform'](value)


def objective(trial):
    """Optuna objective function with configurable objectives."""
    search_space = OPT['search_space']
    tune_config = OPT.get('tune', {})
    optimizer_config = OPT.get('optimizer', {})

    # Get objective configuration
    objective1 = tune_config.get('objective1', 'r2')
    objective2 = tune_config.get('objective2', None)  # Optional second objective

    # Suggest batch size
    batch_size = suggest_param(trial, 'batch_size', search_space['batch_size'])

    # Create dataloaders with suggested batch size
    train_loader, val_loader = create_dataloaders(batch_size)

    # Generate the model
    model = define_model(trial).to(DEVICE)

    # Get optimizer hyperparameters - either from search_space (tunable) or optimizer config (fixed)
    if 'lr' in search_space:
        lr = suggest_param(trial, 'lr', search_space['lr'])
    else:
        lr = optimizer_config.get('lr', 1e-3)

    if 'weight_decay' in search_space:
        weight_decay = suggest_param(trial, 'weight_decay', search_space['weight_decay'])
    else:
        weight_decay = optimizer_config.get('weight_decay', 1e-4)

    # Count model parameters
    n_params = count_parameters(model)

    # Get activation for logging
    activation = trial.params.get('activation', OPT.get('model', {}).get('activation', 'ReLU'))

    # Print trial info
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} ({DEVICE})")
    print(f"{'='*60}")
    print(f"  batch_size:   {batch_size}")
    print(f"  n_hidden:     {trial.params['n_hidden']}")
    print(f"  hidden_dim:   {trial.params['hidden_dim']}")
    print(f"  activation:   {activation}")
    print(f"  lr:           {lr:.2e}")
    print(f"  weight_decay: {weight_decay:.2e}")
    print(f"  n_params:     {n_params:,}")

    # Create optimizer (fixed type from config)
    optimizer_type = optimizer_config.get('type', 'AdamW')
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Training loop
    epochs = tune_config.get('epochs', 50)
    log_freq = tune_config.get('log_freq', 10)  # Log every N epochs
    trial_start = time.time()
    total_iter = 0
    iters_per_epoch = len(train_loader)
    total_iters = epochs * iters_per_epoch

    # Create prefetchers for overlapping data transfer with computation
    use_cuda = DEVICE.type == 'cuda'
    if use_cuda:
        opt_for_prefetcher = {'num_gpu': 1}  # CUDAPrefetcher expects this
        train_prefetcher = CUDAPrefetcher(train_loader, opt_for_prefetcher)
        val_prefetcher = CUDAPrefetcher(val_loader, opt_for_prefetcher)
    else:
        train_prefetcher = CPUPrefetcher(train_loader)
        val_prefetcher = CPUPrefetcher(val_loader)

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        # Use prefetcher pattern
        train_prefetcher.reset()
        batch = train_prefetcher.next()

        while batch is not None:
            params = batch['params']
            targets = batch['target']

            optimizer.zero_grad()
            output = model(params)
            loss = F.mse_loss(output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            total_iter += 1

            batch = train_prefetcher.next()

        train_loss /= n_batches

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        val_prefetcher.reset()
        batch = val_prefetcher.next()

        with torch.no_grad():
            while batch is not None:
                params = batch['params']
                targets = batch['target']

                output = model(params)
                all_preds.append(output)
                all_targets.append(targets)

                batch = val_prefetcher.next()

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_mse = F.mse_loss(all_preds, all_targets).item()
        r2 = compute_r2(all_preds, all_targets).item()
        rmse = compute_rmse(all_preds, all_targets).item()

        epoch_time = time.time() - epoch_start
        avg_iter_time = (time.time() - trial_start) / total_iter * 1000  # ms per iteration

        # Log progress
        if (epoch + 1) % log_freq == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Iter {total_iter:6d}/{total_iters} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val MSE: {val_mse:.6f} | "
                  f"R²: {r2:.4f} | "
                  f"RMSE: {rmse:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"Avg: {avg_iter_time:.1f}ms/it")

        # Report intermediate value and handle pruning (single-objective only)
        # Get primary objective value for pruning
        obj1_value = get_objective_value(objective1, r2, rmse, n_params)
        obj1_transformed = transform_for_optuna(objective1, obj1_value)

        if objective2 is None:
            trial.report(obj1_transformed, epoch)
            if trial.should_prune():
                trial_time = time.time() - trial_start
                print(f"  >>> Trial pruned at epoch {epoch+1} ({objective1}: {obj1_value:{METRIC_CONFIG[objective1]['format']}}) | Total time: {trial_time:.1f}s")
                raise optuna.exceptions.TrialPruned()

    trial_time = time.time() - trial_start

    # Get final objective values
    obj1_value = get_objective_value(objective1, r2, rmse, n_params)
    print(f"  >>> Trial completed | {objective1}: {obj1_value:{METRIC_CONFIG[objective1]['format']}} | n_params: {n_params:,} | Total time: {trial_time:.1f}s")

    # Store metrics as user attributes for later analysis
    trial.set_user_attr('n_params', n_params)
    trial.set_user_attr('r2', r2)
    trial.set_user_attr('rmse', rmse)

    # Return objective value(s)
    if objective2 is not None:
        # Multi-objective mode
        obj2_value = get_objective_value(objective2, r2, rmse, n_params)
        return transform_for_optuna(objective1, obj1_value), transform_for_optuna(objective2, obj2_value)
    else:
        # Single-objective mode
        return transform_for_optuna(objective1, obj1_value)


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for emulators')
    parser.add_argument('-opt', type=str, required=True, help='Path to tune options YAML file')
    args = parser.parse_args()

    # Load configuration
    opt = load_config(args.opt)
    tune_config = opt.get('tune', {})

    # Get tuning settings from config
    n_trials = tune_config.get('n_trials', 100)
    timeout = tune_config.get('timeout', 3600)
    study_name = tune_config.get('study_name', 'emulator_tuning')

    # Get objective configuration
    objective1 = tune_config.get('objective1', 'r2')
    objective2 = tune_config.get('objective2', None)
    multi_objective = objective2 is not None

    # Check for overwrite option
    overwrite = tune_config.get('overwrite', False)

    # Create output directory
    output_dir = Path('param_search') / study_name

    # Handle overwrite
    if overwrite and output_dir.exists():
        print(f"\nOverwriting existing study at: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file to output directory
    shutil.copy(args.opt, output_dir / 'tune_config.yml')

    # SQLite storage for persistence
    storage_path = output_dir / 'study.db'
    storage = f"sqlite:///{storage_path}"

    print(f"\nTuning settings:")
    print(f"  n_trials: {n_trials}")
    print(f"  timeout: {timeout}s")
    print(f"  epochs per trial: {tune_config.get('epochs', 50)}")
    print(f"  log_freq: {tune_config.get('log_freq', 10)}")
    print(f"  study_name: {study_name}")
    print(f"  output_dir: {output_dir}")
    print(f"  objective1: {objective1} ({METRIC_CONFIG[objective1]['direction']})")
    if multi_objective:
        print(f"  objective2: {objective2} ({METRIC_CONFIG[objective2]['direction']})")
    print(f"  multi_objective: {multi_objective}")

    # Print search space
    print(f"\nSearch space:")
    search_space = opt.get('search_space', {})
    for param_name, param_config in search_space.items():
        ptype = param_config['type']
        if ptype == 'categorical':
            print(f"  {param_name}: {param_config['choices']}")
        else:
            log_str = " (log)" if param_config.get('log', False) else ""
            print(f"  {param_name}: [{param_config['low']}, {param_config['high']}]{log_str}")

    # Create or load Optuna study
    if multi_objective:
        # Multi-objective: always maximize (we transform values in objective function)
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            # Note: Pruning not supported for multi-objective
        )
    else:
        # Single-objective: always maximize (we transform values in objective function)
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

    # Check if resuming
    if len(study.trials) > 0:
        print(f"\nResuming study with {len(study.trials)} existing trials")
        if not multi_objective:
            best_raw = study.best_trial.user_attrs.get(objective1, study.best_value)
            print(f"  Best so far: {objective1} = {best_raw:{METRIC_CONFIG[objective1]['format']}}")
        else:
            print(f"  Pareto front size: {len(study.best_trials)}")

    # Callback to print best trial after each trial
    def print_best_callback(study, trial):
        if multi_objective:
            # For multi-objective, check if trial is on Pareto front
            if trial in study.best_trials:
                obj1_val = trial.user_attrs.get(objective1, 'N/A')
                obj2_val = trial.user_attrs.get(objective2, 'N/A')
                print(f"\n  *** Trial on Pareto front! {objective1}: {obj1_val}, {objective2}: {obj2_val} ***")
            print(f"  Pareto front size: {len(study.best_trials)}")
        else:
            if study.best_trial.number == trial.number:
                obj1_val = trial.user_attrs.get(objective1, trial.value)
                print(f"\n  *** New best trial! {objective1}: {obj1_val:{METRIC_CONFIG[objective1]['format']}} ***")
            else:
                best_val = study.best_trial.user_attrs.get(objective1, study.best_value)
                print(f"  Current best: Trial {study.best_trial.number} with {objective1}: {best_val:{METRIC_CONFIG[objective1]['format']}}")

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[print_best_callback])

    # Print results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "=" * 60)
    print("Study statistics:")
    print("=" * 60)
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    model_config = opt.get('model', {})
    optimizer_config = opt.get('optimizer', {})

    if multi_objective:
        # Print Pareto front
        print(f"\nPareto front ({len(study.best_trials)} solutions):")
        print("-" * 60)

        # Sort by first objective (transformed value, higher is better)
        pareto_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)

        for i, t in enumerate(pareto_trials):
            obj1_val = t.user_attrs.get(objective1, 'N/A')
            obj2_val = t.user_attrs.get(objective2, 'N/A')
            # Format based on metric type
            if objective1 == 'n_params':
                obj1_str = f"{obj1_val:,}" if isinstance(obj1_val, (int, float)) else str(obj1_val)
            else:
                obj1_str = f"{obj1_val:{METRIC_CONFIG[objective1]['format']}}" if isinstance(obj1_val, (int, float)) else str(obj1_val)
            if objective2 == 'n_params':
                obj2_str = f"{obj2_val:,}" if isinstance(obj2_val, (int, float)) else str(obj2_val)
            else:
                obj2_str = f"{obj2_val:{METRIC_CONFIG[objective2]['format']}}" if isinstance(obj2_val, (int, float)) else str(obj2_val)
            print(f"  {i+1}. Trial {t.number}: {objective1} = {obj1_str}, {objective2} = {obj2_str}")

        # Select best: highest transformed objective1 (first in sorted list)
        trial = pareto_trials[0]
        print(f"\nSelected best (best {objective1}): Trial {trial.number}")

        # Save all Pareto solutions
        pareto_configs = []
        for t in pareto_trials:
            pareto_configs.append({
                'trial_number': t.number,
                objective1: t.user_attrs.get(objective1, None),
                objective2: t.user_attrs.get(objective2, None),
                'r2': t.user_attrs.get('r2', None),
                'rmse': t.user_attrs.get('rmse', None),
                'n_params': t.user_attrs.get('n_params', None),
                'params': dict(t.params),
            })

        pareto_path = output_dir / 'pareto_front.yml'
        with open(pareto_path, 'w') as f:
            yaml.dump(pareto_configs, f, default_flow_style=False, sort_keys=False)
        print(f"Pareto front saved to: {pareto_path}")

    else:
        print("\nBest trial:")
        trial = study.best_trial

    # Print all metrics for the best trial
    print(f"\nBest trial metrics:")
    print(f"  R²:      {trial.user_attrs.get('r2', 'N/A'):.4f}" if isinstance(trial.user_attrs.get('r2'), float) else f"  R²:      {trial.user_attrs.get('r2', 'N/A')}")
    print(f"  RMSE:    {trial.user_attrs.get('rmse', 'N/A'):.6f}" if isinstance(trial.user_attrs.get('rmse'), float) else f"  RMSE:    {trial.user_attrs.get('rmse', 'N/A')}")
    print(f"  n_params: {trial.user_attrs.get('n_params', 'N/A'):,}" if isinstance(trial.user_attrs.get('n_params'), int) else f"  n_params: {trial.user_attrs.get('n_params', 'N/A')}")

    print("\n  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Get activation - from tuned params or fixed config
    activation = trial.params.get('activation', model_config.get('activation', 'ReLU'))
    out_activation = model_config.get('out_activation', 'ReLU')

    # Get lr/weight_decay - from tuned params or fixed config
    lr = trial.params.get('lr', optimizer_config.get('lr', 1e-3))
    weight_decay = trial.params.get('weight_decay', optimizer_config.get('weight_decay', 1e-4))

    # Print suggested config for training options file
    print("\n" + "=" * 60)
    print("Suggested config for training YAML file:")
    print("=" * 60)
    print(f"network_opt:")
    print(f"  hidden_dim: !!int {trial.params['hidden_dim']}")
    print(f"  n_hidden: !!int {trial.params['n_hidden']}")
    print(f"  out_dim: !!int {model_config.get('out_dim', 1)}")
    print(f"  activation: {activation}")
    print(f"  out_activation: {out_activation}")
    print(f"\ndataset:")
    print(f"  batch_size_per_gpu: !!int {trial.params['batch_size']}")
    print(f"\ntrain:")
    print(f"  optimizer_opt:")
    print(f"    type: {optimizer_config.get('type', 'AdamW')}")
    print(f"    lr: !!float {lr:.6e}")
    print(f"    weight_decay: !!float {weight_decay:.6e}")

    # Save best config to YAML file
    best_config = {
        'network_opt': {
            'hidden_dim': int(trial.params['hidden_dim']),
            'n_hidden': int(trial.params['n_hidden']),
            'out_dim': int(model_config.get('out_dim', 1)),
            'activation': activation,
            'out_activation': out_activation,
        },
        'dataset': {
            'batch_size_per_gpu': int(trial.params['batch_size']),
        },
        'train': {
            'optimizer_opt': {
                'type': optimizer_config.get('type', 'AdamW'),
                'lr': float(lr),
                'weight_decay': float(weight_decay),
            }
        },
        'best_trial': {
            'number': trial.number,
            'r2': trial.user_attrs.get('r2', None),
            'rmse': trial.user_attrs.get('rmse', None),
            'n_params': trial.user_attrs.get('n_params', None),
        },
        'objectives': {
            'objective1': objective1,
            'objective2': objective2,
        }
    }

    best_config_path = output_dir / 'best_config.yml'
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest config saved to: {best_config_path}")
    print(f"Study database saved to: {storage_path}")


if __name__ == "__main__":
    main()
