"""
Optuna hyperparameter tuning for MLP emulators.

This script optimizes the neural network architecture and optimizer
configuration for emulator training. All settings are read from a YAML config file.

Usage:
    python tune.py -opt options/tune/tune_Pk_SDC3b.yml
"""

import argparse
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

    train_loader = DataLoader(
        TRAIN_SET,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        VAL_SET,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=use_cuda,
    )

    return train_loader, val_loader


def suggest_param(trial, name, config):
    """Suggest a parameter based on its configuration."""
    param_type = config['type']
    low = config['low']
    high = config['high']

    if param_type == 'int':
        return trial.suggest_int(name, low, high)
    elif param_type == 'float':
        log_scale = config.get('log', False)
        return trial.suggest_float(name, low, high, log=log_scale)
    elif param_type == 'categorical':
        return trial.suggest_categorical(name, config['choices'])
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def define_model(trial):
    """Define MLP model with Optuna-suggested hyperparameters."""
    search_space = OPT['search_space']
    model_config = OPT.get('model', {})

    # Get tunable parameters
    n_hidden = suggest_param(trial, 'n_hidden', search_space['n_hidden'])
    hidden_dim = suggest_param(trial, 'hidden_dim', search_space['hidden_dim'])

    # Get fixed parameters
    activation = model_config.get('activation', 'ReLU')
    out_dim = model_config.get('out_dim', 1)

    model = MLP(
        in_dim=IN_DIM,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        out_dim=out_dim,
        activation=activation,
    )

    return model


def compute_r2(predictions, targets):
    """Compute R² score."""
    mse = F.mse_loss(predictions, targets)
    var = targets.var()
    return 1 - (mse / var)


def objective(trial):
    """Optuna objective function - maximize R²."""
    search_space = OPT['search_space']
    tune_config = OPT.get('tune', {})
    optimizer_config = OPT.get('optimizer', {})

    # Suggest batch size
    batch_size = suggest_param(trial, 'batch_size', search_space['batch_size'])

    # Create dataloaders with suggested batch size
    train_loader, val_loader = create_dataloaders(batch_size)

    # Generate the model
    model = define_model(trial).to(DEVICE)

    # Suggest optimizer hyperparameters
    lr = suggest_param(trial, 'lr', search_space['lr'])
    weight_decay = suggest_param(trial, 'weight_decay', search_space['weight_decay'])

    # Count model parameters
    n_params = sum(p.numel() for p in model.parameters())

    # Print trial info
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print(f"  batch_size:   {batch_size}")
    print(f"  n_hidden:     {trial.params['n_hidden']}")
    print(f"  hidden_dim:   {trial.params['hidden_dim']}")
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

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            params = batch['params'].to(DEVICE)
            targets = batch['target'].to(DEVICE)

            optimizer.zero_grad()
            output = model(params)
            loss = F.mse_loss(output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            total_iter += 1

        train_loss /= n_batches

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                params = batch['params'].to(DEVICE)
                targets = batch['target'].to(DEVICE)

                output = model(params)
                all_preds.append(output)
                all_targets.append(targets)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_mse = F.mse_loss(all_preds, all_targets).item()
        r2 = compute_r2(all_preds, all_targets).item()

        epoch_time = time.time() - epoch_start

        # Log progress
        if (epoch + 1) % log_freq == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Iter {total_iter:6d}/{total_iters} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val MSE: {val_mse:.6f} | "
                  f"R²: {r2:.4f} | "
                  f"Time: {epoch_time:.2f}s")

        # Report intermediate value
        trial.report(r2, epoch)

        # Handle pruning
        if trial.should_prune():
            trial_time = time.time() - trial_start
            print(f"  >>> Trial pruned at epoch {epoch+1} (R²: {r2:.4f}) | Total time: {trial_time:.1f}s")
            raise optuna.exceptions.TrialPruned()

    trial_time = time.time() - trial_start
    print(f"  >>> Trial completed | Final R²: {r2:.4f} | Total time: {trial_time:.1f}s")

    return r2


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

    # Create output directory
    output_dir = Path('param_search') / study_name
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

    # Create or load Optuna study (maximize R²)
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
        print(f"  Best so far: R² = {study.best_value:.4f}")

    # Callback to print best trial after each trial
    def print_best_callback(study, trial):
        if study.best_trial.number == trial.number:
            print(f"\n  *** New best trial! R²: {trial.value:.4f} ***")
        else:
            print(f"  Current best: Trial {study.best_trial.number} with R²: {study.best_trial.value:.4f}")

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

    print("\nBest trial:")
    trial = study.best_trial

    print(f"  R² Value: {trial.value:.6f}")

    print("\n  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Print suggested config for training options file
    model_config = opt.get('model', {})
    optimizer_config = opt.get('optimizer', {})

    print("\n" + "=" * 60)
    print("Suggested config for training YAML file:")
    print("=" * 60)
    print(f"network_opt:")
    print(f"  hidden_dim: !!int {trial.params['hidden_dim']}")
    print(f"  n_hidden: !!int {trial.params['n_hidden']}")
    print(f"  out_dim: !!int {model_config.get('out_dim', 1)}")
    print(f"  activation: {model_config.get('activation', 'ReLU')}")
    print(f"\ndataset:")
    print(f"  batch_size_per_gpu: !!int {trial.params['batch_size']}")
    print(f"\ntrain:")
    print(f"  optimizer_opt:")
    print(f"    type: {optimizer_config.get('type', 'AdamW')}")
    print(f"    lr: !!float {trial.params['lr']:.6e}")
    print(f"    weight_decay: !!float {trial.params['weight_decay']:.6e}")

    # Save best config to YAML file
    best_config = {
        'network_opt': {
            'hidden_dim': int(trial.params['hidden_dim']),
            'n_hidden': int(trial.params['n_hidden']),
            'out_dim': int(model_config.get('out_dim', 1)),
            'activation': model_config.get('activation', 'ReLU'),
        },
        'dataset': {
            'batch_size_per_gpu': int(trial.params['batch_size']),
        },
        'train': {
            'optimizer_opt': {
                'type': optimizer_config.get('type', 'AdamW'),
                'lr': float(trial.params['lr']),
                'weight_decay': float(trial.params['weight_decay']),
            }
        },
        'best_trial': {
            'number': trial.number,
            'r2': float(trial.value),
        }
    }

    best_config_path = output_dir / 'best_config.yml'
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest config saved to: {best_config_path}")
    print(f"Study database saved to: {storage_path}")


if __name__ == "__main__":
    main()
