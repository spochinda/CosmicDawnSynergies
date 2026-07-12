# CosmicDawnSynergies

This package is used for model inference and includes likelihoods for 21-cm power spectrum observations (HERA), radio background temperature (Table 2 of Dowell & Taylor (2018)), integrated X-ray background (Hickox & Markevitch (2006) and Harrison et al. (2016)), and SARAS 3 (Singh et al. 2022). In addition, the code contains the likelihood function used for the Cantabrigians parameter inference analysis in the SKA Science Data Challenge 3b.

**This branch (`jax`) is the JAX implementation**: emulators are flax NNX MLPs trained with optax and checkpointed with orbax; inference uses GPU-vectorized nested slice sampling from blackjax instead of PolyChord/MPI. The former PyTorch/PolyChord stack is frozen in `src/CosmicDawnSynergies/legacy/` for reference and remains runnable on `main`. Design decisions are recorded in `docs/adr/`; domain vocabulary in `CONTEXT.md`.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Notes:
- `blackjax` is pinned to the [handley-lab fork](https://github.com/handley-lab/blackjax) (`nested_sampling` branch) — nested sampling is not in the PyPI release yet (see `docs/adr/0002`).
- `torch` is **not** a dependency. It is only needed to convert legacy `.pth` emulators: `pip install -e '.[convert]'`.
- On a machine with CUDA, install the matching jaxlib (e.g. `pip install -U "jax[cuda12]"`); the code auto-detects devices.

## Training emulators

```bash
python train.py -opt options/emulators/Pk_SDC3b_jax.yml
```

All training options live in `options/emulators/*.yml` using an **optax-native schema** (legacy torch-style ymls do not run on this branch — see `docs/adr/0001`; `*_jax.yml` files are the ported configs). The key blocks:

```yaml
model: MLPModel            # models/ registry
arch:                      # archs/ registry (in_dim is inferred from the dataset)
  type: MLP
  hidden_dim: 100
  n_hidden: 6
  activation: relu
  init: torch_default
train:
  total_iter: 3200
  loss: mse                # mse | mae | huber
  grad_clip_norm: 1.0
  ema_decay: ~             # e.g. 0.999 to enable EMA
  optimizer: {name: adam, weight_decay: 1.0e-4}
  schedule:  {name: piecewise_constant, init_value: 1.0e-3, boundaries_and_scales: {60000: 1.0}}
dataset:                   # unchanged from the legacy schema (seeds included)
  type: BaseDataset
  batch_size: 20000
  ...
```

Useful flags:
- `--debug` — short val/log intervals, `debug_` prefix on the run name
- `--auto_resume` — pick up the latest step checkpoint if one exists
- `--force_yml train:total_iter=100 manual_seed=7` — override any yml key from the CLI

Outputs go to `trained_emulators/<name>/`:

```
trained_emulators/<name>/
├── <name>.yml           # copy of the options used
├── param_stats.json     # input normalization stats (used by inference)
├── train_<...>.log
└── checkpoints/
    ├── best/            # best-validation weights — the canonical emulator
    └── steps/<iter>/    # periodic checkpoints incl. optimizer state (resume)
```

The whole dataset lives on device; on a multi-GPU node the batch is sharded across local devices automatically (no launcher, no MPI). TensorBoard logs go to `tb_logger/<name>/`.

### Converting a legacy .pth emulator

```bash
pip install -e '.[convert]'
python scripts/convert_pth_to_orbax.py trained_emulators/<legacy_dir> \
    [--opt <legacy yml>] [--weights net_g_latest.pth]
```

This writes `<legacy_dir>_jax/` in the standard layout above and verifies torch-vs-JAX forward parity as it goes.

## Inference

```bash
python inference.py -opt options/inference/sdc3b_jax_PS1.yml
```

Inference options live in `options/inference/*.yml`. The `emulator` fields point at emulator *directories* (JAX-trained or converted). Sampler settings replace the old `polychord_settings`:

```yaml
sampler:
  seed: 51
  num_live: 500
  num_delete: 250          # particles replaced per (vectorized) NS step
  num_inner_steps: 25      # slice steps per new particle, ~5 x ndims
  stop_dlogZ: 3.0          # stop when logZ_live - logZ < -3
```

Each run creates `inferences/<inference_id>/` containing the options copy, copies of the emulators used, a log, the chains as `run.csv` (anesthetic format — load with `anesthetic.read_chains("inferences/<id>/run.csv")`, mixes freely with PolyChord roots in `triangle_plot`), and a triangle plot. Inference runs in float64 (`jax_enable_x64`); training stays float32.

Available configs: `sdc3b_jax_PS1.yml` / `sdc3b_jax_PS2.yml` (SDC3b with xHI derived fractions) and `joint_jax.yml` (HERA + radio background + X-ray background + SARAS3).

## Adding a new likelihood

Add a `*_likelihood.py` file in `src/CosmicDawnSynergies/likelihoods/` with a class registered via `@LIKELIHOOD_REGISTRY.register()` that subclasses `BaseLikelihood`:

- `extract_data(self)` — host-side numpy setup; precompute static (pre-normalized) input blocks
- `loglikelihood(self, particle)` — **pure JAX** (jit/vmap-safe) map from a `{param_name: scalar}` dict to a scalar logL
- `derived(self, particle)` — optional derived quantities, vmapped over the posterior after the run
- `prior_bounds` — extend if the module adds nuisance parameters (see `LikelihoodSARAS3`)

The registry auto-imports the file; reference it by class name under `LikelihoodModules:` in an inference yml. Parity checks against the legacy implementations are in `scripts/likelihood_parity_sdc3b.py` and `scripts/likelihood_parity_phase2.py` (both require `.[convert]`).

## Legacy stack

The PyTorch/PolyChord implementation (including `tune.py`, the Optuna tuning harness) lives unmodified in `src/CosmicDawnSynergies/legacy/` and runs on the `main` branch. PolyChord/HPC installation notes for it are in the `main` branch README.
