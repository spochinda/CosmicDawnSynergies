# CosmicDawnSynergies

Bayesian astrophysical parameter inference for 21cm cosmology: neural-network emulators stand in for expensive simulations inside nested-sampling likelihoods over observational datasets (SDC3b, HERA, SARAS3, X-ray/radio backgrounds).

## Language

### Emulation

**Emulator**:
A trained neural network that maps astrophysical parameters (plus data dimensions) to a simulated observable, replacing the simulation at likelihood-evaluation time. One emulator = one directory under `trained_emulators/` containing weights, its options file, and logs.
_Avoid_: model (ambiguous with Model below), surrogate, network

**Arch**:
The bare network architecture (e.g. the MLP) — layer structure and forward pass only, no training logic.
_Avoid_: network, net_g

**Model**:
The wrapper around an Arch that owns the emulator's full contract: training (loss, optimizer, EMA, validation, checkpointing) and prediction (input transforms, normalization, output unscaling). Likelihoods consume predictions in physical units and never normalize.
_Avoid_: trainer

**Data dimension**:
An input axis of an emulator that is swept over at prediction time rather than sampled as a parameter — e.g. `z`, `kperp`, `kpar`. Listed under `data_dims` in options and occupying the leading columns of the input vector.
_Avoid_: coordinate, grid axis

**Parameter stats (`param_stats`)**:
Per-input min/max/mean/std computed from the training split, stored with the emulator's weights. The single source of truth for input normalization at inference time and for default prior bounds.
_Avoid_: scaler, scale_opt (legacy)

**Normalization**:
The input scaling applied before the Arch: one of `norm_minmax` ([0,1]), `norm_minmax_extended` ([-1,1]), or `norm_standard` (z-score). Named identically in dataset code and likelihood code; the legacy `Scaler.standardize` was in fact `norm_minmax_extended`.

### Inference

**Likelihood module**:
A class binding one observational dataset to one (or two) emulators and exposing `computeLikelihood(params) -> (logL, derived)`. Setup (data extraction) is host-side; the evaluation must be pure JAX on the new stack.
_Avoid_: likelihood class, dataset likelihood

**Prior dict**:
The ordered mapping from sampled parameter name to uniform prior bounds, assembled from emulator `param_stats` plus nuisance parameters (`lognoise`, foreground coefficients). Its ordering defines the sampled vector layout.

**Inference run**:
One execution of the inference pipeline: a directory under `inferences/<inference_id>/` holding the copied inference options, copies of the emulators used, and the sampler output chains.
_Avoid_: experiment

**Derived parameter**:
A quantity computed alongside the log-likelihood and stored with each sample (e.g. per-band `xHI` predictions), not sampled.

**Parity**:
The migration acceptance standard. Level 1: converted legacy weights reproduce forward passes numerically. Level 2: a freshly trained emulator plus nested sampling reproduces legacy posteriors statistically (overlapping contours), not bitwise.

### Legacy

**Legacy**:
The frozen PyTorch/PolyChord implementation preserved under `src/CosmicDawnSynergies/legacy/` on the JAX branch. Reference only; not maintained.
