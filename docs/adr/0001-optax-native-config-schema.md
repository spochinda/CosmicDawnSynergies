# Optax-native training config schema (breaks legacy yml compatibility)

The JAX branch redesigns the `train:` block of emulator options around optax
vocabulary (schedules as first-class config, optimizer by optax name) instead of
mapping the old torch-flavoured keys (`optimizer_opt: type: Adam`,
`scheduler: type: MultiStepLR`) internally. Legacy options files therefore do
**not** run on this branch; only the parity set (Pk SDC3b, xHI SDC3b, seed51
inference opts) was hand-ported, and no converter tool exists. We accepted the
compatibility break to avoid maintaining a torch-vocabulary emulation layer
forever in a single-user research code; the rejected alternative (same schema,
internal mapping) would have let every old yml run unchanged but frozen torch
naming into a torch-free codebase.
