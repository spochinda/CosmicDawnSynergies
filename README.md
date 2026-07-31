# CosmicDawnSynergies
This package is used for model inference and includes likelihoods for 21-cm power specrum observations (HERA), radio background temperature (Table 2 of Dowell & Taylor (2018)), integrated X-ray background (Hickox & Markevitch (2006) and Harrison et al. (2016)), and SARAS 3 (Singh et al. 2022). In addition, the code contains the likelihood function used for the Cantabrigians parameter inference analysis in the SKA Science Data Challenge 3b.

## Installation
Dependencies are managed with [uv](https://docs.astral.sh/uv/) via `pyproject.toml`/`uv.lock`:
```bash
uv sync                    # everything except pypolychord — see below
uv sync --extra polychord  # + pypolychord, once you have a compiler and MPI (see Installing pypolychord)
```
`requires-python` is capped at `<3.13`: `basicsr`'s legacy build script fails on 3.13+.

Plain `pip` also works: `pip install -e .` (or `pip install -e ".[polychord]"`).

## Reproducing the SDC3b results
`analysis/reproduce_sdc3b.sh` trains both emulators, runs PS1/PS2 inference, and generates all comparison
plots in one go.

**With a working environment already** (PolyChord, PyTorch, etc. installed):
```bash
git clone https://github.com/spochinda/CosmicDawnSynergies.git && ./CosmicDawnSynergies/analysis/reproduce_sdc3b.sh
```
The script finds its own location, so it can be run from anywhere; it uses whatever `python` is first on
`PATH`.

**Building the environment from scratch with `uv`**: `uv sync --extra polychord` currently fails on the
upstream PolyChordLite bug below, so install pypolychord separately first (patched clone method under
[Installing pypolychord](#installing-pypolychord)), then:
```bash
git clone https://github.com/spochinda/CosmicDawnSynergies.git
uv sync --project CosmicDawnSynergies
VIRTUAL_ENV=CosmicDawnSynergies/.venv uv pip install --no-build-isolation -e /tmp/PolyChordLite
uv run --project CosmicDawnSynergies ./CosmicDawnSynergies/analysis/reproduce_sdc3b.sh
```
The `VIRTUAL_ENV=` prefix matters: if another venv is already active in your shell (e.g. `(cosmicdawn)` in
your prompt), `uv pip install --project <dir>` silently installs into *that* venv instead — `--project`
doesn't override an already-set `VIRTUAL_ENV` for that subcommand. Skip it and `pypolychord` ends up in the
wrong place, and the final `uv run` fails with `ModuleNotFoundError: No module named 'pypolychord'` despite
the install appearing to succeed.

## Emulators
The emulator code is based on the BasicSR framework. For example the SDC3b cylindrical power spectrum emulator can be trained using the command from the root directory:
```
python train.py -opt options/emulators/Pk_SDC3b.yml
```
All options required to train emulators are contained in the .yml files. Trained emulators will be located in trained_emulators/

## Inference
The inference part of the code uses polychord. Similar to emulator training, all options for inference are contained within .yml files in options/inference/*.yml. For example the SDC3b inference can be done using a similar command:
```
python inference.py -opt options/inference/sdc3b.yml
```
Results will be located in the inferences/ directory

## Adding more likelihoods
To perform inference with new data, new likelihood classes can be added in the likelihood.py file. New likelihoods should have a computeLikelihood method which is used during inference. For examples of other likelihood classes have a look at the likelihood.py file.

## Installing pypolychord
Needs a Fortran/C++ compiler and MPI — how much setup that takes depends on where you're running.

### On a laptop
Install a compiler + MPI, then a plain pip install — most of the Azimuth steps below (module system,
`libhwloc.so.15`, `sudo dnf install`) are specific to that cluster and don't apply here:
- **macOS**: `brew install gcc open-mpi`
- **Linux (Debian/Ubuntu)**: `sudo apt install gfortran libopenmpi-dev openmpi-bin`
```bash
pip install git+https://github.com/PolyChord/PolyChordLite@master
# or, from this repo: uv sync --extra polychord
```

**If that fails with `opal_init failed` / `Unable to get the user home directory`**: a bug in PolyChordLite's
own `setup.py`, not this repo, `uv`, or `pip`. `PyPolyChordExtension.run()` builds a near-empty environment
for its internal `make` call (`PATH`/`MPI`/`CURDIR`/`CC`/`CXX`/`FC` only) that omits `HOME`, which newer Open
MPI needs during `mpicc`/`mpifort`. Confirmed on a plain, unsandboxed machine, across `pip`, `uv`, isolated
and `--no-build-isolation` builds, and wheel and editable installs — it's specifically the missing `HOME`.
Workaround: clone locally, patch `setup.py` to also copy `HOME`, install from the patched clone:
```bash
git clone https://github.com/PolyChord/PolyChordLite /tmp/PolyChordLite
python3 -c "
content = open('/tmp/PolyChordLite/setup.py').read()
content = content.replace(
    'env[\"PATH\"] = os.environ[\"PATH\"]',
    'env[\"PATH\"] = os.environ[\"PATH\"]\n        env[\"HOME\"] = os.environ.get(\"HOME\", \"\")'
)
open('/tmp/PolyChordLite/setup.py', 'w').write(content)
"
pip install --no-build-isolation -e /tmp/PolyChordLite
```

### On Azimuth
1. Load compilers: `module load gnu12/12.2.0` and `module load openmpi4/4.1.5`.
   If `libhwloc.so.15` is missing: `module load hwloc/2.9.3`; if that's not enough, check
   `echo $LD_LIBRARY_PATH` includes `/opt/ohpc/pub/libs/hwloc/lib` (add it if not), confirm the file exists
   via `find /opt/ohpc/pub/libs/hwloc/lib -name libhwloc.so.15`, then symlink it in if needed:
   `sudo ln -s /opt/ohpc/pub/libs/hwloc/lib/libhwloc.so.15 /usr/lib64/libhwloc.so.15`.
2. Missing Python headers: `sudo dnf install python3-devel`.
3. Install: `pip install git+https://github.com/PolyChord/PolyChordLite@master`.
