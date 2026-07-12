from setuptools import setup, find_packages

setup(
    name='CosmicDawnSynergies',
    version='0.2.0',
    description='A package for astrophysical parameter inference with 21cmSPACE (JAX)',
    author='spochinda',
    author_email='sp2053@cam.ac.uk',
    url='https://github.com/CosmicDawnLab/CosmicDawnSynergies',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    extras_require={
        # Only needed by scripts/convert_pth_to_orbax.py to read legacy .pth emulators
        'convert': ['torch'],
    },
)
