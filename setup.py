from setuptools import setup, find_packages

setup(
    name='CosmicDawnSynergies',  # Change this to your package's name
    version='0.1.0',
    description='A package for astrophysical parameter inference with 21cmSPACE',
    author='spochinda',
    author_email='sp2053@cam.ac.uk',
    url='https://github.com/CosmicDawnLab/CosmicDawnSynergies',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires = ">=3.8",
    #install_requires=open('requirements.txt').read().splitlines(),
)

