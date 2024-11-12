# CosmicDawnSynergies
Fork of Stefan's old powerspectra_analysis. This package is used for model inference and includes likelihoods for 21cm power specrum observations (HERA), radio background temperature (Table 2 of Dowell & Taylor (2018)), integrated X-ray background (Hickox & Markevitch (2006) and Harrison et al. (2016)), and SARAS 3 (Singh et al. 2022).

## Emulators
For the joint analysis I used 4 emulators.

* A global signal emulator trained using Harry's 'globalemu'. See https://github.com/htjb/globalemu
* Stefan's 'poweremu' emulator, inspired by Harry's globalemu idea but for powerspectra (2D input). The main trick is to treat k and z as input parameters, instead of treating the spectra as output arrays.
* An X-ray background emulator which uses the same approach as the power spectrum emulator, but takes just the Emin energies as an additional input (instead of k and z)
* A star formation rate emulator which also uses the same approach as the power spectrum emulator but takes just the redshift as input.

## Usage
Add any new likelihoods as a class in the likelihood.py file. It should have a method called computeLikelihood which returns a single log likelihood.

The nested sampling is run with run_mcmc.py in scripts/.

More info will follow...





## How to install pypolychord on Azimuth

### Step-by-Step Instructions

1. **Load MPI compilers:**
    ```bash
    module load gnu12/12.2.0
    module load openmpi4/4.1.5
    ```

    If you encounter an error indicating that `libhwloc.so.15` shared library is missing:

    a. Verify `hwloc/2.9.3` is loaded or load it:

        module load hwloc/2.9.3
        

    b. If `libhwloc.so.15` is still not found, check if `/opt/ohpc/pub/libs/hwloc/lib` is in the `LD_LIBRARY_PATH`:
        
        
        echo $LD_LIBRARY_PATH
        

    c. If it isn't, add it to the `LD_LIBRARY_PATH`:
        
        
        export LD_LIBRARY_PATH=/opt/ohpc/pub/libs/hwloc/lib:$LD_LIBRARY_PATH


    d. If `libhwloc.so.15` is still not found, check if it exists:

        find /opt/ohpc/pub/libs/hwloc/lib -name libhwloc.so.15


    e. If it does exist and is still not found, create a symbolic link:
        ```bash
        sudo ln -s /opt/ohpc/pub/libs/hwloc/lib/libhwloc.so.15 /usr/lib64/libhwloc.so.15
        ```

2. **If you are missing the Python headers, install them:**
    ```bash
    sudo dnf install python3-devel
    ```

3. **Install pypolychord:**
    ```bash
    pip install git+https://github.com/PolyChord/PolyChordLite@master
    ```
