# CosmicDawnSynergies
This package is used for model inference and includes likelihoods for 21-cm power specrum observations (HERA), radio background temperature (Table 2 of Dowell & Taylor (2018)), integrated X-ray background (Hickox & Markevitch (2006) and Harrison et al. (2016)), and SARAS 3 (Singh et al. 2022). In addition, the code contains the likelihood function used for the Cantabrigians parameter inference analysis in the SKA Science Data Challenge 3b.

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