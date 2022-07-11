# Powerspectra Analysis
Model inference from 21cm power specrum observations, notably (i) power spectrum emulator and (ii) likelihood for data (e.g. HERA upper limits)

## Emulators
I so far used 3 emulators, 1 of which is uploaded here (poweremu).

* Old 'classic' emulator, vaguely based on Sudipta Sidker's emulator but without the PCR (did not improve performance now) and also completely rewritten.
* Harry's 'globalemu' emulator. See https://github.com/htjb/globalemu
* My new 'poweremu' emulator, inspired by Harry's globalemu idea but for powerspectra (2D input). The main trick is to treat k and z as input parameters, instead of treating the spectra as output arrays.

### Notes on final (TM) emulator files

Final PS emulators for HERA, trained on z=7-11 and k=0.02-3:

* Pk_emu_m_Sims_adaptive.pkl (initially different name)
* Pk_emu_m_RadLyA_adaptive.pkl (initially different name)
    
Final PS emulator for SARAS comparisons, trained on z=7-31 and k=0.1-0.5:

* Pk_emu_m4_RadLyA_adaptive.pkl


## 21cmSim runs
The data can be found on `AFdata/21cmDATA/` (?), containing 3 directories:
* `EmulatorPS` (July 2020?, used by Sudipta Sikder's emulator)
* `Radio_and_LyAheating_Itamar` (March to July 2021, used by Stefan's initial emulator). This dataset incorporates new effects (LyA heating), and part of the data includes RSDs
* `Sims2021` (November 2021, used by Peter Sims' analysis with Harry's globalemu and other emulators, now also used by Stefan's emulators)

All data sets have some NaNs and zeros in places where they should not have zeros (e.g. temperatures) as shown here for the latest `Sims2021` data set
![Sims2021 NaNs and Zeros](https://github.com/CosmicDawnLab/powerspectra_analysis/blob/main/images/gaps_Sims2021.png)

## Emulators
These colormaps show the width/2 of the confidence intervals for different z and k, i.e. a value of 0.1 means a confidence interval of +/-10%, or +15%/-5% etc.
### Sims2021 data
![Sims2021 image](https://github.com/CosmicDawnLab/powerspectra_analysis/blob/main/images/emulator_zkmap_Sims2021.png)
### Radio_and_LyAheating_Itamar data
![Radio_and_LyAheating_Itamar image](https://github.com/CosmicDawnLab/powerspectra_analysis/blob/main/images/emulator_zkmap_Radio_and_LyAheating_Itamar_m4.png)

## Observational data
HERA IDR2 (Public Data Release 1), IDR3 not yet public.
![IDR2 image](https://github.com/CosmicDawnLab/powerspectra_analysis/blob/main/images/data_HERA_IDR2.png)
