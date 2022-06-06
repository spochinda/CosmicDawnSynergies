# Powerspectra Analysis
Model inference from 21cm power specrum observations, notably (i) power spectrum emulator and (ii) likelihood for data (e.g. HERA upper limits)

## Emulators
I used ~3 emulators, one of which is uploaded here.

* Old 'classic' emulator, vaguely based on Sudipta Sidker's emulator but without the PCR (did not improve performance now) and also completely rewritten.
* Harry's 'globalemu' emulator. See https://github.com/htjb/globalemu
* My new 'poweremu' emulator, inspired by Harry's globalemu idea but for powerspectra (2D input). The main trick is to treat k and z as input parameters, instead of treating the spectra as output arrays.

## Model data
The data can be found on `AFdata/21cmDATA/` (?), containing 3 directories:
* `EmulatorPS` (July 2020?, used by Sudipta Sikder's emulator)
* `Radio_and_LyAheating_Itamar` (March to July 2021, used by Stefan's initial emulator). This dataset incorporates new effects (LyA heating), and part of the data includes RSDs
* `Sims2021` (November 2021, used by Peter Sims' analysis with Harry's globalemu and other emulators, now also used by Stefan's emulators)