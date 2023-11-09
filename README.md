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