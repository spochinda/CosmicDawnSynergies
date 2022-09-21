# Load the emulator wrapper
from codes.emulator_poweremu import *
# And a particular emulator file
emu = poweremu(loadfile="data/trained_emulators_poweremu/pk_emu_sims_5x100_offset10_nsample1000_v2.pkl",
                   preprocesss_log_x=False, preprocess_y=True, offset=10)

# Simple wrappers to explain which parameter is what
def model_of_z_and_k(z, k, Rmfp=50, log10fStar=-1, log10Vc=1.3, log10fX=-1, powerInd=1.3, numin=0.6, tau=0.0561, log10fradio=2):
    ''' Take redshift, wave number, and parameters (dimensionless, and k in h/Mpc) and return power spectrum Delta² (in mK²) '''
    zlow = 6
    zhigh = 36
    klow = 0.0445
    khigh = 1.633
    # Check that k nd z are within range covered by emulator, otherwise return 0
    if k>khigh or k<klow or z>zhigh or z<zlow:
        return 0
    else:
        return emu.predict([z, k, Rmfp, log10fStar, log10Vc, log10fX, powerInd, numin, tau, log10fradio])

model_of_z_and_k(z=10, k=0.2, Rmfp=50, log10fStar=np.log10(0.3), log10Vc=np.log10(16.5), log10fX=np.log10(0.1), powerInd=1, numin=0.8, tau=0.0561, log10fradio=np.log10(100))

# Note: These are the priors we place on the parameters / extends of the training data:
#   "Rmfp": [10, 70],
#   "log10fStar": [-4, np.log10(0.5)],
#   "log10Vc": [np.log10(4.2), 2],
#   "log10fX": [-5, 3],
#   # powerInd and numin are discrete parameters, only trained for these values:
#   "powerInd": [1, 1.3, 1.5],
#   "numin": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0],
#   "tau": [0.02, 0.1],
#   "log10Fr": [-1, 6]


def model_of_k_array(z, k_array, Rmfp=50, log10fStar=-1, log10Vc=1.3, log10fX=-1, powerInd=1.3, numin=0.6, tau=0.0561, log10fradio=2):
    ''' Take one redshift, a wave number array, and parameters (dimensionless, and k in h/Mpc) and return power spectrum Delta² (in mK²) '''
    par0 = np.array([z, None, Rmfp, log10fStar, log10Vc, log10fX, powerInd, numin, tau, log10fradio])
    params=np.tile(par0, (len(k_array), 1))
    params[:,1] = k_array
    return emu.predict(params)

def model_of_z_array(z_array, k, Rmfp=50, log10fStar=-1, log10Vc=1.3, log10fX=-1, powerInd=1.3, numin=0.6, tau=0.0561, log10fradio=2):
    ''' Take a redshift array, one wave number, and parameters (dimensionless, and k in h/Mpc) and return power spectrum Delta² (in mK²) '''
    par0 = np.array([None, k, Rmfp, log10fStar, log10Vc, log10fX, powerInd, numin, tau, log10fradio])
    params=np.tile(par0, (len(z_array), 1))
    params[:,0] = z_array
    return emu.predict(params)

# Plot demos
import matplotlib.pyplot as plt
plt.figure(constrained_layout=True)
kplot_emulator = np.linspace(0.0445, 1.633, 100)
kplot_extrapolated = np.linspace(0, 2, 100)

for log10fStar in np.linspace(-3, np.log10(0.5), 10):
    dsq = model_of_k_array(z=20, k_array=kplot_extrapolated, Rmfp=50, log10fStar=log10fStar, log10Vc=np.log10(16.5), log10fX=np.log10(0.1), powerInd=1, numin=0.8, tau=0.0561, log10fradio=np.log10(100))
    plt.semilogy(kplot_extrapolated, dsq, ls=":")
for log10fStar in np.linspace(-3, np.log10(0.5), 10):
    dsq = model_of_k_array(z=20, k_array=kplot_emulator, Rmfp=50, log10fStar=log10fStar, log10Vc=np.log10(16.5), log10fX=np.log10(0.1), powerInd=1, numin=0.8, tau=0.0561, log10fradio=np.log10(100))
    plt.semilogy(kplot_emulator, dsq, label="$f_*=10^{{{0:.1f}}}$".format(log10fStar))

plt.legend()
plt.ylabel(r"Dimensionless power spectrum $\Delta^2$ [mK²]")
plt.ylim(1,4e3)
plt.title("Extrapolation in dotted, do not use for inference")
plt.xlabel("Redshift z")
plt.savefig("images/demo_k.png", dpi=600)
plt.show()

# Plot demo
import matplotlib.pyplot as plt
plt.figure(constrained_layout=True)
zplot = np.linspace(7,30,100)
for log10fStar in np.linspace(-3, np.log10(0.5), 10):
	dsq = model_of_z_array(z_array=zplot, k=0.3, Rmfp=50, log10fStar=log10fStar, log10Vc=np.log10(16.5), log10fX=np.log10(0.1), powerInd=1, numin=0.8, tau=0.0561, log10fradio=np.log10(100))
	plt.semilogy(zplot, dsq, label="$f_*=10^{{{0:.1f}}}$".format(log10fStar))

plt.legend()
plt.ylabel(r"Dimensionless power spectrum $\Delta^2$ [mK²]")
plt.ylim(1,4e3)
plt.xlabel("Redshift z")
plt.savefig("images/demo_z.png", dpi=600)
plt.show()
