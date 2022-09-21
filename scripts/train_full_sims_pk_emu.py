from codes.emulator_poweremu import *
from codes.loader_21cmSim import *
from codes.tools import *
import matplotlib.pyplot as plt
from copy import deepcopy
# Redshift and k ranges used in data, and load params and powerspectra
## 21cmSim uses these redshifts for all outputs, except xHI.
z_array = np.arange(6,50.01,1)
## And these ones for xHI.
z_xHI_array = np.arange(0,30.001,0.1)
## Finally get the wavenumbers [1/cMpc] from the files. They
## should be all identical but double check for new data.
k_array = load_files('data/models_21cmSim/Sims2021/', middle="_sims_", name="K", key='Kout', endings=["fRad"])
k_array = k_array[0]
# Little h for wave number conversions, use h from simulation
h=0.6704


# Load Sims2021 models (8 parameters)
Pk = load_files("data/models_21cmSim/Sims2021/", name="Pk", middle="_sims_", endings=["fRad"])
PT = load_files("data/models_21cmSim/Sims2021/", name="PT", middle="_sims_", endings=["fRad"])
TS = load_files("data/models_21cmSim/Sims2021/", name="TS", middle="_sims_", endings=["fRad"])
TK = load_files("data/models_21cmSim/Sims2021/", name="TK", middle="_sims_", endings=["fRad"])
Trad = load_files("data/models_21cmSim/Sims2021/", name="Trad", key="Tradout", middle="_sims_", endings=["fRad"])
xA = load_files("data/models_21cmSim/Sims2021/", name="xA", middle="_sims_", endings=["fRad"])
xHI = load_files("data/models_21cmSim/Sims2021/", name="xHI", middle="_sims_", endings=["fRad"])
Pk_Sims, [PT, TS_Sims, TK_Sims, Trad_Sims, xA_Sims, xHI_Sims] = remove_powerspectra_nans(Pk, [PT, TS, TK, Trad, xA, xHI])
PL_Sims = PT9_to_PL8(PT)

# Judge emulator quality, checking 68 and 95% limits
# Note: When saying which parameters you rule out, the emulator underestimating the
#       powerspectrum (+..%) is "conservative", so rather large + tail than - tail.
def calculate_accuracy(emu, test_x, test_y, add_rsd=None):
    print(np.shape(test_x))
    if add_rsd is None:
        pred_y = emu.predict(test_x)
    else:
        pred_y = emu.predict(np.hstack((test_x, np.ones([len(test_x), 1])*add_rsd)))

    deltas = np.log10((test_y+10)/(pred_y+10))
    return deltas

def score(emu, test_x, test_y, add_rsd=None):
    deltas = calculate_accuracy(emu, test_x, test_y, add_rsd=add_rsd)
    limit68 = 10**confidence_level(deltas, level=0.68)
    limit95 = 10**confidence_level(deltas, level=0.95)
    limit997 = 10**confidence_level(deltas, level=0.997)
    print("68% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit68-1)), np.sum(np.abs(limit68-1)))+"\n95% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit95-1)), np.sum(np.abs(limit95-1)))+"\n99.7% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit997-1)), np.sum(np.abs(limit997-1)))+"\n(assuming 10 mK² level threshold; +% means test>pred)")
    return limit68, limit95, limit997

def hist(emu, test_x, test_y, add_rsd=None):
    deltas = calculate_accuracy(emu, test_x, test_y, add_rsd=add_rsd)
    plt.hist(deltas, bins=100)
    plt.show()
    return deltas

# Emulator error as function of z and k
def zkmap(emu, full_x, full_y, zlow=6, zhigh=36, klow=0.0445, khigh=1.633, add_rsd=None, geomspace=False):
    # Make a colormap showing emulator error as a function of k and z
    def test_emu_kz(emu,z,k, PL=full_x, Pk=full_y):
        test_x, test_y = gen_training(1, PL, Pk, seed=0, fix_k=k, fix_z=z)
        limit68, limit95, limit997 = score(emu, test_x, test_y, add_rsd=add_rsd)
        return (limit68[1]-limit68[0])/2, (limit95[1]-limit95[0])/2, (limit997[1]-limit997[0])/2
    # Compute values
    zarr = np.arange(zlow, zhigh+0.1, 1)
    if geomspace:
        karr = np.geomspace(klow, khigh, 10)
    else:
        karr = np.arange(klow, khigh+0.01, 0.1)
    tarr1 = np.ones([len(zarr), len(karr)])
    tarr2 = np.ones([len(zarr), len(karr)])
    tarr3 = np.ones([len(zarr), len(karr)])
    for i in range(len(zarr)):
        for j in range(len(karr)):
            print("z =", zarr[i], "k =", karr[j])
            tarr1[i,j], tarr2[i,j], tarr3[i,j] = test_emu_kz(emu,zarr[i],karr[j])
    # Make plot
    zax, kax = make_axes_pcolor(zarr, karr)
    plt.subplot(211)
    plt.suptitle("Emulator average confidence interval sizes (e.g. +15/-5% is 0.1)")
    plt.title("68% CLs")
    plt.pcolormesh(zax, kax, tarr1.T)
    plt.ylabel("Wavenumber k h/cMpc")
    plt.colorbar(label="Error bar size")
    plt.subplot(212)
    plt.title("95% CLs")
    plt.pcolormesh(zax, kax, tarr2.T)
    plt.ylabel("Wavenumber k h/cMpc")
    plt.colorbar(label="Error bar size")
    plt.tight_layout()
    plt.savefig("zkmap.png", dpi=600)
    plt.show()

# Train an emulator on the whole available k and z range
# Make it one layer deeper to improve performance
layers = (100, 100, 100, 100, 100)
# Make training data. Sample 1000 k-z pairs per data set due to the large range
nsample = 1000
# Offse 10 mK² for dynamical range
offset = 10

# Train emulator or load from file?
run_training = False

PL_Sims_train, PL_Sims_test, Pk_Sims_train, Pk_Sims_test = train_test_split(PL_Sims, Pk_Sims, test_size=0.2, random_state=42)
print("Generating training data ...")
if run_training:
    train_x, train_y = gen_training(nsample, PL_Sims_train, Pk_Sims_train, zlow=6, zhigh=36, klow=0.0445, khigh=1.633, progress=True)
    ## An additional run with fixed k fixed z data set to make sure the large range is not compromising emulator performance
    ptrain_x, ptrain_y = gen_training(100, PL_Sims_train, Pk_Sims_train, fix_k=0.192, fix_z=8)
test_x, test_y = gen_training(10, PL_Sims_test, Pk_Sims_test, seed=0, zlow=6, zhigh=36, klow=0.0445, khigh=1.633, progress=True)
ptest_x, ptest_y = gen_training(1, PL_Sims_test, Pk_Sims_test, seed=1, fix_k=0.192, fix_z=8)

# Emulator. Use SGD so we can do adaptive learning rate. Use preprocessing as required.
if run_training:
    emu = poweremu(learning_rate="adaptive", solver="sgd", hidden_layer_sizes = layers,
    			   preprocesss_log_x=False, preprocess_y=True, offset=offset,
    			   n_iter_no_change=5, max_iter=9999, batch_size=200)
    
    
    emu.train(train_x, train_y)
    emu.save("data/trained_emulators_poweremu/pk_emu_sims_5x100_offset10_nsample1000_v2.pkl")
else:
    emu = poweremu(loadfile="data/trained_emulators_poweremu/pk_emu_sims_5x100_offset10_nsample1000_v1.pkl",
                   preprocesss_log_x=False, preprocess_y=True, offset=offset)
    score(emu, test_x, test_y)
    zkmap(emu, PL_Sims_test, Pk_Sims_test)

