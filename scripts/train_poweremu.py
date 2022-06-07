from codes.loader_21cmSim import *
import matplotlib.pyplot as plt

# Redshift and k ranges used in data, and load params and powerspectra
## 21cmSim uses these redshifts for all outputs, except xHI.
z_array = np.arange(6,50.01,1)
## And these ones for xHI.
z_xHI_array = np.arange(0,30.001,0.1)
## Finally get the wavenumbers [1/cMpc] from the files. They
## should be all identical but double check for new data.
#k_array = load_files('data/models_21cmSim/EmulatorPS/', name='KK', key='KK', middle="", endings=[""])[0]
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Fr", model_generation="new")
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Ar", model_generation="new")
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Fr", model_generation="old")
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Ar", model_generation="old")
k_array = load_files('data/models_21cmSim/Sims2021/', middle="_sims_", name="K", key='Kout', endings=["fRad"])
k_array = k_array[0]
# Little h for wave number conversions, use h from simulation
h=0.6704

from codes.emulator_poweremu import *
from codes.tools import *

# Load Radio_and_LyAheating_Itamar models (5 parameters)
## These models are without RSDs (9927+800-26):
PT = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="PT", model_type="Fr", model_generation="new")
Pk = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="Pk", model_type="Fr", model_generation="new")
Pk_noRSD_Itamar, [PT] = remove_powerspectra_nans(Pk, [PT])
PL_noRSD_Itamar = PT9_to_PL5(PT)
## And these with RSDs (2181-6):
PT = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="PT", endings=["fRad_RSDrand"], middle=None, key="PTout")
Pk = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="Pk", endings=["fRad_RSDrand1"], middle=None, key="PKout1")
Pk_RSD_Itamar, [PT] = remove_powerspectra_nans(Pk, [PT])
PL_RSD_Itamar = PT9_to_PL5(PT)
# Load Sims2021 models (8 parameters)
Pk = load_files("data/models_21cmSim/Sims2021/", name="Pk", middle="_sims_", endings=["fRad"])
PT = load_files("data/models_21cmSim/Sims2021/", name="PT", middle="_sims_", endings=["fRad"])
TS = load_files("data/models_21cmSim/Sims2021/", name="TS", middle="_sims_", endings=["fRad"])
TK = load_files("data/models_21cmSim/Sims2021/", name="TK", middle="_sims_", endings=["fRad"])
Trad = load_files("data/models_21cmSim/Sims2021/", name="Trad", key="Tradout", middle="_sims_", endings=["fRad"])
Pk_Sims, [PT, TS_Sims, TK_Sims, Trad_Sims] = remove_powerspectra_nans(Pk, [PT, TS, TK, Trad])
PL_Sims = PT9_to_PL8(PT)

# Judge emulator quality, checking 68 and 95% limits
# Note: When saying which parameters you rule out, the emulator underestimating the
#       powerspectrum (+..%) is "conservative", so rather large + tail than - tail.

def score(emu, test_x, test_y):
    pred_y = emu.predict(test_x)
    deltas = np.log10((test_y+10)/(pred_y+10))
    limit68 = 10**confidence_level(deltas, level=0.68)
    limit95 = 10**confidence_level(deltas, level=0.95)
    print("68% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit68-1)), np.sum(np.abs(limit68-1)))+"\n95% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit95-1)), np.sum(np.abs(limit95-1)))+"\n(assuming 10 mK² level threshold; +% means test>pred)")
    return deltas


# Emulator architectures:
## We dit a lot of manual optimization previously:
## Also the first two here are for one specific SED only
emu01 = poweremu(loadfile="data/trained_emulators_poweremu/Sims_data_v03_150it_23.02.2022.pkl",preprocesss_log_x=False)
emu02 = poweremu(loadfile="data/trained_emulators_poweremu/PK_emu_Sims_prelim_v3_01.06.2022.pkl",preprocesss_log_x=False)
emu03 = poweremu(loadfile="data/trained_emulators_poweremu/PK_all_emu_Sims_prelim_v4_01.06.2022.pkl",preprocesss_log_x=False)
## But actually the automatic settings work just fine
emu04 = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_Sims_auto100100100.pkl",preprocesss_log_x=False)
emu05 = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_Sims_auto10030105_better.pkl",preprocesss_log_x=False)
emu06 = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_Sims_auto100505_better.pkl",preprocesss_log_x=False)
emu07a = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_Sims_June06_adaptive.pkl",preprocesss_log_x=False)

## (100, 30, 10, 5) layers, with adaptive (SGD) or constant (Adam) learning rates
emu_c = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_RadLyA_June06_constant.pkl",preprocesss_log_x=False)
emu_a = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_RadLyA_June06_adaptive.pkl",preprocesss_log_x=False)
emu_f = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_fixkz_RadLyA_adaptive.pkl",preprocesss_log_x=False)
emu_m = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_moarkz_Sims_adaptive.pkl",preprocesss_log_x=False)

# Training data
#model_generation = "Sims"
model_generation = "RadLyA"

def model_of_k(z, karr, p, emu=None):
    par0 = np.array([z, np.NaN, *p])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    return emu.predict(params)

def model_of_z(zarr, k, p, emu=None):
    par0 = np.array([np.NaN, k, *p])
    params=np.tile(par0, (len(karr), 1))
    params[:,0] = zarr
    return emu.predict(params)

if model_generation == "Sims":
    # Sims training data: [z, k, Rmfp, log10fStar, log10Vc, log10fX, powerInd (discrete), numin (discrete), tau, log10Fr]
    PL_Sims_train, PL_Sims_test, Pk_Sims_train, Pk_Sims_test = train_test_split(PL_Sims, Pk_Sims, test_size=0.2, random_state=42)
    test_x, test_y = gen_training(10, PL_Sims_test, Pk_Sims_test, seed=0)
    train_x, train_y = gen_training(100, PL_Sims_train, Pk_Sims_train)
    ptest_x, ptest_y = gen_training(1, PL_Sims_test, Pk_Sims_test, fix_k=0.192, fix_z=8)

    for e in [emu03, emu04, emu05, emu06, emu07a]:
        print(e.mlp[1].hidden_layer_sizes)
        print(score(e, test_x, test_y))

elif model_generation == "RadLyA":
    # RadLyA training data: [z, k, log10fStar, log10Vc, log10fX, tau, log10Fr, flag (1 for RSD on, 0 for RSD off)]
    PL_RSD_Itamar_train, PL_RSD_Itamar_test, Pk_RSD_Itamar_train, Pk_RSD_Itamar_test = train_test_split(PL_RSD_Itamar, Pk_RSD_Itamar, test_size=0.2, random_state=42)
    train0_x, train0_y = gen_training(10, PL_noRSD_Itamar, Pk_noRSD_Itamar, seed=0, flag=0)
    train1_x, train1_y = gen_training(100, PL_RSD_Itamar_train, Pk_RSD_Itamar_train, seed=1, flag=1)
    train_x = np.concatenate([train0_x, train1_x])
    train_y = np.concatenate([train0_y, train1_y])
    # Special case to test maximum performance possible if using 1 particular point only
    ptrain0_x, ptrain0_y = gen_training(1, PL_noRSD_Itamar, Pk_noRSD_Itamar, seed=0, flag=0, fix_k=0.192, fix_z=8)
    ptrain1_x, ptrain1_y = gen_training(1, PL_RSD_Itamar_train, Pk_RSD_Itamar_train, seed=1, flag=1, fix_k=0.192, fix_z=8)
    ptrain_x = np.concatenate([ptrain0_x, ptrain1_x])
    ptrain_y = np.concatenate([ptrain0_y, ptrain1_y])
    # Reproduce that performance with lots of samples
    mtrain_x, mtrain_y = gen_training(1000, PL_RSD_Itamar_train, Pk_RSD_Itamar_train, seed=1, flag=1)
    m2train_x, m2train_y = gen_training(1000, PL_RSD_Itamar_train, Pk_RSD_Itamar_train, seed=1, flag=1, zhigh=21)

    test_x, test_y = gen_training(10, PL_RSD_Itamar_test, Pk_RSD_Itamar_test, seed=2, flag=1)
    ptest_x, ptest_y = gen_training(1, PL_RSD_Itamar_test, Pk_RSD_Itamar_test, seed=3, flag=1, fix_k=0.192, fix_z=8)

    for e in [emu_c, emu_a, emu_f, emu_m]:
        print(e.mlp[1].hidden_layer_sizes)
        print(score(e, test_x, test_y))
        print(score(e, ptest_x, ptest_y))
    plt.title("Deviation in log10; 0.3 corresponds to Factor 2")
    plt.hist(score(emu_m, ptest_x, ptest_y), bins=50, alpha=0.5, range=(-0.3, 0.3))
    plt.hist(score(emu_f, ptest_x, ptest_y), bins=50, alpha=0.5, range=(-0.3, 0.3))
    plt.show()

# Make a new emulator
layers = (100, 30, 10, 5)
## Adaptive
emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, learning_rate="adaptive", solver="sgd")
## Constant
#emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, solver="adam")

# Train
#emu.train(m2train_x, m2train_y)

# Save
#emu.save("data/trained_emulators_poweremu/x.pkl")

# Question: What's the maximum we can achieve at fix_k=0.192, fix_z=8? [emu_f]
#               68% samples within +14% / -8% of true --> 0.22
#               95% samples within +44% / -33% of true --> 0.77
#           Do we achieve this with the general emulator? [emu_a]
#               68% samples within +24% / -5% of true --> 0.30
#               95% samples within +59% / -19% of true --> 0.77
#           No. Let's run an emulator with 1k oversampling
#               68% samples within +13% / -8% of true --> 0.21
#               95% samples within +54% / -19% of true --> 0.73
#           Awesome! Can we increase the z bounds and still achieve this?
