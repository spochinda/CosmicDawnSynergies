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
#k_array = load_files('data/models_21cmSim/EmulatorPS/', name='KK', key='KK', middle="", endings=[""])[0]
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Fr", model_generation="new")
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Ar", model_generation="new")
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Fr", model_generation="old")
#k_array = load_files('data/models_21cmSim/Radio_and_LyAheating_Itamar/', name='K', key='Kout', model_type="Ar", model_generation="old")
k_array = load_files('data/models_21cmSim/Sims2021/', middle="_sims_", name="K", key='Kout', endings=["fRad"])
k_array = k_array[0]
# Little h for wave number conversions, use h from simulation
h=0.6704

if False:
    # Load Radio_and_LyAheating_Itamar models (5 parameters)
    ## These models are without RSDs (9927+800-26):
    PT = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="PT", model_type="Fr", model_generation="new") #Deltak
    Pk = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="Pk", model_type="Fr", model_generation="new") #parameters mat
    Trad = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="Trad", key="Tradout", model_type="Fr", model_generation="new")
    TK = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="TK", model_type="Fr", model_generation="new")
    T21 = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="T21", model_type="Fr", model_generation="new")
    xA = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="xA", model_type="Fr", model_generation="new")
    xHI = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="xHI", model_type="Fr", model_generation="new")
    SFR = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="SFR", model_type="Fr", model_generation="new", key="meanSFRout")
    meanSFR = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="meanSFR", model_type="Fr", model_generation="new", key="meanSFRout")
    assert np.all(meanSFR==SFR)
    Pk_noRSD_Itamar, [PT, Trad_noRSD_Itamar, TK_noRSD_Itamar, T21_noRSD_Itamar, xA_noRSD_Itamar, xHI_noRSD_Itamar, SFR_noRSD_Itamar] = remove_powerspectra_nans(Pk, [PT, Trad, TK, T21, xA, xHI, SFR])
    PL_noRSD_Itamar = PT9_to_PL5(PT)
    TS_noRSD_Itamar = derive_TS_xRad(xA_noRSD_Itamar, xHI_noRSD_Itamar, TK_noRSD_Itamar, Trad_noRSD_Itamar)[0]

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


# Sims emulators
## PS HERA-range zlow=7, zhigh=11, klow=0.02, khigh=3
Pk_emu_m_Sims = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_m_Sims_adaptive.pkl",preprocesss_log_x=False)
## PS fixed to z=8 and k=0.192
Pk_emu_fixed_Sims = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emuL_fixkz_Sims_adam_2001005025_v2.pkl",preprocesss_log_x=False)

## T, range: z = 6..31
TS_emu_Sims = poweremu(loadfile="data/trained_emulators_poweremu/TSemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TK_emu_Sims = poweremu(loadfile="data/trained_emulators_poweremu/TKemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TR_emu_Sims = poweremu(loadfile="data/trained_emulators_poweremu/TRemu_m3_converged.pkl", preprocesss_log_x=False, offset=1e-3)

# RadLyA Itamar emulators
## PS HERA-range zlow=7, zhigh=11, klow=0.02, khigh=3
Pk_emu_RadLyA_m = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_m_RadLyA_adaptive.pkl",preprocesss_log_x=False)
## PS SARAS-range zlow=7, zhigh=31, klow=0.1, khigh=0.5
Pk_emu_RadLyA_m4 = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_m4_RadLyA_adaptive.pkl",preprocesss_log_x=False)
## PS fixed to z=8 and k=0.192
Pk_emu_RadLyA_fixed = poweremu(loadfile="data/trained_emulators_poweremu/Pk_emu_fixkz_RadLyA_adaptive.pkl",preprocesss_log_x=False)
## T, range: z = 6..31
TK_emu_RadLyA = poweremu(loadfile="data/trained_emulators_poweremu/TK_emu_RayLyA_v1_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TR_emu_RadLyA = poweremu(loadfile="data/trained_emulators_poweremu/TR_emu_RayLyA_v1_converged.pkl", preprocesss_log_x=False, offset=1e-3)
TS_emu_RadLyA = poweremu(loadfile="data/trained_emulators_poweremu/TS_emu_RayLyA_v1_converged.pkl", preprocesss_log_x=False, offset=1e-3)
SFR_emu_RadLyA = poweremu(loadfile="data/trained_emulators_poweremu/SFR_emu_RayLyA_v1_converged.pkl", preprocesss_log_x=False, offset=1e-25)



# Training data
model_generation = "Sims"
#model_generation = "RadLyA"
#model_generation = "TS" "TK" "TR" for Sims
#model_generation = "TempRadLyA" #with manually setting TK or TR, or TS
#model_generation = "SFRRadLyA" #different offset
offset = 1e-3

if model_generation == "Sims":
    layers = (100, 30, 10, 5)
    # Sims training data: [z, k, Rmfp, log10fStar, log10Vc, log10fX, powerInd (discrete), numin (discrete), tau, log10Fr]
    PL_Sims_train, PL_Sims_test, Pk_Sims_train, Pk_Sims_test = train_test_split(PL_Sims, Pk_Sims, test_size=0.2, random_state=42)
    print(PL_Sims_train.shape, Pk_Sims_train.shape)
    train_x, train_y = gen_training(100, PL_Sims_train, Pk_Sims_train)
    ptrain_x, ptrain_y = gen_training(1, PL_Sims_train, Pk_Sims_train, fix_k=0.192, fix_z=8)
    #mtrain_x, mtrain_y = gen_training(1000, PL_Sims_train, Pk_Sims_train)
    #m2train_x, m2train_y = gen_training(1000, PL_Sims_train, Pk_Sims_train, zhigh=21)

    test_x, test_y = gen_training(10, PL_Sims_test, Pk_Sims_test, seed=0)
    ptest_x, ptest_y = gen_training(1, PL_Sims_test, Pk_Sims_test, seed=1, fix_k=0.192, fix_z=8)

    #for e in [emu03, emu04, emu05, emu06, emu07a]:
    #    print(e.mlp[1].hidden_layer_sizes)
    #    print(score(e, test_x, test_y))

    # Emulator error as function of z and k
    def zkmap(emu=Pk_emu_m_Sims, full_x=PL_Sims_test, full_y=Pk_Sims_test, zmin=7, zmax=11, add_rsd=None):
        # Make a colormap showing emulator error as a function of k and z
        def test_emu_kz(emu,z,k, PL=full_x, Pk=full_y):
            test_x, test_y = gen_training(1, PL, Pk, seed=0, fix_k=k, fix_z=z)
            limit68, limit95, limit997 = score(emu, test_x, test_y, add_rsd=add_rsd)
            return (limit68[1]-limit68[0])/2, (limit95[1]-limit95[0])/2, (limit997[1]-limit997[0])/2
        # Compute values
        zarr = np.arange(zmin, zmax+0.1, 1)
        karr = np.arange(0.1, 0.51, 0.1)
        tarr1 = np.ones([len(zarr), len(karr)])
        tarr2 = np.ones([len(zarr), len(karr)])
        tarr3 = np.ones([len(zarr), len(karr)])
        for i in range(len(zarr)):
            for j in range(len(karr)):
                tarr1[i,j], tarr2[i,j], tarr3[i,j] = test_emu_kz(emu,zarr[i],karr[j])
        # Make plot
        zax, kax = make_axes_pcolor(zarr, karr)
        plt.subplot(311)
        plt.suptitle("Emulator average CL sizes (e.g. +15/-5% is 0.1)")
        plt.title("68% CLs")
        plt.pcolormesh(zax, kax, tarr1.T)
        plt.ylabel("Wavenumber k h/cMpc")
        plt.colorbar()
        plt.subplot(312)
        plt.title("95% CLs")
        plt.pcolormesh(zax, kax, tarr2.T)
        plt.ylabel("Wavenumber k h/cMpc")
        plt.colorbar()
        plt.subplot(313)
        plt.title("99.7% CLs")
        plt.pcolormesh(zax, kax, tarr3.T)
        plt.xlabel("Redshift z")
        plt.ylabel("Wavenumber k h/cMpc")
        plt.colorbar()
        plt.show()


elif model_generation == "RadLyA":
    #m: (100, 30, 10, 5)
    #m4: (150, 50, 15, 5)
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
    #mtrain_x, mtrain_y = gen_training(1000, PL_RSD_Itamar_train, Pk_RSD_Itamar_train, flag=1)
    #m2train_x, m2train_y = gen_training(1000, PL_RSD_Itamar_train, Pk_RSD_Itamar_train, flag=1, zhigh=21)
    #m3train_x, m3train_y = gen_training(1000, PL_RSD_Itamar_train, Pk_RSD_Itamar_train, flag=1, zhigh=31)
    #m4train_x, m4train_y = gen_training(1000, PL_RSD_Itamar_train, Pk_RSD_Itamar_train, flag=1, zlow=7, zhigh=31, klow=0.1, khigh=0.5)

    test_x, test_y = gen_training(10, PL_RSD_Itamar_test, Pk_RSD_Itamar_test, seed=2, flag=1)
    ptest_x, ptest_y = gen_training(1, PL_RSD_Itamar_test, Pk_RSD_Itamar_test, seed=3, flag=1, fix_k=0.192, fix_z=8)

    for e in [emu_c, emu_a, emu_f, emu_m]:
        print(e.mlp[1].hidden_layer_sizes)
        print(score(e, test_x, test_y))
        print(score(e, ptest_x, ptest_y))
    plt.title("Deviation in log10 (0.3 corresponds to factor 2)")
    plt.hist(score(emu_m, ptest_x, ptest_y), bins=100, alpha=0.5, range=(-0.7, 0.4), label="Fixed k=0.192 and z=8")
    plt.hist(score(emu_f, ptest_x, ptest_y), bins=100, alpha=0.5, range=(-0.7, 0.4), label="Emulator for 'all' k and z")
    plt.legend()
    plt.show()
elif model_generation == "TempRadLyA":
    layers = (100, 30, 10, 5)
    #T = Trad_noRSD_Itamar[:,:31]
    #T = TK_noRSD_Itamar[:,:31]
    T = TS_noRSD_Itamar[:,:31]
    #layers = (100, 50, 5) #For TS only, and also change 1k to 500
    #T = T21_noRSD_Itamar[:,:31]
    def zmap(emu=TS_emu_RadLyA, full_x=PL_noRSD_Itamar, full_y=T, zmin=6, zmax=36):
        # Make a colormap showing emulator error as a function of k and z
        zarr = np.arange(zmin, zmax+0.1, 1)
        def test_emu_z(emu, z, PL=full_x, Pk=full_y):
            test_x, test_y = gen_training_1d(1, PL, Pk, seed=0, fix_z=z, zarr=zarr)
            limit68, limit95, limit997 = score(emu, test_x, test_y)
            return (limit68[1]-limit68[0])/2, (limit95[1]-limit95[0])/2, (limit997[1]-limit997[0])/2
        # Compute values
        tarr1 = np.ones([len(zarr), 1])
        tarr2 = np.ones([len(zarr), 1])
        tarr3 = np.ones([len(zarr), 1])
        for i in range(len(zarr)):
            tarr1[i,0], tarr2[i,0], tarr3[i,0] = test_emu_z(emu,zarr[i])
        # Make plot
        zax, kax = make_axes_pcolor_1d(zarr, [0])
        plt.subplot(311)
        plt.suptitle("Emulator average CL sizes (e.g. +15/-5% is 0.1)")
        plt.ylabel("68% CLs")
        plt.pcolormesh(zax, kax, tarr1.T)
        plt.colorbar()
        plt.subplot(312)
        plt.ylabel("95% CLs")
        plt.pcolormesh(zax, kax, tarr2.T)
        plt.colorbar()
        plt.subplot(313)
        plt.ylabel("99.7% CLs")
        plt.pcolormesh(zax, kax, tarr3.T)
        plt.colorbar()
    print(np.shape(T))
    print(np.shape(PL_noRSD_Itamar))
    zarr = z_array[:31]
    mask = np.all(np.logical_not(np.logical_or(np.isnan(T), T==0)), axis=-1)
    print("Using", np.sum(mask), "out of", len(mask), "samples")
    print("Defaults zlow=6, zhigh=31")
    print("with zarr in", np.min(zarr), np.max(zarr))
    PL_train, PL_test, T_train, T_test = train_test_split(PL_noRSD_Itamar[mask], T[mask], test_size=0.2, random_state=42)
    train_x, train_y = gen_training_1d(1000, PL_train, T_train, zarr=zarr)
    ptrain_x, ptrain_y = gen_training_1d(1, PL_train, T_train, fix_z=8, zarr=zarr)
    test_x, test_y = gen_training_1d(10, PL_test, T_test, seed=0, zarr=zarr)
    ptest_x, ptest_y = gen_training_1d(1, PL_test, T_test, seed=1, fix_z=8, zarr=zarr)
    score(TS_emu_RadLyA, test_x, test_y)
    zmap(TS_emu_RadLyA); plt.savefig("accuracy_TS_emu_RadLyA.png", dpi=600); plt.close()
    # Always no RSD
elif model_generation == "SFRRadLyA":
    layers = (100, 30, 10, 5)
    #T = Trad_noRSD_Itamar[:,:31]
    #T = TK_noRSD_Itamar[:,:31]
    T = SFR_noRSD_Itamar[:,:31]
    offset = 1e-25
    #T = T21_noRSD_Itamar[:,:31]
    print(np.shape(T))
    print(np.shape(PL_noRSD_Itamar))
    zarr = z_array[:31]
    mask = np.all(np.logical_not(np.logical_or(np.isnan(T), T==0)), axis=-1)
    print("Using", np.sum(mask), "out of", len(mask), "samples")
    print("Defaults zlow=6, zhigh=31")
    print("with zarr in", np.min(zarr), np.max(zarr))
    PL_train, PL_test, T_train, T_test = train_test_split(PL_noRSD_Itamar[mask], T[mask], test_size=0.2, random_state=42)
    train_x, train_y = gen_training_1d(1000, PL_train, T_train, zarr=zarr)
    ptrain_x, ptrain_y = gen_training_1d(1, PL_train, T_train, fix_z=8, zarr=zarr)
    test_x, test_y = gen_training_1d(10, PL_test, T_test, seed=0, zarr=zarr)
    ptest_x, ptest_y = gen_training_1d(1, PL_test, T_test, seed=1, fix_z=8, zarr=zarr)
    # Always no RSD
else:
    key = model_generation
    print("Training key", key)
    T_Sims = {"TS":TS_Sims, "TK":TK_Sims, "TR":Trad_Sims}[key][:,:31]
    zarr = z_array[:31]
    mask = np.all(np.logical_not(np.logical_or(np.isnan(T_Sims), T_Sims==0)), axis=-1)
    print("Using", np.sum(mask), "out of", len(mask), "samples")
    PL = deepcopy(PL_Sims[mask])
    T = T_Sims[mask]
    print("Defaults zlow=6, zhigh=31")
    print("with zarr in", np.min(zarr), np.max(zarr))
    PL_Sims_train, PL_Sims_test, T_Sims_train, T_Sims_test = train_test_split(PL, T, test_size=0.2, random_state=42)
    train_x, train_y = gen_training_1d(1000, PL_Sims_train, T_Sims_train, zarr=zarr)
    ptrain_x, ptrain_y = gen_training_1d(1, PL_Sims_train, T_Sims_train, fix_z=8, zarr=zarr)

    test_x, test_y = gen_training_1d(10, PL_Sims_test, T_Sims_test, seed=0, zarr=zarr)
    ptest_x, ptest_y = gen_training_1d(1, PL_Sims_test, T_Sims_test, seed=1, fix_z=8, zarr=zarr)

    score(TR_emu0, ptest_x[:,1:], ptest_y) #0.09
    score(TR_emu1, ptest_x, ptest_y) #0.10
    score(TR_emu1, test_x, test_y) #0.06
    score(TS_emu0, ptest_x[:,1:], ptest_y) #0.10
    score(TS_emu1, ptest_x, ptest_y) #0.16 0.12
    score(TS_emu1, test_x, test_y) #0.10
    score(TK_emu0, ptest_x[:,1:], ptest_y) #0.10
    score(TK_emu1, ptest_x, ptest_y) #0.20 0.1...
    score(TK_emu1, test_x, test_y) #0.09


#zkmap()
#zkmap(emu=Pk_emu_RadLyA_m, full_x=PL_RSD_Itamar, full_y=Pk_RSD_Itamar, add_rsd=True)
#zkmap(emu=Pk_emu_RadLyA_m4, full_x=PL_RSD_Itamar, full_y=Pk_RSD_Itamar, add_rsd=True, zmax=30)


# Make a new emulator

## Adaptive. Temperature:
#emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers,
#    max_iter=9999, learning_rate="adaptive", solver="sgd", n_iter_no_change=5,
#    tol=0.00001, offset=offset)
    # currently m2 converged runs forgot offset! Otherwise really good after ~75 it (TS)
    # emu_m3_converged done
    #emu.save("data/trained_emulators_poweremu/"+key+"emu_m3_converged.pkl")
#emu.train(train_x, train_y)
#T21
#emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers,
#    max_iter=9999, learning_rate="adaptive", solver="sgd", n_iter_no_change=5,
#    tol=0.00001, offset=1e-3, preprocess_y=False)
#emu.train(train_x, train_y)
#emu.save("data/trained_emulators_poweremu/TK_emu_RayLyA_v1_unconverged.pkl")
# v1 converged: Good models with

## Constant
#emu = poweremu(loadfile=None,preprocesss_log_x=False, hidden_layer_sizes=layers, max_iter=9999, solver="adam")

# Train & Save

#emu.train(ptrain_x, ptrain_y)
#emu.save("data/trained_emulators_poweremu/Pk_emu_fixkz_Sims_adaptive.pkl")

#emu.train(mtrain_x, mtrain_y)
#emu.save("data/trained_emulators_poweremu/Pk_emu_m_Sims_adaptive.pkl")


# Discussion, here focused on RadLyA models:
# Question: What's the maximum we can achieve at fix_k=0.192, fix_z=8? [emu_f]
#               68% samples within +14% / -8% of true --> 0.22
#               95% samples within +44% / -33% of true --> 0.77
#           Do we achieve this with the general emulator? [emu_a]
#               68% samples within +24% / -5% of true --> 0.30
#               95% samples within +59% / -19% of true --> 0.77
#           No. Let's run an emulator with 1k oversampling [emu_m]
#               68% samples within +13% / -8% of true --> 0.21
#               95% samples within +54% / -19% of true --> 0.73
#           Awesome! Can we increase the z bounds and still achieve this?
#           Now with 1k oversampling up to z=21, converged @ 232 iterations
#               68% samples within +15% / -9% of true --> 0.24
#               95% samples within +61% / -19% of true --> 0.80
#           That's alright for now --> Do SARAS+HERA tests with Pk_emu_evenmorez_RadLyA_adaptive.
#           Ah wait need to z=31. Here we go Pk_emu_m3_RadLyA_adaptive.pkl @ 234 converged
#               68% samples within +22% / -11% of true --> 0.33
#               95% samples within +68% / -30% of true --> 0.98
#           Hmm OK. But give it one more try with less k space (m4train) and slightly larger layers (150, 50, 15, 5)
#               ("data/trained_emulators_poweremu/Pk_emu_m4_RadLyA_adaptive.pkl")
#               68% samples within +16% / -10% of true --> 0.26
#               95% samples within +54% / -22% of true --> 0.77
#           Yeah here we go.

#           Let's apply this to Sims though! Let's see what is the best score we can get with fixed k & z:
#               (100, 30, 10, 5), adaptive
#                   68% samples within +23% / -15% of true --> 0.38
#                   95% samples within +87% / -53% of true --> 1.40
#                   99.7% samples within +529% / -82% of true --> 6.11
#               (200, 100, 50, 25), adaptive
#                   68% samples within +18% / -14% of true --> 0.32
#                   95% samples within +87% / -45% of true --> 1.32
#                   99.7% samples within +282% / -79% of true --> 3.61
#               repeat
#                   68% samples within +18% / -15% of true --> 0.33
#                   95% samples within +75% / -48% of true --> 1.24
#                   99.7% samples within +190% / -74% of true --> 2.64
#               Pk_emuL_fixkz_Sims_adaptive
#               (200, 100, 50, 25), adam
#                   68% samples within +15% / -11% of true --> 0.26
#                   95% samples within +60% / -38% of true --> 0.98
#                   99.7% samples within +281% / -69% of true --> 3.51
#               repeat (save as Pk_emuL_fixkz_Sims_adam_2001005025)
#                   68% samples within +19% / -11% of true --> 0.30
#                   95% samples within +65% / -42% of true --> 1.07
#                   99.7% samples within +206% / -80% of true --> 2.86
#               repeat (save as Pk_emuL_fixkz_Sims_adam_2001005025_v2)
#                   68% samples within +17% / -8% of true --> 0.26
#                   95% samples within +67% / -37% of true --> 1.04
#                   99.7% samples within +323% / -75% of true --> 3.98
#              (400, 200, 100, 50), adam
#                  68% samples within +13% / -9% of true --> 0.22
#                  95% samples within +57% / -35% of true --> 0.92
#                  99.7% samples within +179% / -86% of true --> 2.66
#           Okay, (200, 100, 50, 25) w/ adam is the score to beat, saved as Pk_emuL_fixkz_Sims_adam_2001005025_v2
#           Run Sims emulator aiming for 68% ~ 0.26 and 95% ~ 1 scores.
#               Here we go! (100, 30, 10, 5) layers with adaptive sgd for 234 iterations, trained on mtrain
#               gives excellent numbers, took ages. Pk_emu_m_Sims_adaptive.pkl
#                   68% samples within +14% / -10% of true --> 0.24
#                   95% samples within +55% / -30% of true --> 0.85
#                   99.7% samples within +229% / -75% of true --> 3.04



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

#todo: make some plots of emulators
