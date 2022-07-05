from codes.tools import *
import matplotlib.pyplot as plt
from codes.loader_21cmSim import *
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


_,converged,ini = derive_TS_xRad(xA_Sims, xHI_Sims, TK_Sims, Trad_Sims)

limits95 = []
for i in range(45):
    ini_tmp = ini[:,i]
    converged_tmp = converged[:,i]
    true_tmp=TS_Sims[:,i]
    deltas = np.log10((converged_tmp+1e-3)/(true_tmp+1e-3))
    limit95 = 10**confidence_level((deltas.flatten()), level=0.95)
    limits95.append(limit95)

limits95 = np.array(limits95)

plt.figure()
plt.title("T_S formula accuracy (Sims data), limits in log-space")
plt.plot(z_array, limits95[:,0], label="Lower 95% deviation")
plt.plot(z_array, limits95[:,1], label="Upper 95% deviation")
plt.fill_between(z_array, y1=limits95[:,0], y2=limits95[:,1], alpha=0.5)
plt.axhline(1, ls="dashed", color="black", label="TS from simulation")
plt.xlabel("Redshift z")
plt.ylabel("Ratio of Formula-TS over Simulation-TS")
plt.legend()

true = TS_Sims
plt.figure()
plt.scatter(true,converged-ini, marker=".", label="Converged - Initial")
plt.plot([1e-8,1e8], [1e-8,1e8])
plt.plot([1e-8,1e8], [1e-9,1e7])
plt.plot([1e-8,1e8], [1e-10,1e6])
plt.loglog()
plt.legend()
plt.figure()
plt.scatter(true,converged-true, marker=".", label="Converged - True")
plt.plot([1e-8,1e8], [1e-8,1e8])
plt.plot([1e-8,1e8], [1e-9,1e7])
plt.plot([1e-8,1e8], [1e-10,1e6])
plt.loglog()
plt.legend()
plt.figure()
plt.scatter(true[:,2],converged[:,2]-ini[:,2], marker=".", label="z=8 Converged - Initial")
plt.plot([1e-8,1e8], [1e-8,1e8])
plt.plot([1e-8,1e8], [1e-9,1e7])
plt.plot([1e-8,1e8], [1e-10,1e6])
plt.loglog()
plt.legend()
plt.figure()
plt.scatter(true[:,2],converged[:,2]-true[:,2], marker=".", label="z=8 Converged - True")
plt.plot([1e-8,1e8], [1e-8,1e8])
plt.plot([1e-8,1e8], [1e-9,1e7])
plt.plot([1e-8,1e8], [1e-10,1e6])
plt.loglog()
plt.legend()
deltas = np.log10((converged+1e-3)/(true+1e-3))
limit68 = 10**confidence_level((deltas.flatten()), level=0.68)
limit95 = 10**confidence_level((deltas.flatten()), level=0.95)
limit997 = 10**confidence_level((deltas.flatten()), level=0.997)
print("68% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit68-1)), np.sum(np.abs(limit68-1)))+"\n95% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit95-1)), np.sum(np.abs(limit95-1)))+"\n99.7% samples within +{1:.0f}% / {0:.0f}% of true --> {2:.2f}".format(*(100*(limit997-1)), np.sum(np.abs(limit997-1)))+"\n(assuming 10 mK² level threshold; +% means test>pred)")
plt.show()
