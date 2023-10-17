from codes.loader_21cmSim import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_gaps(Pk, x=k_array/h, y=z_array, label="Gaps", Temps=None, labels=None):
	nans = np.sum(np.isnan(Pk), axis=0)
	zeros = np.sum(Pk==0, axis=0)
	xdiff = np.diff(x)[0]
	ydiff = np.diff(y)[0]
	xplot = [x[0]-xdiff, *np.array(x+xdiff)]
	yplot = [y[0]-ydiff, *np.array(y+ydiff)]
	if Temps is None:
		fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(16,8))
	else:
		nTemps = len(Temps)
		fig = plt.figure(figsize=(16,8+nTemps))
		ax1  = plt.subplot2grid((3+nTemps, 2), (0, 0), rowspan=3)
		ax2  = plt.subplot2grid((3+nTemps, 2), (0, 1), rowspan=3)
		ax3a = []; ax3b = []
		for i in range(nTemps):
			ax3a.append(plt.subplot2grid((3+nTemps, 2), (3+i, 0)))
			ax3b.append(plt.subplot2grid((3+nTemps, 2), (3+i, 1)))
	fig.suptitle(label)
	im = ax1.pcolormesh(y, x, nans.T)
	divider=make_axes_locatable(ax1); cax = divider.append_axes('right', size='5%', pad=0.05); fig.colorbar(im, cax=cax, orientation="vertical")
	ax1.set_yscale("log")
	ax1.set_xlabel("Redshift z")
	ax1.set_ylabel("Wavenumber k [h/Mpc]")
	ax1.set_title("NaNs")
	im = ax2.pcolormesh(y, x, zeros.T)
	divider=make_axes_locatable(ax2); cax = divider.append_axes('right', size='5%', pad=0.05); fig.colorbar(im, cax=cax, orientation="vertical")
	ax2.set_yscale("log")
	ax2.set_xlabel("Redshift z")
	ax2.set_ylabel("Wavenumber k [h/Mpc]")
	ax2.set_title("Zeros")
	if Temps is not None:
		for i in range(nTemps):
			nans = np.sum(np.isnan(Temps[i]), axis=0)
			zeros = np.sum(Temps[i]==0, axis=0)
			im = ax3a[i].pcolormesh(yplot,[0,1],np.array([nans]))
			ax3a[i].set_yticklabels([])
			ax3a[i].set_ylabel(labels[i])
			divider=make_axes_locatable(ax3a[i]); cax = divider.append_axes('right', size='5%', pad=0.05); fig.colorbar(im, cax=cax, orientation="vertical")
			im = ax3b[i].pcolormesh(yplot,[0,1],np.array([zeros]))
			ax3b[i].set_yticklabels([])
			ax3b[i].set_ylabel(labels[i])
			divider=make_axes_locatable(ax3b[i]); cax = divider.append_axes('right', size='5%', pad=0.05); fig.colorbar(im, cax=cax, orientation="vertical")
		ax3a[-1].set_xlabel("Redshift z")
		ax3b[-1].set_xlabel("Redshift z")
	plt.tight_layout()

# Load Radio_and_LyAheating_Itamar models (5 parameters)
## These models are without RSDs (9927+800-26):
Pk = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="Pk", model_type="Fr", model_generation="new")
plot_gaps(Pk, label="Gaps in Radio_and_LyAheating_Itamar (NoRSD) training data")
plt.savefig("images/gaps_Radio_and_LyAheating_Itamar_noRSD.png", dpi=600)
## And these with RSDs (2181-6):
Pk = load_files("data/models_21cmSim/Radio_and_LyAheating_Itamar/", name="Pk", endings=["fRad_RSDrand1"], middle=None, key="PKout1")
plot_gaps(Pk, label="Gaps in Radio_and_LyAheating_Itamar (RSD) training data")
plt.savefig("images/gaps_Radio_and_LyAheating_Itamar_RSD.png", dpi=600)

# Load Sims2021 models (8 parameters)
Pk = load_files("data/models_21cmSim/Sims2021/", name="Pk", middle="_sims_", endings=["fRad"])
TS = load_files("data/models_21cmSim/Sims2021/", name="TS", middle="_sims_", endings=["fRad"])
TK = load_files("data/models_21cmSim/Sims2021/", name="TK", middle="_sims_", endings=["fRad"])
Trad = load_files("data/models_21cmSim/Sims2021/", name="Trad", key="Tradout", middle="_sims_", endings=["fRad"])
plot_gaps(Pk, label="Gaps in Sims2021 training data", Temps=[TS, TK, Trad], labels=["T_S", "T_K", "T_rad"])
plt.savefig("images/gaps_Sims2021.png", dpi=600)
plt.show()


