from codes.tools import *
import matplotlib.pyplot as plt

print("=== Checking priors ===")
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

def testprior(pD, PL, pn):
	N = len(pn)
	fig, axes = plt.subplots(nrows=N)
	for i in range(N):
		k = pn[i]
		axes[i].hist(PL.T[i], color="blue", bins=100)
		for j in range(len(pD[k])):
			axes[i].axvline(pD[k][j], color="red")
	plt.show()
	
testprior(priorDict_Sims, PL_Sims, paramNames_Sims_full)
testprior(priorDict_RadLyA, PL_RSD_Itamar, paramNames_RadLyA)
testprior(priorDict_RadLyA, PL_noRSD_Itamar, paramNames_RadLyA)


print("=== Checking trapezoidal PDF function ===")
print("    Case: Real numbers")
fX = np.random.uniform(-5,3,1000000)
fR = np.random.uniform(-1,6,1000000)
fS = np.random.uniform(-5,-0.3,1000000)
fXfS = fX+fS
fRfS = fR+fS
plt.hist2d(fXfS, fRfS, bins=50)
plt.xlim(-10,3)
plt.ylim(-6,6)

plt.figure()
N = 50
alphaplot = np.linspace(-10, 3.3, N)
betaplot = np.linspace(-6, 5.7, N)
ff = np.reshape(np.meshgrid(alphaplot, betaplot), (2, N**2))
Z = [sum_pdf_2d(a,b,-5,3, -1,6, -5,-0.3) for a,b in ff.T]
Z=np.reshape(Z, (N,N))
plt.pcolormesh(alphaplot,betaplot,Z)
plt.xlim(-10,3)
plt.ylim(-6,6)
plt.show()

print("    Case: Diagonal")
plt.figure()
fX = np.random.uniform(0,1,10000000)
fR = np.random.uniform(0,5,10000000)
fS = np.random.uniform(0,8,10000000)
fXfS = fX+fS
fRfS = fR+fS
plt.hist2d(fXfS, fRfS, bins=50)

plt.figure()
N = 50
alphaplot = np.linspace(0, 9, N)
betaplot = np.linspace(0, 13, N)
ff = np.reshape(np.meshgrid(alphaplot, betaplot), (2, N**2))
Z = [sum_pdf_2d(a,b,0,1,0,5,0,8, debug=True) for a,b in ff.T]
Z=np.reshape(Z, (N,N))
plt.pcolormesh(alphaplot,betaplot,Z)
plt.show()

print("    Case: Rectangular")
plt.figure()
fX = np.random.uniform(0,3,1000000)
fR = np.random.uniform(0,2,1000000)
fS = np.random.uniform(0,1,1000000)
fXfS = fX+fS
fRfS = fR+fS
plt.hist2d(fXfS, fRfS, bins=50)
plt.xlim(0,4)
plt.ylim(0,4)

plt.figure()
N = 30
alphaplot = np.linspace(0, 4, N)
betaplot = np.linspace(0, 4, N)
ff = np.reshape(np.meshgrid(alphaplot, betaplot), (2, N**2))
Z = [sum_pdf_2d(a,b,0,3,0,2,0,1, debug=False) for a,b in ff.T]
Z=np.reshape(Z, (N,N))
plt.pcolormesh(alphaplot,betaplot,Z)
plt.xlim(0,4)
plt.ylim(0,4)
plt.show()
