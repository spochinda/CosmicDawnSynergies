import os
import sys
#sys.path.append("/home/sp2053/rds/hpc-work/CosmicDawnSynergies/src/")
from CosmicDawnSynergies.likelihood import LikelihoodNeutralFraction, LikelihoodRadioBackground, LikelihoodSARAS3, LikelihoodXRB
from CosmicDawnSynergies.likelihood_hera import likelihood
import numpy as np
#from codes.likelihood import *
#from C.likelihood_hera import *
from margarine.maf import MAF
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior,LogUniformPrior
import time
from mpi4py import MPI

import torch
from CosmicDawnSynergies.train_tools import poweremu_torch, MLP
from CosmicDawnSynergies.train_tools import Scaler

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--band_idx", type=int, default=1, help="Band index")
args = parser.parse_args()
band_idx = args.band_idx


path = os.path.dirname(os.getcwd()) #"/home/sp2053/rds/hpc-work/powerspectra_analysis/" #"/Users/simonpochinda/venvs/testenv/lib/python3.8/site-packages/powerspectra_analysis/"

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    print(f"Running script with number of processes: {size}",flush=True)

network_opt = dict(in_dim=11, hidden_dim=100, n_hidden = 6, out_dim = 1, dropout = 0.1, use_norm_dropout = False, use_attn = False)
optimizer_opt = dict(lr=1e-3, weight_decay=1e-4)
train_opt = dict(epochs=10000, profiling=False, loss_fn=torch.nn.MSELoss())
scale_opt = dict()

poweremu = poweremu_torch(network=MLP, network_opt=network_opt, 
                          optimizer=torch.optim.Adam, optimizer_opt=optimizer_opt, 
                          train_opt=train_opt, scale_opt=scale_opt,
                          device='cpu')
poweremu.load_network("/home/sp2053/rds/hpc-work/CosmicDawnSynergies/data/trained_emulators_poweremu/MLP_7_2.pth")
scaler = Scaler(poweremu.scale_opt)



texDict = {"log10fstarII": r"$\log_{10} \left(f_{\rm \ast, II}\right)$",
        "log10fstarIII": r"$\log_{10} \left(f_{\rm \ast, III}\right)$",
        "log10Vc": r"$\log_{10}\left(V_c\right)$",
        "log10fX": r"$\log_{10}\left( f_{\rm X}\right)$",
        "alpha": r"$\alpha$",
        "nu_0": r"$\nu_{\rm 0}$",
        "tau": r"$\tau$",
        "log10fradio": r"$\log_{10} \left(f_{\rm r}\right)$",
        "pop": r"$\rm pop$",
        }

texDict_SARAS3 = {
    "a0": r"$a_{0}$",
    "a1": r"$a_{1}$",
    "a2": r"$a_{2}$",
    "a3": r"$a_{3}$",
    "a4": r"$a_{4}$",
    "a5": r"$a_{5}$",
    "a6": r"$a_{6}$",
    "std21": r"$\sigma_{21}$",
    }

priorDict = {
            "log10fstarII": scaler.standardize(np.log10([1e-3, 0.5]), **scaler.scale_opt["logf_star_II"]["stats"]),
            "log10fstarIII": scaler.standardize(np.log10([1e-3, 0.5]), **scaler.scale_opt["logf_star_III"]["stats"]),
            "log10Vc": scaler.standardize(np.log10([4.2, 100]), **scaler.scale_opt["logVc"]["stats"]),
            "log10fX": scaler.standardize(np.log10([1e-3, 1e3]), **scaler.scale_opt["logfx"]["stats"]),
            "alpha": [-0.5, 2.5],
            "nu_0": [-0.5, 16.5],#[100:100:1500, 2000, 3000],
            "tau": scaler.normalize(np.array([0.054-3*0.007, 0.054+3*0.007]), **scaler.scale_opt["tau"]["stats"]),#??
            "log10fradio": scaler.standardize(np.log10([1e-1, 99990.]), **scaler.scale_opt["logfrad"]["stats"]),
            "pop": [-0.5, 2.5],#[231, 232, 233],#??
            }

if rank==0:
    print("Prior dict: ", priorDict, flush=True)


priorDict_SARAS3 = {
    "a0": [-10,10],
    "a1": [-10,10],
    "a2": [-10,10],
    "a3": [-10,10],
    "a4": [-10,10],
    "a5": [-10,10],
    "a6": [-10,10],
    "std21": [1e-2, 0.5],
}
    
discrete_params = {
            "alpha": scaler.standardize(np.array([1, 1.3, 1.5])),
            "nu_0": scaler.standardize(np.array([*range(100,1500,100), 1500, 2000, 3000])),
            "pop": scaler.standardize(np.array([231, 232, 233])),
            #"feed": [0, 1],
            #"delay": [0, 0.75]
}


output_names_HERA = {"logL_HERA": r"\log L_\mathrm{HERA}", 
                        #"logL_HERA_B1": r"\log L_\mathrm{HERA,1}", "logL_HERA_B2": r"\log L_\mathrm{HERA,2}", 
                        }

output_names_Chandra = {"logL_Chandra": r"\log L_\mathrm{Chandra,tot}", 
                        #"S_{1-2}": r"S(1-2 keV)", "S_{2-8}": r"S(2-8 keV)", "S_{8-24}": r"S(8-24 keV)", "S_{20-50}": r"S(20-50 keV)"
                        }

output_names_LWA = {"logL_LWA": r"\log L_\mathrm{LWA}"}

output_names_SARAS3 = {"logL_SARAS": r"\log L_\mathrm{SARAS3}"}

output_names_xHI = {"logL_xHI": r"\log L_\mathrm{x_{HI}}"}
    

#selection = [{"1": {"D": {"kstart":0.36}}}, {"2": {"C": {"kstart":0.34}}}]
#dpath = [
#    'data/observations_H1C_IDR3/Deltasq_Band_1_Field_D_idr3.h5',
#    'data/observations_H1C_IDR3/Deltasq_Band_2_Field_C_idr3.h5']
#like_hera = likelihood(
#    datapath=dpath,
#    decimation_factor=2,
#    selections=selection,
#    return_individual_loglikes=False, #Can only use false with this new likelihood module approach
#    emupath='data/trained_emulators_poweremu/dsq_emu_n500_l100100100100_t1e-05_o0.pkl',
#    output_names = output_names_HERA
#)

dpath = [f"/home/sp2053/rds/hpc-work/CosmicDawnSynergies/scripts/data/observations_H6C_IDR2/Deltasq_Band_{i}.h5" for i in range(7,0,-1)] #8 is z=5.6
#dpath = [f"/home/azimuth/venvs/inference/lib/python3.9/site-packages/CosmicDawnSynergies/scripts/data/observations_H6C_IDR2/Deltasq_Band_{band_idx}.h5",]
selection = len(dpath) * [None,] #len(range(3,0,-1))*[None,] #7:0
like_hera = likelihood(datapath=dpath, selections=selection, zero_fill=1e-50,
                decimation_factor=2, set_negative_to_zero=True, theory_err=0.2, kstart_modulo=True,
                return_individual_loglikes=False, debug=False,
                emupath=None,#'data/trained_emulators_poweremu/dsq_emu_n500_l100100100100_t1e-05_o0.pkl',
                output_names = {"logL_HERA": r"\log L_\mathrm{HERA}"},
                scaler = scaler,
                rank = rank
                 )
like_hera.model_dsq = poweremu.model
like_hera.model_dsq.eval()

#Setup sampling
if rank==0: print([like_hera.data[key]["0"]["z"] for key in like_hera.data.keys()], flush=True)

class UniformDiscretePrior:
    def __init__(self, a):
        self.a = a
        self.b = len(a)
        #self.b = b

    def __call__(self, x):
        return self.a[ np.floor(self.b * x).astype(int) ] #+ (self.b-self.a) * x


use_MAFs = False
"""hera_maf = MAF.load("data/margarine/HERA_MAF_IDR3.pkl")
Chandra_maf = MAF.load("data/margarine/Chandra_MAF.pkl")
LWA_maf = MAF.load("data/margarine/LWA_MAF.pkl")
saras_maf = MAF.load("data/margarine/SARAS_MAF.pkl") 

mafs = [hera_maf, saras_maf]
theta_lims = np.stack(
    [np.max([maf.theta_min for maf in mafs],axis=0), 
    np.min([maf.theta_max for maf in mafs],axis=0)],
    axis=1)
#maf_param_indices = [0,1,2,3,6,7]

"""
def dumper(live, dead, logweights, logZ, logZerr):
    # params, derived, b0 (lowest loglikelihood at birth), l0 (loglike)
    print("Last dead point:", dead[-1], flush=True)



#constraints = np.array(list(product([0,1], repeat=4))) #repeats=number of constraints
#constraints = np.unique(constraints, axis=0)
#constraints = np.array([set for set in constraints.tolist() if set!=[0,0,0,0]]) #remove 0 constraints



LikelihoodModules = np.array([like_hera, 
                        LikelihoodXRB(use_MAFs=use_MAFs, output_names = output_names_Chandra), 
                        LikelihoodRadioBackground(use_MAFs=use_MAFs, output_names = output_names_LWA), 
                        LikelihoodSARAS3(use_MAFs=use_MAFs, output_names = output_names_SARAS3),
                        LikelihoodNeutralFraction(use_MAFs=use_MAFs, output_names = output_names_xHI)
                        ])

#constraints = np.array([(1,1,1,1), (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]).astype(bool) #HERA, Chandra, LWA, SARAS
constraints = np.array([(1,0,0,0,0)]).astype(bool) #HERA, Chandra, LWA, SARAS, xHI
nlives = [1000,]

for i,(nlive,(HERA,Chandra,LWA,SARAS,xHI)) in enumerate(zip(nlives,constraints)):
    priorDict_ = priorDict.copy() if not SARAS else {**priorDict.copy(), **priorDict_SARAS3.copy()}
    texDict_ = texDict.copy() if not SARAS else {**texDict.copy(), **texDict_SARAS3.copy()}

    if use_MAFs:
        pop_params = list(discrete_params.keys()) if not SARAS else [*list(discrete_params.keys()), *list(priorDict_SARAS3.keys())]
        [priorDict_.pop(p) for p in pop_params]
        [texDict_.pop(p) for p in pop_params]
    
    paramNames = list(priorDict_.keys())

    def prior(cube):
        theta = np.zeros_like(cube)
        if not use_MAFs:
            for i,(name,bounds) in enumerate(priorDict_.items()):
                if name in discrete_params.keys():
                    theta[i] = UniformDiscretePrior(discrete_params[name])(cube[i])
                else:
                    if name!="std21":
                        theta[i] = UniformPrior(*bounds)(cube[i])    
                    else:
                        theta[i] = LogUniformPrior(*bounds)(cube[i])
        else:
            for i,t in enumerate(theta_lims):
                theta[i] = UniformPrior(*t)(cube[i]) #edit bounds with maf.theta_min, maf.theta_max
        return theta


    def loglikelihood(p, include_HERA=HERA, include_Chandra=Chandra,  include_LWA=LWA, include_SARAS=SARAS):
        try:
            logL_nDerived = [likelihood.computeLikelihood(p) for likelihood in LikelihoodModules[constraints[i]]]
            logL = np.sum([item[0] if isinstance(item,list) else item for item in logL_nDerived])

            logL_nDerived_flattened = np.array([])
            for item in logL_nDerived:
                if isinstance(item,list):
                    for sub_item in item:
                        logL_nDerived_flattened = np.append(logL_nDerived_flattened, sub_item)
                else:
                    logL_nDerived_flattened = np.append(logL_nDerived_flattened, item)
        except Exception as e:
            print("error was: ", e)

        return logL, logL_nDerived_flattened#, *T_emus]


    bandsNfields = 0
    redshifts=[]
    for band in like_hera.data.keys():
        for field in like_hera.data[band].keys():
            redshifts.append(like_hera.data[band][field]["z"])
            bandsNfields = bandsNfields+1
    temp_redshifts, index = np.unique(redshifts, return_index=True)
    index.sort()
    redshifts = np.array(redshifts)[index] #preserve redshift order
    nDims = len(paramNames) #if not use_MAFs else len(np.array(paramNames)[maf_param_indices])
    nDerived = np.sum([likelihood.nDerived for likelihood in LikelihoodModules[constraints[i]]]) #2 #(bandsNfields*HERA-1)*0 + LikelihoodXRB(use_MAFs=use_MAFs).nDerived + LikelihoodRadioBackground(use_MAFs=use_MAFs).nDerived + LikelihoodSARAS3(use_MAFs=use_MAFs).nDerived #if not use_MAFs else 2 #+3*len(redshifts) #2*9 + 3*9 # (selections, number of bands*fields, +6 temperature outputs) # idr4=(9bands*2fields+3temps*9bands), idr2=(2bands*1fields+3temps*9redshifts(AKA bands)) #2
    settings = PolyChordSettings(nDims, nDerived)
    settings.nlive = nlive #00 #2000
    settings.base_dir = path+'/scripts/non-public/{0}HERA_{1}Chandra_{2}LWA_{3}SARAS_{4}xHI_bidx{6}_nlive_{5}'.format(*constraints[i].astype(int),settings.nlive,"all2")
    settings.file_root = 'run'
    settings.do_clustering = True
    settings.read_resume = False    
    
    if rank==0:
        start_time = time.time()
        print("Redshifts: ", redshifts, flush=True) 
        print("Constraints, nDerived:", HERA,Chandra,LWA,SARAS, nDerived, flush=True) 
        print("Starting sampling. Base dir: {0}".format(settings.base_dir), flush=True)
    
    if True:
        output = run_polychord(loglikelihood, nDims, nDerived, settings, prior, dumper)
        #output.logZ for evidence
        redshifts_str = [str(z) for z in redshifts]
        polychordnames = []
        for name,texStr in texDict_.items():
            polychordnames.append((name, texStr[1:-1]))
        for likelihood in LikelihoodModules[constraints[i]]:
            for item in likelihood.output_names.items():
                polychordnames.append(item)

        output.make_paramnames_files(polychordnames)
        if rank==0:
            print("{0} took {1:.2f} hours ---".format( settings.base_dir.split("/")[-1], (time.time() - start_time) / 3600))
