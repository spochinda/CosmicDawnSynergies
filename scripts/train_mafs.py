import anesthetic
from margarine.maf import MAF
from margarine.marginal_stats import calculate
from tensorflow import keras
from scipy.stats import ks_2samp



paramNames = [
             "log10fstarII",
             "log10fstarIII",
             "log10Vc",
             "log10fX",
             #"alpha", "nu_0",#
             #"zeta",
             "tau",
             "log10fradio",
             #"pop",#
    #"a0", "a1", "a2", "a3", "a4", "a5", "a6", "std21",
]

files = [#"non-public/h1c_idr2_1Chandra_1LWA_1SARAS_globalemufinal_nlive_1000/run_h1c_idr2",
         #"non-public/h1c_idr2_0Chandra_0LWA_0SARAS_globalemu_nlive_10000/run_h1c_idr2",
         #"non-public/no_idr_1Chandra_0LWA_0SARAS_globalemufinal_nlive_10000/run_no_idr",
         #"non-public/no_idr_0Chandra_1LWA_0SARAS_globalemuz6_nlive_10000/run_no_idr",
         #"non-public/no_idr_0Chandra_0LWA_1SARAS_globalemu_nlive_1000/run_no_idr",
         
         "non-public/1HERA_1Chandra_1LWA_1SARAS_globalemu315emu14test2idr3_nlive_1000/run",
         #"non-public/1HERA_0Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_h1c_idr2",
         #"non-public/0HERA_1Chandra_0LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr",
         #"non-public/0HERA_0Chandra_1LWA_0SARAS_globalemu315emu14_nlive_10000/run_no_idr",
         #"non-public/0HERA_0Chandra_0LWA_1SARAS_globalemu315emu14_nlive_1000/run_no_idr", #**emu13
        ]

samples = [anesthetic.read_chains(root=file) for file in files]

dkl = []
dkl_lower = []
dkl_upper = []
 
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=25,
    decay_rate=0.9)

for i,(sample,file) in enumerate(zip(samples,files)):
    #if "1SARAS" in file:
    fn = file.split("/")[-2].split("_globalemu")[0].split("_")
    fn = "_".join([element[1:] for element in fn if "1" in element])+"_MAF.pkl"
    print(fn)
    sample_values = sample[paramNames].values #just the Astro params as an np.array for margarine
    sample_weights = sample.get_weights() # weights as np for margarine.
    sample_maf = MAF(theta=sample_values, weights=sample_weights, learning_rate=lr_schedule)
    sample_maf.train(15000, early_stop=True)#True)
    sample_stats = calculate(sample_maf).statistics()
    sample_maf.save("data/margarine/" + fn)
    #dkl.append(sample_stats.iloc[0,0])
    #dkl_lower.append(sample_stats.iloc[0,1])
    #dkl_upper.append(sample_stats.iloc[0,2])


#mafs = [MAF.load("data/margarine/" + file.split("/")[2] + "_MAF2.pkl") for file in files]

for i,(sample,file) in enumerate(zip(samples,files)):
    #fn = file.split("/")[-2].split("_globalemu")[0].split("_")
    #fn = "_".join([element[:] for element in fn if "1" in element])+"_MAF2.pkl"
    fn = file.split("/")[-2].split("_globalemu")[0].split("_")
    fn = "_".join([element[1:] for element in fn if "1" in element])+"_MAF.pkl"
    print(fn)
    maf = MAF.load("data/margarine/"+fn)
    for j,name in enumerate(paramNames):
        kstest = ks_2samp(data1=sample.posterior_points()[name].values, 
                          data2=maf.sample(100000)[:,j]#round( np.sum( samples[1].get_weights() * np.ones(len(samples[1])) ) ))[:,-3]
                         )

        if kstest.pvalue < 0.05:
            print(name,round(kstest.pvalue,3))


