import radio_cutoff_calc as rad
import numpy as np

path = '../models_21cmSim/data/Sims2021/'
filematch="fRad"

all_params = rad.get_data_params_run(path, data_type='params', match=filematch, antimatch="RSD")
all_sfr = rad.get_data_params_run(path, data_type='meanSFR', match=filematch, antimatch="RSD")
all_pk = rad.get_data_params_run(path, data_type='pk', match=filematch, antimatch="RSD")
fr = all_params[:,8]
z = np.arange(6,51)


zmin_z8 = rad.get_z_all(all_sfr, fr, z=8)
np.save("lwa_z8_checks_2sigma.npy", {"params":all_params, "allowed":zmin_z8})

zmin_z10 = rad.get_z_all(all_sfr, fr, z=10)
np.save("lwa_z10_checks_2sigma.npy", {"params":all_params, "allowed":zmin_z10})

#np.save("Tot_fRadRSD_LWA.npy", {"params":all_params, "allowed":zmin_test})

#get_zmin_all returns something called zoffs for each of the models passed to it.
# The argument is a) SFR(z) as 45-long array and b) f_R as number
# The it interpolates SFR in more accuracy (sfr_dense, z_dense)
# From that it computes nu_today and T_today [???]
# The it iterates through all measured frequencies, computes T_model,
#    and compares whether the current observational limit allows for max(T_model)
#    If not it increases zoff to the redshift where T_model goes above T_obs_curr
# So zoffs is, for each of the model, the redshift when the model emission runs over
#    measurements and when we'd have to turn it off. 


