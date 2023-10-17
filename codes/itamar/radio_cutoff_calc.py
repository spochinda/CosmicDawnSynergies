import numpy
import astropy.constants as c
import astropy.units as u
import os
from scipy.io import loadmat
from tqdm import trange
import json

def get_cosmo():
    f = open("codes/itamar/cosmo.json","r")
    cosmo = json.load(f)
    f.close()
    return cosmo

cosmo = get_cosmo()
H0 = cosmo['H0']
Om = cosmo['Om']
OLambda = cosmo['OLambda']

def Hubble_const(z, H0=H0, Om=Om, OLambda=OLambda):
    Hz = H0*numpy.sqrt(Om*(1+z)**3) # EdS
    return Hz

def get_mat_data(mat):
    if type(mat) == str:
        mat = loadmat(mat)

    for v,k in zip(mat.values(), mat.keys()):
        if type(v) == numpy.ndarray:
            #print('Got mat data: {}'.format(k))
            data = v
    return data


def get_data_params_run(p, data_type='pk', k_plot = 0.1, match="", antimatch=""):
    print("get_data_params_run: Trying to load files of type", data_type, "in path", p)
    print("get_data_params_run: Only considering files with the string", match, "and without the string", antimatch)

    files = os.listdir(p)
    if data_type == 'pk':
        fname = 'Pk_'
    elif data_type == 'ps':
        fname = 'PS_'
    elif data_type == 'k':
        fname = 'KK'
    elif data_type == 't21':
        fname = 'T21'
    elif data_type == 'params':
        fname = 'PT_'
    elif data_type == 'Param':
        fname = 'Param_'
    elif data_type == 'sfr':
        fname = 'SFR'
    else:
        fname = data_type

    print("get_data_params_run: Assuming the filename starts with", fname)

    k = None
    data_random_order = []
    sizes = []
    print("get_data_params_run: Available files are", files)
    for f in files:
        if match not in f:
            continue
        if antimatch in f:
            continue
        else:
            print("get_data_params_run: Considering", f)
        if f[:len(fname)] == fname:
            print("get_data_params_run: Filename works, extracting data.")
            mat = loadmat('{}{}'.format(p,f))
            data_single_file = get_mat_data(mat)
            sizes += [data_single_file.shape[0]]
            data_random_order += [data_single_file]
            #print(f, sizes[-1])

        if f[:2] == 'K_':
            mat = loadmat('{}{}'.format(p,f))
            k = get_mat_data(mat)
        elif f[:2] == 'KK':
            mat = loadmat('{}{}'.format(p,f))
            k = get_mat_data(mat)

    if len(data_random_order)>1:
        all_data = []
        for i in numpy.argsort(sizes)[::-1]:
            all_data += [data_random_order[i]]
        all_data = numpy.concatenate(all_data)
    else:
        all_data = data_random_order[0]

    if (data_type == 'pk') or (data_type == 'ps') :
        if k_plot:
            if k is None:
                k = numpy.load('k_param_runs.npy')

            k_idx = numpy.argmin(abs(k_plot - k))

            all_data = all_data[:,:,k_idx]

    return all_data


def get_radio_sed(sed_type, power=-0.7):

    '''
    Load raw SED
    Eqn. 5 in "What does the first highly-redshifted 21-cm detection tell us about early galaxies?"
    Gives the valie at 150 MHz, extrapolate to lower frequencies with spectral index = -0.7
    '''
    if sed_type == 'power_law':
        nu = numpy.logspace(6,13,1000)
        sed = (nu/(150*10**6))**(power) # times frad times SFR/(m_solar yr-1)
        log_sed = 22 + numpy.log10(sed)
    else:
        nu, sed = 0,0
        
    return numpy.log10(nu), log_sed
        
def get_T_radio_today(z_dense, sfr_dense):
    nu_today = numpy.logspace(-2, 1.1, 100)*10**9 * u.Hz # Hz
    lambda_today  = c.c/nu_today
    log_nu, log_sed = get_radio_sed('power_law')
    dz = abs(z_dense[1] - z_dense[0])
    T = numpy.zeros([z_dense.size,nu_today.size])
    for t_idx, (ldba, nu) in enumerate(zip(lambda_today, nu_today)):
        factor = (  ldba**2/(2*c.k_B)  )*(  c.c/(4*numpy.pi)  )
        for z_idx, z in enumerate(z_dense):
            sfr = sfr_dense[z_idx]
            Hz = Hubble_const(z) * u.km/u.s/u.Mpc
            A = (1/Hz)*(1/(1+z)) * dz
            log_nu_emmit = numpy.log10(nu.to(u.Hz).value*(1 + z))
            log_sed_interp = numpy.interp(log_nu_emmit, log_nu, log_sed)
            val = A*factor*10**(log_sed_interp)*(u.W  /u.Hz) *sfr /(u.Mpc)**3
            T[z_idx:, t_idx] += val.to(u.K).value

    return nu_today, T


def get_zmin_all(sfrs,  frs, print_flag = False):

    [nu_obs, T_obs, dT_obs] = numpy.load('codes/itamar/LWA1_with_err.npy')
    T_obs = T_obs+2*dT_obs
    z_sfrs = numpy.arange(6,51)
    zoffs = []
    nof_models = len(sfrs)
    for i in trange(nof_models):
        sfr = sfrs[i]
        fr = frs[i]

        # only the early redshifts have sfr
        z_has_sfr = z_sfrs[sfr > 10**(-7)]
        sfr_has_sfr = sfr[sfr > 10**(-7)]

        # Interpolate sfr for the new dense z sampling
        z_dense = numpy.linspace(numpy.min(z_has_sfr), numpy.max(z_has_sfr), 100)
        sfr_dense = 10**(numpy.interp(z_dense, z_has_sfr, numpy.log10(sfr_has_sfr) ))

        nu_today, T_today = get_T_radio_today(z_dense[::-1], sfr_dense[::-1])

        # Incrementally increase zoff from tightest constraint
        zoff = 6
        for nu_obs_curr, T_obs_curr in zip(nu_obs,T_obs):
            nu_idx = numpy.argmin(abs(nu_obs_curr - nu_today.value))
            T_model = T_today[:,nu_idx]*fr
            if T_obs_curr > numpy.max(T_model):
                pass
            else:
                zoff_curr = z_dense[::-1][numpy.argmin(T_model < T_obs_curr)]
                if zoff_curr > zoff:
                    zoff = zoff_curr

        # This is useless
        zoff_idx = numpy.argmin(abs(zoff - z_dense[::-1]) )
        # This is just append
        zoffs += [zoff]

    return zoffs

def get_z_all(sfrs,  frs, z=8, print_flag = False):
    [nu_obs, T_obs, dT_obs] = numpy.load('codes/itamar/LWA1_with_err.npy')
    T_obs = T_obs+2*dT_obs
    z_sfrs = numpy.arange(6,51)
    zoffs = []
    nof_models = len(sfrs)
    for i in trange(nof_models):
        try:
            sfr = sfrs[i]
            fr = frs[i]
    
            # only the early redshifts have sfr
            z_has_sfr = z_sfrs[sfr > 10**(-7)]
            sfr_has_sfr = sfr[sfr > 10**(-7)]
    
            # Interpolate sfr for the new dense z sampling
            z_dense = numpy.linspace(z-0.01,z+0.01,2)
            sfr_dense = 10**(numpy.interp(z_dense, z_has_sfr, numpy.log10(sfr_has_sfr) ))
    
            nu_today, T_today = get_T_radio_today(z_dense[::-1], sfr_dense[::-1])
    
            # Incrementally increase zoff from tightest constraint
            allowed = True
            for nu_obs_curr, T_obs_curr in zip(nu_obs,T_obs):
                nu_idx = numpy.argmin(abs(nu_obs_curr - nu_today.value))
                T_model = T_today[:,nu_idx]*fr
                if T_obs_curr > numpy.max(T_model):
                    pass
                else:
                    allowed = False
                    continue
        except:
            allowed = numpy.nan
            print("NaN at i =", i)
        zoffs += [allowed]
    return zoffs
