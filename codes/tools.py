import scipy.interpolate as sip
import scipy.optimize as sop
import scipy.integrate as sin
import numpy as np

def confidence_level(samples, weights=None, level=0.68):
    assert level<1, "Level >= 1!"
    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = np.array(samples)[order]
    weights = np.array(weights)[order]/np.sum(weights)
    # Compute inverse cumulative distribution function
    cumsum = np.cumsum(weights)
    S = np.array([np.min(samples), *samples, np.max(samples)])
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
    invcdf = sip.interp1d(CDF, S)
    # Find smallest interval
    distance = lambda a, level=level: invcdf(a+level)-invcdf(a)
    res = sop.minimize(distance, (1-level)/2, bounds=[(0,1-level)], method="Nelder-Mead")
    return np.array([invcdf(res.x[0]), invcdf(res.x[0]+level)])

def powerInd_and_numin_from_index(index):
    powerInds = [1, 1.3, 1.5]
    numins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0]
    powerInd = powerInds[int(index/len(numins))]
    numin = numins[index % len(numins)]
    return powerInd, numin

powerIndPrior = 1/3
def numinPrior(index):
    numins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0]
    d = np.diff(np.log10(numins))
    weights = np.array([0, *(d/2)]) + np.array([*(d/2), 0])
    numin_index = index % len(numins)
    return weights[numin_index]/np.sum(weights)


from codes.loader_21cmSim import *
## 21cmSim uses these redshifts for all outputs, except xHI.
z_array = np.arange(6,50.01,1)
## And these ones for xHI.
z_xHI_array = np.arange(0,30.001,0.1)
# Get the wavenumbers [1/cMpc] from the files. They
# should be all identical but double check for new data.
k_array = load_files('data/models_21cmSim/Sims2021/', middle="_sims_", name="K", key='Kout', endings=["fRad"])[0]
# Little h for wave number conversions, use h from simulation
h=0.6704

# Tools useful for emulator training data sampling
def powerspec_of_z_k_hovercMpc(data, z_array=z_array, k_array_over_h=k_array/h):
    # Interpolate a given power spectrum (data) at z and k within the respective bounds
    # Make sure to convert to h/cMpc and never use non-h units anywhere anymore
    f = sip.interp2d(z_array, np.log(k_array_over_h), np.log(data+1).T, kind="linear", fill_value=0, bounds_error=False)
    return lambda z,k: np.exp(f(z, np.log(k)))-1

def gen_training(n_over, params, data, fix_z=False, fix_k=False, seed=None, flag=None,
                 zlow=7, zhigh=11, klow=0.02, khigh=3):
    # Sample random z and k from the power spectra interpolations
    # Note: Use k in h/cMpc !
    # n_over = number of random (z,k) samples per model
    # params, data: Parameters and power spectra of models
    # Returns n_over*len(params) samples
    training_x = []
    training_y = []
    if seed is not None:
        np.random.seed(seed)
    for m in np.random.permutation(len(params)):
        p = params[m]
        z = [fix_z]*n_over if fix_z else np.random.uniform(low=zlow, high=zhigh, size=n_over)
        k = [fix_k]*n_over if fix_k else np.random.uniform(low=klow, high=khigh, size=n_over)
        f = powerspec_of_z_k_hovercMpc(data=data[m])
        for j in range(n_over):
            if flag is None:
                training_x.append([z[j],k[j], *p])
            else:
                training_x.append([z[j],k[j], *p, flag])
            r = f(z[j],k[j])
            training_y.append(r)
    indices = np.random.choice(len(training_y), size=len(training_y), replace=False)
    return np.array(training_x)[indices], np.array(training_y)[indices,0]

def Tinterp1d(z, Tarr, zarr=z_array):
    f = sip.interp1d(zarr, Tarr)
    return f(z)

def gen_training_1d(n_over, params, data, fix_z=False, seed=None, flag=None,
                    zlow=6, zhigh=31, zarr=z_array):
    # n_over = number of random (z,k) samples per model
    # params, data: Parameters and power spectra of models
    # Returns n_over*len(params) samples
    training_x = []
    training_y = []
    if seed is not None:
        np.random.seed(seed)
    for m in np.random.permutation(len(params)):
        p = params[m]
        z = [fix_z]*n_over if fix_z else np.random.uniform(low=zlow, high=zhigh, size=n_over)
        Ti = Tinterp1d(z, data[m], zarr=zarr)
        for j in range(n_over):
            if flag is None:
                training_x.append([z[j], *p])
            else:
                training_x.append([z[j], *p, flag])
            training_y.append(Ti[j])
    indices = np.random.choice(len(training_y), size=len(training_y), replace=False)
    return np.array(training_x)[indices], np.array(training_y)[indices]


paramNames_Sims_poly = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
paramNames_Sims_full = ["Rmfp", "log10fStar", "log10Vc", "log10fX", "powerInd", "numin", "tau", "log10Fr"]
priorDict_Sims = {
             "Rmfp": [10, 70],
             "log10fStar": [-4, np.log10(0.5)],
             "log10Vc": [np.log10(4.2), 2],
             "log10fX": [-5, 3],
             "powerInd": [1, 1.3, 1.5], #discrete
             "numin": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0], #discrete
             "tau": [0.02, 0.1],
             "log10Fr": [-1, 6]}

paramNames_RadLyA = ["log10fStar", "log10Vc", "log10fX", "tau", "log10Fr"]
priorDict_RadLyA = {
             "log10fStar": [-3, np.log10(0.5)],
             "log10Vc": [np.log10(4.2), 2],
             "log10fX": [-4, 3],
             "tau": [0.035, 0.088],
             "log10Fr": [0, 5]}

texDict = {"Rmfp": r"$R_{\rm mfp}$",
           "log10fStar": r"$\log_{10} f_{\rm star}$",
           "log10Vc": r"$V_c$",
           "log10fX": r"$\log_{10} f_{\rm X}$",
           "powerInd": r"\alpha_X",
           "numin": r"\nu_{\rm min}",
           "tau": r"$\tau$",
           "log10fr": r"$\log_{10} f_{\rm r}$",
           "log10Fr": r"$\log_{10} f_{\rm r}$",
           "log10Ar": r"$\log_{10} A_{\rm r}$",
           "log10TS": r"$\log_{10} T_{\rm spin}$",
           "log10TK": r"$\log_{10} T_{\rm gas}$",
           "log10TR": r"$\log_{10} T_{\rm rad}$",
           "log10Trad": r"$\log_{10} T_{\rm rad}$"}

def make_axes_pcolor(x,y):
    # Expand x and y by 1 each
    def add(z):
        d = np.diff(z)/2
        new = [z[0]-d[0]]+list(np.array(z[:-1])+d)+[z[-1]+d[-1]]
        return new
    return add(x), add(y)

def trapezoidal_bump(a,b,c,d, peak=1):
    return sip.interp1d([a,b,c,d], [0,peak,peak,0], fill_value=(0,0), bounds_error=False)

def sum_pdf_1d(alpha, xmin, xmax, ymin, ymax):
    # PDF for alpha which is the sum of two uniformly distributed
    # random variables x and y, alpha = x + y
    a = xmin+ymin
    b = ymin+xmax#np.minimum(xmin+ymax, ymin+xmax)
    c = xmin+ymax#np.maximum(xmin+ymax, ymin+xmax)
    d = xmax+ymax
    f = trapezoidal_bump(a,b,c,d)
    norm = sin.quad(f,a,d)[0]
    return f(alpha)/norm

def sum_pdf_2d(alpha, beta, xmin, xmax, ymin, ymax, zmin, zmax, debug=False):
    # 2D PDF in (alpha, beta) where the alpha = x + z and beta = y + z
    # where x, y and z are uniformly distributed random variables.
    # Approach to calculate this: Compute the beta-1d-pdf for every alpha:
    # There are (up to) 5 distinct regimes. Use empirical formulas. Proof: Todo.
    alphamin = xmin+zmin 
    betamin = ymin+zmin
    alphamax = xmax+zmax
    betamax = ymax+zmax
    alphalow = np.minimum(xmax+zmin, xmin+zmax)
    alphaup = np.maximum(xmax+zmin, xmin+zmax)
    betalow = np.minimum(ymax+zmin, ymin+zmax)
    betaup = np.maximum(ymax+zmin, ymin+zmax)
    p1_prelim = xmax+zmin
    p2_prelim = xmin+zmax
    if p1_prelim>p2_prelim:
        yellow = "rect"
    else:
        yellow = "diag"
    if debug:
        print("yellow =", yellow)
    p1 = np.minimum(p1_prelim, p2_prelim)
    p2 = np.maximum(p1_prelim, p2_prelim)
    if debug:
        print("p1", p1)
        print("p2", p2)
        print("alphamin", alphamin)
    overallnorm = trapezoidal_bump(alphamin, p1, p2, alphamax)(alpha)
    if yellow=="rect":
        a = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betamin, betamin, betamin, betalow], fill_value=0, bounds_error=False)(alpha)
        b = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betamin, betalow, betalow, betalow], fill_value=0, bounds_error=False)(alpha)
        c = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betaup, betaup, betaup, betamax], fill_value=0, bounds_error=False)(alpha)
        d = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betaup, betamax, betamax, betamax], fill_value=0, bounds_error=False)(alpha)
        if debug:
            print("a,b,c,d", a,b,c,d)
        overallnorm = trapezoidal_bump(alphamin, alphalow, alphaup, alphamax)(alpha)
        return trapezoidal_bump(a,b,c,d,peak=overallnorm)(beta)
    elif yellow=="diag":
        assert betalow+(alphaup-alphamin) == betamax
        a = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betamin, betamin, betamin+(alphaup-alphalow), betamin+(alphamax-alphalow)], fill_value=0, bounds_error=False)(alpha)
        b = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betamin, betamin+(alphalow-alphamin), betamin+(alphaup-alphamin), betamin+(alphaup-alphamin)], fill_value=0, bounds_error=False)(alpha)
        c = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betalow, betalow, betalow+(alphaup-alphalow), betalow+(alphamax-alphalow)], fill_value=0, bounds_error=False)(alpha)
        d = sip.interp1d([alphamin, alphalow, alphaup,alphamax],[betalow, betalow+(alphalow-alphamin), betalow+(alphaup-alphamin), betalow+(alphaup-alphamin)], fill_value=0, bounds_error=False)(alpha)
        overallnorm = trapezoidal_bump(alphamin, alphalow, alphaup, alphamax)(alpha)
        return trapezoidal_bump(a,b,c,d,peak=overallnorm)(beta)
    else:
        assert False, yellow
 

def derive_TS_xRad(xA_Sims, xHI_Sims, TK_Sims, Trad_Sims):
    def get_TS(TK, TR, xA, xRad, z):
        Om=0.3168681398488275
        Ob=0.04902142275334499
        h=0.6704
        delta=0
        deltab=0
        xC=0
        Tse=0.402
        xAeff = xA * (1+Tse/TK)**(-1)*np.exp(-2.06*(Ob*h/0.0327)**(1/3)*(Om/0.307)**(-1/6)*np.sqrt((1+z)/10)*(TK/Tse)**(-2/3)*(1+deltab)**(1/3)*(1+delta)**(1/9.))
        xtot = xAeff+xC
        Ts8_array = (xRad+xtot)/(xRad/TR+xtot/TK)
        Ts8_array[TK==0]=0
        return Ts8_array
    def get_tau21(TS, xHI, z):
        c = 3e5; # speed of light km/s
        A10 = 2.85e-15; #1/s spontaneous emission coefficient
        lambda21 = 21.106 #cm, code
        h = 0.6704;
        Oc = 0.12038/h**2;
        Ob = 0.022032/h**2;
        Om = Ob+Oc;
        OLambda=1-Om;
        hpl =  4.136e-15; #eV s
        mp = 8.40969762e-58; #proton mass in M_sol
        rhoc = 1.36e11*(h/0.7)**2; #M_sol/cMpc^3
        rhob = Ob*rhoc;
        #nb = rhob/mb;# 1/cMpc^3
        Y = 0.247; # Helium abundance by mass
        nH=(rhoc/mp)*(1-Y)*Ob*(1+z)**3; # 1/Mpc^3
        nH *= (3.24e-25)**3.*xHI #1/cm^3 now
        #nH = rhoc/mp*(1-0.247)*Ob*(1+z)**3
        kBoltzmann = 8.617e-5;#eV K-1
        H0=100*h; # in km/s/Mpc
        pi = np.pi
        H = H0*np.sqrt(Om*(1+z)**3+OLambda+8.5522e-05*(1+z)**4);
        dvdr = H/(1+z)*1e5/3e24;
        # this should be 3.08e24 to convert Mpc to cm, 
        return (3*hpl*(c*1e5)*A10*(lambda21)**2.*nH)/(32*pi*kBoltzmann*(1+z)*dvdr)/TS
    def get_xRad(TS, xHI, z=8):
        tau21 = get_tau21(TS, xHI, z)
        xRad = (1-np.exp(-tau21))/tau21
        return xRad

    # z_array for used everything except zHI
    z_tile = np.tile(z_array, (len(xA_Sims),1))
    # get xHI at z_array, filling high-z with 1 and double checking
    # slope to make sure no values were forgotten to fill
    xHI_normalz = np.ones(np.shape(xA_Sims))
    for i in range(len(z_array)):
        zi = z_array[i]
        if zi in z_xHI_array:
            xHI_normalz[:,i] = xHI_Sims[:,np.where(zi == z_xHI_array)[0][0]]
    assert np.all(np.diff(xHI_normalz, axis=-1) >= -1e-5)
    # Iteratively find xRad and TS
    xRad_Sims = np.ones(np.shape(TK_Sims))
    TS_initial_Sims = get_TS(TK_Sims, Trad_Sims, xA_Sims, xRad_Sims, z_tile)
    for i in range(10):
        TS_converged_Sims = get_TS(TK_Sims, Trad_Sims, xA_Sims, xRad_Sims, z_tile)
        xRad_Sims = get_xRad(TS_converged_Sims, xHI_normalz, z_tile)

    return xRad_Sims, TS_converged_Sims, TS_initial_Sims