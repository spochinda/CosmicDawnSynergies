# HERA-Stack
import hera_pspec as hp
#import simpleqe #https://github.com/nkern/simpleQE
import numpy as np
import scipy.special as ssp
import matplotlib.pyplot as plt
from .emulator_poweremu import *

# Likelihood and data extraction code based on notebook in archive here:
# http://reionization.org/science/public-data-release-1/

def extract_data(field="1", band=1,
                 kstart=0.192, decimation_factor=2,
                 set_negative_to_zero=True, kstart_modulo=True,
                 datapath='IDR2/pspec_h1c_idr2_field{}.h5'):
    """
    Extract data decimated data for a specific field and band from the files.
    Adapted for IDR2 and IDR3.

    Args:
        field : integer or string
            Which field to use. Field 1 used to give the best constraints in IDR2.
        band : integer, either 1 or 2
            Which band to use. Band 1 corresponds to z~10.37, band 2 to z~7.93.
            You can combine both bands in the likelihood.
        kstart : float
            Which wavenumber to start at when decimating the data -- i.e. included
            this k value and every other one after it in the data set. E.g.
            0.128 or 0.192 for IDR2.
        kstart_modulo : bool
            Include points before and after kstart, according to decimation factor,
            except for when their uncertainty is equal to zero (nonsense data).
            Default: True
        decimation_factor : integer
            Set to 2 to use every other k, set to 100 to use only the
            specified wave number.

    Returns:
        dsq : N-dimensional array
            The data delta^2 (set to 0 if negative)
        std : N-dimensional array
            The 1 sigma error bars (not variance) for the data
        wfn : NxM-dimensional array
            The "window functions" i.e. matrix
        wfn_kbins: M-dimensional array
            The wavenumbers to evaluate the theory at before
            multiplying with data
        z : float
            Redshift at which the corresponding model/theory must be evaluated
    """
    if ("idr2" in datapath) or ("idr3" in datapath):
        assert field in ["1","2","3"] or field in ['A', 'B', 'C', 'D', 'E'], "field name not recognized"
        assert band in [1,2], "band must be 1 or 2."
        if field in ["1","2","3"]:
            data_type = "idr2"
        else:
            data_type = "idr3"
        # Read the hdf5 file
        uvp = hp.UVPSpec()
        if data_type == "idr2":
            uvp.read_hdf5(datapath.format(field))
        else:
            uvp.read_hdf5(datapath.format(field, band))
        # Access the right band
        if data_type == "idr2":
            band_key = uvp.get_all_keys()[band-1]
            spw_index = uvp.spw_array[band-1]
        else:
            band_key = uvp.get_all_keys()[0]
            spw_index = uvp.spw_array[0]
        # Get redshift (i.e. spherical window)
        spw_frequencies = uvp.get_spw_ranges()[spw_index][:2] #:2 because we only want frequency range
        z = uvp.cosmo.f2z(np.mean(spw_frequencies))
        # Get wave number, data, error, and (k-space) window functions
        kbins_data = uvp.get_kparas(spw_index)
        kbins_model = uvp.get_kparas(spw_index)
        dsq = uvp.get_data(band_key)[0].real.copy()
        std = np.sqrt(uvp.get_cov(band_key)[0].diagonal().real.copy())
        wfn = uvp.get_window_function(band_key)[0]
        # Negative power spectra values are physically impossible
        if set_negative_to_zero:
            dsq[dsq < 0] = 0
    elif ('idr4' in datapath) or ('idr6' in datapath): #added by SP
        band = str(band)
        data = np.load(datapath, allow_pickle=True).item()
        z = data[band][field]["z"]
        kbins_data = data[band][field]["k_data"]
        kbins_model = data[band][field]["k_model"]
        dsq = data[band][field]["dsq"]
        std = data[band][field]["std"]
        wfn = data[band][field]["wfn"]
    elif ('Deltasq_Band_' in datapath): #added by SP
        x = hp.UVPSpec()
        x.read_hdf5(datapath)

        key = x.get_all_keys()[0]
        spw_index = x.spw_array[0]
        spw_frequencies = x.get_spw_ranges()[spw_index][:2]
        z = x.cosmo.f2z(np.mean(spw_frequencies))
        # Get the delta^2
        x.get_data(key)
        # Get the (diagonal) covariance
        std = np.sqrt(x.get_stats('P_SN', key).real[0])
        x.convert_to_deltasq()
        dsq = x.data_array[0].real[0,:,0]

        k_para = x.get_kparas(spw_index)
        k_perp = x.get_kperps(spw_index)
        k_mag = np.sqrt(k_perp**2 + k_para**2)
        
        dsq_mask = dsq+std < 1e10
        k_mask = np.logical_and(k_mag < 1.47, k_mag > 0.027)
        mask = np.logical_and(dsq_mask, k_mask)

        dsq = dsq[mask]
        std = std[mask]
        k_mag = k_mag[mask]

        kbins_model = kbins_data = k_mag
        wfn = np.identity(len(kbins_model))
    


    # Decimate data to assume diagonal covariance matrix
    if decimation_factor is not None:
        initial_index = (np.argmin(np.abs(kbins_data - kstart)))
        if kstart_modulo:
            assert std[initial_index]!=0, "Error: Chosen kstart has uncertainty std==0"
            while initial_index >= decimation_factor and std[initial_index-decimation_factor]!=0:
                initial_index -= decimation_factor
        name_even_odd = "even" if initial_index%2==0 else "odd"
        kbins_indices = slice(initial_index, None, decimation_factor)
        dsq = dsq[kbins_indices]
        std = std[kbins_indices]
        wfn = wfn[kbins_indices]
        kbins_data = kbins_data[kbins_indices] # just for plots
    # Cutoff k=0 model point
    if ("idr2" in datapath) or ("idr3" in datapath):
        wfn = wfn[:,1:]
        kbins_model = kbins_model[1:]
    return {"z": z, "k_model": kbins_model, "dsq": dsq, "std": std, "wfn": wfn, "k_data": kbins_data}

def compare_data(datapath='IDR3/Deltasq_Band_{1:}_Field_{0:}.h5', theory_err=0.2,
    set_negative_to_zero=True, band=1, fields=['A', 'B', 'C', 'D', 'E'], errorbar=True):
    import matplotlib.pyplot as plt
    cdefault = plt.rcParams['axes.prop_cycle'].by_key()['color']
    data = {}
    for field in fields:
        data[field] = extract_data(datapath=datapath,
                        band=band,
                        field=field,
                        kstart=0,
                        kstart_modulo=False,
                        decimation_factor=1,
                        set_negative_to_zero=set_negative_to_zero)
    karr = data[fields[0]]["k_data"]
    z = data[fields[0]]["z"]
    for i in range(len(fields)):
        f = fields[i]
        assert z == data[f]["z"]
        assert np.all(karr == data[f]["k_data"])
        plt.title("Band "+str(band)+", z="+str(data[f]["z"]))
        if errorbar:
            plt.errorbar(data[f]["k_data"]+0.002*i, data[f]["dsq"], yerr=data[f]["std"], fmt="o", color=cdefault[i])
        plt.scatter(data[f]["k_data"]+0.002*i, data[f]["dsq"]+2*data[f]["std"], marker='x', color=cdefault[i], label="Field "+f)
    plt.xticks(karr, rotation=90)
    plt.yscale('symlog', linthresh=100)
    plt.ylim(-100, 1e7)
    plt.xlabel("k [h/cMpc]")
    plt.ylabel(r"$\Delta^2 \mathrm{[mK^2]}$")
    plt.grid()
    plt.legend(loc="lower right")

def emulatorModel2d(emu, z, karr, p):
    par0 = np.array([z, np.NaN, *p[:9]])
    params=np.tile(par0, (len(karr), 1))
    params[:,1] = karr
    #print(params)
    return emu.predict(params)

class likelihood:
    def __init__(self, datapath='IDR2/pspec_h1c_idr2_field{}.h5', selections=None, zero_fill=1e-50,
                decimation_factor=2, set_negative_to_zero=True, theory_err=0.2, kstart_modulo=True,
                return_individual_loglikes=False, debug=False,
                emupath='data/trained_emulators_poweremu/dsq_emu_n500_l100100100100_t1e-05_o0.pkl',
                output_names = {"logL_HERA": r"\log L_\mathrm{HERA}"}
                 ):
        self.output_names = output_names
        self.theory_err = theory_err
        self.datapath = datapath
        self.decimation_factor = decimation_factor
        self.set_negative_to_zero = set_negative_to_zero
        self.zero_fill = zero_fill
        self.return_individual_loglikes = return_individual_loglikes
        self.debug = debug
        self.model_dsq = poweremu(loadfile=emupath, preprocesss_log_x=False, tol=1e-5, offset=0)
        self.data = {}
        self.nDerived = len(self.output_names.items())

        if isinstance(self.datapath,list) and isinstance(selections,list):
            for i,(dpath,sel) in enumerate(zip(self.datapath,selections)):
                if ('idr4' in dpath) or ('idr6' in dpath): #added by SP
                    self.data = np.load(dpath, allow_pickle=True).item()
                elif ("idr2" in dpath) or ("idr3" in dpath):
                    for band, selection in sel.items():
                        self.data[band] = {}
                        for field, sel in selection.items():
                            self.data[band][field] = extract_data(datapath=dpath,
                                    band=int(band),
                                    field=field,
                                    kstart=sel["kstart"],
                                    kstart_modulo=kstart_modulo,
                                    decimation_factor=self.decimation_factor,
                                    set_negative_to_zero=self.set_negative_to_zero)
                elif ('Deltasq_Band_' in dpath):
                    self.data[i] = {}
                    self.data[i]["0"] = extract_data(datapath=dpath, 
                                                       band=0,
                                                       field=0,
                                                       kstart=0,
                                                       kstart_modulo=False,
                                                       decimation_factor=None,
                                                       set_negative_to_zero=True)
        else:
            if ('idr4' in self.datapath) or ('idr6' in self.datapath): #added by SP
                self.data = np.load(self.datapath, allow_pickle=True).item()
            elif ("idr2" in self.datapath) or ("idr3" in self.datapath):
                for band, selection in selections.items():
                    self.data[band] = {}
                    for field, sel in selection.items():
                        self.data[band][field] = extract_data(datapath=self.datapath,
                                band=int(band),
                                field=field,
                                kstart=sel["kstart"],
                                kstart_modulo=kstart_modulo,
                                decimation_factor=self.decimation_factor,
                                set_negative_to_zero=self.set_negative_to_zero)

        #self.logL0, self.logL0_individual = self.loglike(lambda z,k: np.zeros(len(k)))

    def computeLikelihood(self, p):
        # Important: model must take k as h/cMpc!
        loglike = 0
        if self.return_individual_loglikes:
            individual_loglikes = []
        for band in self.data.keys():
            for field in self.data[band].keys():
                dsq = self.data[band][field]["dsq"]
                std = self.data[band][field]["std"]
                wfn = self.data[band][field]["wfn"]
                z = self.data[band][field]["z"]
                k = self.data[band][field]["k_model"]
                m = emulatorModel2d(emu=self.model_dsq, z=z, karr=k, p=p)
                theory = wfn @ m #theory=model for diag(wfn), @=matrix multiplication
                r = dsq - theory
                l = 0.5 * (1 + ssp.erf(r / np.sqrt(2) / np.sqrt(std**2+(self.theory_err*theory)**2)))
                assert np.shape(theory) == np.shape(dsq), "Shape mismatch"
                # where likelihood == 0 replace w/ zero_fill: note that float equivalence is okay here
                if self.zero_fill>0:
                    l[l == 0.0] = self.zero_fill
                loglike += np.sum(np.log(l))
                if self.return_individual_loglikes:
                    individual_loglikes.append(np.sum(np.log(l)))

                if self.debug:
                    print("Model:", m)
                    print("Theory:", theory)
                    print("Data:", dsq)
                    print("Std:", std)
                    print("Adding logL", loglike, "from", np.shape(l), "likelihoods:", np.log(l))
        if self.return_individual_loglikes:
            return [loglike, *individual_loglikes]
        else:
            return loglike

    def plot_data(self, axes=None, color="green"):
        data = self.data
        if axes is None:
            # Just make a panel for each band
            ncols = len(self.data)
            fig, ax = plt.subplots(ncols=ncols, figsize=(10,5))
            ax = [ax] if ncols==1 else ax
        else:
            if len(np.shape(axes)) == 2:
                formatting = "2D"
            else:
                formatting = "1D"
        for b in range(len(data.keys())):
            band = list(data.keys())[b]
            for f in range(len(data[band].keys())):
                field = list(data[band].keys())[f]
                print(band, field)
                ax = axes[b] if formatting=="1D" else axes[f][b]
                d=data[band][field]
                ax.errorbar(d["k_data"], d["dsq"], yerr=d["std"], color=color, fmt="o", zorder=10)
                ax.scatter(d["k_data"], d["dsq"]+2*d["std"], color=color, marker=7, s=60, zorder=10)
                ax.scatter(d["k_data"], d["dsq"]+2*d["std"], color=color, marker='_', s=60, zorder=10)
                #ax.scatter(d["k_data"], d["dsq"]+3*d["std"], color=color, marker='*', zorder=10)
                #ax.scatter(d["k_data"], d["dsq"]+4*d["std"], color=color, marker='x', zorder=10)
                ax.set_yscale("log")
                ax.set_ylabel("$\Delta^2$")
                if f == 0:
                    ax.set_title("Band "+band+" (z={0:.2f})".format(d["z"]))
                    ax.set_xlabel("k [h/Mpc]")

    def plot_data_violin(self, axes=None):
        self.plot_data(axes=axes, color="black")
        data = self.data
        if axes is None:
            ncols = len(self.data)
            fig, axes = plt.subplots(ncols=ncols, figsize=(10,5))
            axes = [axes] if ncols==1 else axes
        i = 0
        for band in data.keys():
            for field in self.data[band].keys():
                d=data[band][field]
                y = np.geomspace(1,1e7,1000)
                stats = []; positions=[]
                for j in range(len(d["dsq"])):
                    kde = 0.5 * (1 + ssp.erf((d["dsq"][j]-y) / np.sqrt(2) / np.sqrt(d["std"][j]**2)))
                    w = 0.5 if len(d["k_data"])==1 else np.min(np.diff(d["k_data"])/0.75)
                    v = axes[i].violin([{"coords": y, "vals": kde, "mean": 100, "median": -1, "min": -1, "max": -1}],
                        positions=[d["k_data"][j]], widths=w*kde[0])
                    axes[i].axvline(d["k_data"][j]+w/2, color="green", ls=":")
                    for b in v["bodies"]:
                        b.set_alpha(1)
                        # From https://stackoverflow.com/questions/29776114/half-violin-plot-in-matplotlib
                        # half violin
                        b.set_color("green")
                        b.set_facecolor("green")
                        # get the center
                        m = np.mean(b.get_paths()[0].vertices[:, 0])
                        # modify the paths to not go further left than the center
                        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                axes[i].set_title("Band "+band+" (z={0:.2f})".format(d["z"]))
                axes[i].set_yscale("log")
                axes[i].set_ylabel("$\Delta^2$")
                axes[i].set_xlabel("k [h/Mpc]")
            i = (i+1)%len(axes)
