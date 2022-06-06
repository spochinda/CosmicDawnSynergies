from codes.likelihood_hera import *

idr2 = likelihood(
    datapath='data/observations_HERA_IDR2/pspec_h1c_idr2_field{}.h5',
    decimation_factor=2,
    # Band 1 and 2, field 1
    selections={"1": {"1": {"kstart":0.256}},
                "2": {"1": {"kstart":0.192}}})

# Toy likelihood to check likelihood values for Delta^2=0
toy_model_zero = lambda z,k: 1e-10*k**2/(1+z)
assert np.allclose(idr2.loglike(toy_model_zero), idr2.logL0)

fig, axes = plt.subplots(ncols=2, figsize=(10,5))
idr2.plot_data(axes=axes)
fig.suptitle("HERA IDR2")
plt.savefig("images/data_HERA_IDR2.png", dpi=600)

