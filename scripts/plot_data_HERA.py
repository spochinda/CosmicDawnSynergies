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


like_idr3_preliminary = likelihood(
    datapath='data/observations_HERA_IDR3_preliminary/Deltasq_Band_{1:}_Field_{0:}.h5',
    decimation_factor=2,
    selections = {"1": {
            "D": {"kstart":0.356},
            "C": {"kstart":0.356},
            "B": {"kstart":0.294},
            "E": {"kstart":0.417},
            "A": {"kstart":0.417}
        }, "2": {
            "C": {"kstart":0.337},
            "D": {"kstart":0.266},
            "B": {"kstart":0.266},
            "E": {"kstart":0.337},
            "A": {"kstart":0.478}}})

fig, axes = plt.subplots(ncols=2, figsize=(10,5))
like_idr3_preliminary.plot_data(axes=axes)
fig.suptitle("HERA IDR3 preliminary")


like_idr3_all = likelihood(
    datapath='data/observations_HERA_IDR3_final/Deltasq_Band_{1:}_Field_{0:}.h5',
    decimation_factor=1,
    selections = {"1": {
            "D": {"kstart":1},
            "C": {"kstart":1},
            "B": {"kstart":1},
            "E": {"kstart":1},
            "A": {"kstart":1}
        }, "2": {
            "C": {"kstart":1},
            "D": {"kstart":1},
            "B": {"kstart":1},
            "E": {"kstart":1},
            "A": {"kstart":1}}})

like_idr3 = likelihood(
    datapath='data/observations_HERA_IDR3_final/Deltasq_Band_{1:}_Field_{0:}.h5',
    decimation_factor=2,
    selections = {"1": {
            "D": {"kstart":0.356},
            "C": {"kstart":0.356},
            "B": {"kstart":0.294},
            "E": {"kstart":0.417},
            "A": {"kstart":0.417}
        }, "2": {
            "C": {"kstart":0.337},
            "D": {"kstart":0.266},
            "B": {"kstart":0.266},
            "E": {"kstart":0.337},
            "A": {"kstart":0.478}}})

fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(10,5))
like_idr3_all.plot_data(axes=axes, color="blue")
like_idr3.plot_data(axes=axes)
fig.suptitle("HERA IDR3 final")

# Toy likelihood to check likelihood values for Delta^2=0
toy_model_zero = lambda z,k: 1e-10*k**2/(1+z)
assert np.allclose(like_idr3.loglike(toy_model_zero), like_idr3.logL0)

plt.show()
