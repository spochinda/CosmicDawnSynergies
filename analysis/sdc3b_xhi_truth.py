"""
True xHI values for SDC3b Data Challenge scoring, as circulated by the SDC3b
organisers for cross-checking the
team scores in Tables 7, 8, and 9 (Section 5.1.1) of the SDC3b paper draft.

xhi_true_min_max : the [min, max] range used to compute each team's score
                    (see Section 5.1.1 for the score definition).
xhi_true         : the true central xHI value for each redshift bin / PS mode.

NOTE: the z1/z2/z3 labelling here is the SDC3b organisers' own convention and
has not been cross-checked against this repo's band ordering (z~6.53, z~7.18,
z~7.96 for the 181.0-195.9, 166.0-180.9, 151.0-165.9 MHz bands respectively —
see LikelihoodSDC3b.extract_data in src/CosmicDawnSynergies/likelihood.py).
Based on magnitude alone (xHI should increase with z), z1 appears to
correspond to the highest-z band (~7.96) and z3 to the lowest-z band (~6.53),
i.e. the reverse of this repo's z-band ordering — confirm before using these
as truth markers on plots.
"""

xhi_true_min_max = {
    'z1': {
        'PS1': [0.8275, 0.5717],
        'PS2': [0.9493, 0.8730],
        'PS3': [0.8751, 0.6662],
    },
    'z2': {
        'PS1': [0.5717, 0.2308],
        'PS2': [0.8730, 0.7493],
        'PS3': [0.6662, 0.3824],
    },
    'z3': {
        'PS1': [0.2308, 0.0197],
        'PS2': [0.7493, 0.5460],
        'PS3': [0.3824, 0.1059],
    },
}

xhi_true = {
    'z1': {
        'PS1': 0.69960,
        'PS2': 0.91117,
        'PS3': 0.77064,
    },
    'z2': {
        'PS1': 0.40125,
        'PS2': 0.81115,
        'PS3': 0.52431,
    },
    'z3': {
        'PS1': 0.12525,
        'PS2': 0.64765,
        'PS3': 0.24418,
    },
}
