import anesthetic
import matplotlib.pyplot as plt
import textwrap
import numpy as np
import os.path as osp
from CosmicDawnSynergies.tools import confidence_level

def SDC3b_xHI_hist(chain_dir_name = ["LikelihoodSDC3b_SDC3b_2"], xHI_columns = None):
    path = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    path_chain = [osp.join(path, "scripts", "non-public", chain) for chain in chain_dir_name]

    samples = [anesthetic.read_chains(root=chain+"/run") for chain in path_chain]

    samples_prior = samples[0].prior()
    weights_prior = samples_prior.get_weights()

    if xHI_columns == None:
        xHI_columns = [col[0] for col in samples[0].columns if "xHI" in col[0]]
        print(f"Using default xHI columns: {xHI_columns}")
    
    z1 = xHI_columns[0].split("_z")[1]
    z2 = xHI_columns[1].split("_z")[1]
    z3 = xHI_columns[2].split("_z")[1]
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, gridspec_kw={'hspace': 0})

    bins = np.linspace(0, 1, 100)
    
    s1_prior, edges1_prior = np.histogram(samples_prior[xHI_columns[0]], bins=bins, weights=weights_prior)
    s2_prior, edges2_prior = np.histogram(samples_prior[xHI_columns[1]], bins=bins, weights=weights_prior)
    s3_prior, edges3_prior = np.histogram(samples_prior[xHI_columns[2]], bins=bins, weights=weights_prior)
    
    axes[0].hist(edges1_prior[:-1], edges1_prior, weights=s1_prior, color='grey', alpha=0.5, label='Prior', density=True)
    axes[1].hist(edges2_prior[:-1], edges2_prior, weights=s2_prior, color='grey', alpha=0.5, label='Prior', density=True)
    axes[2].hist(edges3_prior[:-1], edges3_prior, weights=s3_prior, color='grey', alpha=0.5, label='Prior', density=True)    
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, sample in enumerate(samples):
        color = np.roll(colors, -i)[0]

        weights = sample.get_weights()

        s1_mean = np.average(sample[xHI_columns[0]].values, weights=weights)
        s2_mean = np.average(sample[xHI_columns[1]].values, weights=weights)
        s3_mean = np.average(sample[xHI_columns[2]].values, weights=weights)

        #list of q95 bounds
        s1_q95 = confidence_level(samples=sample[xHI_columns[0]].values, weights=weights, level=0.95, method="iso-probability")
        s2_q95 = confidence_level(samples=sample[xHI_columns[1]].values, weights=weights, level=0.95, method="iso-probability")
        s3_q95 = confidence_level(samples=sample[xHI_columns[2]].values, weights=weights, level=0.95, method="iso-probability")

        #np histgram
        s1, edges1 = np.histogram(sample[xHI_columns[0]], bins=bins, weights=weights)
        s2, edges2 = np.histogram(sample[xHI_columns[1]], bins=bins, weights=weights)
        s3, edges3 = np.histogram(sample[xHI_columns[2]], bins=bins, weights=weights)
        
        axes[0].axvspan(s1_q95[0], s1_q95[1], color='blue', alpha=0.2, label=r'95\% Confidence Interval')
        axes[0].hist(edges1[:-1], edges1, weights=s1, color=color, alpha=0.8, density=True, label=f'$\\mathrm{{xHI}}_{{z={z1}}}\\approx{s1_mean:.2f}^{{+{s1_q95[1]-s1_mean:.2f}}}_{{-{s1_mean-s1_q95[0]:.2f}}}$')

        axes[1].axvspan(s2_q95[0], s2_q95[1], color='blue', alpha=0.2, label=r'95\% Confidence Interval')
        axes[1].hist(edges2[:-1], edges2, weights=s2, color=color, alpha=0.8, density=True, label=f'$\\mathrm{{xHI}}_{{z={z2}}}\\approx{s2_mean:.2f}^{{+{s2_q95[1]-s2_mean:.2f}}}_{{-{s2_mean-s2_q95[0]:.2f}}}$')

        axes[2].axvspan(s3_q95[0], s3_q95[1], color='blue', alpha=0.2, label=r'95\% Confidence Interval')
        axes[2].hist(edges3[:-1], edges3, weights=s3, color=color, alpha=0.8, density=True, label=f'$\\mathrm{{xHI}}_{{z={z3}}}\\approx{s3_mean:.2f}^{{+{s3_q95[1]-s3_mean:.2f}}}_{{-{s3_mean-s3_q95[0]:.2f}}}$')



    axes[0].legend(loc='upper left')
    axes[1].legend(loc='upper left')
    axes[2].legend(loc='upper left')
    axes[2].set_xlabel(r'$x_\mathrm{HI}$')
    axes[0].set_ylabel('PDF')
    axes[1].set_ylabel('PDF')
    axes[2].set_ylabel('PDF')
    axes[0].set_title(", ".join(chain_dir_name))

    return fig, axes

if __name__ == "__main__":
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'cm',
        'font.size': 18,
    })

    chain = "LikelihoodSDC3b_SDC3b_PS2_2026"
    fig, axes = SDC3b_xHI_hist(chain_dir_name=[chain], xHI_columns=None)
    
    # Set the caption text
    caption = "Constraints on the reionisation fraction at three redshifts derived from these cosmological parameters"
    # Get the figure width in inches and estimate characters per line
    fig_width = fig.get_size_inches()[0]
    # Estimate average character width in inches (approximate, depends on font)
    avg_char_width = 0.12  # tweak if needed
    max_line_chars = int((1. * fig_width) / avg_char_width)
    # Wrap the caption text
    wrapped_caption = "\n".join(textwrap.wrap(caption, width=max_line_chars))
    caption = wrapped_caption
    fig.text(.5, 0.03, caption, ha='center', fontsize=plt.rcParams['font.size']+2, wrap=True)

    # resize the figure to match the aspect ratio of the Axes    
    #fig.set_size_inches(7, 8, forward=True)
    path = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    path_plots = osp.join(path, "images")
    plot_name = f"{chain}_xHI_latest.png"
    plot_path = osp.join(path_plots, plot_name)
    fig.savefig(plot_path, bbox_inches="tight")

