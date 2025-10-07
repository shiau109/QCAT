# Visualization functions for frequency-dependent readout analysis
# All references to 'amp_prefactor' are changed to 'frequency'
import matplotlib.pyplot as plt
import numpy as np

def plot_norm_res_vs_frequency(norm_res_da):
    """
    Plot normalized fit residue (norm_res) vs frequency for each state.
    Args:
        norm_res_da: xarray.DataArray with dims ('frequency', 'state') or similar
    Returns:
        fig: matplotlib Figure
    """
    frequency_values = norm_res_da['frequency'].values
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    if 'state' in norm_res_da.dims:
        state_dim = norm_res_da['state'].values if hasattr(norm_res_da['state'], 'values') else norm_res_da['state']
        colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
        for idx, state in enumerate(state_dim):
            norm_res_values = norm_res_da.sel(state=state).values
            color = colors[idx % len(colors)]
            ax.plot(frequency_values, norm_res_values, marker='o', linestyle='-', color=color, label=f'state {state}')
        ax.legend()
    else:
        norm_res_values = norm_res_da.values
        ax.plot(frequency_values, norm_res_values, marker='o', linestyle='-', color='tab:blue')
    ax.set_xlabel('frequency')
    ax.set_ylabel('Normalized fit residue (norm_res)')
    ax.set_title('Normalized Fit Residue vs frequency')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig

def plot_gaussian_norms_and_direct_counts_vs_frequency(frequencies, gaussian_norms, direct_counts):
    """
    Plot gaussian_norms and direct_counts as a function of frequencies.
    Args:
        frequencies: 1D array-like of frequency values
        gaussian_norms: 2D array-like, shape (n_freq, n_state)
        direct_counts: 2D array-like, shape (n_freq, n_state)
    Returns:
        fig: matplotlib Figure
    """
    frequencies = np.asarray(frequencies)
    gaussian_norms = np.asarray(gaussian_norms)
    direct_counts = np.asarray(direct_counts)
    n_state = gaussian_norms.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green']
    # Plot gaussian_norms
    for state in range(n_state):
        axes[0].plot(frequencies, gaussian_norms[:, state], marker='o', linestyle='-', color=colors[state % len(colors)], label=f'state {state}')
    axes[0].set_xlabel('frequency')
    axes[0].set_ylabel('gaussian_norms')
    axes[0].set_title('Gaussian Norms vs frequency')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    # Plot direct_counts
    for state in range(n_state):
        axes[1].plot(frequencies, direct_counts[:, state], marker='o', linestyle='-', color=colors[state % len(colors)], label=f'state {state}')
    axes[1].set_xlabel('frequency')
    axes[1].set_ylabel('direct_counts')
    axes[1].set_title('Direct Counts vs frequency')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig

def plot_std_vs_frequency(std_da):
    """
    Plot std (sqrt(covariances)) vs frequency.
    Args:
        std_da: xarray.DataArray with dims ('frequency', 'state') or similar
    """
    frequency_values = std_da['frequency'].values
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    std_values = std_da.values
    ax.plot(frequency_values, std_values, marker='o', linestyle='-', color='tab:orange')
    ax.set_xlabel('frequency')
    ax.set_ylabel('std (sqrt(covariances))')
    ax.set_title('Std vs frequency')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig

def plot_means_distance_vs_frequency(mean_da):
    """
    Plot distance between two means vs frequency.
    Args:
        mean_da: xarray.DataArray with dims ('frequency', 'state', 'iq')
    """
    means_values = mean_da.values  # shape (N, 2, 2)
    frequency_values = mean_da['frequency'].values
    means0 = means_values[:, 0, :]
    means1 = means_values[:, 1, :]
    distances = np.linalg.norm(means0 - means1, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(frequency_values, distances, marker='o', linestyle='-', color='tab:green')
    ax.set_xlabel('frequency')
    ax.set_ylabel('Distance between means')
    ax.set_title('Means Distance vs frequency')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig

def plot_means_on_IQ_plane_vs_frequency(summary_dataset):
    """
    Plot the two means for each frequency on the IQ plane, with std as radius and p_outlier as alpha for the circle.
    Optionally, plot the fitted I-Q lines for each state using fit_paras.
    Args:
        summary_dataset: xarray.Dataset with variables 'mean', 'std', 'p_outlier'
        fit_paras: xarray.Dataset with variables 'slope' and 'intercept' (dims: state, iq)
    """
    import matplotlib.patches as mpatches
    means_values = summary_dataset['mean'].values  # shape (N, 2, 2)
    std_values = summary_dataset['std'].values     # shape (N, 2)
    p_outlier_values = summary_dataset['p_outlier'].values  # shape (N, 2)
    frequency_values = summary_dataset['frequency'].values
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    # Plot state=0 (blue), state=1 (red)
    for i, freq in enumerate(frequency_values):
        for state, color, marker in [(0, 'blue', 'x'), (1, 'red', 'x')]:
            mean = means_values[i, state, :]
            std = std_values[i]
            p_outlier = p_outlier_values[i, state]
            alpha = max(0.1, 1 - p_outlier * 10)  # Scale p_outlier to [0,1] for alpha
            # Draw circle with std as radius, p_outlier as alpha
            circle = mpatches.Circle((mean[0], mean[1]), std/10, color=color, alpha=alpha, fill=True, linewidth=0, zorder=1)
            ax.add_patch(circle)
            # Draw center point with constant alpha
            ax.scatter(mean[0], mean[1], color=color, alpha=0.9, marker=marker, label=f'state {state}' if (i==0) else None, zorder=2)
        # Optionally, connect the two means with a gray line
        ax.plot([means_values[i, 0, 0], means_values[i, 1, 0]], [means_values[i, 0, 1], means_values[i, 1, 1]], color='gray', alpha=0.3, zorder=0)
    # Show (0,0) as a black plus marker
    ax.scatter(0, 0, color='black', marker='+', s=80, zorder=3)
    # Plot fitted I-Q lines if fit_paras is provided
    
    ax.set_aspect('equal')
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title('Means on IQ plane vs frequency\n(circle: std as radius, alpha=p_outlier)')
    ax.grid(True, linestyle='--', alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    fig.tight_layout()
    return fig

def plot_p_outlier_vs_frequency(p_outlier_da):
    """
    Plot p_outlier vs frequency.
    Args:
        p_outlier_da: xarray.DataArray with dims ('frequency', 'state') or similar
    """
    frequency_values = p_outlier_da['frequency'].values
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    if 'state' in p_outlier_da.dims:
        state_dim = p_outlier_da['state'].values if hasattr(p_outlier_da['state'], 'values') else p_outlier_da['state']
        colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
        for idx, state in enumerate(state_dim):
            outlier_values = p_outlier_da.sel(state=state).values
            color = colors[idx % len(colors)]
            ax.plot(frequency_values, outlier_values, marker='o', linestyle='-', color=color, label=f'state {state}')
        ax.legend()
    else:
        outlier_values = p_outlier_da.values
        ax.plot(frequency_values, outlier_values, marker='o', linestyle='-', color='tab:blue')
    ax.set_xlabel('frequency')
    ax.set_ylabel('p_outlier')
    ax.set_title('Outlier Probability vs frequency')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig
