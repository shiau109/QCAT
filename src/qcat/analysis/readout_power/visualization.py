def plot_norm_res_vs_amp_prefactor(norm_res_da):
	"""
	Plot normalized fit residue (norm_res) vs amp_prefactor for each state.
	Args:
		norm_res_da: xarray.DataArray with dims ('amp_prefactor', 'state') or similar
	Returns:
		fig: matplotlib Figure
	"""
	import matplotlib.pyplot as plt
	amp_prefactor_values = norm_res_da['amp_prefactor'].values
	fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
	if 'state' in norm_res_da.dims:
		state_dim = norm_res_da['state'].values if hasattr(norm_res_da['state'], 'values') else norm_res_da['state']
		colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
		for idx, state in enumerate(state_dim):
			norm_res_values = norm_res_da.sel(state=state).values
			color = colors[idx % len(colors)]
			ax.plot(amp_prefactor_values, norm_res_values, marker='o', linestyle='-', color=color, label=f'state {state}')
		ax.legend()
	else:
		norm_res_values = norm_res_da.values
		ax.plot(amp_prefactor_values, norm_res_values, marker='o', linestyle='-', color='tab:blue')
	ax.set_xlabel('amp_prefactor')
	ax.set_ylabel('Normalized fit residue (norm_res)')
	ax.set_title('Normalized Fit Residue vs amp_prefactor')
	ax.grid(True, linestyle='--', alpha=0.5)
	fig.tight_layout()
	return fig
def plot_gaussian_norms_and_direct_counts_vs_amp_prefactor(amp_prefactors, gaussian_norms, direct_counts):
	"""
	Plot gaussian_norms and direct_counts as a function of amp_prefactors.
	Args:
		amp_prefactors: 1D array-like of amp_prefactor values
		gaussian_norms: 2D array-like, shape (n_amp, n_state)
		direct_counts: 2D array-like, shape (n_amp, n_state)
	Returns:
		fig: matplotlib Figure
	"""
	import matplotlib.pyplot as plt
	amp_prefactors = np.asarray(amp_prefactors)
	gaussian_norms = np.asarray(gaussian_norms)
	direct_counts = np.asarray(direct_counts)
	n_state = gaussian_norms.shape[1]
	fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
	colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green']
	# Plot gaussian_norms
	# for state in range(n_state):
	axes[0].plot(amp_prefactors, gaussian_norms[:, 0, 1], marker='o', linestyle='-', color=colors[0], label=f'prepare 0, measure 1')
	axes[0].plot(amp_prefactors, gaussian_norms[:, 1, 0], marker='o', linestyle='-', color=colors[1], label=f'prepare 1, measure 0')

	axes[0].set_xlabel('amp_prefactor')
	axes[0].set_ylabel('gaussian_norms')
	# Use log scale for better dynamic range
	axes[0].set_yscale('log')
	axes[0].set_title('Gaussian Norms vs amp_prefactor')
	axes[0].legend()
	axes[0].grid(True, linestyle='--', alpha=0.5)
	# Plot direct_counts
	# for state in range(n_state):
	axes[1].plot(amp_prefactors, direct_counts[:, 0, 1], marker='o', linestyle='-', color=colors[0], label=f'prepare 0, measure 1')
	axes[1].plot(amp_prefactors, direct_counts[:, 1, 0], marker='o', linestyle='-', color=colors[1], label=f'prepare 1, measure 0')
	axes[1].set_xlabel('amp_prefactor')
	axes[1].set_ylabel('direct_counts')
	# Use log scale for direct counts as well
	axes[1].set_yscale('log')
	axes[1].set_title('Direct Counts vs amp_prefactor')
	axes[1].legend()
	axes[1].grid(True, linestyle='--', alpha=0.5)
	fig.tight_layout()
	return fig

def plot_std_vs_amp_prefactor(std_da):
	"""
	Plot std (sqrt(covariances)) vs amp_prefactor.
	Args:
		std_da: xarray.DataArray with dims ('amp_prefactor', 'state') or similar
	"""
	amp_prefactor_values = std_da['amp_prefactor'].values
	fig, ax = plt.subplots(figsize=(6, 4), dpi=150)


	std_values = std_da.values
	ax.plot(amp_prefactor_values, std_values, marker='o', linestyle='-', color='tab:orange')

	ax.set_xlabel('amp_prefactor')
	ax.set_ylabel('std (sqrt(covariances))')
	ax.set_title('Std vs amp_prefactor')
	ax.grid(True, linestyle='--', alpha=0.5)
	fig.tight_layout()
	return fig

def plot_means_distance_vs_amp_prefactor(mean_da):
	"""
	Plot distance between two means vs amp_prefactor.
	Args:
		mean_da: xarray.DataArray with dims ('amp_prefactor', 'state', 'iq')
	"""
	means_values = mean_da.values  # shape (N, 2, 2)
	amp_prefactor_values = mean_da['amp_prefactor'].values
	means0 = means_values[:, 0, :]
	means1 = means_values[:, 1, :]
	distances = np.linalg.norm(means0 - means1, axis=1)
	fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
	ax.plot(amp_prefactor_values, distances, marker='o', linestyle='-', color='tab:green')
	ax.set_xlabel('amp_prefactor')
	ax.set_ylabel('Distance between means')
	ax.set_title('Means Distance vs amp_prefactor')
	ax.grid(True, linestyle='--', alpha=0.5)
	fig.tight_layout()
	return fig

def plot_means_on_IQ_plane_vs_amp_prefactor(summary_dataset, fit_paras=None):
	"""
	Plot the two means for each amp_prefactor on the IQ plane, with std as radius and p_outlier as alpha for the circle.
	Optionally, plot the fitted I-Q lines for each state using fit_paras.
	Args:
		summary_dataset: xarray.Dataset with variables 'mean', 'std', 'p_outlier'
		fit_paras: xarray.Dataset with variables 'slope' and 'intercept' (dims: state, iq)
	"""
	import matplotlib.patches as mpatches
	means_values = summary_dataset['mean'].values  # shape (N, 2, 2)
	std_values = summary_dataset['std'].values     # shape (N, 2)
	p_outlier_values = summary_dataset['p_outlier'].values  # shape (N, 2)
	amp_prefactor_values = summary_dataset['amp_prefactor'].values
	fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
	# Plot state=0 (blue), state=1 (red)
	for i, amp in enumerate(amp_prefactor_values):
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
	if fit_paras is not None:
		for state, color in zip([0, 1], ['blue', 'red']):
			# Get slope/intercept for I and Q
			slope_I = fit_paras['slope'].sel(state=state, iq='I').item()
			intercept_I = fit_paras['intercept'].sel(state=state, iq='I').item()
			slope_Q = fit_paras['slope'].sel(state=state, iq='Q').item()
			intercept_Q = fit_paras['intercept'].sel(state=state, iq='Q').item()
			# Extend amp_prefactor_values to include 0 if not present
			amp_fit = amp_prefactor_values
			if 0 not in amp_prefactor_values:
				amp_fit = np.insert(amp_prefactor_values, 0, 0)
			amp_fit = np.sort(amp_fit)
			I_fit = slope_I * amp_fit + intercept_I
			Q_fit = slope_Q * amp_fit + intercept_Q
			ax.plot(I_fit, Q_fit, color=color, linestyle='--', linewidth=2, label=f'state {state} fit')
	ax.set_aspect('equal')
	ax.set_xlabel('I')
	ax.set_ylabel('Q')
	ax.set_title('Means on IQ plane vs amp_prefactor\n(circle: std as radius, alpha=p_outlier)')
	ax.grid(True, linestyle='--', alpha=0.5)
	# Legend for states
	handles, labels = ax.get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	ax.legend(by_label.values(), by_label.keys())
	fig.tight_layout()
	return fig
import matplotlib.pyplot as plt
import numpy as np

def plot_p_outlier_vs_amp_prefactor(p_outlier_da):
	"""
	Plot p_outlier vs amp_prefactor.
	Args:
		p_outlier_da: xarray.DataArray with dims ('amp_prefactor', 'state') or similar
	"""
	amp_prefactor_values = p_outlier_da['amp_prefactor'].values
	fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
	if 'state' in p_outlier_da.dims:
		state_dim = p_outlier_da['state'].values if hasattr(p_outlier_da['state'], 'values') else p_outlier_da['state']
		colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
		for idx, state in enumerate(state_dim):
			outlier_values = p_outlier_da.sel(state=state).values
			color = colors[idx % len(colors)]
			ax.plot(amp_prefactor_values, outlier_values, marker='o', linestyle='-', color=color, label=f'state {state}')
		ax.legend()
	else:
		outlier_values = p_outlier_da.values
		ax.plot(amp_prefactor_values, outlier_values, marker='o', linestyle='-', color='tab:blue')
	ax.set_xlabel('amp_prefactor')
	ax.set_ylabel('p_outlier')
	ax.set_title('Outlier Probability vs amp_prefactor')
	ax.grid(True, linestyle='--', alpha=0.5)
	fig.tight_layout()
	return fig

def plot_means_vs_amp_prefactor(mean_da, fit_paras=None):
	"""
	Plot the mean of I and Q for both states as a function of amp_prefactor.
	Optionally plot the fitted lines if fit_paras is provided.
	Args:
		mean_da: xarray.DataArray with dims ('amp_prefactor', 'state', 'iq')
		fit_paras: xarray.Dataset with variables 'slope' and 'intercept' (dims: state, iq), optional
	Returns:
		fig: matplotlib Figure
	"""
	import matplotlib.pyplot as plt
	import numpy as np
	amp_prefactor_values = mean_da['amp_prefactor'].values
	means = mean_da.values  # shape (N, 2, 2)
	fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
	# State 0: red, State 1: blue; I: circle, Q: triangle; fit: dashed
	state_colors = {0: 'red', 1: 'blue'}
	markers = {0: 'o', 1: '^'}  # I: circle, Q: triangle
	labels = ['I, state 0', 'I, state 1', 'Q, state 0', 'Q, state 1']
	for idx, (iq, state) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
		y = means[:, state, iq]
		color = state_colors[state]
		marker = markers[iq]
		# Data points: solid marker
		ax.plot(amp_prefactor_values, y, color=color, linestyle='None', marker=marker, markersize=8, label=labels[idx])
		# Optionally plot fit curve and annotate slope/intercept
		if fit_paras is not None:
			slope = fit_paras['slope'].sel(state=state, iq=['I','Q'][iq]).item()
			intercept = fit_paras['intercept'].sel(state=state, iq=['I','Q'][iq]).item()
			# Extend fit x to include 0 if not present
			fit_x = amp_prefactor_values
			if 0 not in fit_x:
				fit_x = np.insert(fit_x, 0, 0)
			fit_x = np.sort(fit_x)
			fit_y = slope * fit_x + intercept
			# Fit: dashed line, same color
			ax.plot(fit_x, fit_y, color=color, linestyle='--', linewidth=2)
			# Annotate slope and intercept
			textstr = f"{labels[idx]}\nslope={slope:.4g}\nintercept={intercept:.4g}"
			yloc = 0.97 - 0.1*idx
			ax.text(1.02, yloc, textstr, transform=ax.transAxes, fontsize=9, color=color,
					verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor=color))
	ax.set_xlabel('amp_prefactor')
	ax.set_ylabel('Mean Value')
	ax.set_title('Means of I and Q vs amp_prefactor (states 0 and 1)')
	ax.legend()
	ax.grid(True, linestyle='--', alpha=0.5)
	fig.tight_layout()
	return fig