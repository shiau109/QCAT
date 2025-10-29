
#%%
# from Hardware_setting import*
# from SQ_RB_seq import *

import numpy as np
import xarray as xr
from scipy import special

from scipy.integrate import quad
from lmfit import Model,Parameter


import matplotlib.pyplot as plt
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
from qcat.common_calculator.convertor import PetoT
from qcat.common_calculator.analytical import Relax_cal

from sklearn.mixture import GaussianMixture

import numpy as np


class StateDiscrimination():

    
    """
    Class for analyzing exponential decay data with the flux coordinate.
    This is adapted from the repetition analysis code but replaces "repetition" with "flux".
    """

    def __init__(self, data: xr.Dataset, user_mean=None, user_std=None):
        # super().__init__()
        self.user_mean = user_mean
        self.user_std = user_std
        self._import_data(data)

    def _import_data(self, data):
        # Ensure input data is an xarray Dataset and has 'shot_idx' and 'prepared_state' coordinates.
        if not isinstance(data, xr.Dataset):
            raise ValueError("Input data must be an xarray.Dataset.")

        for coords_name in ["shot_idx", "prepared_state"]:
            if coords_name not in data.coords:
                raise ValueError(f"No {coords_name} coordinate in the input xarray.Dataset.")

        # Store original data in self.data
        self.data = data


    def _preprocess_data(self, bins=20):
        # Compute mean and std for initialization (in original units)
        mean_I = self.data['I'].mean(dim='shot_idx').values
        mean_Q = self.data['Q'].mean(dim='shot_idx').values
        prepared_state_num = self.data.coords['prepared_state'].size
        mean_init = []
        std_init = []
        std_I = self.data['I'].std(dim='shot_idx').values
        std_Q = self.data['Q'].std(dim='shot_idx').values
        for i in range(prepared_state_num):
            mean_init.append(np.array([mean_I[i], mean_Q[i]]))
            std_init.append(np.array([std_I[i], std_Q[i]]))
        self.mean_init = np.array(mean_init)
        self.std_init = np.min(np.array(std_init))

        # Compute 2D histograms for each prepared_state and build a dataset
        prepared_states = self.data.coords['prepared_state'].values
        # Find global I/Q min/max for binning
        I_all = self.data['I'].values.ravel()
        Q_all = self.data['Q'].values.ravel()
        I_min, I_max = I_all.min(), I_all.max()
        Q_min, Q_max = Q_all.min(), Q_all.max()

        # Use std/5 as step for bins, prefer user_std if set
        std_val = self.user_std if self.user_std is not None else self.std_init
        step = std_val / 5
        # Ensure step is positive and not too small
        if step <= 0:
            step = 1e-3
        xedges = np.arange(I_min, I_max + step, step)
        yedges = np.arange(Q_min, Q_max + step, step)
        # If only one bin, fallback to linspace
        if len(xedges) < 2:
            xedges = np.linspace(I_min, I_max, 2)
        if len(yedges) < 2:
            yedges = np.linspace(Q_min, Q_max, 2)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        bins_x = len(xcenters)
        bins_y = len(ycenters)
        density_arr = np.zeros((len(prepared_states),bins_y,  bins_x))
        for i, state in enumerate(prepared_states):
            I = self.data['I'].sel(prepared_state=state).values
            Q = self.data['Q'].sel(prepared_state=state).values
            H, _, _ = np.histogram2d(I, Q, bins=[xedges, yedges], density=True)
                    # Plot density_all for visual inspection

            density_arr[i, :, :] = H.T
        self.hist_dataset = xr.Dataset(
            {'density': (['prepared_state', 'y', 'x'], density_arr)},
            coords={
                'prepared_state': prepared_states,
                'x': xcenters,
                'y': ycenters
            }
        )

    def _start_analysis(self):
        self._preprocess_data()
        self._analysis_by_multi_2Dgaussian( outlier_sigma = 3 )
        print(self.analysis_result['trained_paras'])

    def _plot_results(self, fig_group_name=None, save_path=None):
        from qcat.analysis.state_discrimination.visualization import axis_formatter, compute_shared_axis_limits, plot_prepared_state_scatter, plot_2d_histogram, plot_outliers, plot_2d_fit_residue
        figs = {}
        

        fig_raw, axes_raw = plot_prepared_state_scatter( self.data, self.analysis_result )
        fig_2Dhist, axes_2Dhist = plot_2d_histogram( self.hist_dataset, analysis_result=self.analysis_result)
        fig_outliers, axes_outliers = plot_outliers(self.data, self.analysis_result["outlier_mask"], analysis_result=self.analysis_result)
        fig_residue, axes_residue = plot_2d_fit_residue(self.analysis_result['fit_residues'],self.analysis_result['norm_res'])
        
        
        lim_I, lim_Q = compute_shared_axis_limits(self.data)

        for i in range(2):
            axis_formatter(axes_raw[i], lim_I, lim_Q, i)
            axis_formatter(axes_2Dhist[i], lim_I, lim_Q, i)
            axis_formatter(axes_outliers[i], lim_I, lim_Q, i)
            axis_formatter(axes_residue[i], lim_I, lim_Q, i)

        figs["2DHist"] = fig_2Dhist
        figs["raw"] = fig_raw
        figs["outliers"] = fig_outliers
        figs["fit_residue"] = fig_residue
        if save_path is not None:
            for plot_name, fig in figs.items():
                fig.savefig(f"{save_path}\\{fig_group_name}_{plot_name}.png", bbox_inches='tight')
        return figs


    def _analysis_by_multi_2Dgaussian(self, outlier_sigma=3):
        """
        1. Fit the 2D multi-Gaussian model on all data (concatenated over prepared_state).
        2. Use the trained model's parameters as initial guess to fit prepared_state=0 and prepared_state=1 separately.
        3. Store or return the fit results for each case.
        All fitting uses the density dataset (var 'density', coords: prepared_state, x, y).
        """
        # 1. Fit on all data (concatenated)
        # Concatenate all prepared_state densities for global fit
        trained_multi_2Dgaussian_params = self._train_by_multi_2Dgaussian()

        # 2. Fit on each prepared_state using the density dataset
        fit_results = []
        x = self.hist_dataset['x'].values
        y = self.hist_dataset['y'].values
        fit_residues_list = []
        norm_res = []
        for i, state in enumerate(self.hist_dataset['prepared_state'].values):
            density = self.hist_dataset['density'].sel(prepared_state=state).values
            fit_result, fitter = self._fit_histogram_by_multi_2Dgaussian(
                density, x, y, mean=trained_multi_2Dgaussian_params['mean'], std=trained_multi_2Dgaussian_params['std']
            )
            fit_results.append(self._extract_multi_2Dgaussian_params(fit_result, n_gauss=len(self.mean_init)))

            best_fit = fit_result.best_fit.reshape(density.shape)
            residue = density - best_fit
            fit_residues_list.append(residue)
            norm_res.append(np.nansum(residue) / np.nansum(density) if np.nansum(density) != 0 else np.nan)

        # Convert fit_residues_list to a DataArray with dims (prepared_state, y, x)
        fit_residues = xr.DataArray(
            np.stack(fit_residues_list, axis=0),
            dims=["prepared_state", "y", "x"],
            coords={
                "prepared_state": self.hist_dataset["prepared_state"].values,
                "y": self.hist_dataset["y"].values,
                "x": self.hist_dataset["x"].values,
            },
        )
        
        # 3. Use the trained model to assign state labels and count populations
        distance_dataset = self.calc_distances_to_mean(mean_trained=trained_multi_2Dgaussian_params['mean'])
        state_label = distance_dataset['distance'].argmin(dim='center')

        # Get population counts for each state label
        def bincount_1d(arr, minlength=None):
            return np.bincount(arr, minlength=minlength)

        max_label = int(state_label.max().item())
        minlength = max_label + 1

        counts = xr.apply_ufunc(
            bincount_1d,
            state_label,
            input_core_dims=[['idx_shot']],
            output_core_dims=[['count']],
            vectorize=True,
            kwargs={'minlength': minlength},
            output_dtypes=[int]
        )
        gaussian_amp = []
        for i, state in enumerate(self.hist_dataset['prepared_state'].values):
            gaussian_amp.append(fit_results[i]['amp'])
        gaussian_norms = np.array(gaussian_amp) / np.sum(gaussian_amp, axis=1, keepdims=True)

        # Outlier probability
        outlier_mask = distance_dataset['distance'].min(dim='center') > (outlier_sigma * np.mean(trained_multi_2Dgaussian_params['std']))
        n_outlier = np.count_nonzero(outlier_mask, axis=1)
        # print(f"Number of outliers detected: {n_outlier}")
        self.p_outlier = n_outlier / self.data['shot_idx'].size


        self.analysis_result = {
            'trained_paras': trained_multi_2Dgaussian_params,
            'fitted_paras': fit_results,
            'gaussian_norms': gaussian_norms,
            'direct_counts': counts.values/ self.data['shot_idx'].size,
            'state_label': state_label.values,
            'outlier_mask': outlier_mask.values,
            'outlier_probability': self.p_outlier,
            'norm_res': norm_res,
            'fit_residues': fit_residues,
        }

    def _train_by_multi_2Dgaussian(self, mean=None, std=None):
        """
        Fit the 2D multi-Gaussian model on all data (concatenated over prepared_state).
        If both mean and std are provided, skip training and use them directly.
        If only one is provided, fix that value during fitting.
        Returns trained_multi_2Dgaussian_params dict.
        """
        density_all = self.hist_dataset['density'].values
        density_all = np.sum(density_all, axis=0)  # sum over prepared_state

        x = self.hist_dataset['x'].values
        y = self.hist_dataset['y'].values

        # Use class properties if not provided as arguments
        use_mean = mean if mean is not None else self.user_mean
        use_std = std if std is not None else self.user_std

        # If both mean and std are provided (via argument or class property), skip fitting and use them directly
        if use_mean is not None and use_std is not None:
            trained_multi_2Dgaussian_params = {
                'mean': np.array(use_mean),
                'std': use_std,
                'covariance': use_std**2,
                'amp': np.ones(len(use_mean)),  # dummy amplitude
            }
            return trained_multi_2Dgaussian_params

        # If only one is provided, fix that value during fitting
        fit_all_result, fit_all_fitter = self._fit_histogram_by_multi_2Dgaussian(
            density_all, x, y, mean=use_mean, std=use_std
        )
        trained_multi_2Dgaussian_params = self._extract_multi_2Dgaussian_params(fit_all_fitter, n_gauss=len(self.mean_init))

        return trained_multi_2Dgaussian_params

    def _extract_multi_2Dgaussian_params(self, fit_result, n_gauss=None):
        """
        Extract mean, std, and amp from a fit result (lmfit.ModelResult) and return a dict in the format of self.trained_multi_2Dgaussian_params.
        Args:
            fit_result: lmfit ModelResult object with .params attribute
            n_gauss: number of Gaussians (if None, use self.mean_init)
        Returns:
            dict with keys 'mean', 'std', 'amp'
        """
        if n_gauss is None:
            n_gauss = len(self.mean_init)
        mean = []
        std = []
        amp = []
        for i in range(n_gauss):
            x0 = fit_result.params[f'g{i}_x0'].value
            y0 = fit_result.params[f'g{i}_y0'].value
            g_amp = fit_result.params[f'g{i}_amp'].value
            mean.append(np.array([x0, y0]))
            amp.append(g_amp)
        std = fit_result.params[f'g0_sigma_x'].value
        return {
            'mean': np.array(mean),
            'std': std,
            'covariance': std**2,
            'amp': np.array(amp),
        }
    
    def _fit_histogram_by_multi_2Dgaussian(self, density, x, y, mean=None, std=None):
        """
        Fit a 2D histogram (density) using FitMultiGaussian2D.
        Args:
            density: 2D numpy array (shape: [len(x), len(y)])
            x: 1D array of x bin centers
            y: 1D array of y bin centers
            mean: list/array of initial mean (optional)
            std: list/array of initial std (optional)
        Returns:
            fit_result: lmfit ModelResult from FitMultiGaussian2D.fit()
            fitter: the FitMultiGaussian2D instance
        """
        vary_mean = False
        vary_std = False
        if mean is None:
            mean = self.mean_init
            vary_mean = True
            print("Using default mean_init for fitting.", mean)
        if std is None:
            std = self.std_init
            vary_std = True
            print("Using default std_init for fitting.", std)

        from qcat.utilities.function_fitting.fit_gaussian2d import FitMultiGaussian2D
        n_gauss = len(mean)
        fitter = FitMultiGaussian2D(density, x, y, n_gauss=n_gauss)
        fitter.params['offset'].set(value=0, vary=False)
        for i in range(n_gauss):
            fitter.params[f'g{i}_x0'].set(value=mean[i][0], vary=vary_mean)
            fitter.params[f'g{i}_y0'].set(value=mean[i][1], vary=vary_mean)
            # fitter.params[f'g{i}_amp'].set(value=np.max(density), vary=True)
            if i == 0:
                fitter.params[f'g{i}_sigma_x'].set(value=std, vary=vary_std)
            else:
                fitter.params[f'g{i}_sigma_x'].set(expr='g0_sigma_x')
            fitter.params[f'g{i}_sigma_y'].set(expr='g0_sigma_x')

        fit_result = fitter.fit()
        return fit_result, fitter
      
    def _export_result(self, save_path=None):
        # Implement result export functionality if needed.
        pass


    
    def rotate_data_to_x_axis(self):
        """
        Return a rotated copy of self.data so that the vector between the two GMM mean (in scaled space) aligns with the x-axis.
        self.data is never modified in-place.
        Returns:
            rotated_data: xarray.Dataset with rotated 'I' and 'Q' variables
            angle: rotation angle in radians (counterclockwise)
        """
        import copy
        # Get mean in scaled space
        mean = self.gmm_model.mean_  # shape (2, 2), columns: I, Q
        v = mean[1] - mean[0]  # vector from mean 0 to mean 1
        angle = np.arctan2(v[1], v[0])  # angle to x-axis
        # Build rotation matrix (counterclockwise)
        R = np.array([[np.cos(-angle), -np.sin(-angle)],
                      [np.sin(-angle),  np.cos(-angle)]])
        # Deep copy to avoid modifying self.data
        rotated_data = copy.deepcopy(self.data)
        for i in self.data.coords['prepared_state']:
            I = self.data['I'].sel(prepared_state=i).values
            Q = self.data['Q'].sel(prepared_state=i).values
            IQ = np.stack([I, Q], axis=-1)
            IQ_rot = IQ @ R.T  # shape (..., 2)
            rotated_data['I'].loc[dict(prepared_state=i)] = IQ_rot[..., 0]
            rotated_data['Q'].loc[dict(prepared_state=i)] = IQ_rot[..., 1]
        return rotated_data, angle
    
    def calc_distances_to_mean(self, mean_trained=None):
        """
        Calculate the Euclidean distances from each (I, Q) point (for all shot_idx and prepared_state)
        to each of the two mean_trained points.
        Args:
            mean_trained: list or array of two mean points [[x0, y0], [x1, y1]]. If None, uses self._trained_multi_2Dgaussian_params['mean'].
        Returns:
            distances: dict with keys 'prepared_state', each value is a (n_shots, 2) array of distances to each mean.
        """
        if mean_trained is None:
            mean_trained = self._trained_multi_2Dgaussian_params['mean']
        # Get coordinate values
        prepared_states = self.data.coords['prepared_state'].values
        n_center = len(mean_trained)
        n_state = len(prepared_states)
        n_shot = self.data.sizes['shot_idx']

        # Allocate array: (center, prepared_state, idx_shot)
        dist_arr = np.zeros((n_center, n_state, n_shot))
        for i_center, mean in enumerate(mean_trained):
            for i_state, state in enumerate(prepared_states):
                I = self.data['I'].sel(prepared_state=state).values.ravel()
                Q = self.data['Q'].sel(prepared_state=state).values.ravel()
                dist_arr[i_center, i_state, :] = np.sqrt((I - mean[0])**2 + (Q - mean[1])**2)

        # Build xarray Dataset
        dis = xr.Dataset({
            'distance': (['center', 'prepared_state', 'idx_shot'], dist_arr)
        }, coords={
            'center': np.arange(n_center),
            'prepared_state': prepared_states,
            'idx_shot': np.arange(n_shot)
        })
        return dis
    




plt.show()
if __name__ == '__main__':
    import xarray as xr
    from qcat.parser.qm_reader import load_xarray_h5, repetition_data
    # Load the dataset
    path_name = r"D:\github\ASQMDriver\data\MIST\2025-09-23\#778_LCH_readout_fidelity_095728"
    ds = load_xarray_h5(path_name+"\\ds_raw.h5")
    sep_data = repetition_data(ds, repetition_dim="qubit")


    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        if qubit_name == "q1":

            # Rename n_runs to shot_idx if present
            sq_data = sq_data.rename({'n_runs': 'shot_idx'})
            # print(sq_data)
            analysis = StateDiscrimination(sq_data)
            analysis._start_analysis()
            analysis._plot_results(qubit_name,path_name)

           
    plt.show()

    # plt.show()
# %%
