
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
from qcat.NCU.Fit_library import gauss_func, gauss2d_func_model, gauss2d_func, bigauss2d_func_model, bigauss2d_func
from qcat.utilities.data_processing import find_nearest, rot, IQ_data_dis
from qcat.common_calculator.convertor import PetoT
from qcat.common_calculator.analytical import Relax_cal

from sklearn.mixture import GaussianMixture

import numpy as np


class StateDiscrimination():

    
    """
    Class for analyzing exponential decay data with the flux coordinate.
    This is adapted from the repetition analysis code but replaces "repetition" with "flux".
    """

    def __init__(self, data: xr.Dataset):
        # super().__init__()
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
        self._preprocess_data()

    def _preprocess_data(self, bins=50):
        # Compute means and stds for initialization (in original units)
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
        self.std_init = np.array(std_init)

        scale = np.abs(np.sqrt((mean_I[0] - mean_I[1]) ** 2 + (mean_Q[0] - mean_Q[1]) ** 2))
        if scale == 0:
            raise ValueError("Scale factor is zero; cannot normalize data.")

        # Scale data for fitting
        rescaled_data = self.data.copy()
        rescaled_data['I'] = rescaled_data['I'] / scale
        rescaled_data['Q'] = rescaled_data['Q'] / scale
        self.rescaled_data = rescaled_data
        self.scale = scale

        # Compute means and stds for initialization (in scaled units)
        mean_I_s = self.rescaled_data['I'].mean(dim='shot_idx').values
        mean_Q_s = self.rescaled_data['Q'].mean(dim='shot_idx').values
        prepared_state_num = self.rescaled_data.coords['prepared_state'].size
        mean_s_init = []
        std_s_init = []
        std_I_s = self.rescaled_data['I'].std(dim='shot_idx').values
        std_Q_s = self.rescaled_data['Q'].std(dim='shot_idx').values
        self.precisions_init = [1 / ((np.mean(std_I_s[0]) + np.mean(std_Q_s[0])) / 2)] * 2
        for i in range(prepared_state_num):
            mean_s_init.append([mean_I_s[i], mean_Q_s[i]])
            std_s_init.append((std_I_s[i] + std_Q_s[i]))
        self.mean_s_init = mean_s_init
        self.std_s_init = std_s_init

        # Compute 2D histograms for each prepared_state and build a dataset
        bins = int(bins)
        prepared_states = self.data.coords['prepared_state'].values
        # Find global I/Q min/max for binning
        I_all = self.data['I'].values.ravel()
        Q_all = self.data['Q'].values.ravel()
        I_min, I_max = I_all.min(), I_all.max()
        Q_min, Q_max = Q_all.min(), Q_all.max()
        xedges = np.linspace(I_min, I_max, bins + 1)
        yedges = np.linspace(Q_min, Q_max, bins + 1)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        density_arr = np.zeros((len(prepared_states), bins, bins))
        for i, state in enumerate(prepared_states):
            I = self.data['I'].sel(prepared_state=state).values.ravel()
            Q = self.data['Q'].sel(prepared_state=state).values.ravel()
            H, _, _ = np.histogram2d(I, Q, bins=[xedges, yedges], density=True)
            density_arr[i, :, :] = H
        self.hist_dataset = xr.Dataset(
            {'density': (['prepared_state', 'x', 'y'], density_arr)},
            coords={
                'prepared_state': prepared_states,
                'x': xcenters,
                'y': ycenters
            }
        )

    def _start_analysis(self):


        # print("Scaled means_init:", means_init)

        # self._analysis_gmm(outlier_sigma=3)
        self._analysis_by_multi_2Dgaussian( outlier_sigma = 3 )

    def _plot_results(self, fig_group_name=None, save_path=None):
        from qcat.analysis.state_discrimination.visualization import axis_formatter, compute_shared_axis_limits, plot_prepared_state_scatter, plot_2d_histogram, plot_outliers
        figs = {}
        

        fig_raw, axes_raw = plot_prepared_state_scatter( self.data, self.analysis_result )
        fig_2Dhist, axes_2Dhist = plot_2d_histogram( self.hist_dataset, analysis_result=self.analysis_result)
        fig_outliers, axes_outliers = plot_outliers(self.data, self.analysis_result["outlier_mask"], analysis_result=self.analysis_result)

        lim_I, lim_Q = compute_shared_axis_limits(self.data)
        for i in range(2):
            axis_formatter(axes_raw[i], lim_I, lim_Q, i)
            axis_formatter(axes_2Dhist[i], lim_I, lim_Q, i)
            axis_formatter(axes_outliers[i], lim_I, lim_Q, i)

        figs["2DHist"] = fig_2Dhist
        figs["raw"] = fig_raw
        figs["outliers"] = fig_outliers
        if save_path is not None:
            for plot_name, fig in figs.items():
                fig.savefig(f"{save_path}\\{fig_group_name}_{plot_name}.png", bbox_inches='tight')
        return figs



    def _analysis_gmm(self, outlier_sigma=3):
        self._train_gmm()
        state_label = []
        outlier_mask = []
        for i in self.rescaled_data.coords['prepared_state']:
            state_label.append(self._predict_by_gmm(i))
            outlier_mask.append(self._get_outlier_by_gmm(i,outlier_sigma)[0])

        self.analysis_result = {
            'means': self.gmm_params['means'],
            'covariances': self.gmm_params['covariances'],
            # 'rotated_data': self.rotate_data_to_x_axis(),
            'state_label': state_label,
            'outlier_mask': outlier_mask,
            'outlier_probability': self.p_outlier,
        }
        # print('GMM (original units):', self.analysis_result)

    def _train_gmm( self ):
        # Fit GMM in scaled space
        self.gmm_model = GaussianMixture(
            n_components=2,
            covariance_type="spherical",
            means_init=self.mean_s_init,
            precisions_init=self.precisions_init,
            tol=1e-5,
            reg_covar=1e-12,
        )
        X_scaled = np.array([self.rescaled_data['I'].values.ravel(), self.rescaled_data['Q'].values.ravel()]).T
        self.gmm_model.fit(X_scaled)

        self._modify_gmm()
        mean_covariances = np.mean(self.gmm_model.covariances_)
        self.gmm_model.covariances_[0] = mean_covariances
        self.gmm_model.covariances_[1] = mean_covariances
        self.gmm_model.weights_[0] = 0.5
        self.gmm_model.weights_[1] = 0.5
        # Rescale GMM parameters to original units
        means_unscaled = self.gmm_model.means_ * self.scale
        covariances_unscaled = self.gmm_model.covariances_ * (self.scale ** 2)
        precisions_unscaled = self.gmm_model.precisions_ / (self.scale ** 2)


        self.gmm_params = {
            'means': means_unscaled,
            'covariances': covariances_unscaled,
            'weights': self.gmm_model.weights_,
            'precisions': precisions_unscaled,
            'scale': self.scale,
            
        }
        # print('GMM parameters (original units):', self.gmm_params)

    def _modify_gmm( self ):
        # Refine GMM by enforcing equal covariances and equal weights
        mean_covariances = np.mean(self.gmm_model.covariances_)
        self.gmm_model.covariances_[0] = mean_covariances
        self.gmm_model.covariances_[1] = mean_covariances
        self.gmm_model.weights_[0] = 0.5
        self.gmm_model.weights_[1] = 0.5

    def _predict_by_gmm( self, prepared_state ):
        """
        Use the fitted GMM to predict component labels for data with prepared_state=0 (in scaled space).
        Returns:
            labels: array of predicted component indices for each shot in prepared_state=0
        """
        # Extract I and Q for prepared_state=0, flatten to 1D
        I = self.rescaled_data['I'].sel(prepared_state=prepared_state).values.ravel()
        Q = self.rescaled_data['Q'].sel(prepared_state=prepared_state).values.ravel()
        X = np.stack([I, Q], axis=1)
        labels = self.gmm_model.predict(X)
        return labels

    def _get_outlier_by_gmm(self, prepared_state, sigma_level=3):
        """
        Return a boolean mask for outliers and the number of outliers, where the GMM predict_proba for all components is below the threshold defined as the value of a Gaussian PDF at x=sigma*sigma_level.
        Args:
            prepared_state: int or label for which prepared_state to check
            sigma_level: number of standard deviations (default 3)
        Returns:
            outlier_mask: boolean array, True for outlier points
            n_outlier: int, number of outliers
        """
        from scipy.stats import norm
        I = self.rescaled_data['I'].sel(prepared_state=prepared_state).values.ravel()
        Q = self.rescaled_data['Q'].sel(prepared_state=prepared_state).values.ravel()
        X = np.stack([I, Q], axis=1)
        # Use the minimum variance among GMM components (in scaled space)
        if hasattr(self.gmm_model, 'covariances_'):
            sigma = np.sqrt(np.min(self.gmm_model.covariances_))
        else:
            sigma = 1.0
        threshold = norm.pdf(sigma_level * sigma, loc=0, scale=sigma)
        
        max_proba = np.exp(self.gmm_model._estimate_log_prob(X))[np.arange(X.shape[0]), self.gmm_model.predict(X)]
        outlier_mask = max_proba < threshold
        # print(threshold, max_proba)
        n_outlier = np.count_nonzero(outlier_mask)
        self.p_outlier = n_outlier / len(I)
        # print(f"Prepared state {prepared_state}: {n_outlier} outliers detected (threshold={threshold:.3e})")
        return outlier_mask, n_outlier
    
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

        for i, state in enumerate(self.hist_dataset['prepared_state'].values):
            density = self.hist_dataset['density'].sel(prepared_state=state).values
            fit_result, fitter = self._fit_histogram_by_multi_2Dgaussian(
                density, x, y, mean_init=trained_multi_2Dgaussian_params['means'], std_init=trained_multi_2Dgaussian_params['stds']
            )
            fit_results.append(self._extract_multi_2Dgaussian_params(fit_result, n_gauss=len(self.mean_init)))

            
        
        # 3. Use the trained model to assign state labels and count populations
        distance_dataset = self.calc_distances_to_means(means_trained=trained_multi_2Dgaussian_params['means'])
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
        gaussian_amps = []
        for i, state in enumerate(self.hist_dataset['prepared_state'].values):
            gaussian_amps.append(fit_results[i]['amps'])
        gaussian_norms = np.array(gaussian_amps) / np.sum(gaussian_amps, axis=1, keepdims=True)

        # Outlier probability
        outlier_mask = distance_dataset['distance'].min(dim='center') > (outlier_sigma * np.mean(trained_multi_2Dgaussian_params['stds']))
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
        }

    def _train_by_multi_2Dgaussian(self):
        """
        Fit the 2D multi-Gaussian model on all data (concatenated over prepared_state).
        Store the fit result and fitter as attributes.
        """
        # Use the global density for all prepared_state for fitting
        density_all = self.hist_dataset['density'].values.reshape(-1, self.hist_dataset['density'].shape[-2], self.hist_dataset['density'].shape[-1])
        density_all = np.sum(density_all, axis=0)  # sum over prepared_state
        x = self.hist_dataset['x'].values
        y = self.hist_dataset['y'].values
        fit_all_result, fit_all_fitter = self._fit_histogram_by_multi_2Dgaussian(density_all, x, y)
        trained_multi_2Dgaussian_params = self._extract_multi_2Dgaussian_params(fit_all_fitter, n_gauss=len(self.mean_init))
        return trained_multi_2Dgaussian_params

    def _extract_multi_2Dgaussian_params(self, fit_result, n_gauss=None):
        """
        Extract means, stds, and amps from a fit result (lmfit.ModelResult) and return a dict in the format of self.trained_multi_2Dgaussian_params.
        Args:
            fit_result: lmfit ModelResult object with .params attribute
            n_gauss: number of Gaussians (if None, use self.mean_init)
        Returns:
            dict with keys 'means', 'stds', 'amps'
        """
        if n_gauss is None:
            n_gauss = len(self.mean_init)
        means = []
        stds = []
        amps = []
        for i in range(n_gauss):
            x0 = fit_result.params[f'g{i}_x0'].value
            y0 = fit_result.params[f'g{i}_y0'].value
            sigma_x = fit_result.params[f'g{i}_sigma_x'].value
            sigma_y = fit_result.params[f'g{i}_sigma_y'].value
            amp = fit_result.params[f'g{i}_amp'].value
            means.append(np.array([x0, y0]))
            stds.append(np.array([sigma_x, sigma_y]))
            amps.append(amp)

        return {
            'means': np.array(means),
            'stds': np.array(stds),
            'covariances': np.array(stds)[0]**2,
            'amps': np.array(amps),
        }
    
    def _fit_histogram_by_multi_2Dgaussian(self, density, x, y, mean_init=None, std_init=None):
        """
        Fit a 2D histogram (density) using FitMultiGaussian2D.
        Args:
            density: 2D numpy array (shape: [len(x), len(y)])
            x: 1D array of x bin centers
            y: 1D array of y bin centers
            mean_init: list/array of initial means (optional)
            std_init: list/array of initial stds (optional)
        Returns:
            fit_result: lmfit ModelResult from FitMultiGaussian2D.fit()
            fitter: the FitMultiGaussian2D instance
        """
        fix_means = False
        fix_std = False
        if mean_init is None:
            mean_init = self.mean_init
            fix_means = True
            print("Using default mean_init for fitting.", mean_init)
        if std_init is None:
            std_init = self.std_init
            fix_std = True
            print("Using default std_init for fitting.", std_init)

        std = np.min(std_init)

        from qcat.utilities.function_fitting.fit_gaussian2d import FitMultiGaussian2D
        n_gauss = len(mean_init)
        fitter = FitMultiGaussian2D(density, x, y, n_gauss=n_gauss)
        fitter.params['offset'].set(value=0, vary=False)
        for i in range(n_gauss):
            fitter.params[f'g{i}_x0'].set(value=mean_init[i][0], vary=fix_means)
            fitter.params[f'g{i}_y0'].set(value=mean_init[i][1], vary=fix_means)
            fitter.params[f'g{i}_sigma_x'].set(value=std, vary=fix_std)
            fitter.params[f'g{i}_sigma_y'].set(expr=f'g{i}_sigma_x')
        fit_result = fitter.fit()
        return fit_result, fitter
      
    def _export_result(self, save_path=None):
        # Implement result export functionality if needed.
        pass


    
    def rotate_data_to_x_axis(self):
        """
        Return a rotated copy of self.data so that the vector between the two GMM means (in scaled space) aligns with the x-axis.
        self.data is never modified in-place.
        Returns:
            rotated_data: xarray.Dataset with rotated 'I' and 'Q' variables
            angle: rotation angle in radians (counterclockwise)
        """
        import copy
        # Get means in scaled space
        means = self.gmm_model.means_  # shape (2, 2), columns: I, Q
        v = means[1] - means[0]  # vector from mean 0 to mean 1
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
    
    def calc_distances_to_means(self, means_trained=None):
        """
        Calculate the Euclidean distances from each (I, Q) point (for all shot_idx and prepared_state)
        to each of the two means_trained points.
        Args:
            means_trained: list or array of two mean points [[x0, y0], [x1, y1]]. If None, uses self._trained_multi_2Dgaussian_params['means'].
        Returns:
            distances: dict with keys 'prepared_state', each value is a (n_shots, 2) array of distances to each mean.
        """
        if means_trained is None:
            means_trained = self._trained_multi_2Dgaussian_params['means']
        # Get coordinate values
        prepared_states = self.data.coords['prepared_state'].values
        n_center = len(means_trained)
        n_state = len(prepared_states)
        n_shot = self.data.sizes['shot_idx']

        # Allocate array: (center, prepared_state, idx_shot)
        dist_arr = np.zeros((n_center, n_state, n_shot))
        for i_center, mean in enumerate(means_trained):
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
    



def Single_shot_ref_fit_analysis(data:tuple):

    I, Q= np.array(data[0]), np.array(data[1])
    bins=101
    I_=np.linspace(I.min(),I.max(),bins)  
    Q_=np.linspace(Q.min(),Q.max(),bins)
    hist, xedges, yedges = np.histogram2d(I,Q, bins=(bins,bins), density=True)
    I_guess, Q_guess, sig_guess=np.mean(I),np.mean(Q),(np.std(I)+np.std(Q))/2
    c_I=Parameter(name='c_I', value= I_guess, min=I_guess-5*sig_guess, max=I_guess+5*sig_guess) 
    c_Q=Parameter(name='c_Q', value= Q_guess, min=Q_guess-5*sig_guess, max=Q_guess+5*sig_guess)
    sigma=Parameter(name='sigma', value= sig_guess, min=0.1*sig_guess, max=3*sig_guess)
    X,Y= np.meshgrid(I_,Q_)
    result= gauss2d_func_model.fit(hist.transpose(),I=X,Q=Y,c_I=c_I,c_Q=c_Q,sigma=sigma,A=Parameter(name='A',value=np.max(hist), min=0.01*np.max(hist)) )     
    c_I_fit=result.best_values['c_I']
    c_Q_fit=result.best_values['c_Q']
    sigma_fit=result.best_values['sigma']
    print('Fit result: c_I=%.3e, c_Q=%.3e, sigma=%.3e'%(c_I_fit,c_Q_fit,sigma_fit))
    A_fit=result.best_values['A']
    fit_pack=[c_I_fit,c_Q_fit,sigma_fit]
    fitting= gauss2d_func(X,Y,c_I_fit,c_Q_fit,sigma_fit,A_fit)
    #print ('Total prob. =',np.sum(hist)*((max(xedges)-min(xedges))/bins*(max(yedges)-min(yedges))/bins))
    
    return dict(data=[I,Q],data_hist=hist,coords=[X,Y],fitting=fitting,fit_pack=fit_pack)

def Outlier(I,Q,Ig,Qg,Ie,Qe,sigma): # only use in rotated IQ signals on I axis
    I,Q= np.array(I),np.array(Q)
    Dis_g= IQ_data_dis(I,Q,Ig,Qg)
    Dis_e= IQ_data_dis(I,Q,Ie,Qe)
    Outlier_event_I=[]
    Outlier_event_Q=[]
    Inner_event_I=[]
    Inner_event_Q=[]
    for i in range(len(I)): 
        if (I[i]<Ig and Dis_g[i]>3*sigma) or (I[i]>Ie and Dis_e[i]>3*sigma) or np.abs(Q[i])>3*sigma:
            Outlier_event_I.append(I[i])
            Outlier_event_Q.append(Q[i])
        else:
            Inner_event_I.append(I[i])
            Inner_event_Q.append(Q[i])
    Outlier_event_I,Outlier_event_Q,Inner_event_I,Inner_event_Q= np.array(Outlier_event_I),np.array(Outlier_event_Q),np.array(Inner_event_I),np.array(Inner_event_Q)
    return dict(Outlier_event=[Outlier_event_I,Outlier_event_Q],Inner_event=[Inner_event_I,Inner_event_Q],Outlier_P=len(Outlier_event_I)/len(I))




def Qubit_state_single_shot_fit_analysis(data:dict, T1:float=None,tau:float=None,f01:float=None,fixed_sigma_on=False,fixed_sigma_value=None):
    Ig_data,Qg_data,Ie_data,Qe_data= np.array(data['g'][0]), np.array(data['g'][1]) ,np.array(data['e'][0]) , np.array(data['e'][1])
    if len(Ig_data)<=2000:
        bins=21
    elif 2000<len(Ig_data)<10000:
        bins=101
    elif 10000<=len(Ig_data)<=20000:
        bins=151
    else:
        bins=201
    g_predict= Single_shot_ref_fit_analysis(data['g'])['fit_pack']
    Ig_guess, Qg_guess, sig_guess= g_predict[0],g_predict[1],g_predict[2]
    I_mixdata, Q_mixdata= np.hstack([Ig_data,Ie_data]), np.hstack([Qg_data,Qe_data])
    I_=np.linspace(I_mixdata.min(),I_mixdata.max(),bins)  
    Q_=np.linspace(Q_mixdata.min(),Q_mixdata.max(),bins)
    X,Y= np.meshgrid(I_,Q_)
    hist_Ie, edges_Ie= np.histogram(Ie_data, bins=bins, density=True)
    hist_Qe, edges_Qe= np.histogram(Qe_data, bins=bins, density=True)
    hist, xedges, yedges = np.histogram2d(I_mixdata, Q_mixdata, bins=(bins,bins), density=True)
    Ie_guess_idx, Qe_guess_idx=find_nearest(hist_Ie,np.max(hist_Ie)),find_nearest(hist_Qe,np.max(hist_Qe))
    Ie_guess, Qe_guess= edges_Ie[Ie_guess_idx],edges_Qe[Qe_guess_idx]
    #Parameter ini-guess
    cg_I=Parameter(name='cg_I', value= Ig_guess, min=Ig_guess-1*sig_guess, max=Ig_guess+1*sig_guess) 
    cg_Q=Parameter(name='cg_Q', value= Qg_guess, min=Qg_guess-1*sig_guess, max=Qg_guess+1*sig_guess)
    ce_I=Parameter(name='ce_I', value= Ie_guess, min=Ie_guess-5*sig_guess, max=Ie_guess+5*sig_guess) 
    ce_Q=Parameter(name='ce_Q', value= Qe_guess, min=Qe_guess-5*sig_guess, max=Qe_guess+5*sig_guess)
    
    Ag=Parameter(name='Ag',value=np.max(hist), min=0.1*np.max(hist)) 
    Ae=Parameter(name='Ae',value=np.max(hist), min=0.1*np.max(hist))
    if fixed_sigma_on:
        sigma=Parameter(name='sigma', value= fixed_sigma_value,vary=False)
    else:
        sigma=Parameter(name='sigma', value= sig_guess, min=0.01*sig_guess, max=3*sig_guess)
    # mixed data fit
    result= bigauss2d_func_model.fit(hist.transpose(),I=X,Q=Y,cg_I=cg_I,cg_Q=cg_Q,sigma=sigma,Ag=Ag,ce_I=ce_I,ce_Q=ce_Q,Ae=Ae)
    cg_I_fit=result.best_values['cg_I']
    cg_Q_fit=result.best_values['cg_Q']
    ce_I_fit=result.best_values['ce_I']
    ce_Q_fit=result.best_values['ce_Q']
    sigma_fit=result.best_values['sigma']
    # displace + rotate
    angle= np.angle(ce_I_fit-cg_I_fit+(ce_Q_fit-cg_Q_fit)*1j)
    rot_e_center= rot(ce_I_fit-cg_I_fit,ce_Q_fit-cg_Q_fit,angle)
    rot_g_IQ= rot(Ig_data-cg_I_fit,Qg_data-cg_Q_fit,angle)
    rot_e_IQ= rot(Ie_data-cg_I_fit,Qe_data-cg_Q_fit,angle)
    Ig_data_new, Qg_data_new, Ie_data_new, Qe_data_new= rot_g_IQ[0],rot_g_IQ[1],rot_e_IQ[0],rot_e_IQ[1]
    # along single quadrature fit
    R= 20*sigma_fit #range_factor
    xmin, xmax = np.minimum(0,rot_e_center[0])-R,np.maximum(0,rot_e_center[0])+R
    ymin, ymax = np.minimum(0,rot_e_center[1])-R,np.maximum(0,rot_e_center[1])+R
    hist_r_g, xedges, yedges = np.histogram2d(Ig_data_new, Qg_data_new, bins=(bins,bins),range=[[xmin, xmax], [ymin, ymax]], density=True)
    hist_r_e, xedges, yedges = np.histogram2d(Ie_data_new, Qe_data_new, bins=(bins,bins),range=[[xmin, xmax], [ymin, ymax]], density=True)
    New_axe_g_hist= np.sum(hist_r_g.transpose(), axis=0)
    New_axe_e_hist= np.sum(hist_r_e.transpose(), axis=0)
    I_ro = np.linspace(xmin, xmax,bins)
    I_fit= np.linspace(xmin, xmax,bins*5)
    def Reduced_bimodal_func(x,Ag,Ae):
        return gauss_func(x,0,sigma_fit,Ag)+gauss_func(x,rot_e_center[0],sigma_fit,Ae)
    bimodal_func_model = Model(Reduced_bimodal_func)
    result_g= bimodal_func_model.fit(New_axe_g_hist,x=I_ro,Ag=Parameter(name='Ag',value=np.max(New_axe_g_hist), min=0.5*np.max(New_axe_g_hist)) ,Ae=Parameter(name='Ae',value=0.05*np.max(New_axe_g_hist), min=0.001*np.max(New_axe_g_hist)))
    result_e= bimodal_func_model.fit(New_axe_e_hist,x=I_ro,Ag=Parameter(name='Ag',value=0.5*np.max(New_axe_e_hist), min=0.001*np.max(New_axe_e_hist)),Ae=Parameter(name='Ae',value=np.max(New_axe_e_hist), min=0.5*np.max(New_axe_e_hist)))
    Agg_fit=result_g.best_values['Ag']
    Aeg_fit=result_g.best_values['Ae']
    Age_fit=result_e.best_values['Ag']
    Aee_fit=result_e.best_values['Ae']
    
    inter_point = rot_e_center[0]/2
    def G_gg(I):
        return Agg_fit*np.exp(-I**2/(2*sigma_fit**2))
    def G_eg(I):
        return Aeg_fit*np.exp(-(I-rot_e_center[0])**2/(2*sigma_fit**2))
    def G_ge(I):
        return Age_fit*np.exp(-I**2/(2*sigma_fit**2))
    def G_ee(I):
        return Aee_fit*np.exp(-(I-rot_e_center[0])**2/(2*sigma_fit**2))

    Ggg= quad(G_gg,rot_e_center[0]-R,rot_e_center[0]+R)
    Geg= quad(G_eg,rot_e_center[0]-R,rot_e_center[0]+R)
    Gge= quad(G_ge,rot_e_center[0]-R,rot_e_center[0]+R)
    Gee= quad(G_ee,rot_e_center[0]-R,rot_e_center[0]+R)
    overlap_gg= quad(G_gg,inter_point,rot_e_center[0]+R)
    transi_eg= quad(G_eg,inter_point,rot_e_center[0]+R)
    transi_ge= quad(G_ge,rot_e_center[0]-R,inter_point)
    overlap_ee= quad(G_ee,rot_e_center[0]-R,inter_point)
    overlap= (overlap_gg[0])/(Ggg[0])
    Peg=  (Geg[0])/(Geg[0]+Ggg[0])
    Pge= (Gge[0])/(Gee[0]+Gge[0])
    Thermal= (Geg[0])/(Geg[0]+Ggg[0])
    Relax= (Gge[0])/(Gee[0]+Gge[0])-Thermal




    if f01 is None or tau is None or T1 is None:
        Wa = None
        Pre_decay = None
        T = None
    else:
        Wa = 2 * np.pi * f01
        T = PetoT(Thermal, Wa)
        Relax_predict= Relax_cal(0,tau,T1)
        Pre_decay= Relax-Relax_predict

    D = rot_e_center[0]
    SNR = D / sigma_fit
    overlap_predict = (1 / 2) * (1 - special.erf(np.sqrt(SNR ** 2 / 8)))
    M = np.array([[1 - Peg, Peg], [Pge, 1 - Pge]])
    F_s = 1 - overlap
    F_g = 1 - Peg
    F_e = 1 - Pge
    F = 1 - (1 / 2) * (Peg + Pge)

    Outlier_g_info = Outlier(Ig_data_new, Qg_data_new, 0, 0, rot_e_center[0], rot_e_center[1], sigma_fit)
    Outlier_e_info = Outlier(Ie_data_new, Qe_data_new, 0, 0, rot_e_center[0], rot_e_center[1], sigma_fit)
    fit_pack = [rot_e_center, Agg_fit, Aeg_fit, Age_fit, Aee_fit, sigma_fit, New_axe_g_hist, New_axe_e_hist]
    error_pack = dict(Ig=cg_I_fit, Qg=cg_Q_fit, Ie=ce_I_fit, Qe=ce_Q_fit,
                     D=D, sigma=sigma_fit, SNR=SNR, overlap=overlap, overlap_predict=overlap_predict,
                     Pgg=1-Peg, Peg=Peg, Pge=Pge, Pee=1-Pge, Thermal=Thermal, eff_T_mK=T,
                     Relax=Relax, Relax_predict=Relax_predict, Pre_decay=Pre_decay,
                     F_s=F_s, F_g=F_g, F_e=F_e, F=F,
                     pre_g_outlier=Outlier_g_info['Outlier_P'], pre_e_outlier=Outlier_e_info['Outlier_P'])

    return dict(rot_IQdata=[rot_g_IQ, rot_e_IQ], I_ro=I_ro, I_fit=I_fit, fit_pack=fit_pack, error_pack=error_pack, M=M,
                Ig=cg_I_fit, Qg=cg_Q_fit, Ie=ce_I_fit, Qe=ce_Q_fit, Outlier_g_info=Outlier_g_info, Outlier_e_info=Outlier_e_info)

plt.show()
if __name__ == '__main__':
    import xarray as xr
    from qcat.analysis.state_discrimination.visualization import Qubit_state_single_shot_1Q
    from qcat.parser.qm_reader import load_xarray_h5, repetition_data
    # Load the dataset
    file_path = r"d:\github\ASQMDriver\data\MIST\2025-09-08\#68_07_iq_blobs_210029\ds_raw.h5"
    ds = load_xarray_h5(file_path)


    Ig = ds['Ig']
    Ie = ds['Ie']
    Qg = ds['Qg']
    Qe = ds['Qe']

    # Stack Ig/Ie and Qg/Qe into new variables I and Q with a new axis 'prepared_state'
    # prepared_state: 0 = g, 1 = e
    I = xr.concat([Ig, Ie], dim=xr.DataArray([0, 1], dims='prepared_state', name='prepared_state'))
    Q = xr.concat([Qg, Qe], dim=xr.DataArray([0, 1], dims='prepared_state', name='prepared_state'))

    # Create a new dataset with only I and Q
    ds = xr.Dataset({'I': I, 'Q': Q})
    sep_data = repetition_data(ds, repetition_dim="qubit")



    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        if qubit_name == "q1":

            # Rename n_runs to shot_idx if present
            sq_data = sq_data.rename({'n_runs': 'shot_idx'})
            # print(sq_data)
            analysis = StateDiscrimination(sq_data)
            analysis._start_analysis()
            analysis._plot_results(qubit_name,r"d:\github\ASQMDriver\data\MIST\2025-09-22\#769_LCH_readout_fidelity_152716")

            # Ig = sq_data['I'].sel(prepared_state=0).values
            # Ie = sq_data['I'].sel(prepared_state=1).values
            # Qg = sq_data['Q'].sel(prepared_state=0).values
            # Qe = sq_data['Q'].sel(prepared_state=1).values
            # raw_data = {'g': [Ig, Qg], 'e': [Ie, Qe]}
            # if all(v is not None for v in [Ig, Ie, Qg, Qe]):
            #     print(f"Extracted Ig, Ie, Qg, Qe for qubit {qubit_name}.")
            #     result = Qubit_state_single_shot_fit_analysis(raw_data, T1=100, tau=10, f01=3.5)
            #     Qubit_state_single_shot_1Q(raw_data, result, None, False, None, None)
            # else:
            #     print(f"Could not find all required variables: Ig, Ie, Qg, Qe for qubit {qubit_name}.")

    plt.show()
# %%
