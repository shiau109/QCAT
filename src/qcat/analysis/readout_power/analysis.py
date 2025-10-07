# from qcat.analysis.base import QCATAna
from qcat.utilities.function_fitting.fit_damped_oscillation import FitDampedOscillation
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS
from qcat.parser.qm_reader import load_xarray_h5, repetition_data

class ROFidelityPower():
    """
    Class for analyzing exponential decay data with the flux coordinate.
    This is adapted from the repetition analysis code but replaces "repetition" with "flux".
    """

    def __init__(self, data: xr.Dataset, user_mean=None, user_std=None):
        super().__init__()
        self._import_data(data)
        self.weight = 'cov'  # 'none', 'cov', 'p_outlier', 'mix'
        self.user_mean = user_mean
        self.user_std = user_std
    def _import_data(self, data):
        # Ensure input data is an xarray Dataset and has required coordinates.
        if not isinstance(data, xr.Dataset):
            raise ValueError("Input data must be an xarray.Dataset.")

        for coords_name in ["shot_idx", "amp_prefactor"]:
            if coords_name not in data.coords:
                raise ValueError(f"No {coords_name} coordinate in the input Dataset.")

        self.data = data

    def _start_analysis(self):
        """
        Separate the data into a list of subdata along sweep_coord coordinate.
        For each subdata, use StateDiscrimination to do the analysis.
        Store the results in self.state_discrimination_results (list).
        Also gather std_list and mean_list for summary plotting.
        """
        from qcat.analysis.state_discrimination.analysis import StateDiscrimination
        self.state_discrimination_results = []
        sweep_values = self.data.coords["amp_prefactor"].values
        self.sweep_values = sweep_values
        p_outlier_list = []
        std_list = []
        mean_list = []
        norm_res_list = []
        for val in sweep_values:
            subdata = self.data.sel({"amp_prefactor": val})
            # If subdata is DataArray, convert to Dataset with I and Q if needed
            if isinstance(subdata, xr.DataArray):
                # Assume subdata has variables 'I' and 'Q' or is already suitable
                if 'I' in subdata or 'Q' in subdata:
                    subdata = subdata.to_dataset()
            # Pass user_mean and user_std if set
            kwargs = {}
            if self.user_mean is not None:
                kwargs['user_mean'] = self.user_mean
            if self.user_std is not None:
                kwargs['user_std'] = self.user_std
            analysis = StateDiscrimination(subdata, **kwargs)
            analysis._start_analysis()
            self.state_discrimination_results.append(analysis)
            # Gather std (sqrt(covariances)) and means for each sweep value
            result = analysis.analysis_result
            trained_paras = result["trained_paras"]
            std_list.append(trained_paras["std"])
            mean_list.append(trained_paras["mean"])
            p_outlier_list.append(result["outlier_probability"])
            norm_res_list.append(result['norm_res'])
        self.p_outlier = np.array(p_outlier_list)
        self.std_list = np.array(std_list)
        self.mean_list = np.array(mean_list)
        # Build a summary xarray.Dataset for all sweep values
        sweep_dim = np.array(sweep_values)

        self.summary_dataset = xr.Dataset(
            {
                'p_outlier': (["amp_prefactor", 'state'], np.array(p_outlier_list)),
                'std': (["amp_prefactor"], np.array(std_list)),
                'mean': (["amp_prefactor", 'state', 'iq'], np.array(mean_list)),
                'norm_res': (["amp_prefactor", 'state'], np.array(norm_res_list))
            },
            coords={
                "amp_prefactor": sweep_dim,
                'state': [0, 1],
                'iq': ['I', 'Q'],
            }
        )
        print(self.summary_dataset)
        self.fit_paras, fit_curve = self.fit_means_vs_amp_prefactor()

    def _plot_results(self, fig_group_name=None, save_path=None, plot_all=False ):
        from qcat.analysis.readout_power.visualization import (
            plot_std_vs_amp_prefactor,
            plot_means_distance_vs_amp_prefactor,
            plot_means_on_IQ_plane_vs_amp_prefactor,
            plot_gaussian_norms_and_direct_counts_vs_amp_prefactor,
            plot_norm_res_vs_amp_prefactor,
            plot_p_outlier_vs_amp_prefactor
        )

        figs = {}
        figs["outlier"] = plot_p_outlier_vs_amp_prefactor(self.summary_dataset['p_outlier'])
        figs["std_vs_amp"] = plot_std_vs_amp_prefactor(self.summary_dataset['std'])
        figs["means_distance_vs_amp"] = plot_means_distance_vs_amp_prefactor(self.summary_dataset['mean'])
        figs["means_on_IQ_plane"] = plot_means_on_IQ_plane_vs_amp_prefactor(self.summary_dataset, self.fit_paras)
        figs["norm_res_vs_amp"] = plot_norm_res_vs_amp_prefactor(self.summary_dataset["norm_res"])
        # Gather gaussian_norms and direct_counts from each analysis_result
        gaussian_norms = np.array([res.analysis_result['gaussian_norms'] for res in self.state_discrimination_results])
        direct_counts = np.array([res.analysis_result['direct_counts'] for res in self.state_discrimination_results])
        figs["fidelity_vs_amp"] = plot_gaussian_norms_and_direct_counts_vs_amp_prefactor(
            self.data.coords['amp_prefactor'].values, gaussian_norms, direct_counts
        )

        if plot_all:
            for i in range(len(self.state_discrimination_results)):
                analysis = self.state_discrimination_results[i]
                analysis._plot_results(fig_group_name=f"{fig_group_name}_{self.sweep_values[i]}", save_path=save_path)
        if save_path is not None:
            for plot_name, fig in figs.items():
                fig.savefig(f"{save_path}\\{fig_group_name}_{plot_name}.png", bbox_inches='tight')
        return figs

    def fit_means_vs_amp_prefactor(self):
        """
        Fit straight lines to I and Q as a function of amp_prefactor for both means in self.summary_dataset using lmfit's LinearModel.
        Optionally use 1/std**2 as weights if self.is_cov_weight is True.
        Returns:
            fit_results: dict with keys 'I0', 'Q0', 'I1', 'Q1', each value is lmfit.ModelResult
            fit_dataset: xarray.Dataset with dims ('state', 'iq') and variables 'slope', 'intercept'
            fit_curve_dataset: xarray.Dataset with dims ('amp_prefactor', 'state', 'iq') and variable 'mean_fit'
        """
        from lmfit.models import LinearModel
        ds = self.summary_dataset
        amp_prefactor_values = ds['amp_prefactor'].values
        means = ds['mean'].values  # shape (N, 2, 2)
        stds = ds['std'].values    # shape (N, 2)
        p_outlier = ds['p_outlier'].values  # shape (N, 2)
        fit_results = {}
        slopes = np.full((2,2), np.nan)
        intercepts = np.full((2,2), np.nan)
        mean_fit = np.full((len(amp_prefactor_values), 2, 2), np.nan)
        labels = ['I0', 'Q0', 'I1', 'Q1']
        # (state, iq): (0,0)=I0, (0,1)=Q0, (1,0)=I1, (1,1)=Q1
        for idx, (state, iq) in enumerate([(0,0), (0,1), (1,0), (1,1)]):

            y = means[:, state, iq]
            mask = np.isfinite(y) & np.isfinite(amp_prefactor_values)

            match self.weight:
                case 'p_outlier':
                    weights = np.zeros_like(y)
                    weights[mask] = 1.0 / p_outlier[:, state]**2
                case 'cov':
                    weights = np.zeros_like(y)
                    weights[mask] = 1.0 / stds**2
                case 'mix':
                    weights = np.zeros_like(y)
                    weights[mask] = 1.0 / (stds**2 *p_outlier[:, state]**2)
                case _:
                    weights = None


            print(f"Using {self.weight} weights for state {state}, iq {iq}")
            if np.sum(mask) >= 2:
                model = LinearModel()
                if weights is not None:
                    result = model.fit(y[mask], x=amp_prefactor_values[mask], weights=weights[mask])
                else:
                    result = model.fit(y[mask], x=amp_prefactor_values[mask])
                slopes[state, iq] = result.params['slope'].value
                intercepts[state, iq] = result.params['intercept'].value
                # Calculate fitted curve at all sweep_values
                mean_fit[:, state, iq] = result.eval(x=amp_prefactor_values)
            else:
                result = None
            fit_results[labels[idx]] = result
        # Build xarray.Dataset for fit parameters
        fit_paras = xr.Dataset(
            {
                'slope': (['state', 'iq'], slopes),
                'intercept': (['state', 'iq'], intercepts),
            },
            coords={
                'state': ds['state'].values,
                'iq': ds['iq'].values,
            }
        )
        # Build xarray.Dataset for fitted curve
        fit_curve_dataset = xr.Dataset(
            {
                'mean_fit': (['amp_prefactor', 'state', 'iq'], mean_fit),
            },
            coords={
                'amp_prefactor': ds['amp_prefactor'].values,
                'state': ds['state'].values,
                'iq': ds['iq'].values,
            }
        )
        return fit_paras, fit_curve_dataset
    
    def _export_result(self, save_path=None):
        # Implement result export functionality if needed.
        pass


if __name__ == '__main__':
    # Open the netCDF dataset with your data.
    import matplotlib.pyplot as plt
    path_name = r"D:\data\MIST\charge_ramsey_power_fidelity\41_500_1\#1436_LCH_const_charge_readout_power_28_010738"
    ds = load_xarray_h5(path_name+"\\ds_raw.h5")
    sep_data = repetition_data(ds, repetition_dim="qubit")



    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        # Rename n_runs to shot_idx if present
        # sq_data = sq_data.rename({'n_runs': 'shot_idx','state': 'prepared_state'})
        print(sq_data)
        analysis = ROFidelityPower(sq_data)
        analysis._start_analysis()
        analysis._plot_results(qubit_name, path_name, plot_all=True)
        

    # plt.show()
