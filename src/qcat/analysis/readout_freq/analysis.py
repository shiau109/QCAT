# Copy of power_dep_state/analysis.py, adapted for frequency-dependent readout
# All references to 'amp_prefactor' are changed to 'frequency'

from qcat.utilities.function_fitting.fit_damped_oscillation import FitDampedOscillation
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS
from qcat.parser.qm_reader import load_xarray_h5, repetition_data

class ROFidelityFreq():
    """
    Class for analyzing exponential decay data with the frequency coordinate.
    This is adapted from the repetition analysis code but replaces "repetition" with "frequency".
    """

    def __init__(self, data: xr.Dataset):
        super().__init__()
        self._import_data(data)
        self.weight = 'mix'  # 'none', 'cov', 'p_outlier', 'mix'
    def _import_data(self, data):
        # Ensure input data is an xarray Dataset and has 'time' and 'frequency' coordinates.
        if not isinstance(data, xr.Dataset):
            raise ValueError("Input data must be an xarray.Dataset.")

        for coords_name in ["shot_idx", "frequency"]:
            if coords_name not in data.coords:
                raise ValueError(f"No {coords_name} coordinate in the input Dataset.")

        self.data = data

    def _start_analysis(self):
        """
        Separate the data into a list of subdata along frequency coordinate.
        For each subdata, use StateDiscrimination to do the analysis.
        Store the results in self.state_discrimination_results (list).
        Also gather std_list and mean_list for summary plotting.
        """
        from qcat.analysis.state_discrimination.analysis import StateDiscrimination
        self.state_discrimination_results = []
        self.frequencies = frequencies = self.data.coords['frequency'].values
        p_outlier_list = []
        std_list = []
        mean_list = []
        norm_res_list = []
        for freq in frequencies:
            subdata = self.data.sel(frequency=freq)
            # If subdata is DataArray, convert to Dataset with I and Q if needed
            if isinstance(subdata, xr.DataArray):
                # Assume subdata has variables 'I' and 'Q' or is already suitable
                if 'I' in subdata or 'Q' in subdata:
                    subdata = subdata.to_dataset()
            analysis = StateDiscrimination(subdata)
            analysis._start_analysis()
            self.state_discrimination_results.append(analysis)
            # Gather std (sqrt(covariances)) and means for each frequency
            result = analysis.analysis_result
            trained_paras = result["trained_paras"]
            std_list.append(trained_paras["std"])
            mean_list.append(trained_paras["mean"])
            p_outlier_list.append(result["outlier_probability"])
            norm_res_list.append(result['norm_res'])
        self.p_outlier = np.array(p_outlier_list)
        self.std_list = np.array(std_list)
        self.mean_list = np.array(mean_list)
        # Build a summary xarray.Dataset for all frequencies
        freq_dim = np.array(frequencies)

        self.summary_dataset = xr.Dataset(
            {
                'p_outlier': (['frequency', 'state'], np.array(p_outlier_list)),
                'std': (['frequency'], np.array(std_list)),
                'mean': (['frequency', 'state', 'iq'], np.array(mean_list)),
                'norm_res': (['frequency', 'state'], np.array(norm_res_list))
            },
            coords={
                'frequency': freq_dim,
                'state': [0, 1],
                'iq': ['I', 'Q'],
            }
        )
        print(self.summary_dataset)

    def _plot_results(self, fig_group_name=None, save_path=None, plot_all=False ):
        from qcat.analysis.readout_freq.visualization import (
            plot_p_outlier_vs_frequency,
            plot_std_vs_frequency,
            plot_means_distance_vs_frequency,
            plot_means_on_IQ_plane_vs_frequency,
            plot_gaussian_norms_and_direct_counts_vs_frequency,
            plot_norm_res_vs_frequency
        )

        figs = {}
        figs["outlier"] = plot_p_outlier_vs_frequency(self.summary_dataset['p_outlier'])
        figs["std_vs_frequency"] = plot_std_vs_frequency(self.summary_dataset['std'])
        figs["means_distance_vs_frequency"] = plot_means_distance_vs_frequency(self.summary_dataset['mean'])
        figs["means_on_IQ_plane"] = plot_means_on_IQ_plane_vs_frequency(self.summary_dataset)
        figs["norm_res_vs_frequency"] = plot_norm_res_vs_frequency(self.summary_dataset["norm_res"])
        # Gather gaussian_norms and direct_counts from each analysis_result
        gaussian_norms = np.array([res.analysis_result['gaussian_norms'] for res in self.state_discrimination_results])
        direct_counts = np.array([res.analysis_result['direct_counts'] for res in self.state_discrimination_results])
        figs["fidelity_vs_frequency"] = plot_gaussian_norms_and_direct_counts_vs_frequency(
            self.frequencies, gaussian_norms, direct_counts
        )

        if plot_all:
            for i in range(len(self.state_discrimination_results)):
                analysis = self.state_discrimination_results[i]
                analysis._plot_results(fig_group_name=f"{fig_group_name}_{self.frequencies[i]}", save_path=save_path)
        if save_path is not None:
            for plot_name, fig in figs.items():
                fig.savefig(f"{save_path}\\{fig_group_name}_{plot_name}.png", bbox_inches='tight')
        return figs

    
    
    def _export_result(self, save_path=None):
        # Implement result export functionality if needed.
        pass


if __name__ == '__main__':
    # Open the netCDF dataset with your data.
    import matplotlib.pyplot as plt
    path_name = r"d:\github\ASQMDriver\data\MIST\2025-09-30\#1074_LCH_readout_frequency_224043"
    ds = load_xarray_h5(path_name+"\\ds_raw.h5")
    sep_data = repetition_data(ds, repetition_dim="qubit")



    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        # Rename n_runs to shot_idx if present
        # sq_data = sq_data.rename({'n_runs': 'shot_idx','state': 'prepared_state'})
        print(sq_data)
        analysis = ROFidelityFreq(sq_data)
        analysis._start_analysis()
        analysis._plot_results(qubit_name, path_name, plot_all=True)
        

    # plt.show()
