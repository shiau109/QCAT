# from qcat.analysis.base import QCATAna
from qcat.utilities.function_fitting.fit_damped_oscillation import FitDampedOscillation
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS
from qcat.parser.qm_reader import load_xarray_h5, repetition_data

class ZZinteractionEcho():
    """
    Class for analyzing exponential decay data with the flux coordinate.
    This is adapted from the repetition analysis code but replaces "repetition" with "flux".
    """

    def __init__(self, data: xr.DataArray):
        super().__init__()
        self._import_data(data)

    def _import_data(self, data):
        # Ensure input data is an xarray DataArray and has 'time' and 'flux' coordinates.
        if not isinstance(data, xr.DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")

        for coords_name in ["time", "flux"]:
            if coords_name not in data.coords:
                raise ValueError(f"No {coords_name} coordinate in the input DataArray.")

        self.data = data
        # Convert time coordinate units if needed (here, dividing by 1000 to convert ms to us)
        self.data.coords["time"] = self.data.coords["time"] / 1000

    def _start_analysis(self):
        # Iterate over the flux points instead of repetition
        flux_points = self.data.coords["flux"].values.shape[0]
        param_dict = {}


        for i in range(flux_points):
            # For each flux value, perform a damped oscillation fit.
            fit_ramsey = FitDampedOscillation(self.data.isel(flux=i).rename({"time": "x"}))
            # (Optional) Get initial parameter guesses
            guess_params = fit_ramsey.guess()
            guess_params["f"].min=0
            # Fit the data and extract the decay time parameter (f)
            fit_ramsey.params = guess_params
            fit_result = fit_ramsey.fit()
            self.fit_result = fit_result

            # Collect all parameter values
            for param_name, param in fit_result.params.items():
                param_dict.setdefault(param_name, []).append(param.value)

        # Create a dataset containing the analysis results with "flux" as coordinate.
        self.statistic_result = xr.Dataset(
            {param: ("flux", np.array(values)) for param, values in param_dict.items()},
        coords={"flux": self.data.coords["flux"].values}
        )


    def _plot_results(self):
        from qcat.analysis.zz_interaction.visualization import plot_results
        return plot_results(self.data, self.statistic_result)
    
    def _export_result(self, save_path=None):
        # Implement result export functionality if needed.
        pass


if __name__ == '__main__':
    # Open the netCDF dataset with your data.
    from qcat.utilities.simple_visualization import plot_2d_colormap_from_h5
    import matplotlib.pyplot as plt

    ds = load_xarray_h5(r"d:\github\ASQMDriver\data\QPU_project\2025-08-25\#862_LCH_zz_interaction_withCouplerOffset_195540\ds_raw.h5")
    print(ds)
    sep_data = repetition_data(ds, repetition_dim="qubit")


    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        data = sq_data["state"].transpose('coupler_z','idle_time')
        # Rename coordinates for compatibility with ZZinteractionEcho
        data = data.rename({'coupler_z': 'flux', 'idle_time': 'time'})
        # print(data)

        # plot_2d_colormap_from_h5(sq_data, data_var="state", x_dim='coupler_z', y_dim="idle_time")
        data.attrs = sq_data.attrs
        data.name = qubit_name
        # Create an instance of the RamseyFluxAnalysis class
        analysis = ZZinteractionEcho(data)
        analysis._start_analysis()
        figs = analysis._plot_results()
    # output_dataarray.to_netcdf(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\20250112_122247_find_ZZfree_q1_q2\zz_freq.nc")
    plt.show()


