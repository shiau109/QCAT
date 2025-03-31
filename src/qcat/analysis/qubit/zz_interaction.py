from qcat.analysis.base import QCATAna
from qcat.analysis.function_fitting.fit_damped_oscillation import FitDampedOscillation
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS

class ZZinteractionEcho(QCATAna):
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
        statistic_freq = []

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

            statistic_freq.append(fit_result.params['f'].value)

        # Create a dataset containing the analysis results with "flux" as coordinate.
        dims = ["flux"]
        coords = {
            "flux": self.data.coords["flux"].values,
        }
        self.statistic_result = xr.Dataset(
            {
                "frequency": (dims, np.array(statistic_freq)),
            },
            coords=coords,
        )

        # self._plot_results()

    def _plot_results(self):
        time = self.data.coords["time"].values
        flux = self.data.coords["flux"]
        t2 = self.statistic_result["frequency"].values

        # Multiply by 1000 to return the data to its original scale (if needed)
        raw_data = self.data.transpose() * 1000
        median_ans, std_ans = np.median(t2), np.std(t2)

        fig = plt.figure(dpi=100, figsize=(12, 9))
        gs = GS(2, 2, width_ratios=[2, 1], height_ratios=[1, 5])
        ax1 = fig.add_subplot(gs[:, 1])
        ax1.hist(t2, bins='auto', density=False)
        ax1.axvline(median_ans, c='k', ls='--', lw=1)
        ax1.set_ylabel("Counts", fontsize=20)
        ax1.set_xlabel("frequency (MHz)", fontsize=20)
        ax1.set_title(f"frequency = {round(median_ans,1)} ± {round(std_ans,1)} us", fontsize=20)

        ax2 = fig.add_subplot(gs[:, 0])
        c = ax2.pcolormesh(flux, time, raw_data, cmap='RdBu')
        fig.colorbar(c, ax=ax2, label='Contrast (mV)', location='bottom')
        ax2.plot(flux, t2, c='green')
        # ax2.axhline(median_ans + std_ans, linestyle='--', c='orange')
        # ax2.axhline(median_ans, linestyle='-', c='orange')
        # ax2.axhline(median_ans - std_ans, linestyle='--', c='orange')
        ax2.set_xlabel('Flux', fontsize=20)
        ax2.set_ylabel("Free evolution time (us)", fontsize=20)

        for ax in [ax1, ax2]:
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)

        fig.tight_layout()
        self.fig = fig

    def _export_result(self, save_path=None):
        # Implement result export functionality if needed.
        pass


if __name__ == '__main__':
    # Open the netCDF dataset with your data.
    dataset = xr.open_dataset(r"d:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\20250112_122247_find_ZZfree_q1_q2\find_ZZfree_q1_q2.nc")
    dataset = dataset.sel(mixer="I")
    print(dataset)

    # Loop over the data variables.
    # (If needed, adjust selection. For example, the original code used .sel(mixer="I") if applicable.)
    # for dataset, data in dataset.data_vars.items():
    data = dataset["q1_ro"]
    data.attrs = dataset.attrs
    data.name = "q1_ro"
    # Create an instance of the RamseyFluxAnalysis class
    analysis = ZZinteractionEcho(data)
    analysis._start_analysis()
    output_dataarray = xr.DataArray(
        data = analysis.statistic_result["frequency"].values,
        dims=["flux"],
        coords=dict(
            flux = data.coords["flux"].values,
        )
    )
    output_dataarray.to_netcdf(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\20250112_122247_find_ZZfree_q1_q2\zz_freq.nc")
    plt.show()


