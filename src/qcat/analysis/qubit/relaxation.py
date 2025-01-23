



from qcat.analysis.base import QCATAna
from qcat.analysis.function_fitting.fit_exp_decay import FitExponentialDecay
from xarray import DataArray
import numpy as np
import matplotlib.pyplot as plt

class RelaxationAnalysis(QCATAna):
    """
    Class for analyzing exponential decay data.
    """

    def __init__(self, data: DataArray):
        super().__init__()
        self._import_data(data)

    def _import_data(self, data):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")

        if "time" not in data.coords:
            raise ValueError("No 'time' coordinate in the input DataArray.")

        self.data = data
        self.data.coords["time"] = self.data.coords["time"] /1000
    def _start_analysis(self):
        # Initialize the fitting class
        fit_exp_decay = FitExponentialDecay(self.data.rename({"time": "x"}))

        # Generate initial parameter guesses
        guess_params = fit_exp_decay.guess()

        # Perform the fitting process
        fit_result = fit_exp_decay.fit()
        self.fit_result = fit_result

        # Plot the results
        self._plot_results()

    def _plot_results(self):
        x = self.data.coords["time"].values
        y = self.data.values

        # Create a figure
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', label="Data", markersize=4)

        if self.fit_result is not None:
            ax.plot(x, self.fit_result.best_fit, '-', label="Fit")
            a = self.fit_result.params['a'].value
            tau = self.fit_result.params['tau'].value
            c = self.fit_result.params['c'].value

            ax.text(0.05, 0.95, f"a: {a:.3e}\nτ: {tau:.3f}\nc: {c:.3f}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top')

        ax.set_title("Exponential Decay Analysis")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Signal")
        ax.legend()
        self.fig = fig

    def _export_result(self, save_path=None):
        pass



if __name__ == '__main__':
    import xarray as xr
    dataset = xr.open_dataset(r"d:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\Q1\20250112_154052_T1_rep\T1_rep.nc")
    print(dataset)
    for ro_name, data in dataset.data_vars.items():
        data.attrs = dataset.attrs
        data.name = ro_name
        my_ana = RelaxationAnalysis(data.sel(mixer="I"))
        my_ana._start_analysis()
        plt.show()
