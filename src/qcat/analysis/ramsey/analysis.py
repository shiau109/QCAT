
import xarray as xr
from qcat.utilities.function_fitting.fit_damping_beat import FitDampingBeat
import numpy as np
from numpy import fft, asarray, mean, nan, pi

class RamseyAnalysis:

    def __init__(self, data: xr.DataArray):
        self.data = data
        self.fit_result = None
        self.fitter = None
        self._fit()


    
    def _fit(self):
        fit_data = self.data["state"].rename({"idle_time": "x"}).squeeze()
        self.fitter = FitDampingBeat(fit_data)
        self.fit_result = self.fitter.fit()

    def get_fft_data(self):
        idle_times = self.data["state"].coords["idle_time"].values
        y = self.data["state"].values
        n = len(idle_times)
        dt = idle_times[1] - idle_times[0] if n > 1 else 1.0
        amp = np.fft.fft(y)[:n // 2]
        freq = np.fft.fftfreq(n, dt)[:len(amp)]
        amp[0] = 0  # Remove DC part
        return freq, np.abs(amp)
    
    def get_fit_report(self):
        return self.fit_result.fit_report()
    
    def _plot_results(self):
        from qcat.analysis.ramsey.visualization import plot_results, plot_fft
        freq, amp = self.get_fft_data()
        freq = freq*1e6 #GHz to kHz
        # Convert fit_result.params to a simple dictionary
        analysis_result = {k: v.value for k, v in self.fit_result.params.items()} if self.fit_result is not None else None
        if analysis_result is not None:
            analysis_result['f_1'] = analysis_result['f_1']*1e6 #GHz to kHz
            analysis_result['f_2'] = analysis_result['f_2']*1e6 #GHz to kHz
            analysis_result['kappa_1'] = analysis_result['kappa_1']*1e3 #GHz to MHz
            analysis_result['kappa_2'] = analysis_result['kappa_2']*1e3 #GHz to MHz
            analysis_result['best_fit'] = self.fit_result.best_fit
        spec_fig = plot_fft(freq, amp)
        time_fig = plot_results(self.data["state"], analysis_result)
        return {"time_fig":time_fig, "spec_fig":spec_fig}
    
    def get_fit_data(self):
        return self.fitter.x, self.fitter.y, self.fit_result.best_fit





if __name__ == "__main__":
    from qcat.parser.qm_reader import load_xarray_h5, repetition_data
    import matplotlib.pyplot as plt
    ds = load_xarray_h5(r"d:\github\ASQMDriver\data\MIST\2025-09-15\#641_LCH_Ramsey_222721\ds_raw.h5")
    print(ds)
    sep_data = repetition_data(ds, repetition_dim="qubit")
    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        print(qubit_name)
        print(sq_data)
        analysis = RamseyAnalysis(sq_data)
        analysis._plot_results()

    plt.show()