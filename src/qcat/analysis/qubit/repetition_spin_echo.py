



from qcat.analysis.base import QCATAna
from qcat.analysis.function_fitting.fit_exp_decay import FitExponentialDecay
from xarray import DataArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS


class SpinEchoRepetitionAnalysis(QCATAna):
    """
    Class for analyzing exponential decay data.
    """

    def __init__(self, data: DataArray):
        super().__init__()
        self._import_data(data)

    def _import_data(self, data):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")

        for coords_name in ["time", "repetition"]:
            if coords_name not in data.coords:
                raise ValueError(f"No {coords_name} coordinate in the input DataArray.")

        self.data = data
        self.data.coords["time"] = self.data.coords["time"] /1000

    def _start_analysis(self):
        
        repetition_times = self.data.coords["repetition"].values.shape[0]
        statistic_t1 = []
        for i in range(repetition_times):
            # Initialize the fitting class
            fit_exp_decay = FitExponentialDecay(self.data.isel(repetition=i).rename({"time": "x"}))

            # Generate initial parameter guesses
            guess_params = fit_exp_decay.guess()

            # Perform the fitting process
            fit_result = fit_exp_decay.fit()
            self.fit_result = fit_result
            statistic_t1.append(fit_result.params['tau'].value)


        dims = ["repetition"]
            # Define dimensions and coordinates
        coords = {
            "repetition": self.data.coords["repetition"].values,  # e.g., time indices
        }
        self.statistic_result = xr.Dataset(
            {
                "T2_echo": (dims, np.array(statistic_t1)),  # Variable with dimensions
            },
            coords=coords,  # Shared coordinates
)
        # Plot the results
        self._plot_results()

    def _plot_results(self):
        time = self.data.coords["time"].values
        rep = self.data.coords["repetition"]

        t2e = self.statistic_result["T2_echo"].values

        raw_data = self.data.transpose()*1000
        median_ans, std_ans =np.median(t2e), np.std(t2e)


        fig = plt.figure(dpi=100, figsize=(12,9))
    
        gs = GS(2,2, width_ratios=[2,1],height_ratios=[1,5])
        ax1 = fig.add_subplot(gs[:,1])
        ax1.hist(t2e, bins='auto', density=False)
        ax1.axvline(median_ans,c='k',ls='--',lw='1')
        ax1.set_ylabel("Counts",fontsize=20)
        ax1.set_xlabel(f"T2_echo (us)", fontsize=20)
        ax1.set_title(f"T2_echo = {round(median_ans,1)} $\pm$ {round(std_ans,1)} us",fontsize=20)

        ax2 = fig.add_subplot(gs[:,0])

        c = ax2.pcolormesh(rep, time, raw_data, cmap='RdBu')
        fig.colorbar(c, ax=ax2, label='Contrast (mV)',location='bottom',)
        ax2.plot(rep, raw_data, c='green')
        ax2.axhline(median_ans+std_ans,linestyle='--',c='orange')
        ax2.axhline(median_ans,linestyle='-',c='orange')
        ax2.axhline(median_ans-std_ans,linestyle='--',c='orange')
        ax2.set_xlabel('Time past (min)',fontsize=20)
        ax2.set_ylabel("Free evolution time (us)",fontsize=20)
        

        for Ax in [ax1, ax2]:
            Ax:plt.Axes
            Ax.xaxis.set_tick_params(labelsize=16)
            Ax.yaxis.set_tick_params(labelsize=16)

        # fig.title(f"Time dependent {exp.upper()}",fontsize=20)
        fig.tight_layout()
        # plt.savefig(os.path.join(raw_data_folder,f"{exp.upper()}_{q}_timeDep.png"))

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
        my_ana = SpinEchoRepetitionAnalysis(data.sel(mixer="I"))
        my_ana._start_analysis()
        plt.show()
