import matplotlib.pyplot as plt
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import xarray as xr
from qcat.NCU.Fit_library import QS_fit_analysis
import numpy as np





# --- FFT Analysis and 2D Colormap Plotting ---
class ChargeGateRamseyAnalysis:
    
    def __init__(self, data: xr.DataArray):
        self.data = data
        self.spectrum = None
        self.freqs = None
        self._fft()
    


    def _fft(self):
        # Assume data dims: ('charge_gate', 'idle_time')
        charge_gates = self.data.coords['charge_gate'].values
        idle_times = self.data.coords['idle_time'].values
        n_idle = len(idle_times)
        dt = idle_times[1] - idle_times[0] if n_idle > 1 else 1.0
        spectra = []
        for cg in charge_gates:
            y = self.data.sel(charge_gate=cg).values
            amp = np.fft.fft(y)[:n_idle // 2]
            freq = np.fft.fftfreq(n_idle, dt)[:len(amp)]
            amp[0] = 0  # Remove DC part
            spectra.append(np.abs(amp))
        self.freqs = freq
        self.spectrum = np.array(spectra)

    def plot_2d_spectrum(self):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        im = ax.imshow(
            self.spectrum,
            aspect='auto',
            origin='lower',
            extent=[self.freqs[0], self.freqs[-1], self.data.coords['charge_gate'].values[0], self.data.coords['charge_gate'].values[-1]],
            cmap='viridis'
        )
        ax.set_xlabel('Frequency', fontsize=20)
        ax.set_ylabel('Charge Gate', fontsize=20)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        fig.colorbar(im, ax=ax, label='FFT Amplitude')
        fig.tight_layout()
        return fig
    
    def plot_raw_2d_colormap(self):
        # Plot raw data as 2D color map (charge_gate vs idle_time)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        im = ax.imshow(
            self.data.values,
            aspect='auto',
            origin='lower',
            extent=[
                self.data.coords['idle_time'].values[0],
                self.data.coords['idle_time'].values[-1],
                self.data.coords['charge_gate'].values[0],
                self.data.coords['charge_gate'].values[-1]
            ],
            cmap='viridis'
        )
        ax.set_xlabel('Idle Time', fontsize=20)
        ax.set_ylabel('Charge Gate', fontsize=20)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        fig.colorbar(im, ax=ax, label='Raw Amplitude')
        fig.tight_layout()
        return fig
# --- Test code ---
if __name__ == "__main__":
    ds = load_xarray_h5(r"d:\github\ASQMDriver\data\MIST\2025-09-13\#260_LCH_charge_gate_Ramsey_235929\ds_raw.h5")
    print(ds)
    sep_data = repetition_data(ds, repetition_dim="qubit")
    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        print(qubit_name)
        print(sq_data)
        # Assume 'I' is the signal variable
        analysis = ChargeGateRamseyAnalysis(sq_data['I'])
        analysis.plot_raw_2d_colormap()
        analysis.plot_2d_spectrum()
    plt.show()