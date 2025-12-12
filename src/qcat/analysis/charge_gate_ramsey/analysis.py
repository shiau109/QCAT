from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import xarray as xr
import numpy as np

from qcat.analysis.ramsey.analysis import RamseyAnalysis




# --- FFT Analysis and 2D Colormap Plotting ---
class ChargeGateRamseyAnalysis:
    
    def __init__(self, data: xr.DataArray):
        self.data = data
        self.spectrum = None
        self.freqs = None
        # self._fft()
        self.all_ave_freq = None
        self.fixed_frequency = None
        self.abscos_fit_result_dict = None
        
    def _start_analysis(self):
        self._get_frequency()
        self._fit_abscos()

    def _fit_abscos(self):
        """
        Fit the merged f1 and f2 frequencies vs charge_gate with |cos| function.
        Model: freq = offset + amplitude * |cos(2*pi * frequency * (charge_gate - phase))|
        """
        from lmfit import Model
        
        # Get valid data points (non-NaN)
        charge_gates = self.fit_results_dataset.coords['charge_gate'].values
        f1_vals = self.fit_results_dataset['f1'].values
        f2_vals = self.fit_results_dataset['f2'].values
        
        # Merge f1 and f2 data points
        merged_charge_gates = []
        merged_freqs = []
        
        # Add valid f1 points
        valid_f1 = ~np.isnan(f1_vals)
        if np.any(valid_f1):
            merged_charge_gates.extend(charge_gates[valid_f1])
            merged_freqs.extend(f1_vals[valid_f1])
        
        # Add valid f2 points
        valid_f2 = ~np.isnan(f2_vals)
        if np.any(valid_f2):
            merged_charge_gates.extend(charge_gates[valid_f2])
            merged_freqs.extend(f2_vals[valid_f2])
        
        if len(merged_freqs) < 4:  # Need minimum points for fitting
            self.abscos_fit_result = None
            return
        
        merged_charge_gates = np.array(merged_charge_gates)
        merged_freqs = np.abs(np.array(merged_freqs)-self.all_ave_freq)
        
        # Get index of minimum value in merged_freqs
        min_freq_index = np.argmin(merged_freqs)

        # Define |cos| model function
        def abscos_func(x, amplitude, frequency, phase):
            return  amplitude * np.abs(np.cos(2 * np.pi * frequency * (x - phase)))
        
        # Create lmfit model
        model = Model(abscos_func)
        
        # Initial parameter guesses
        guess_freq = 0.55 if self.fixed_frequency is None else self.fixed_frequency
        guess_period = 1/guess_freq
        freq_range = np.max(merged_freqs) - np.min(merged_freqs)
        charge_range = np.max(merged_charge_gates) - np.min(merged_charge_gates)
        guess_offset = merged_charge_gates[min_freq_index] -guess_period/4
        if guess_offset>guess_period/4:
            guess_offset -= guess_period/2
        if guess_offset<-guess_period/4:
            guess_offset += guess_period/2
        # Set initial parameters
        params = model.make_params(
            amplitude=freq_range,
            frequency=guess_freq,  # Use fixed frequency if provided, otherwise guess
            phase=guess_offset
        )

        # Set parameter bounds
        params['amplitude'].set(min=0.5*freq_range, max=1.5*freq_range)
        
        # Fix frequency parameter if manual value is provided
        if self.fixed_frequency is not None:
            params['frequency'].set(value=self.fixed_frequency, vary=False)  # Fix the parameter
        else:
            params['frequency'].set(min=0.8*guess_freq, max=1.2*guess_freq)  # Allow variation
            
        params['phase'].set(min=guess_offset-guess_period/10, max=guess_offset+guess_period/10)
        try:
            # Perform the fit
            self.abscos_fit_result = model.fit(merged_freqs, params, x=merged_charge_gates)
            
            # Store fit parameters in the dataset attributes
            if self.abscos_fit_result.success:
                self.abscos_fit_result_dict = {
                    'abscos_amplitude': self.abscos_fit_result.params['amplitude'].value,
                    'abscos_frequency': self.abscos_fit_result.params['frequency'].value,
                    'abscos_phase': self.abscos_fit_result.params['phase'].value,
                    'abscos_fit_success': True,
                    'abscos_chisqr': self.abscos_fit_result.chisqr,
                    'abscos_redchi': self.abscos_fit_result.redchi
                }
                self.fit_results_dataset.attrs.update(self.abscos_fit_result_dict)
            else:
                self.fit_results_dataset.attrs['abscos_fit_success'] = False
                
        except Exception as e:
            print(f"AbsCos fit failed: {e}")
            self.abscos_fit_result = None
            self.fit_results_dataset.attrs['abscos_fit_success'] = False

    def _get_frequency(self):
        spectrum = []
        f1_list = []
        f2_list = []
        a_1_list = []
        a_2_list = []
        kappa_1_list = []
        kappa_2_list = []
        charge_gates = self.data.coords["charge_gate"].values
        idle_time_axis = self.data["idle_time"].values
        
        for idx, charge_gate in enumerate(charge_gates):
            single_ds = self.data.sel(charge_gate=charge_gate)
            analysis = RamseyAnalysis(single_ds)
            freq, amp = analysis.get_fft_data()
            spectrum.append(np.abs(amp))
            fit_result = analysis.fit_result
            if fit_result is not None:
                a_1_list.append(fit_result.params.get('a_1', np.nan))
                a_2_list.append(fit_result.params.get('a_2', np.nan))
                kappa_1 = fit_result.params.get('kappa_1', None)
                kappa_2 = fit_result.params.get('kappa_2', None)
                kappa_1_list.append(kappa_1.value if kappa_1 is not None else np.nan)
                kappa_2_list.append(kappa_2.value if kappa_2 is not None else np.nan)
                f1 = fit_result.params.get('f_1', None)
                f2 = fit_result.params.get('f_2', None)
                f1_list.append(f1.value if f1 is not None else np.nan)
                if fit_result.params.get('a_2', None) == 0:
                    f2_list.append(np.nan)
                else:
                    f2_list.append(f2.value if f2 is not None else np.nan)
            else:
                f1_list.append(np.nan)
                f2_list.append(np.nan)
                kappa_1_list.append(np.nan)
                kappa_2_list.append(np.nan)

        spectrum_array = np.array(spectrum)

        self.f1 = np.array(f1_list)
        self.f2 = np.array(f2_list)
        self.a_1 = np.array(a_1_list)
        self.a_2 = np.array(a_2_list)
        self.kappa_1 = np.array(kappa_1_list)
        self.kappa_2 = np.array(kappa_2_list)

        # Calculate all_ave_freq ignoring NaN values
        if self.all_ave_freq is None:
            self.all_ave_freq = np.nanmean((self.f1 + self.f2) / 2)

        # Package 1D arrays into Dataset with charge_gates coordinate
        self.fit_results_dataset = xr.Dataset(
            {
                'f1': (['charge_gate'], self.f1),
                'f2': (['charge_gate'], self.f2),
                'a_1': (['charge_gate'], self.a_1),
                'a_2': (['charge_gate'], self.a_2),
                'kappa_1': (['charge_gate'], self.kappa_1),
                'kappa_2': (['charge_gate'], self.kappa_2),
                'ave_freq': (['charge_gate'], (self.f1 + self.f2)/2)
            },
            coords={
                'charge_gate': (['charge_gate'], charge_gates),
                'qubit': self.data.coords.get('qubit', 'unknown')
            },
            attrs={
                'all_ave_freq': self.all_ave_freq
            }
        )
        
        # Keep backward compatibility

        self._construnct_spectrum_dataset(spectrum_array, freq, charge_gates)
    def _fft(self):
        # Assume data dims: ('charge_gate', 'idle_time')
        charge_gates = self.data.coords['charge_gate'].values
        idle_times = self.data.coords['idle_time'].values
        n_idle = len(idle_times)
        dt = idle_times[1] - idle_times[0] if n_idle > 1 else 1.0
        spectrum = []
        for cg in charge_gates:
            y = self.data.sel(charge_gate=cg).values
            amp = np.fft.fft(y)[:n_idle // 2]
            freq = np.fft.fftfreq(n_idle, dt)[:len(amp)]
            amp[0] = 0  # Remove DC part
            spectrum.append(np.abs(amp))
        
        # Create xarray Dataset with spectrum, freqs, and charge_gates
        spectrum_array = np.array(spectrum)
        
        # Keep backward compatibility
        self.freqs = freq
        self.spectrum = spectrum_array
        self._construnct_spectrum_dataset(spectrum_array, freq, charge_gates)

    def _construnct_spectrum_dataset(self, spectrum_array, freq, charge_gates):
        self.spectrum_dataset = xr.Dataset(
            {
                'spectrum': (['charge_gate', 'frequency'], spectrum_array)
            },
            coords={
                'charge_gate': (['charge_gate'], charge_gates),
                'frequency': (['frequency'], freq),
            }
        )
        


    def _plot_results(self):
        from qcat.analysis.charge_gate_ramsey.visualization import plot_raw_2d_colormap, plot_2d_spectrum, plot_1d_frequencies
        spec_fig = plot_2d_spectrum(self.spectrum_dataset, self.fit_results_dataset)
        time_fig = plot_raw_2d_colormap(self.data["signal"])
        freq_diff_fig = plot_1d_frequencies(self.fit_results_dataset)
        return {"time_fig":time_fig, "spec_fig":spec_fig, "freq_diff_fig": freq_diff_fig}
    
# --- Test code ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = load_xarray_h5(r"d:\github\ASQMDriver\data\MIST\2025-12-01\#2750_LCH_charge_gate_ramsey_163341\ds_raw.h5")
    # print(ds)
    sep_data = repetition_data(ds, repetition_dim="qubit")
    for sqdata in sep_data:
        qubit_name = sqdata["qubit"].values.item()
        print(qubit_name)
        sqdata = sqdata.rename({"state": "signal"}) 
        # Assume 'I' is the signal variable
        analysis = ChargeGateRamseyAnalysis(sqdata)
        analysis.all_ave_freq = 0.25*1e-3
        analysis._start_analysis()
        analysis._plot_results()
    plt.show()