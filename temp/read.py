
import os
import json
from qcat.parser.qm_reader import load_xarray_h5
import numpy as np
import matplotlib.pyplot as plt
# Folder to check
folder_path = r"d:\github\ASQMDriver\data\2Q1C_17\2025-10-29\#907_LCH_parity_switch_ramsey_2_001732"

# Build paths
file_path = os.path.join(folder_path, 'ds_raw.h5')
json_state_path = os.path.join(folder_path, 'quam_state\\state.json')


ds = load_xarray_h5(file_path)
with open(json_state_path, 'r') as f:
    json_dict = json.load(f)

qubit_name = ds.coords['qubit'].values[0]

charge_dispersion = json_dict["qubits"][qubit_name]["charge_dispersion"]/1e6  # Convert Hz to MHz
readout_time = json_dict["qubits"][qubit_name]["resonator"]["operations"]["readout"]["length"]/1e3  # Convert ns to us
readout_depletion_time = json_dict["qubits"][qubit_name]["resonator"]["depletion_time"]/1e3  # Convert ns to us
pi_pulse_time = json_dict["qubits"][qubit_name]["xy"]["operations"]["x180_DragCosine"]["length"]/1e3  # Convert ns to us

ramsey_time = 1/charge_dispersion/4
total_time = ramsey_time + readout_time + readout_depletion_time + pi_pulse_time
state_array = ds["state"].sel(qubit=qubit_name).values

print(f"Qubit name: {qubit_name}")
print(f"Charge dispersion: {charge_dispersion} MHz")
print(f"Ramsey time: {ramsey_time} us")
print(f"Estimated total time per shot: {total_time} us")

# Ensure state_array is 1D
state_array = np.ravel(state_array)

print(f"shot number: {state_array.shape}")
print(f"average: {np.mean(state_array)}")

# Build time axis (units: microseconds) and plot state vs time
time_axis = np.arange(state_array.size) * total_time
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(time_axis, state_array, marker='.', linestyle='None', markersize=0.5, color='C0', alpha=0.2)
ax.set_xlabel('Time (µs)')
ax.set_ylabel('Prepared state')
ax.set_title(f"Prepared state vs time — {qubit_name}")
ax.grid(True, alpha=0.3)
fig.tight_layout()

# Save figure next to the data for convenience
out_path = os.path.join(folder_path, 'state_vs_time.png')
try:
    fig.savefig(out_path, dpi=200)
    print(f"Saved state vs time plot to: {out_path}")
except Exception as e:
    print(f"Failed to save plot: {e}")
plt.close(fig)

# Compute FFT of the prepared-state time series
# total_time is in microseconds; convert to seconds for frequency axis
dt_sec = total_time * 1e-6
if state_array.size > 1 and dt_sec > 0:
    N = state_array.size
    fs = 1.0 / dt_sec
    # Use rFFT for real-valued input
    state_demeaned = state_array - np.mean(state_array)
    # One-sided FFT (rfft) already drops the mirrored negative freqs.
    fft_vals = np.fft.rfft(state_demeaned)
    fft_freqs = np.fft.rfftfreq(N, d=dt_sec)
    power = np.abs(fft_vals) ** 2

    # Remove zero frequency to allow log-scale plotting on x axis
    mask = fft_freqs > 0
    fft_freqs_plot = fft_freqs[mask]
    power_plot = power[mask]

    # Avoid zeros in power (log scale) by flooring to a tiny positive value
    if power_plot.size > 0:
        eps = np.maximum(power_plot.max() * 1e-12, 1e-20)
        power_plot = np.clip(power_plot, a_min=eps, a_max=None)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    # Plot FFT as dots (marker-only) to show discrete spectral samples
    ax2.plot(fft_freqs_plot, power_plot, linestyle='None', marker='.', markersize=4, color='C1', alpha=0.9)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Frequency (Hz, log)')
    ax2.set_ylabel('Power (log)')
    ax2.set_title(f"Power spectrum of prepared state — {qubit_name}")
    ax2.set_xlim(fft_freqs_plot.min(), fft_freqs_plot.max())
    ax2.grid(True, alpha=0.3, which='both')
    fig2.tight_layout()

    fft_out = os.path.join(folder_path, 'state_fft.png')
    try:
        fig2.savefig(fft_out, dpi=200)
        print(f"Saved FFT plot to: {fft_out}")
    except Exception as e:
        print(f"Failed to save FFT plot: {e}")
    plt.close(fig2)
else:
    print("Skipping FFT: insufficient data or invalid total_time")