
import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import json
from lmfit import Model
import numpy as np



import xarray as xr
import matplotlib.pyplot as plt
from qcat.analysis.ramsey.analysis import RamseyAnalysis

import datetime

def parse_timestamp(ts):
    # Remove timezone info for parsing
    if '+' in ts:
        ts = ts.split('+')[0]
    return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")


base_dir = r'D:\data\MIST\repeat_ramsey\with_connect'
dataset_list = []

for root, dirs, files in os.walk(base_dir):
    if 'ds_raw.h5' in files and 'node.json' in files:
        file_path = os.path.join(root, 'ds_raw.h5')
        json_path = os.path.join(root, 'node.json')
        try:
            ds = load_xarray_h5(file_path)
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
            ds = ds.rename({'state': 'signal'})
            dataset_list.append((ds, json_dict))
            print(f"Loaded: {file_path}, loaded node.json")
        except Exception as e:
            print(f"Failed to load {file_path} or {json_path}: {e}")

print(f"Total datasets loaded: {len(dataset_list)}")

# Get all start times
start_times = [parse_timestamp(json_dict["metadata"]["run_start"]) for _, json_dict in dataset_list]
print(start_times)
t0 = start_times[0]
relative_times = [(t - t0).total_seconds() for t in start_times]


# Assume all datasets have the same qubit coordinates
qubit_names = dataset_list[0][0].coords['qubit'].values
num_qubits = len(qubit_names)

# For each qubit, collect the corresponding data from all datasets
qubit_data_list = [[] for _ in range(num_qubits)]
json_list = [[] for _ in range(num_qubits)]
for ds, json_dict in dataset_list:
    for i, qubit_name in enumerate(qubit_names):
        # Extract the data for this qubit
        sq_data = ds.sel(qubit=qubit_name)
        qubit_data_list[i].append(sq_data)
        json_list[i].append(json_dict)

# Now, for each qubit, analyze and plot
for i, qubit_name in enumerate(qubit_names):
    spectra = []
    f1_list = []
    f2_list = []
    a_1_list = []
    a_2_list = []
    kappa_1_list = []
    kappa_2_list = []
    times = []
    for sq_data, json_dict in zip(qubit_data_list[i], json_list[i]):
        analysis = RamseyAnalysis(sq_data)
        freq, amp = analysis.get_fft_data()
        spectra.append(np.abs(amp))
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

        # Get relative time
        run_start = parse_timestamp(json_dict["metadata"]["run_start"])
        times.append((run_start - t0).total_seconds())

    spectra_arr = np.array(spectra)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        spectra_arr,
        aspect='auto',
        origin='lower',
        extent=[freq[0], freq[-1], times[0], times[-1]],
        cmap='viridis'
    )
    print(times)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Time (s since first run)')
    ax.set_title(f'Ramsey Power Spectrum - Qubit {qubit_name}')
    fig.colorbar(im, ax=ax, label='Power')
    ax.plot(f1_list, times, 'ro', label='f_1 (fit)')
    ax.plot(f2_list, times, 'bo', label='f_2 (fit)')

    avg_freq = (np.array(f1_list) + np.array(f2_list)) / 2
    ax.plot(avg_freq, times, 'ko-', label='(f1+f2)/2')

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, f'ramsey_spectrum_{qubit_name}.png'))

    # Plot (f1_list + f2_list)/2 vs times and save
    avg_freq = (np.array(f1_list) + np.array(f2_list)) / 2
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(times, a_1_list, 'ro', label='a_1')
    ax2.plot(times, a_2_list, 'bo', label='a_2')

    ax2.set_xlabel('Time (s since first run)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Amplitude vs Time - Qubit {qubit_name}')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(base_dir, f'amp_vs_time_{qubit_name}.png'))
    # Plot kappa_1 and kappa_2 vs time and save
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(times, kappa_1_list, 'go-', label='kappa_1')
    ax3.plot(times, kappa_2_list, 'mo-', label='kappa_2')
    ax3.set_xlabel('Time (s since first run)')
    ax3.set_ylabel('Decay Rate (kappa)')
    ax3.set_title(f'Decay Rate vs Time - Qubit {qubit_name}')
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(base_dir, f'kappa_vs_time_{qubit_name}.png'))
plt.show()