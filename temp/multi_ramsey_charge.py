

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


base_dir = r'D:\data\MIST\charge_gate_ramsey\SC_RF_20251026\RT_DCblock'
dataset_list = []

for root, dirs, files in os.walk(base_dir):
    # Only process subfolders with the pattern 'LCH_const_charge_gate_ramsey' in their path
    if 'LCH_const_charge_gate_ramsey' not in root:
        continue
    if 'ds_raw.h5' in files and 'node.json' in files:
        file_path = os.path.join(root, 'ds_raw.h5')
        json_path = os.path.join(root, 'node.json')
        try:
            ds = load_xarray_h5(file_path)
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
            dataset_list.append((ds, json_dict))
            print(f"Loaded: {file_path}, loaded node.json")
        except Exception as e:
            print(f"Failed to load {file_path} or {json_path}: {e}")

print(f"Total datasets loaded: {len(dataset_list)}")

# Get all start times
start_times = [parse_timestamp(json_dict["metadata"]["run_start"]) for _, json_dict in dataset_list]
t0 = start_times[0]
relative_times = [(t - t0).total_seconds() for t in start_times]


ds_list = []
charge_volt_list = []
for ds, json_dict in dataset_list:
    if "state" in list(ds.data_vars):
        ds = ds.rename({"state": "signal"})
    else:
        ds = ds.rename({"I": "signal"})

    # Build a list of charge_volt values
    charge_volt_list.append(json_dict["data"]["parameters"]["model"]["charge_gate_in_v"])
    ds_list.append(ds)

# Concatenate along a new 'charge_volt' coordinate
merged_ds = xr.concat(ds_list, dim=xr.DataArray(charge_volt_list, dims="charge_volt", name="charge_volt"))

# Save merged_ds to an HDF5 file
merged_ds_save_path = os.path.join(base_dir, "charge_gate_ramsey.h5")
try:
    merged_ds.to_netcdf(merged_ds_save_path)
    print(f"Merged dataset saved to {merged_ds_save_path}")
except Exception as e:
    print(f"Failed to save merged dataset: {e}")

    

# Use repetition_data to get per-qubit data from merged_ds
from qcat.parser.qm_reader import repetition_data
qubit_datasets = repetition_data(merged_ds, repetition_dim="qubit")

for sq_data in qubit_datasets:
    qubit_name = sq_data["qubit"].values.item()
    spectra = []
    f1_list = []
    f2_list = []
    a_1_list = []
    a_2_list = []
    kappa_1_list = []
    kappa_2_list = []
    times = []
    charge_volt_list = sq_data.coords["charge_volt"].values
    rawdata_matrix = []
    idle_time_axis = sq_data["idle_time"].values
    for idx, charge_volt in enumerate(charge_volt_list):
        single_ds = sq_data.sel(charge_volt=charge_volt)
        analysis = RamseyAnalysis(single_ds)
        freq, amp = analysis.get_fft_data()
        spectra.append(np.abs(amp))
        fit_result = analysis.fit_result
        yvals = single_ds["signal"].values
        rawdata_matrix.append(yvals)
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
        # times can be filled if you have a time coord per charge_volt
        # times.append(single_ds.coords["time"].item())
    # Plot raw data as 2D colormap: x=idle_time, y=charge_volt_list
    rawdata_matrix = np.array(rawdata_matrix)
    fig_raw, ax_raw = plt.subplots(figsize=(10, 6))
    im_raw = ax_raw.imshow(
        rawdata_matrix,
        aspect='auto',
        origin='lower',
        extent=[idle_time_axis[0], idle_time_axis[-1], charge_volt_list[0], charge_volt_list[-1]],
        cmap='RdBu_r'
    )
    ax_raw.set_xlabel('Idle Time')
    ax_raw.set_ylabel('Charge Gate Voltage (V)')
    ax_raw.set_title(f'Raw Ramsey Data - Qubit {qubit_name}')
    fig_raw.colorbar(im_raw, ax=ax_raw, label='Signal')
    fig_raw.tight_layout()
    fig_raw.savefig(os.path.join(base_dir, f'rawdata_2d_{qubit_name}.png'))

    spectra_arr = np.array(spectra)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        spectra_arr,
        aspect='auto',
        origin='lower',
        extent=[freq[0], freq[-1], charge_volt_list[0], charge_volt_list[-1]],
        cmap='viridis'
    )
    print(charge_volt_list)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Charge Gate Voltage (V)')
    ax.set_title(f'Ramsey Power Spectrum - Qubit {qubit_name}')
    fig.colorbar(im, ax=ax, label='Power')
    ax.plot(f1_list, charge_volt_list, 'ro', label='f_1 (fit)')
    ax.plot(f2_list, charge_volt_list, 'bo', label='f_2 (fit)')

    avg_freq = (np.array(f1_list) + np.array(f2_list)) / 2
    ax.plot(avg_freq, charge_volt_list, 'ko-', label='(f1+f2)/2')

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, f'ramsey_spectrum_{qubit_name}.png'))

    # Plot (f1_list + f2_list)/2 vs charge_volt_list and save
    avg_freq = (np.array(f1_list) + np.array(f2_list)) / 2
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(charge_volt_list, a_1_list, 'ro', label='a_1')
    ax2.plot(charge_volt_list, a_2_list, 'bo', label='a_2')

    ax2.set_xlabel('Charge Gate Voltage (V)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Amplitude vs Charge Gate Voltage - Qubit {qubit_name}')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(base_dir, f'amp_vs_charge_{qubit_name}.png'))
    # Plot kappa_1 and kappa_2 vs charge_volt_list and save
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(charge_volt_list, kappa_1_list, 'go-', label='kappa_1')
    ax3.plot(charge_volt_list, kappa_2_list, 'mo-', label='kappa_2')
    ax3.set_xlabel('Charge Gate Voltage (V)')
    ax3.set_ylabel('Decay Rate (kappa)')
    ax3.set_title(f'Decay Rate vs Charge Gate Voltage - Qubit {qubit_name}')
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(base_dir, f'kappa_vs_charge_{qubit_name}.png'))
# plt.show()