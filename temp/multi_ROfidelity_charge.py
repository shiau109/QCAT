
import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data, parse_timestamp
import json
from lmfit import Model
import numpy as np



import xarray as xr
import matplotlib.pyplot as plt
from qcat.analysis.state_discrimination.analysis import StateDiscrimination

import datetime


base_dir = r'D:\data\MIST\charge_ramset_fidelity'
dataset_list = []

for root, dirs, files in os.walk(base_dir):
    # Only process subfolders with the pattern 'LCH_const_charge_readout_fidelity' in their path
    if 'LCH_const_charge_readout_fidelity' not in root:
        continue
    if 'ds_raw.h5' in files and 'node.json' in files:
        file_path = os.path.join(root, 'ds_raw.h5')
        json_path = os.path.join(root, 'node.json')
        try:
            ds = load_xarray_h5(file_path)
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
            ds = ds.rename({'n_runs': 'shot_idx'})
            dataset_list.append((ds, json_dict))
            print(f"Loaded: {file_path}, loaded node.json")
        except Exception as e:
            print(f"Failed to load {file_path} or {json_path}: {e}")

print(f"Total datasets loaded: {len(dataset_list)}")

# Build merged_ds and save it
ds_list = []
charge_volt_list = []
for ds, json_dict in dataset_list:
    charge_volt_list.append(json_dict["data"]["parameters"]["model"]["charge_gate_in_v"])
    ds_list.append(ds)

import xarray as xr
merged_ds = xr.concat(ds_list, dim=xr.DataArray(charge_volt_list, dims="charge_volt", name="charge_volt"))
merged_ds_save_path = os.path.join(base_dir, "charge_fidelity_merged.h5")
try:
    merged_ds.to_netcdf(merged_ds_save_path)
    print(f"Merged dataset saved to {merged_ds_save_path}")
except Exception as e:
    print(f"Failed to save merged dataset: {e}")


# Get all start times
start_times = [parse_timestamp(json_dict["metadata"]["run_start"]) for _, json_dict in dataset_list]
t0 = start_times[0]
relative_times = [(t - t0).total_seconds() for t in start_times]

# Assume all datasets have the same qubit coordinates
qubit_names = dataset_list[0][0].coords['qubit'].values
num_qubits = len(qubit_names)


# Use repetition_data to get per-qubit data from merged_ds
from qcat.parser.qm_reader import repetition_data
qubit_datasets = repetition_data(merged_ds, repetition_dim="qubit")

for sq_data in qubit_datasets:
    qubit_name = sq_data["qubit"].values.item()
    charge_volt_list = sq_data.coords["charge_volt"].values
    outlier_prob_list = []
    for charge_volt in charge_volt_list:
        single_ds = sq_data.sel(charge_volt=charge_volt)
        analysis = StateDiscrimination(single_ds)
        analysis._start_analysis()
        outlier_prob = analysis.analysis_result.get('outlier_probability', np.nan)
        if hasattr(outlier_prob, 'shape') and outlier_prob.shape != ():
            outlier_prob = np.mean(outlier_prob)
        outlier_prob_list.append(outlier_prob)

    # Plot outlier_probability vs charge_volt_list
    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.plot(charge_volt_list, outlier_prob_list, 'o-', label='Outlier Probability')
    ax.set_xlabel('Charge Gate Voltage (V)')
    ax.set_ylabel('Outlier Probability')
    ax.set_title(f'Qubit {qubit_name}: Outlier Probability vs Charge Gate Voltage')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    # Save figure to base_dir
    fig_path = os.path.join(base_dir, f"outlier_probability_vs_charge_{qubit_name}.png")
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    