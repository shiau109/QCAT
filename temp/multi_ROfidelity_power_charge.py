import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from qcat.analysis.readout_power.analysis import ROFidelityPower
import datetime

def parse_timestamp(ts):
    # Remove timezone info for parsing
    if '+' in ts:
        ts = ts.split('+')[0]
    return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")

base_dir = r'D:\data\MIST\charge_ramsey_power_fidelity\41_500_1'
dataset_list = []

for root, dirs, files in os.walk(base_dir):
    # Only process subfolders with the pattern 'LCH_const_charge_readout_power' in their path
    if 'LCH_const_charge_readout_power' not in root:
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

# Build merged_ds and save it
ds_list = []
charge_volt_list = []
for ds, json_dict in dataset_list:
    charge_volt_list.append(json_dict["data"]["parameters"]["model"]["charge_gate_in_v"])
    ds_list.append(ds)

merged_ds = xr.concat(ds_list, dim=xr.DataArray(charge_volt_list, dims="charge_volt", name="charge_volt"))
merged_ds_save_path = os.path.join(base_dir, "charge_fidelity_merged.h5")
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
    charge_volt_list = sq_data.coords["charge_volt"].values
    summary_list = []
    for charge_volt in charge_volt_list:
        single_ds = sq_data.sel(charge_volt=charge_volt)
        analysis = ROFidelityPower(single_ds)
        analysis._start_analysis()
        # Add charge_volt as a coordinate to the summary_dataset
        summary_ds = analysis.summary_dataset.expand_dims({'charge_volt': [charge_volt]})
        summary_list.append(summary_ds)

    # Concatenate all summary_datasets along charge_volt
    merged_summary = xr.concat(summary_list, dim='charge_volt')

    # Plot 2D colormaps for p_outlier and norm_res
    amp_prefactor = merged_summary['amp_prefactor'].values
    charge_volt = merged_summary['charge_volt'].values
    # p_outlier: shape (charge_volt, amp_prefactor, state)
    # norm_res: shape (charge_volt, amp_prefactor, state)
    # We'll plot for state=0 and state=1 separately
    for state in [0, 1]:
        # p_outlier
        z = merged_summary['p_outlier'].sel(state=state).transpose('charge_volt', 'amp_prefactor').values
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(z, aspect='auto', origin='lower',
                      extent=[amp_prefactor[0], amp_prefactor[-1], charge_volt[0], charge_volt[-1]],
                      cmap='viridis')
        ax.set_xlabel('amp_prefactor')
        ax.set_ylabel('charge_volt')
        ax.set_title(f'Qubit {qubit_name} State {state}: p_outlier')
        fig.colorbar(im, ax=ax, label='p_outlier')
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, f'p_outlier_2d_{qubit_name}_state{state}.png'))
        plt.close(fig)

        # norm_res
        z2 = merged_summary['norm_res'].sel(state=state).transpose('charge_volt', 'amp_prefactor').values
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im2 = ax2.imshow(z2, aspect='auto', origin='lower',
                        extent=[amp_prefactor[0], amp_prefactor[-1], charge_volt[0], charge_volt[-1]],
                        cmap='RdBu_r')
        ax2.set_xlabel('amp_prefactor')
        ax2.set_ylabel('charge_volt')
        ax2.set_title(f'Qubit {qubit_name} State {state}: norm_res')
        fig2.colorbar(im2, ax=ax2, label='norm_res')
        fig2.tight_layout()
        fig2.savefig(os.path.join(base_dir, f'norm_res_2d_{qubit_name}_state{state}.png'))
        plt.close(fig2)
