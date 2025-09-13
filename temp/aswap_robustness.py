import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import json
from lmfit import Model
import numpy as np







base_dir = r'D:\data\RSWAP\start_value'
dataset_list = []

for root, dirs, files in os.walk(base_dir):
    if 'ds_raw.h5' in files and 'node.json' in files:
        file_path = os.path.join(root, 'ds_raw.h5')
        json_path = os.path.join(root, 'quam_state\\state.json')
        try:
            ds = load_xarray_h5(file_path)
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
            dataset_list.append((ds, json_dict))
            print(f"Loaded: {file_path}, loaded state.json")
        except Exception as e:
            print(f"Failed to load {file_path} or {json_path}: {e}")

print(f"Total datasets loaded: {len(dataset_list)}")


import xarray as xr
import matplotlib.pyplot as plt

# Collect datasets and operation_times
mean_signal_list = []
mean_ref_list = []
r_swap_length_list = []
r_swap_amp_list = []
for i, (ds, json_dict) in enumerate(dataset_list):
    
    I = ds["I"].squeeze("qubit")  # Removes 'qubit' dimension if length 1

    full_freq = ds["full_freq"].squeeze("qubit")
    mask = (full_freq >= 3.540e9) & (full_freq <= 3.555e9)
    I_signal = I.where(mask, drop=True)
    mean_signal = I_signal.mean().item()

    mask = (full_freq >= 3.68e9) | (full_freq <= 3.42e9)
    I_ref = I.where(mask, drop=True)
    mean_ref= I_ref.mean().item()    
    std_ref= I_ref.std().item()  
    r_swap_length = json_dict["qubits"]["q1"]["z"]["operations"]['r_swap']["length"]
    r_swap_length_list.append(r_swap_length)
    r_swap_amp = json_dict["qubits"]["q1"]["z"]["operations"]['r_swap']["start_value"]
    r_swap_amp_list.append(r_swap_amp)
    if i == 5 or i == 14:
        # print(I)
        result = ds["I"].plot(x="full_freq")
        ax = plt.gca()  # Get current axis
        ax.axhline(mean_signal, color='red', linestyle='--', label=f'signal :{r_swap_length}ns, {r_swap_amp}V')
        ax.axhline(mean_ref, color='black', linestyle='--', label=f'ref :{r_swap_length}ns, {r_swap_amp}V')

        ax.legend()

    mean_signal_list.append(mean_signal)
    mean_ref_list.append(mean_ref)

    print(i, "length, amp, peak",r_swap_length, r_swap_amp, f"{mean_signal*1000:.1f}", f"{mean_ref*1000:.1f}")
# Convert lists to numpy arrays
mean_signal_arr = np.array(mean_signal_list)
mean_ref_arr = np.array(mean_ref_list)
r_swap_length_arr = np.array(r_swap_length_list)
r_swap_amp_arr = np.array(r_swap_amp_list)

# First figure: mean_ref_arr
plt.figure(figsize=(8, 6))
contour1 = plt.tricontourf(r_swap_length_arr, r_swap_amp_arr, mean_ref_arr, levels=14, cmap='viridis')
plt.xlabel('r_swap_length')
plt.ylabel('amp')
plt.title('Mean Ref (tricontourf)')
plt.colorbar(contour1, label='Mean Ref')

# Second figure: mean_signal_arr - mean_ref_arr
plt.figure(figsize=(8, 6))
contour2 = plt.tricontourf(r_swap_length_arr, r_swap_amp_arr, mean_signal_arr - mean_ref_arr, levels=14, cmap='viridis')
plt.xlabel('r_swap_length')
plt.ylabel('amp')
plt.title('Mean Signal - Mean Ref (tricontourf)')
plt.colorbar(contour2, label='Mean Signal - Mean Ref')
plt.show()