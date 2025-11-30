from qcat.utilities.file.folder_finder import get_child_folders, get_child_folder_paths

from qcat.parser.qm_reader import load_xarray_h5, repetition_data, parse_timestamp
import os 
import json
import numpy as np
import matplotlib.pyplot as plt
qubit_name = "q6"
folder_path = f"D:\\data\\6SQ_XYtest\\{qubit_name}"

# # Example usage:
# t1_folder_names = get_child_folders(folder_path, None)
# print(f"Found T1 folders: {t1_folder_names}")

t1_folder_paths = get_child_folder_paths(folder_path, "05_T1")
print(f"Full paths: {t1_folder_paths}")
dataset_list = []
for folder in t1_folder_paths:
    raw_data_path = os.path.join(folder, 'ds_raw.h5')

    node_config_path = os.path.join(folder, 'node.json')
    ana_data_path = os.path.join(folder, 'data.json')

    ds_raw = load_xarray_h5(raw_data_path)
    with open(node_config_path, 'r') as f:
        node_config_dict = json.load(f)
    with open(ana_data_path, 'r') as f:
        ana_data_dict = json.load(f)
    dataset_list.append((ds_raw, node_config_dict, ana_data_dict))

start_times = []
t1_times = []
for _, node_config_dict, ana_data_dict in dataset_list:

    # Get all start times
    start_times.append(parse_timestamp(node_config_dict["metadata"]["run_start"]))
    t1_times.append(ana_data_dict["fit_results"][qubit_name]["t1"])

t0 = start_times[0]
print(t0, t1_times)

relative_times = [(t - t0).total_seconds() for t in start_times]

# Convert to numpy arrays for easier calculations
relative_times_array = np.array(relative_times)
t1_times_array = np.array(t1_times)

# Calculate statistics
t1_mean = np.mean(t1_times_array)
t1_std = np.std(t1_times_array)

print(f"T1 Statistics:")
print(f"Average T1: {t1_mean:.6f}")
print(f"Standard Deviation: {t1_std:.6f}")
print(f"Number of measurements: {len(t1_times_array)}")

# Plot T1 times vs relative times
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(relative_times_array, t1_times_array, 'o-', markersize=6, linewidth=1, color='C0', alpha=0.8)
ax.axhline(y=t1_mean, color='red', linestyle='--', alpha=0.7, label=f'Mean = {t1_mean:.6f}')
ax.fill_between(relative_times_array, t1_mean - t1_std, t1_mean + t1_std, 
                alpha=0.2, color='red', label=f'±1σ = {t1_std:.6f}')

ax.set_xlabel('Relative Time (seconds)')
ax.set_ylabel('T1 Time')
ax.set_title(f'{qubit_name} T1 Time Evolution')
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

# Save figure
out_path = os.path.join(folder_path, f'{qubit_name}_T1_statistics.png')
try:
    fig.savefig(out_path, dpi=200)
    print(f"Saved T1 statistics plot to: {out_path}")
except Exception as e:
    print(f"Failed to save plot: {e}")
plt.close(fig)

