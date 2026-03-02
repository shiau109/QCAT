
import os
import json
from qcat.parser.qm_reader import load_xarray_h5
import numpy as np
import matplotlib.pyplot as plt
# Folder to check
folder_path = [r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\Qrakal\#276_LCH_qubit_spectroscopy_164353",
               ]

# Build paths
fig, ax = plt.subplots(figsize=(10, 6))

for i, folder in enumerate(folder_path):
    file_path = os.path.join(folder, 'ds_raw.h5')
    json_state_path = os.path.join(folder, 'quam_state\\state.json')

    ds = load_xarray_h5(file_path)
    with open(json_state_path, 'r') as f:
        json_dict = json.load(f)

    print(f"Dataset {i+1}: {ds}")
    
    # Get the qubit name (assuming single qubit per dataset)
    qubit_name = ds.coords['qubit'].values[0]
    
    # Extract full_freq and IQ_abs for this qubit
    full_freq = ds['full_freq'].sel(qubit=qubit_name).values
    iq_abs = ds['I'].sel(qubit=qubit_name).values
    
    # Plot IQ_abs vs full_freq
    label = f"Dataset {i+1} ({os.path.basename(folder)})"
    ax.plot(full_freq / 1e9, iq_abs, marker='.', markersize=2, label=label, alpha=0.8)

ax.set_xlabel('Full Frequency (GHz)')
ax.set_ylabel('IQ_abs')
ax.set_title('IQ Amplitude vs Frequency - Qubit Spectroscopy')
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

# Save figure to the parent directory of the first folder
parent_dir = os.path.dirname(folder_path[0])
out_path = os.path.join(parent_dir, 'qubit_spectroscopy_comparison.png')
try:
    fig.savefig(out_path, dpi=200)
    print(f"Saved spectroscopy comparison plot to: {out_path}")
except Exception as e:
    print(f"Failed to save plot: {e}")
plt.show()
plt.close(fig)

