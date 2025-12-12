import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from qcat.analysis.readout_power.analysis import ROFidelityPower



# Folder to check
folder_path = r"D:\data\MIST\20251201\r_rp\#3696_LCH_charge_gate_readout_power_5_221843"

# Build paths
file_path = os.path.join(folder_path, 'ds_raw.h5')
json_state_path = os.path.join(folder_path, 'quam_state\\state.json')


ds = load_xarray_h5(file_path)
with open(json_state_path, 'r') as f:
    json_dict = json.load(f)

print(ds)

# Get coordinates
charge_gates = ds.coords['charge_gate'].values
amp_prefactors = ds.coords['amp_prefactor'].values
prepared_states = ds.coords['prepared_state'].values

# Create meshgrid for plotting
X, Y = np.meshgrid(charge_gates, amp_prefactors)

# Create separate figure for each prepared_state
for state_idx, prepared_state in enumerate(prepared_states):
    # Create 1x2 subplot figure for this prepared state
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Readout Power Analysis - Prepared State {prepared_state}', fontsize=14)
    
    # Average over shot_idx axis for I and Q data
    i_data = ds['I'].sel(prepared_state=prepared_state).mean(dim='shot_idx')
    q_data = ds['Q'].sel(prepared_state=prepared_state).mean(dim='shot_idx')
    
    # Remove qubit dimension if it exists (assuming single qubit)
    if 'qubit' in i_data.dims:
        i_data = i_data.isel(qubit=0)
        q_data = q_data.isel(qubit=0)
    
    # Transpose to match meshgrid orientation (amp_prefactor, charge_gate)
    i_plot_data = i_data.T.values  # Shape: (amp_prefactor, charge_gate)
    q_plot_data = q_data.T.values
    
    # Plot I data (left subplot)
    ax_i = axes[0]
    im_i = ax_i.pcolormesh(X, Y, i_plot_data, shading='auto', cmap='viridis')
    ax_i.set_title('I Data')
    ax_i.set_xlabel('Charge Gate (V)')
    ax_i.set_ylabel('Amplifier Prefactor')
    cbar_i = plt.colorbar(im_i, ax=ax_i)
    cbar_i.set_label('I Signal')
    
    # Plot Q data (right subplot)
    ax_q = axes[1]
    im_q = ax_q.pcolormesh(X, Y, q_plot_data, shading='auto', cmap='plasma')
    ax_q.set_title('Q Data')
    ax_q.set_xlabel('Charge Gate (V)')
    ax_q.set_ylabel('Amplifier Prefactor')
    cbar_q = plt.colorbar(im_q, ax=ax_q)
    cbar_q.set_label('Q Signal')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure to the folder_path
    save_filename = f'readout_power_prepared_state_{prepared_state}.png'
    save_path = os.path.join(folder_path, save_filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {save_path}")

plt.show()