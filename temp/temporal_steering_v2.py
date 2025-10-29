

import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import json
from lmfit import Model
import numpy as np



import xarray as xr
import matplotlib.pyplot as plt

def analyze_temporal_steering_folder(base_dir):

    file_path = os.path.join(base_dir, 'ds_raw.h5')
    json_path = os.path.join(base_dir, 'node.json')
    try:
        ds = load_xarray_h5(file_path)
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        ds = ds.rename({'state': 'signal', 'basis': 'readout_basis'})

        print(f"Loaded: {file_path}, loaded node.json")
    except Exception as e:
        print(f"Failed to load {file_path} or {json_path}: {e}")


    print(ds)
    # merged_ds = xr.concat(ds_list, dim=xr.DataArray(basis_list, dims="readout_basis", name="readout_basis"))
    merged_ds = ds
    merged_ds_save_path = os.path.join(base_dir, "basis_merged.h5")
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
        bases = sq_data.coords['readout_basis'].values
        idle_times = sq_data.coords['idle_time'].values
        signal = sq_data['signal'].mean(dim='shot_idx')  # shape: (readout_basis, idle_time)
        print(signal)
        Sx = 1- 2*signal.isel(readout_basis=0).values
        Sy = 1- 2*signal.isel(readout_basis=1).values
        Sz = 1- 2*signal.isel(readout_basis=2).values
        print(Sx.shape)
        from qutip import Qobj, sigmax, sigmay, sigmaz, qeye
        density_matrices = []
        for x, y, z in zip(Sx, Sy, Sz):
            rho = 0.5*(qeye(2) + x*sigmax() + y*sigmay() + z*sigmaz())
            density_matrices.append(rho.full())
        density_matrices = np.array(density_matrices)  # shape: (len(idle_times), 2, 2)
        # Add a new 'part' coordinate: 0 for real, 1 for imag
        dm_full = np.stack([density_matrices.real, density_matrices.imag], axis=-1)  # shape: (idle_time, 2, 2, 2)
        dm_ds = xr.Dataset(
            {
                'density_matrix': (['idle_time', 'row', 'col', 'part'], dm_full)
            },
            coords={
                'idle_time': idle_times,
                'row': [0, 1],
                'col': [0, 1],
                'part': ['real', 'imag'],
            }
        )
        subdir_name = os.path.basename(os.path.normpath(base_dir))
        dm_save_path = os.path.join(base_dir, f'density_matrix_{subdir_name}.nc')
        dm_ds.to_netcdf(dm_save_path)
        print(f"Saved density matrix dataset for {qubit_name} to {dm_save_path}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(idle_times, Sx, label='Sx')
        ax.plot(idle_times, Sy, label='Sy')
        ax.plot(idle_times, Sz, label='Sz')
        ax.set_xlabel('Idle Time (ns)')
        ax.set_ylabel('Bloch Vector Component')
        ax.set_title(f'Bloch Vector vs Idle Time - Qubit {qubit_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, f'bloch_vector_vs_idle_time_{qubit_name}.png'))
        plt.close(fig)

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(idle_times, (Sx**2+Sy**2)**0.5, label='Sx')
        ax1.set_xlabel('Idle Time (ns)')
        ax1.set_ylabel('Bloch Vector Component Sx**2+Sy**2')
        ax1.set_title(f'Bloch Vector vs Idle Time - Qubit {qubit_name}')
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(os.path.join(base_dir, f'Sx2+Sy2_vs_idle_time_{qubit_name}.png'))
        plt.close(fig1)


# Recursively process all subfolders under the main temporal_steering folder
root_dir = r'D:\data\temporal_steering\v3_1k_test'
subfolders = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))]
for subdir in subfolders:
    # Only process leaf folders (those containing ds_raw.h5 and node.json)
    print(subdir)
    analyze_temporal_steering_folder(subdir)

    # After analysis, load the saved density matrix and reconstruct complex array
    subdir_name = os.path.basename(os.path.normpath(subdir))
    dm_save_path = os.path.join(subdir, f'density_matrix_{subdir_name}.nc')
    if os.path.exists(dm_save_path):
        dm_ds = xr.load_dataset(dm_save_path)
        dm_real = dm_ds['density_matrix'].sel(part='real').values  # shape: (idle_time, 2, 2)
        dm_imag = dm_ds['density_matrix'].sel(part='imag').values
        density_matrix_complex = dm_real + 1j * dm_imag  # shape: (idle_time, 2, 2)
        idle_times = dm_ds['idle_time'].values
        print(f"Loaded density matrix for {subdir_name} with shape {density_matrix_complex.shape}")
        # Save as npz
        npz_save_path = os.path.join(subdir, f'density_matrix_{subdir_name}.npz')
        np.savez(npz_save_path, density_matrix=density_matrix_complex, idle_times=idle_times)
        print(f"Saved density_matrix_complex and idle_times to {npz_save_path}")
