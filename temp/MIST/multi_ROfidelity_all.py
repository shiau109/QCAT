import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from qcat.analysis.readout_power.analysis import ROFidelityPower



base_dir = r'D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\MIST\20251201\r_9_150x50_50_s300_ro_005x18_s100_fb\set_5'
norm_ac_shift = 166 /4912.0  # Example normalization factor shift/f_ro
charge_period = 0.460  # Volt
print(f"Normalization factor for AC shift: {norm_ac_shift}")


experiment_range = (0,50)  # Set to slice(start, stop) to select experiment subset, e.g. slice(0, 50). None = use all.

assign_std = None#0.000396
assign_mean = None#load_xarray_h5(r"D:\data\MIST\20251201\r_9_150x50_50_s300_ro_005x18_s100_fb\set_5\ro_power_analysis_cg_0.410_amp_0.1_to_0.4\fit_mean.nc")

# merged_ds = xr.concat([prepare_0, prepare_1], dim='prepared_state')
merged_ds = load_xarray_h5(os.path.join(base_dir, "final_readout_dataset.h5"))
print(f"Merged dataset I shape: {merged_ds['I'].shape}")
# Use repetition_data to get per-qubit data from merged_ds
from qcat.parser.qm_reader import repetition_data
qubit_datasets = repetition_data(merged_ds, repetition_dim="qubit")
qubit_datasets = [qubit_datasets[0]]
for sq_data in qubit_datasets:
    qubit_name = sq_data["qubit"].values.item()
    
    # Check if summary file already exists first
    exp_suffix = f'_exp{experiment_range[0]}to{experiment_range[1]}' if experiment_range is not None else ''
    summary_filename = os.path.join(base_dir, f'merged_summary_qubit_{qubit_name}{exp_suffix}.h5')
    
    if os.path.exists(summary_filename):
        print(f"Loading existing merged summary from: {summary_filename}")
        merged_summary = load_xarray_h5(summary_filename)
    else:
        print(f"Summary file not found. Performing analysis for qubit {qubit_name}...")
        normalized_charge_gate_list = sq_data.coords["normalized_charge_gate"].values
        summary_list = []
        
        for normalized_charge_gate in normalized_charge_gate_list:
            single_ds = sq_data.sel(normalized_charge_gate=normalized_charge_gate)
            # Select experiment subset if specified
            if experiment_range is not None and 'experiment' in single_ds.dims:
                single_ds = single_ds.isel(experiment=slice(*experiment_range))
                print(single_ds)
            # Stack experiment dimension into shot_idx to increase total shots
            if 'experiment' in single_ds.dims:
                single_ds = single_ds.stack(extended_shot_idx=('experiment', 'shot_idx'))
                single_ds = single_ds.swap_dims({'extended_shot_idx': 'shot_idx'})
                # Create new shot_idx coordinate values
                total_shots = len(single_ds.shot_idx)
                single_ds = single_ds.assign_coords(shot_idx=np.arange(total_shots))
            print(f"Processing charge gate {normalized_charge_gate:.3f}, data shape: I={single_ds['I'].shape}, Q={single_ds['Q'].shape}")
            analysis = ROFidelityPower(single_ds, user_std=assign_std, fit_mean=assign_mean)
            analysis._start_analysis()
            # Add normalized_charge_gate as a coordinate to the summary_dataset
            summary_ds = analysis.summary_dataset.expand_dims({'normalized_charge_gate': [normalized_charge_gate]})
            summary_list.append(summary_ds)

        # Concatenate all summary_datasets along normalized_charge_gate
        merged_summary = xr.concat(summary_list, dim='normalized_charge_gate')
        
        # Save the merged_summary dataset
        merged_summary.to_netcdf(summary_filename, engine='h5netcdf')
        print(f"Saved merged summary for qubit {qubit_name} to: {summary_filename}")

    # Plot 2D colormaps for p_outlier and norm_res
    amp_prefactor = merged_summary['amp_prefactor'].values
    normalized_charge_gate = merged_summary['normalized_charge_gate'].values/(charge_period*4)
    # p_outlier: shape (normalized_charge_gate, amp_prefactor, state)
    # norm_res: shape (normalized_charge_gate, amp_prefactor, state)
    # We'll plot for state=0 and state=1 separately
    
    # Create common meshgrid for all plots
    X, Y = np.meshgrid((amp_prefactor**2)*norm_ac_shift, normalized_charge_gate)
    
    for state in [0, 1]:
        # p_outlier
        z = merged_summary['p_outlier'].sel(state=state).transpose('normalized_charge_gate', 'amp_prefactor').values
        
        # Find and print max p_outlier value and its coordinates
        max_idx = np.unravel_index(np.argmax(z), z.shape)
        max_value = z[max_idx]
        max_charge_gate = normalized_charge_gate[max_idx[0]]
        max_amp_prefactor = amp_prefactor[max_idx[1]]
        print(f"State {state}: Max p_outlier = {max_value:.6f} at charge_gate = {max_charge_gate:.4f}, amp_prefactor = {max_amp_prefactor:.4f}")
        
        # Apply log scale to z values (add small epsilon to avoid log(0))
        z_log = np.log10(z)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(Y, X, z_log, cmap='viridis', vmin=-3, vmax=0, shading='auto')
        ax.set_xlabel('normalized_charge_gate')
        ax.set_ylabel('amp_prefactor')
        ax.set_title(f'Qubit {qubit_name} State {state}: p_outlier (log scale)')
        fig.colorbar(im, ax=ax, label='log10(p_outlier)')
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, f'p_outlier_2d_{qubit_name}_state{state}{exp_suffix}.png'))
        plt.show()
        plt.close(fig)

        # norm_res
        z2 = merged_summary['norm_res'].sel(state=state).transpose('normalized_charge_gate', 'amp_prefactor').values
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im2 = ax2.pcolormesh(Y, X, z2, cmap='RdBu_r', shading='auto')
        ax2.set_xlabel('normalized_charge_gate')
        ax2.set_ylabel('amp_prefactor')
        ax2.set_title(f'Qubit {qubit_name} State {state}: norm_res')
        fig2.colorbar(im2, ax=ax2, label='norm_res')
        fig2.tight_layout()
        fig2.savefig(os.path.join(base_dir, f'norm_res_2d_{qubit_name}_state{state}{exp_suffix}.png'))
        plt.close(fig2)

        # std (standard deviation) - note: std doesn't depend on state
        std = merged_summary['std'].transpose('normalized_charge_gate', 'amp_prefactor').values
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        im3 = ax3.pcolormesh(Y, X, std, cmap='plasma', shading='auto')
        ax3.set_xlabel('normalized_charge_gate')
        ax3.set_ylabel('amp_prefactor')
        ax3.set_title(f'Qubit {qubit_name}: std')
        fig3.colorbar(im3, ax=ax3, label='std')
        fig3.tight_layout()
        fig3.savefig(os.path.join(base_dir, f'std_2d_{qubit_name}{exp_suffix}.png'))
        plt.close(fig3)
