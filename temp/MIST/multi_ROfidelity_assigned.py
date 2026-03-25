import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from qcat.analysis.readout_power.analysis import ROFidelityPower
from qcat.analysis.state_discrimination.analysis import StateDiscrimination

base_dir = r'D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\MIST\20251201\r_9_150x50_50_s300_ro_005x18_s100_fb\set_5'
norm_ac_shift = 166 /4912.0  # Example normalization factor shift/f_ro
charge_period = 0.460  # Volt
print(f"Normalization factor for AC shift: {norm_ac_shift}")

# Define the list of tuples (normalized_charge_gate, amp_prefactor) to analyze
selected_points = [
    # Add your selected points here, for example:
    # (0.41, 1.0),
    # (0.41, 0.4),
    # (0.41, 0.2),
    # (0.35, 1.0),
    # (0.35, 0.4),
    # (0.35, 0.2),
    (0.0, 1.0),
    (0.0, 0.8),
    (0.0, 0.6),
    # Add more tuples as needed
]

# Create output directory for selected analysis
output_dir = os.path.join(base_dir, "selected_RO_fidelity")
os.makedirs(output_dir, exist_ok=True)

assign_std = 0.000396
# assign_mean = load_xarray_h5(r"D:\data\MIST\20251201\r_9_150x50_50_s300_ro_005x18_s100_fb\set_5\ro_power_analysis_cg_0.410_amp_0.1_to_0.4\fit_mean.nc")
assign_mean = None
print(assign_mean)

# merged_ds = xr.concat([prepare_0, prepare_1], dim='prepared_state')
merged_ds = load_xarray_h5(os.path.join(base_dir, "final_readout_dataset.h5"))
print(f"Merged dataset: {merged_ds}")
print(f"Merged dataset coords: {merged_ds.coords["normalized_charge_gate"]}, {merged_ds.coords["amp_prefactor"]}")

# Use repetition_data to get per-qubit data from merged_ds
from qcat.parser.qm_reader import repetition_data
qubit_datasets = repetition_data(merged_ds, repetition_dim="qubit")
qubit_datasets = [qubit_datasets[0]]

for sq_data in qubit_datasets:
    qubit_name = sq_data["qubit"].values.item()
    
    print(f"Processing qubit: {qubit_name}")
    
    # Process each selected point
    for i, (charge_gate_val, amp_prefactor_val) in enumerate(selected_points):
        # Get the actual indices used for selection
        charge_gate_idx = np.argmin(np.abs(sq_data.coords['normalized_charge_gate'].values - charge_gate_val))
        charge_gate_val = sq_data.coords['normalized_charge_gate'].values[charge_gate_idx]
        amp_prefactor_idx = np.argmin(np.abs(sq_data.coords['amp_prefactor'].values - amp_prefactor_val))  
        amp_prefactor_val = sq_data.coords['amp_prefactor'].values[amp_prefactor_idx]

        print(f"Processing point {i+1}/{len(selected_points)}: charge_gate={charge_gate_val}, amp_prefactor={amp_prefactor_val}")

        try:
            # Select the specific data point
            single_point_data = sq_data.isel(
                normalized_charge_gate=charge_gate_idx, 
                amp_prefactor=amp_prefactor_idx,
            )
            
            # Stack experiment dimension into shot_idx if it exists
            if 'experiment' in single_point_data.dims:
                single_point_data = single_point_data.stack(extended_shot_idx=('experiment', 'shot_idx'))
                single_point_data = single_point_data.swap_dims({'extended_shot_idx': 'shot_idx'})
                # Create new shot_idx coordinate values
                total_shots = len(single_point_data.shot_idx)
                single_point_data = single_point_data.assign_coords(shot_idx=np.arange(total_shots))
            
            print(f"   Selected data shape: I={single_point_data['I'].shape}, Q={single_point_data['Q'].shape}")
            
            # Perform StateDiscrimination analysis
                # If self.fit_mean is not None, use its value to set user_mean for this amp_prefactor
            if assign_mean is not None:
                # self.fit_mean['intercept'] is shape (2,2) for (state, iq), slope is (2,2)
                # user_mean should be shape (2,2): user_mean[state, iq] = slope * val + intercept
                slope = assign_mean['slope'].values  # shape (2,2)
                intercept = assign_mean['intercept'].values  # shape (2,2)
                user_mean = slope * amp_prefactor_val + intercept
            else:
                user_mean = None
            analysis = StateDiscrimination(single_point_data, user_std=assign_std, user_mean=user_mean)
            analysis._start_analysis()
            
            # Create subfolder name for this point
            point_name = f"cg{charge_gate_val:.3f}_ap{amp_prefactor_val:.1f}"
            point_dir = os.path.join(output_dir, point_name)
            os.makedirs(point_dir, exist_ok=True)
            
            # Plot and save results
            figs = analysis._plot_results(fig_group_name=f"{qubit_name}_{point_name}", save_path=point_dir)
            
            # Save analysis results as text file
            results_file = os.path.join(point_dir, f"{point_name}_results.txt")
            with open(results_file, 'w') as f:
                f.write(f"StateDiscrimination Analysis Results\n")
                f.write(f"Point Index: {i+1}\n")
                f.write(f"Qubit: {qubit_name}\n")
                f.write(f"Normalized Charge Gate: {charge_gate_val} (index: {charge_gate_idx})\n")
                f.write(f"Amp Prefactor: {amp_prefactor_val} (index: {amp_prefactor_idx})\n")
                f.write(f"Total Shots: {single_point_data.sizes['shot_idx']}\n")
                f.write(f"\nTrained Parameters:\n")
                f.write(f"Mean: {analysis.analysis_result['trained_paras']['mean']}\n")
                f.write(f"Std: {analysis.analysis_result['trained_paras']['std']}\n")
                f.write(f"Outlier Probability: {analysis.analysis_result['outlier_probability']}\n")
                f.write(f"Normalized Residues: {analysis.analysis_result['norm_res']}\n")
            
            print(f"   ✅ Analysis completed and saved to: {point_dir}")
            
        except Exception as e:
            print(f"   ❌ Error processing point ({charge_gate_val}, {amp_prefactor_val}): {e}")
            continue

print(f"\n🎯 All selected points processed. Results saved in: {output_dir}")
    

    