import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from qcat.analysis.readout_power.analysis import ROFidelityPower

# Configuration
base_dir = r'D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\MIST\20251201\r_9_150x50_50_s300_ro_005x18_s100_fb\set_5'

# assign_std = 0.000396
# assign_mean = load_xarray_h5(r"D:\data\MIST\20251201\r_9_150x50_50_s300_ro_005x18_s100_fb\set_5\ro_power_analysis_cg_0.410_amp_0.1_to_0.4\fit_mean.nc")
assign_std = None
assign_mean = None

# Input: specify the normalized_charge_gate value you want to analyze
normalized_charge_gate_value = 0.0  # Change this value as needed

# Input: specify the amp_prefactor range to analyze (None = use all available values)
amp_prefactor_min = 0.10  # Set to a number to limit minimum amp_prefactor (e.g., 0.5)
amp_prefactor_max = 1.1  # Set to a number to limit maximum amp_prefactor (e.g., 1.5)

print(f"Starting ROFidelityPower analysis for normalized_charge_gate = {normalized_charge_gate_value}")

# Load the merged dataset
merged_ds = load_xarray_h5(os.path.join(base_dir, "final_readout_dataset.h5"))
print(f"Loaded dataset I shape: {merged_ds['I'].shape}")
print(f"Available normalized_charge_gate values: {merged_ds.coords['normalized_charge_gate'].values}")
print(f"Available amp_prefactor values: {merged_ds.coords['amp_prefactor'].values}")

# Use repetition_data to get per-qubit data from merged_ds
qubit_datasets = repetition_data(merged_ds, repetition_dim="qubit")
qubit_datasets = [qubit_datasets[0]]  # Use first qubit

for sq_data in qubit_datasets:
    qubit_name = sq_data["qubit"].values.item()
    print(f"Processing qubit: {qubit_name}")
    
    try:
        # Select data for the specific normalized_charge_gate value
        # This will give us all amp_prefactor values for this charge gate
        charge_gate_data = sq_data.sel(
            normalized_charge_gate=normalized_charge_gate_value,
            method='nearest'  # Use nearest neighbor selection
        )
        
        # Get the actual selected normalized_charge_gate value (in case of nearest neighbor)
        actual_charge_gate = charge_gate_data.coords['normalized_charge_gate'].values.item()
        print(f"Selected normalized_charge_gate: {actual_charge_gate} (requested: {normalized_charge_gate_value})")
        print(f"Data shape: I={charge_gate_data['I'].shape}, Q={charge_gate_data['Q'].shape}")
        print(f"Available amp_prefactor values: {charge_gate_data.coords['amp_prefactor'].values}")
        
        # Apply amp_prefactor range filtering if specified
        if amp_prefactor_min is not None or amp_prefactor_max is not None:
            amp_values = charge_gate_data.coords['amp_prefactor'].values
            mask = np.ones(len(amp_values), dtype=bool)
            
            if amp_prefactor_min is not None:
                mask = mask & (amp_values >= amp_prefactor_min)
            if amp_prefactor_max is not None:
                mask = mask & (amp_values <= amp_prefactor_max)
            
            if np.sum(mask) == 0:
                raise ValueError(f"No amp_prefactor values found in specified range [{amp_prefactor_min}, {amp_prefactor_max}]")
            
            selected_amp_values = amp_values[mask]
            charge_gate_data = charge_gate_data.sel(amp_prefactor=selected_amp_values)
            print(f"Applied amp_prefactor filter: min={amp_prefactor_min}, max={amp_prefactor_max}")
            print(f"Filtered amp_prefactor values: {charge_gate_data.coords['amp_prefactor'].values}")
            print(f"Filtered data shape: I={charge_gate_data['I'].shape}, Q={charge_gate_data['Q'].shape}")
        else:
            print(f"Using all available amp_prefactor values (no filtering)")
        
        # Stack experiment dimension into shot_idx if it exists
        if 'experiment' in charge_gate_data.dims:
            charge_gate_data = charge_gate_data.stack(extended_shot_idx=('experiment', 'shot_idx'))
            charge_gate_data = charge_gate_data.swap_dims({'extended_shot_idx': 'shot_idx'})
            # Create new shot_idx coordinate values
            total_shots = len(charge_gate_data.shot_idx)
            charge_gate_data = charge_gate_data.assign_coords(shot_idx=np.arange(total_shots))
            print(f"After stacking experiments: I={charge_gate_data['I'].shape}, Q={charge_gate_data['Q'].shape}")
        
        # Create output directory with amp_prefactor range info
        range_suffix = ""
        if amp_prefactor_min is not None or amp_prefactor_max is not None:
            min_str = f"{amp_prefactor_min:.1f}" if amp_prefactor_min is not None else "min"
            max_str = f"{amp_prefactor_max:.1f}" if amp_prefactor_max is not None else "max"
            range_suffix = f"_amp_{min_str}_to_{max_str}"
        
        output_dir = os.path.join(base_dir, f"ro_power_analysis_cg_{actual_charge_gate:.3f}{range_suffix}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Perform ROFidelityPower analysis
        print("Starting ROFidelityPower analysis...")
        analysis = ROFidelityPower(charge_gate_data, user_std=assign_std, fit_mean=assign_mean)
        analysis._start_analysis(save_path=output_dir)
        
        # Plot and save results
        print("Generating plots...")
        figs = analysis._plot_results(fig_group_name=f"{qubit_name}_cg_{actual_charge_gate:.3f}", 
                                    save_path=output_dir, 
                                    plot_all=False)  # Set to True to plot all individual StateDiscrimination results
        
        # Save summary information
        summary_file = os.path.join(output_dir, f"analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"ROFidelityPower Analysis Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Qubit: {qubit_name}\n")
            f.write(f"Normalized Charge Gate: {actual_charge_gate}\n")
            f.write(f"Requested Charge Gate: {normalized_charge_gate_value}\n")
            f.write(f"Total Shots per Point: {charge_gate_data.sizes['shot_idx']}\n")
            f.write(f"Number of Amp Prefactor Points: {len(charge_gate_data.coords['amp_prefactor'])}\n")
            f.write(f"Amp Prefactor Range: {charge_gate_data.coords['amp_prefactor'].values.min():.3f} to {charge_gate_data.coords['amp_prefactor'].values.max():.3f}\n")
            f.write(f"Amp Prefactor Filter: min={amp_prefactor_min}, max={amp_prefactor_max}\n\n")
            
            f.write(f"Analysis Results:\n")
            f.write(f"-----------------\n")
            f.write(f"Standard Deviations: {analysis.summary_dataset['std'].values}\n")
            f.write(f"Outlier Probabilities:\n")
            for state in [0, 1]:
                f.write(f"  State {state}: {analysis.summary_dataset['p_outlier'].sel(state=state).values}\n")
            
            f.write(f"\nFit Parameters (mean vs amp_prefactor):\n")
            f.write(f"---------------------------------------\n")
            if analysis.fit_mean is not None:
                slopes = analysis.fit_mean['slope'].values
                intercepts = analysis.fit_mean['intercept'].values
                for state in [0, 1]:
                    for iq_idx, iq_label in enumerate(['I', 'Q']):
                        f.write(f"State {state}, {iq_label}: slope = {slopes[state, iq_idx]:.6f}, intercept = {intercepts[state, iq_idx]:.6f}\n")
        
        print(f"✅ ROFidelityPower analysis completed successfully!")
        print(f"   Output directory: {output_dir}")
        print(f"   Files generated:")
        print(f"     - Various analysis plots (.png)")
        print(f"     - summary_dataset.nc (analysis data)")
        print(f"     - fit_mean.nc (fitted parameters)")
        print(f"     - analysis_summary.txt (text summary)")
        
        if analysis.state_discrimination_results:
            print(f"     - Individual StateDiscrimination plots for each amp_prefactor")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

print(f"\n🎯 Analysis completed for normalized_charge_gate = {normalized_charge_gate_value}")
print(f"To analyze a different charge gate value, change the 'normalized_charge_gate_value' variable and run again.")