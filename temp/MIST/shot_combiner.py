import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time
from pathlib import Path



def find_stable_indices(ds, redchi_threshold=1e-10, phase_diff_threshold=0.01):
    """
    Find indices that satisfy stability conditions for Ramsey analysis.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing abscos_redchi and abscos_phase data
    redchi_threshold : float, default 1e-10
        Maximum allowed reduced chi-squared for both i and i+1
    phase_diff_threshold : float, default 0.005
        Maximum allowed phase difference between i and i+1
        
    Returns:
    --------
    list
        Indices that satisfy both conditions
    """
    stable_indices = []
    
    # Get the data arrays
    if 'abscos_redchi' not in ds.data_vars:
        raise ValueError("Dataset must contain 'abscos_redchi' variable")
    if 'abscos_phase' not in ds.data_vars:
        raise ValueError("Dataset must contain 'abscos_phase' variable")
    
    redchi = ds['abscos_redchi'].values
    phase = ds['abscos_phase'].values
    
    # Handle different dimensions
    if redchi.ndim == 1:
        # 1D case
        for i in range(len(redchi) - 1):
            # Condition 1: both i and i+1 have redchi < threshold
            redchi_condition = (redchi[i] < redchi_threshold and 
                              redchi[i+1] < redchi_threshold)
            
            # Condition 2: phase difference < threshold
            phase_diff = abs(phase[i+1] - phase[i])
            phase_condition = phase_diff < phase_diff_threshold
            
            if redchi_condition and phase_condition:
                stable_indices.append(i)
                
    else:
        # Multi-dimensional case - flatten and check
        redchi_flat = redchi.flatten()
        phase_flat = phase.flatten()
        
        for i in range(len(redchi_flat) - 1):
            # Condition 1: both i and i+1 have redchi < threshold
            redchi_condition = (redchi_flat[i] < redchi_threshold and 
                              redchi_flat[i+1] < redchi_threshold)
            
            # Condition 2: phase difference < threshold
            phase_diff = abs(phase_flat[i+1] - phase_flat[i])
            phase_condition = phase_diff < phase_diff_threshold
            
            if redchi_condition and phase_condition:
                stable_indices.append(i)
    
    return stable_indices

def load_readout_power_data_advanced(indices_list, root_path, 
                                   pattern_template="*_LCH_charge_gate_readout_power_{i}_*",
                                   return_metadata=True):
    """
    Advanced version that also returns metadata and handles various file formats.
    
    Parameters:
    -----------
    indices_list : list
        List of indices for subfolder numbers
    root_path : str or Path
        Root directory path
    pattern_template : str
        Folder name pattern template
    return_metadata : bool, default True
        Whether to include metadata about loaded files
        
    Returns:
    --------
    dict
        If return_metadata=False: {index: dataset}
        If return_metadata=True: {
            'data': {index: dataset},
            'metadata': {index: {'folder_path': path, 'file_size': size, 'load_time': time}},
            'failed_indices': [list of failed indices],
            'summary': {'total_requested': int, 'loaded': int, 'failed': int}
        }
    """
    import glob
    import xarray as xr
    import time
    from pathlib import Path
    
    root_path = Path(root_path)
    loaded_data = {}
    metadata = {}
    failed_indices = []
    
    print(f"Loading readout power data from: {root_path}")
    print(f"Target indices: {indices_list}")
    print(f"Folder pattern template: {pattern_template}")
    
    for idx in indices_list:
        start_time = time.time()
        
        # Create the folder pattern for this index
        folder_pattern = pattern_template.format(i=idx)
        search_pattern = root_path / folder_pattern
        
        # Find matching folders
        matching_folders = glob.glob(str(search_pattern))
        if not matching_folders:
            print(f"Warning: No folder found for index {idx} with pattern: {folder_pattern}")
            failed_indices.append(idx)
            continue
        
        if len(matching_folders) > 1:
            print(f"Warning: Multiple folders found for index {idx}: {matching_folders}")
        
        folder_path = Path(matching_folders[0])
        ds_raw_path = folder_path / "ds_raw.h5"
        
        if not ds_raw_path.exists():
            print(f"Warning: ds_raw.h5 not found in {folder_path}")
            failed_indices.append(idx)
            continue
        
        try:
            # Load the dataset
            dataset = xr.open_dataset(ds_raw_path)
            loaded_data[idx] = dataset
            
            # Calculate metadata
            load_time = time.time() - start_time
            file_size = ds_raw_path.stat().st_size
            
            metadata[idx] = {
                'folder_path': str(folder_path),
                'file_path': str(ds_raw_path),
                'file_size': file_size,
                'load_time': load_time,
                'dataset_sizes': dict(dataset.sizes),
                'data_vars': list(dataset.data_vars.keys()),
                'coords': list(dataset.coords.keys())
            }
            
            print(f"✅ Index {idx}: Loaded successfully ({file_size/(1024*1024):.1f} MB, {load_time:.2f}s)")
            
        except Exception as e:
            print(f"❌ Index {idx}: Load error - {e}")
            failed_indices.append(idx)
            continue
    
    # Create summary
    total_requested = len(indices_list)
    loaded_count = len(loaded_data)
    failed_count = len(failed_indices)
    
    print(f"\n📊 Summary:")
    print(f"   Total requested: {total_requested}")
    print(f"   Successfully loaded: {loaded_count}")
    print(f"   Failed: {failed_count}")
    
    if return_metadata:
        return {
            'data': loaded_data,
            'metadata': metadata,
            'failed_indices': failed_indices,
            'summary': {
                'total_requested': total_requested,
                'loaded': loaded_count,
                'failed': failed_count,
                'success_rate': loaded_count / total_requested if total_requested > 0 else 0
            }
        }
    else:
        return loaded_data

def process_stable_readout_data(root_path, 
                              redchi_threshold=1e-10, 
                              phase_diff_threshold=0.005):
    """
    Complete workflow: load netCDF, find stable indices, then load corresponding readout data.
    
    Parameters:
    -----------
    root_path : str  
        Root directory containing readout power subfolders and .h5 file
    redchi_threshold : float
        Reduced chi-squared threshold for stability
    phase_diff_threshold : float
        Phase difference threshold for stability
        
    Returns:
    --------
    dict
        Complete results with stability analysis and loaded readout data
    """
    print("🚀 Starting complete stable readout data processing...")
    
    # Step 1: Load and analyze netCDF file
    print("\n📋 Step 1: Loading Ramsey analysis results...")
    h5_path = f"{root_path}\\ramsey_analysis_results.h5"
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Ramsey analysis h5 file not found: {h5_path}")
    ds_ramsey = xr.load_dataset(h5_path)
    print(ds_ramsey)
    # Step 2: Find stable indices
    print("\n🔍 Step 2: Finding stable indices...")
    stable_indices = find_stable_indices(ds_ramsey, redchi_threshold, phase_diff_threshold)
    print(f"Found {len(stable_indices)} stable indices: {stable_indices}")
    
    # Step 3: Load readout power data for stable indices
    print("\n📂 Step 3: Loading readout power data for stable indices...")
    readout_results = load_readout_power_data_advanced(stable_indices, root_path)
    
    # Step 4: Create comprehensive results
    results = {
        'ramsey_dataset': ds_ramsey,
        'stable_indices': stable_indices,
        'readout_data': readout_results['data'],
        'readout_metadata': readout_results['metadata'],
        'failed_indices': readout_results['failed_indices'],
        'summary': {
            'total_experiments': len(ds_ramsey["status"]),
            'stable_count': len(stable_indices),
            'readout_loaded': readout_results['summary']['loaded'],
            'stability_criteria': {
                'redchi_threshold': redchi_threshold,
                'phase_diff_threshold': phase_diff_threshold
            }
        }
    }
    
    print(f"\n🎯 Complete! Processed {results['summary']['readout_loaded']} stable readout datasets")
    
    return results

def concatenate_readout_data_simple(readout_data):
    """
    Simply concatenate readout datasets along a new experiment dimension.
    Assumes all datasets have the same coordinates structure.
    
    Parameters:
    -----------
    readout_data : dict
        Dictionary with experiment indices as keys and xarray Datasets as values
        All datasets should have the same coordinate structure
        
    Returns:
    --------
    xr.Dataset
        Concatenated dataset with experiment dimension added
    """
    import xarray as xr
    
    if not readout_data:
        print("No readout data to concatenate")
        return None
    
    print(f"📋 Simply concatenating {len(readout_data)} readout datasets...")
    
    # Collect all datasets with experiment indices
    datasets = []
    experiment_indices = []
    
    for exp_idx in sorted(readout_data.keys()):
        exp_dataset = readout_data[exp_idx]
        print(f"Experiment {exp_idx}: Dataset shape I={exp_dataset['I'].shape}, Q={exp_dataset['Q'].shape}")
        
        # Add experiment index as a coordinate to the dataset
        exp_dataset_with_idx = exp_dataset.assign_coords(experiment=exp_idx)
        
        datasets.append(exp_dataset_with_idx)
        experiment_indices.append(exp_idx)
    
    # Simply concatenate all datasets along experiment dimension
    print("Concatenating datasets along experiment dimension...")
    concatenated_dataset = xr.concat(datasets, dim='experiment')
    
    # Rename charge_gate to normalized_charge_gate if it exists
    if 'charge_gate' in concatenated_dataset.coords:
        concatenated_dataset = concatenated_dataset.rename({'charge_gate': 'normalized_charge_gate'})
        print("✅ Renamed 'charge_gate' coordinate to 'normalized_charge_gate'")
    
    # Add metadata as attributes
    concatenated_dataset.attrs.update({
        'description': 'Simple concatenation of readout data from stable experiments',
        'total_experiments': len(experiment_indices),
        'experiment_indices': str(experiment_indices)
    })
    
    print(f"✅ Simple Concatenation Complete:")
    print(f"   Experiments combined: {len(experiment_indices)}")
    print(f"   Final dataset I shape: {concatenated_dataset['I'].shape}")
    print(f"   Final dataset Q shape: {concatenated_dataset['Q'].shape}")
    print(f"   Coordinates: {list(concatenated_dataset.coords.keys())}")
    print(f"   Dimensions: {dict(concatenated_dataset.sizes)}")
    
    return concatenated_dataset



# Example usage and testing
if __name__ == "__main__":
    # Test with the stable indices
    root_path = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\MIST\20251201\r_9_150x50_50_s300_ro_005x18_s100_fb\set_5"
    gate_step=0.005
    results = process_stable_readout_data(root_path, redchi_threshold=1e-11, phase_diff_threshold=gate_step/2)
    
    # Test the new charge gate index finding functionality
    print("\n" + "="*60)
    print("CHARGE GATE INDEX ANALYSIS")
    print("="*60)
    
    if results['readout_data'] and results['ramsey_dataset']:
        # Simply concatenate readout data without any modifications
        print("\n" + "-"*40)
        print("Simply concatenating readout data...")
        readout_dataset = concatenate_readout_data_simple(results['readout_data'])

        # Save the final dataset
        if readout_dataset is not None:
            print("\n" + "-"*40)
            print("Saving final dataset...")
            dataset_save_path = f"{root_path}\\final_readout_dataset.h5"
            try:
                readout_dataset.to_netcdf(dataset_save_path)
                print(f"✅ Final dataset saved to: {dataset_save_path}")
                print(f"   Dataset size: {readout_dataset.nbytes / (1024*1024):.1f} MB")
            except Exception as e:
                print(f"❌ Error saving dataset: {e}")
        
            # Plot I/Q colormaps
            print("\n" + "-"*40)
            print("Creating I/Q colormap plots...")
            print("NOTE: Use the plot_iq_colormaps() function in ploting.ipynb to create visualizations")
            # Don't squeeze prepared_state dimension if it exists
            squeeze_dims = [dim for dim in readout_dataset.sizes if readout_dataset.sizes[dim] == 1 and dim != 'prepared_state']
            if squeeze_dims:
                readout_dataset = readout_dataset.squeeze(squeeze_dims, drop=True)
            print(readout_dataset)
        
        print(f"\nFinal Results Summary:")
        print(f"Stable indices: {results['stable_indices']}")
        if readout_dataset is not None:
            print(f"Concatenated dataset shape: I={readout_dataset['I'].shape}, Q={readout_dataset['Q'].shape}")
            print(f"Dataset created with dimensions: {dict(readout_dataset.sizes)}")
            print(f"Plots saved to: {root_path}")
    else:
        print("No readout data available for charge gate analysis")
        print(f"Available data: readout_data={bool(results['readout_data'])}, ramsey_dataset={bool(results['ramsey_dataset'])}")