import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time
from pathlib import Path

def read_ramsey_analysis_csv(csv_path):
    """
    Specialized function to read Ramsey analysis results CSV files.
    This assumes a simple 1D table format with one row per experiment.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the Ramsey analysis results CSV file
        
    Returns:
    --------
    xr.Dataset
        xarray Dataset with analysis results indexed by experiment number
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Ramsey analysis CSV file not found: {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Create a simple index for each row (experiment number)
    experiment_index = np.arange(len(df))
    
    # Separate coordinate/metadata columns from data columns
    metadata_cols = ['folder_name', 'folder_path', 'qubit_name', 'status', 'abscos_fit_success']
    
    # Identify data columns (numeric analysis results)
    data_cols = []
    coord_cols = []
    
    for col in df.columns:
        if col in metadata_cols:
            coord_cols.append(col)
        else:
            # Try to convert to numeric to identify data columns
            try:
                pd.to_numeric(df[col], errors='raise')
                data_cols.append(col)
            except (ValueError, TypeError):
                coord_cols.append(col)
    
    # Create coordinate dictionary
    coords = {'experiment': experiment_index}
    
    # Add metadata as coordinates (keep as string/object type)
    for coord_col in coord_cols:
        coords[coord_col] = (['experiment'], df[coord_col].values)
    
    # Create data variables dictionary
    data_vars = {}
    for data_col in data_cols:
        # Convert to numeric, handling any conversion issues
        try:
            data_values = pd.to_numeric(df[data_col], errors='coerce').values
        except:
            data_values = df[data_col].values
        
        data_vars[data_col] = (['experiment'], data_values)
    
    # Create the dataset
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add attributes for metadata
    ds.attrs['description'] = 'Ramsey analysis results'
    ds.attrs['source_file'] = str(csv_path)
    ds.attrs['total_experiments'] = len(df)
    
    return ds

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
    Complete workflow: load CSV, find stable indices, then load corresponding readout data.
    
    Parameters:
    -----------
    root_path : str  
        Root directory containing readout power subfolders and CSV file
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
    
    # Step 1: Load and analyze CSV
    print("\n📋 Step 1: Loading Ramsey analysis results...")
    csv_path = f"{root_path}\\ramsey_analysis_results.csv"
    ds_ramsey = read_ramsey_analysis_csv(csv_path)
    
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
            'total_experiments': len(ds_ramsey.experiment),
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

def plot_iq_colormaps(dataset, save_path=None, prepared_state=None):
    """
    Plot I and Q data as 2D colormaps with charge_gate vs amp_prefactor.
    Average over shot_idx dimension. Handle prepared_state dimension if present.
    
    Parameters:
    -----------
    dataset : xr.Dataset
        Dataset containing I and Q data variables
    save_path : str, optional
        Path to save the plot. If None, plot will be displayed
    prepared_state : int, optional
        Which prepared state to plot (0 or 1). If None, plots both states separately
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Check if prepared_state dimension exists
    has_prepared_state = 'prepared_state' in dataset.dims
    
    if has_prepared_state:
        prepared_states = dataset.coords['prepared_state'].values
        print(f"Dataset has prepared_state dimension with values: {prepared_states}")
        
        if prepared_state is not None:
            # Plot specific prepared state
            plot_dataset = dataset.sel(prepared_state=prepared_state)
            state_suffix = f"_state{prepared_state}"
            state_title = f" (State {prepared_state})"
        else:
            # Plot all prepared states separately
            figures = []
            for state in prepared_states:
                print(f"Plotting prepared state: {state}")
                state_save_path = None
                if save_path:
                    # Insert state suffix before file extension
                    base, ext = os.path.splitext(save_path)
                    state_save_path = f"{base}_state{state}{ext}"
                
                fig = plot_iq_colormaps(dataset, save_path=state_save_path, prepared_state=state)
                figures.append(fig)
            return figures
    else:
        # No prepared_state dimension
        plot_dataset = dataset
        state_suffix = ""
        state_title = ""
    
    # Get I and Q data and average over shot_idx and experiment dimensions
    i_mean = plot_dataset['I'].mean(dim=['shot_idx', 'experiment'])
    q_mean = plot_dataset['Q'].mean(dim=['shot_idx', 'experiment'])
    
    # Get coordinates - handle different coordinate names
    if 'charge_gate' in plot_dataset.coords:
        charge_gates = plot_dataset.coords['charge_gate'].values
        x_label = 'Charge Gate (V)'
    elif 'normalized_charge_gate' in plot_dataset.coords:
        charge_gates = plot_dataset.coords['normalized_charge_gate'].values
        x_label = 'Normalized Charge Gate (V)'
    else:
        raise ValueError("No charge gate coordinate found in dataset")
    
    amp_prefactors = plot_dataset.coords['amp_prefactor'].values
    
    # Create meshgrid
    X, Y = np.meshgrid(charge_gates, amp_prefactors)
    
    # Transpose data for pcolormesh (amp_prefactor, charge_gate)
    i_plot_data = i_mean.T.values
    q_plot_data = q_mean.T.values
    
    print(f"Plotting data shapes - I: {i_plot_data.shape}, Q: {q_plot_data.shape}")
    print(f"Meshgrid shapes - X: {X.shape}, Y: {Y.shape}")
    
    # Create figure
    fig, (ax_i, ax_q) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot I data
    im_i = ax_i.pcolormesh(X, Y, i_plot_data, shading='auto', cmap='viridis')
    ax_i.set_title(f'I Signal{state_title}')
    ax_i.set_xlabel(x_label)
    ax_i.set_ylabel('Amplifier Prefactor')
    plt.colorbar(im_i, ax=ax_i, label='I Signal')
    
    # Plot Q data
    im_q = ax_q.pcolormesh(X, Y, q_plot_data, shading='auto', cmap='plasma')
    ax_q.set_title(f'Q Signal{state_title}')
    ax_q.set_xlabel(x_label)
    ax_q.set_ylabel('Amplifier Prefactor')
    plt.colorbar(im_q, ax=ax_q, label='Q Signal')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    return fig

# Example usage and testing
if __name__ == "__main__":
    # Test with the stable indices
    root_path = r"D:\data\MIST\20251201\r_9_150x50_50_s300_ro_005x18_s100_fb\set_5"
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
            # Don't squeeze prepared_state dimension if it exists
            squeeze_dims = [dim for dim in readout_dataset.sizes if readout_dataset.sizes[dim] == 1 and dim != 'prepared_state']
            if squeeze_dims:
                readout_dataset = readout_dataset.squeeze(squeeze_dims, drop=True)
            print(readout_dataset)
            # This will automatically handle prepared_state dimension and create separate plots
            figs = plot_iq_colormaps(readout_dataset, save_path=f"{root_path}\\iq_colormaps.png")
        
        print(f"\nFinal Results Summary:")
        print(f"Stable indices: {results['stable_indices']}")
        if readout_dataset is not None:
            print(f"Concatenated dataset shape: I={readout_dataset['I'].shape}, Q={readout_dataset['Q'].shape}")
            print(f"Dataset created with dimensions: {dict(readout_dataset.sizes)}")
            print(f"Plots saved to: {root_path}")
    else:
        print("No readout data available for charge gate analysis")
        print(f"Available data: readout_data={bool(results['readout_data'])}, ramsey_dataset={bool(results['ramsey_dataset'])}")