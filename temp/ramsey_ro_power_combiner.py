import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

def read_csv_to_xarray(csv_path, index_cols=None, coord_cols=None):
    """
    Read a CSV file and convert it to an xarray Dataset.
    Assumes a simple 1D table format unless specified otherwise.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file
    index_cols : list, optional
        Column names to use as indices/coordinates. If None, will create a simple row index
    coord_cols : list, optional  
        Column names to use as coordinates. If None, will auto-detect
        
    Returns:
    --------
    xr.Dataset
        xarray Dataset with data from CSV
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # If no specific indexing requested, treat as simple 1D table
    if index_cols is None and coord_cols is None:
        # Create a simple row index
        row_index = np.arange(len(df))
        
        # Separate numeric data columns from text/metadata columns
        data_vars = {}
        coords = {'row': row_index}
        
        for col in df.columns:
            try:
                # Try to convert to numeric
                numeric_data = pd.to_numeric(df[col], errors='raise').values
                data_vars[col] = (['row'], numeric_data)
            except (ValueError, TypeError):
                # Non-numeric columns become coordinates
                coords[col] = (['row'], df[col].values)
        
        ds = xr.Dataset(data_vars, coords=coords)
        return ds
    
    # Auto-detect likely coordinate columns if not specified
    if index_cols is None:
        # Common coordinate column patterns
        potential_coords = []
        for col in df.columns:
            col_lower = col.lower()
            if any(coord_name in col_lower for coord_name in 
                   ['index', 'charge_gate', 'folder', 'qubit', 'gate', 'freq', 'time']):
                potential_coords.append(col)
        
        # Use detected coordinates or fall back to row indexing
        index_cols = potential_coords if potential_coords else ['row']
    
    if coord_cols is None:
        coord_cols = index_cols
    
    # Handle row indexing case
    if 'row' in index_cols:
        row_index = np.arange(len(df))
        coords = {'row': row_index}
        
        # All other columns become either data or coordinates
        data_vars = {}
        for col in df.columns:
            try:
                numeric_data = pd.to_numeric(df[col], errors='raise').values
                data_vars[col] = (['row'], numeric_data)
            except (ValueError, TypeError):
                coords[col] = (['row'], df[col].values)
        
        ds = xr.Dataset(data_vars, coords=coords)
        return ds
    
    # Separate coordinate columns from data columns
    data_cols = [col for col in df.columns if col not in coord_cols]
    
    # Handle single coordinate case (1D data)
    if len(coord_cols) == 1:
        coord_name = coord_cols[0]
        data_vars = {}
        
        for data_col in data_cols:
            # Convert to numeric if possible, handling NaN values
            try:
                data_values = pd.to_numeric(df[data_col], errors='coerce').values
            except:
                data_values = df[data_col].values
                
            data_vars[data_col] = ([coord_name], data_values)
        
        ds = xr.Dataset(data_vars, coords={coord_name: df[coord_name].values})
    
    # Handle multiple coordinates (multi-dimensional data)
    else:
        # Create a multi-index approach
        # Set the coordinate columns as index
        df_indexed = df.set_index(coord_cols)
        
        # Convert to xarray Dataset
        ds = df_indexed.to_xarray()
    
    return ds

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
                # Non-numeric columns are treated as coordinates/metadata
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

def find_stable_indices(ds, redchi_threshold=1e-10, phase_diff_threshold=0.005):
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

def analyze_stability_conditions(ds, redchi_threshold=1e-10, phase_diff_threshold=0.005, 
                               verbose=True):
    """
    Comprehensive analysis of stability conditions with detailed output.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing analysis results
    redchi_threshold : float, default 1e-10
        Reduced chi-squared threshold
    phase_diff_threshold : float, default 0.005
        Phase difference threshold
    verbose : bool, default True
        Print detailed analysis
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    results = {
        'stable_indices': [],
        'redchi_values': [],
        'phase_differences': [],
        'conditions_met': [],
        'summary': {}
    }
    
    # Get data
    redchi = ds['abscos_redchi'].values
    phase = ds['abscos_phase'].values

    if redchi.ndim > 1:
        redchi = redchi.flatten()
        phase = phase.flatten()
     
    # Analyze each consecutive pair
    for i in range(len(redchi) - 1):
        # Current and next values
        redchi_current = redchi[i]
        redchi_next = redchi[i+1]
        phase_current = phase[i]
        phase_next = phase[i+1]
        
        # Check conditions
        redchi_condition = (redchi_current < redchi_threshold and 
                          redchi_next < redchi_threshold)
        
        phase_diff = abs(phase_next - phase_current)
        phase_condition = phase_diff < phase_diff_threshold
        
        both_conditions = redchi_condition and phase_condition
        
        # Store results
        results['redchi_values'].append((redchi_current, redchi_next))
        results['phase_differences'].append(phase_diff)
        results['conditions_met'].append({
            'redchi_ok': redchi_condition,
            'phase_ok': phase_condition,
            'both_ok': both_conditions
        })
        
        if both_conditions:
            results['stable_indices'].append(i)
    
    # Summary statistics
    total_pairs = len(results['conditions_met'])
    redchi_ok_count = sum(1 for c in results['conditions_met'] if c['redchi_ok'])
    phase_ok_count = sum(1 for c in results['conditions_met'] if c['phase_ok'])
    both_ok_count = len(results['stable_indices'])
    
    results['summary'] = {
        'total_pairs': total_pairs,
        'redchi_ok_count': redchi_ok_count,
        'phase_ok_count': phase_ok_count,
        'both_ok_count': both_ok_count,
        'redchi_ok_fraction': redchi_ok_count / total_pairs if total_pairs > 0 else 0,
        'phase_ok_fraction': phase_ok_count / total_pairs if total_pairs > 0 else 0,
        'both_ok_fraction': both_ok_count / total_pairs if total_pairs > 0 else 0
    }
    
    if verbose:
        print(f"Stability Analysis Results:")
        print(f"==========================")
        print(f"Total consecutive pairs analyzed: {total_pairs}")
        print(f"Pairs meeting redchi condition (<{redchi_threshold}): {redchi_ok_count} ({redchi_ok_count/total_pairs*100:.1f}%)")
        print(f"Pairs meeting phase condition (<{phase_diff_threshold}): {phase_ok_count} ({phase_ok_count/total_pairs*100:.1f}%)")
        print(f"Pairs meeting BOTH conditions: {both_ok_count} ({both_ok_count/total_pairs*100:.1f}%)")
        print(f"\nStable indices: {results['stable_indices']}")
        
        if results['stable_indices']:
            print(f"\nDetailed results for stable indices:")
            for idx in results['stable_indices']:
                redchi_pair = results['redchi_values'][idx]
                phase_diff = results['phase_differences'][idx]
                print(f"  Index {idx}: redchi=({redchi_pair[0]:.2e}, {redchi_pair[1]:.2e}), phase_diff={phase_diff:.6f}")
    
    return results

def load_readout_power_data(indices_list, root_path, pattern_template="*_LCH_charge_gate_readout_power_{i}_*"):
    """
    Load ds_raw.h5 files from readout power subfolders based on index list.
    
    Parameters:
    -----------
    indices_list : list
        List of indices corresponding to subfolder numbers (from stable_indices)
    root_path : str or Path
        Root directory containing the readout power subfolders
    pattern_template : str, default "*_LCH_charge_gate_readout_power_{i}_*"
        Template for folder name pattern where {i} will be replaced with index
        
    Returns:
    --------
    dict
        Dictionary with index as key and loaded dataset as value
        Format: {index: xr.Dataset, ...}
    """
    import glob
    import xarray as xr
    
    root_path = Path(root_path)
    loaded_data = {}
    
    print(f"Loading readout power data from: {root_path}")
    print(f"Target indices: {indices_list}")
    
    for idx in indices_list:
        # Create the folder pattern for this index
        folder_pattern = pattern_template.format(i=idx)
        search_pattern = root_path / folder_pattern
        
        # Find matching folders
        matching_folders = glob.glob(str(search_pattern))
        
        if not matching_folders:
            print(f"Warning: No folder found for index {idx} with pattern: {folder_pattern}")
            continue
        
        if len(matching_folders) > 1:
            print(f"Warning: Multiple folders found for index {idx}: {matching_folders}")
            print(f"Using the first one: {matching_folders[0]}")
        
        folder_path = Path(matching_folders[0])
        ds_raw_path = folder_path / "ds_raw.h5"
        print(ds_raw_path)
        if not ds_raw_path.exists():
            print(f"Warning: ds_raw.h5 not found in {folder_path}")
            continue
        
        try:
            # Load the dataset
            ds = xr.open_dataset(ds_raw_path, engine='h5netcdf')
            loaded_data[idx] = ds
            print(f"Successfully loaded index {idx}: {folder_path.name}")
            
        except Exception as e:
            print(f"Error loading dataset for index {idx} from {ds_raw_path}: {e}")
            continue
    
    print(f"\nSuccessfully loaded {len(loaded_data)} out of {len(indices_list)} datasets")
    return loaded_data

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
            print(f"❌ Index {idx}: No folder found with pattern '{folder_pattern}'")
            failed_indices.append(idx)
            continue
        
        if len(matching_folders) > 1:
            print(f"⚠️  Index {idx}: Multiple folders found, using first: {Path(matching_folders[0]).name}")
        
        folder_path = Path(matching_folders[0])
        ds_raw_path = folder_path / "ds_raw.h5"
        
        if not ds_raw_path.exists():
            print(f"❌ Index {idx}: ds_raw.h5 not found in {folder_path.name}")
            failed_indices.append(idx)
            continue
        
        try:
            # Load the dataset
            ds = xr.open_dataset(ds_raw_path)
            loaded_data[idx] = ds
            
            load_time = time.time() - start_time
            file_size = ds_raw_path.stat().st_size
            
            if return_metadata:
                metadata[idx] = {
                    'folder_path': str(folder_path),
                    'folder_name': folder_path.name,
                    'file_path': str(ds_raw_path),
                    'file_size_bytes': file_size,
                    'file_size_mb': file_size / (1024*1024),
                    'load_time_seconds': load_time,
                    'dataset_shape': dict(ds.dims),
                    'data_vars': list(ds.data_vars.keys()),
                    'coords': list(ds.coords.keys())
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
    csv_path : str
        Path to ramsey analysis CSV file
    root_path : str  
        Root directory containing readout power subfolders
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

def find_charge_gate_indices(readout_data, ds_ramsey):
    """
    Find the charge_gate coordinate indices that are closest to the abscos_phase values.
    
    Parameters:
    -----------
    readout_data : dict
        Dictionary with experiment indices as keys and xarray Datasets as values
        Each dataset should have 'charge_gate' coordinate
    ds_ramsey : xr.Dataset
        Ramsey analysis dataset containing 'abscos_phase' data variable
        
    Returns:
    --------
    dict
        Dictionary with experiment index as key and charge_gate index as value
        Format: {experiment_idx: charge_gate_idx, ...}
    """
    import numpy as np
    
    charge_gate_indices = {}
    
    print("Finding charge_gate indices closest to abscos_phase values...")
    
    for exp_idx, dataset in readout_data.items():
        try:
            # Get the abscos_phase value for this experiment index
            phase_value = ds_ramsey['abscos_phase'].isel(experiment=exp_idx).values.item()
            
            # Get the charge_gate coordinate values
            charge_gates = dataset.coords['charge_gate'].values
            
            # Find the index of the closest charge_gate value to the phase
            closest_idx = np.argmin(np.abs(charge_gates - phase_value))
            closest_charge_gate = charge_gates[closest_idx]
            
            charge_gate_indices[exp_idx] = closest_idx
            
            print(f"Experiment {exp_idx}: phase={phase_value:.6f}, "
                  f"closest charge_gate[{closest_idx}]={closest_charge_gate:.6f}, "
                  f"diff={abs(phase_value - closest_charge_gate):.6f}")
                  
        except Exception as e:
            print(f"Error processing experiment {exp_idx}: {e}")
            continue
    
    print(f"\nFound charge_gate indices for {len(charge_gate_indices)} experiments")
    return charge_gate_indices

def extract_readout_data_at_phase(readout_data, ds_ramsey, charge_gate_window):
    """
    Extract 2D readout data (I, Q) starting from charge_gate positions closest to abscos_phase values.
    Extracts a window of charge_gate_window points in the forward direction, or backward if out of bounds.
    
    Parameters:
    -----------
    readout_data : dict
        Dictionary with experiment indices as keys and datasets as values
    ds_ramsey : xr.Dataset
        Ramsey analysis dataset with abscos_phase values
    charge_gate_window : int, default 108
        Number of charge_gate points to extract starting from the matched index
        
    Returns:
    --------
    dict
        Extracted readout data with structure:
        {
            'data': {
                experiment_idx: {
                    'I': array_data,
                    'Q': array_data, 
                    'charge_gate_idx_center': int,
                    'charge_gate_slice': slice,
                    'charge_gate_values': array,
                    'charge_gate_direction': str,
                    'phase_value': float,
                    'amp_prefactor': array or float
                }
            },
            'summary': {...}
        }
    """
    import numpy as np
    
    extracted_data = {}
    charge_gate_indices = find_charge_gate_indices(readout_data, ds_ramsey)
    charge_gate_window_pts = int(charge_gate_window[0] / charge_gate_window[1])
    print(f"\nExtracting 2D readout data with {charge_gate_window_pts}-point charge_gate windows...")
    
    for exp_idx, dataset in readout_data.items():
        if exp_idx not in charge_gate_indices:
            print(f"Skipping experiment {exp_idx}: no charge_gate index found")
            continue
        
        try:
            charge_gate_idx_center = charge_gate_indices[exp_idx]
            phase_value = ds_ramsey['abscos_phase'].isel(experiment=exp_idx).values.item()
            
            # Get total number of charge_gate points
            total_charge_gates = len(dataset.coords['charge_gate'])
            
            # Determine the slice direction and range
            if charge_gate_idx_center + charge_gate_window_pts <= total_charge_gates:
                # Forward direction: from center to center+window
                start_idx = charge_gate_idx_center
                end_idx = charge_gate_idx_center + charge_gate_window_pts
                direction = "forward"
                charge_gate_slice = slice(start_idx, end_idx)
            else:
                # Backward direction: from center-window to center
                start_idx = max(0, charge_gate_idx_center - charge_gate_window_pts)
                end_idx = charge_gate_idx_center
                direction = "backward"
                charge_gate_slice = slice(start_idx, end_idx)
            
            # Get the actual charge_gate values in the slice
            charge_gate_values = dataset.coords['charge_gate'].values[charge_gate_slice]
            
            print(f"Experiment {exp_idx}: center_idx={charge_gate_idx_center}, "
                  f"slice=[{start_idx}:{end_idx}], direction={direction}, "
                  f"window_size={len(charge_gate_values)}")
            
            # Extract I and Q data for the charge_gate slice
            i_data = dataset['I'].isel(charge_gate=charge_gate_slice)
            q_data = dataset['Q'].isel(charge_gate=charge_gate_slice)
            amp_prefactor_values = dataset.coords['amp_prefactor'].values

            
            extracted_data[exp_idx] = {
                'I': i_data.values,
                'Q': q_data.values,
                'charge_gate_idx_center': charge_gate_idx_center,
                'charge_gate_slice': charge_gate_slice,
                'charge_gate_slice_indices': (start_idx, end_idx),
                'charge_gate_values': charge_gate_values,
                'charge_gate_direction': direction,
                'phase_value': phase_value,
                'amp_prefactor': amp_prefactor_values,
                'dataset_shape': dict(dataset.dims),
                'extracted_shape': i_data.shape,
                'window_size': len(charge_gate_values)
            }
            
            print(f"✅ Experiment {exp_idx}: Extracted 2D data shape {i_data.shape} "
                  f"({direction}, window={len(charge_gate_values)})")
            
        except Exception as e:
            print(f"❌ Error extracting data for experiment {exp_idx}: {e}")
            continue
    
    # Create summary
    forward_count = sum(1 for data in extracted_data.values() if data['charge_gate_direction'] == 'forward')
    backward_count = sum(1 for data in extracted_data.values() if data['charge_gate_direction'] == 'backward')
    
    summary = {
        'total_experiments': len(readout_data),
        'extracted_count': len(extracted_data),
        'failed_count': len(readout_data) - len(extracted_data),
        'charge_gate_indices': charge_gate_indices,
        'window_size': charge_gate_window_pts,
        'direction_counts': {
            'forward': forward_count,
            'backward': backward_count
        }
    }
    
    result = {
        'data': extracted_data,
        'summary': summary
    }
    
    print(f"\n📊 Extraction Summary:")
    print(f"   Total experiments: {summary['total_experiments']}")
    print(f"   Successfully extracted: {summary['extracted_count']}")
    print(f"   Failed: {summary['failed_count']}")
    print(f"   Window size: {charge_gate_window_pts}")
    print(f"   Forward direction: {forward_count}")
    print(f"   Backward direction: {backward_count}")
    
    return result


def concatenate_extracted_readout_data(extracted_results):
    """
    Concatenate all extracted readout data and extend along shot_idx coordinate.
    
    Parameters:
    -----------
    extracted_results : dict
        Results from extract_readout_data_at_phase function containing extracted data
        
    Returns:
    --------
    dict
        Dictionary containing concatenated datasets:
        {
            'I_concatenated': xr.DataArray,
            'Q_concatenated': xr.DataArray,
            'metadata': {
                'experiment_indices': list,
                'total_shots': int,
                'charge_gate_window_pts': int,
                'amp_prefactor_mode': str,
                'shape': tuple
            }
        }
    """
    import xarray as xr
    import numpy as np
    
    if not extracted_results['data']:
        print("No extracted data to concatenate")
        return {}
    
    print("📋 Concatenating extracted readout data along shot_idx dimension...")
    
    # Collect all I and Q data arrays
    i_arrays = []
    q_arrays = []
    experiment_indices = []
    
    # Get metadata from first experiment
    first_exp_idx = list(extracted_results['data'].keys())[0]
    first_data = extracted_results['data'][first_exp_idx]
    
    for exp_idx in sorted(extracted_results['data'].keys()):
        exp_data = extracted_results['data'][exp_idx]
        
        # Get I and Q data
        i_data = exp_data['I']
        q_data = exp_data['Q']
        
        print(f"Experiment {exp_idx}: I shape {i_data.shape}, Q shape {q_data.shape}")
        
        # Convert to xarray DataArrays with proper coordinates
        # Determine the coordinate structure based on data dimensions
        if i_data.ndim == 3:  # (charge_gate, amp_prefactor, shot_idx) or similar
            charge_gate_values = exp_data['charge_gate_values']
            
            # Create coordinates based on whether amp_prefactor was extracted
            if extracted_results['summary']['amp_prefactor_mode'] == 'all':
                amp_prefactor_values = exp_data['amp_prefactor']
                coords = {
                    'charge_gate': charge_gate_values,
                    'amp_prefactor': amp_prefactor_values,
                    'shot_idx': np.arange(i_data.shape[-1])
                }
                dims = ['charge_gate', 'amp_prefactor', 'shot_idx']
            else:
                coords = {
                    'charge_gate': charge_gate_values,
                    'shot_idx': np.arange(i_data.shape[-1])
                }
                dims = ['charge_gate', 'shot_idx']
        
        elif i_data.ndim == 4:  # (charge_gate, qubit, prepared_state, shot_idx) or similar
            charge_gate_values = exp_data['charge_gate_values']
            coords = {
                'charge_gate': charge_gate_values,
                'qubit': np.arange(i_data.shape[1]),
                'prepared_state': np.arange(i_data.shape[2]),
                'shot_idx': np.arange(i_data.shape[-1])
            }
            dims = ['charge_gate', 'qubit', 'prepared_state', 'shot_idx']
        
        elif i_data.ndim == 5:  # (charge_gate, qubit, prepared_state, amp_prefactor, shot_idx)
            charge_gate_values = exp_data['charge_gate_values']
            amp_prefactor_values = exp_data['amp_prefactor']
            coords = {
                'charge_gate': charge_gate_values,
                'qubit': np.arange(i_data.shape[1]),
                'prepared_state': np.arange(i_data.shape[2]),
                'amp_prefactor': amp_prefactor_values,
                'shot_idx': np.arange(i_data.shape[-1])
            }
            dims = ['charge_gate', 'qubit', 'prepared_state', 'amp_prefactor', 'shot_idx']
        
        else:
            # Fallback for unexpected dimensions
            coords = {f'dim_{i}': np.arange(i_data.shape[i]) for i in range(i_data.ndim)}
            dims = list(coords.keys())
            print(f"Warning: Unexpected data dimensions {i_data.ndim}, using fallback coordinates")
        
        # Create DataArrays
        i_array = xr.DataArray(
            i_data,
            coords=coords,
            dims=dims,
            name=f'I_exp_{exp_idx}'
        )
        
        q_array = xr.DataArray(
            q_data,
            coords=coords,
            dims=dims,
            name=f'Q_exp_{exp_idx}'
        )
        
        # Add experiment index as a coordinate
        i_array = i_array.assign_coords(experiment=exp_idx)
        q_array = q_array.assign_coords(experiment=exp_idx)
        
        i_arrays.append(i_array)
        q_arrays.append(q_array)
        experiment_indices.append(exp_idx)
    
    # Concatenate along a new experiment dimension first, then merge shot_idx
    print("Concatenating arrays along experiment dimension...")
    
    # Concatenate along experiment dimension
    i_combined = xr.concat(i_arrays, dim='experiment')
    q_combined = xr.concat(q_arrays, dim='experiment')
    
    # Now reshape to extend shot_idx dimension
    print("Reshaping to extend shot_idx dimension...")
    
    # Stack experiment and shot_idx dimensions to create extended shot_idx
    i_extended = i_combined.stack(extended_shot_idx=('experiment', 'shot_idx'))
    q_extended = q_combined.stack(extended_shot_idx=('experiment', 'shot_idx'))
    
    # Rename the stacked dimension back to shot_idx
    i_final = i_extended.rename({'extended_shot_idx': 'shot_idx'})
    q_final = q_extended.rename({'extended_shot_idx': 'shot_idx'})
    
    # Calculate metadata
    total_shots = len(i_final.shot_idx)
    charge_gate_window_pts = len(i_final.charge_gate)
    
    result = {
        'I_concatenated': i_final,
        'Q_concatenated': q_final,
        'metadata': {
            'experiment_indices': experiment_indices,
            'total_experiments': len(experiment_indices),
            'total_shots': total_shots,
            'charge_gate_window_pts': charge_gate_window_pts,
            'amp_prefactor_mode': extracted_results['summary']['amp_prefactor_mode'],
            'original_shot_count_per_exp': i_data.shape[-1],
            'final_shape': i_final.shape
        }
    }
    
    print(f"\n✅ Concatenation Complete:")
    print(f"   Experiments combined: {len(experiment_indices)}")
    print(f"   Total shots: {total_shots}")
    print(f"   Charge gate window: {charge_gate_window_pts}")
    print(f"   Final I shape: {i_final.shape}")
    print(f"   Final Q shape: {q_final.shape}")
    print(f"   Coordinates: {list(i_final.coords.keys())}")
    
    return result

def create_concatenated_dataset(concatenated_results):
    """
    Create a properly structured xarray Dataset from concatenated I/Q data.
    
    Parameters:
    -----------
    concatenated_results : dict
        Results from concatenate_extracted_readout_data function
        
    Returns:
    --------
    xr.Dataset
        Dataset with I_concatenated and Q_concatenated as data variables,
        and charge_gate, amp_prefactor as coordinates
    """
    import xarray as xr
    
    if not concatenated_results:
        print("No concatenated results to create dataset")
        return None
    
    print("📋 Creating structured Dataset from concatenated I/Q data...")
    
    # Get the concatenated arrays
    i_concat = concatenated_results['I_concatenated']
    q_concat = concatenated_results['Q_concatenated']
    
    # Extract coordinates that we want to keep
    coords_dict = {}
    
    # Always include charge_gate
    if 'charge_gate' in i_concat.coords:
        coords_dict['charge_gate'] = i_concat.coords['charge_gate']
    
    # Include amp_prefactor if it exists
    if 'amp_prefactor' in i_concat.coords:
        coords_dict['amp_prefactor'] = i_concat.coords['amp_prefactor']
    
    # Keep shot_idx for the extended shots
    if 'shot_idx' in i_concat.coords:
        coords_dict['shot_idx'] = i_concat.coords['shot_idx']
    
    # Create the dataset with I and Q as data variables
    dataset = xr.Dataset(
        data_vars={
            'I': i_concat,
            'Q': q_concat
        },
        coords=coords_dict
    )
    
    # Add metadata as attributes
    metadata = concatenated_results['metadata']
    dataset.attrs.update({
        'description': 'Concatenated readout data from stable Ramsey experiments',
        'total_experiments': metadata['total_experiments'],
        'total_shots': metadata['total_shots'],
        'charge_gate_window_pts': metadata['charge_gate_window_pts'],
        'amp_prefactor_mode': metadata['amp_prefactor_mode'],
        'experiment_indices': str(metadata['experiment_indices'])
    })
    
    print(f"✅ Dataset created:")
    print(f"   Data variables: {list(dataset.data_vars.keys())}")
    print(f"   Coordinates: {list(dataset.coords.keys())}")
    print(f"   Dimensions: {dict(dataset.dims)}")
    print(f"   Total shots: {metadata['total_shots']}")
    
    return dataset

def plot_iq_colormaps(dataset, prepared_state_idx=0, qubit_idx=0, save_path=None):
    """
    Plot I and Q data as 2D colormaps with charge_gate vs amp_prefactor.
    
    Parameters:
    -----------
    dataset : xr.Dataset
        Dataset containing I and Q data variables with charge_gate and amp_prefactor coordinates
    prepared_state_idx : int, default 0
        Index of prepared_state to plot (if dimension exists)
    qubit_idx : int, default 0
        Index of qubit to plot (if dimension exists)
    save_path : str, optional
        Path to save the plot. If None, plot will be displayed
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"📊 Creating I/Q colormaps from dataset...")
    
    # Get I and Q data
    i_data = dataset['I']
    q_data = dataset['Q']
    
    # Select specific indices for qubit and prepared_state if they exist
    if 'qubit' in i_data.dims:
        i_data = i_data.isel(qubit=qubit_idx)
        q_data = q_data.isel(qubit=qubit_idx)
        print(f"   Selected qubit index: {qubit_idx}")
    
    if 'prepared_state' in i_data.dims:
        i_data = i_data.isel(prepared_state=prepared_state_idx)
        q_data = q_data.isel(prepared_state=prepared_state_idx)
        print(f"   Selected prepared_state index: {prepared_state_idx}")
    
    # Average over shot_idx to get mean I/Q values
    if 'shot_idx' in i_data.dims:
        i_mean = i_data.mean(dim='shot_idx')
        q_mean = q_data.mean(dim='shot_idx')
        print(f"   Averaged over {len(i_data.shot_idx)} shots")
    else:
        i_mean = i_data
        q_mean = q_data
    
    # Get coordinate values for plotting
    charge_gates = i_mean.coords['charge_gate'].values
    amp_prefactors = i_mean.coords['amp_prefactor'].values if 'amp_prefactor' in i_mean.coords else np.arange(i_mean.shape[-1])
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(charge_gates, amp_prefactors)
    
    # Transpose data to match meshgrid orientation (amp_prefactor, charge_gate)
    i_plot_data = i_mean.T.values if 'amp_prefactor' in i_mean.dims else i_mean.values
    q_plot_data = q_mean.T.values if 'amp_prefactor' in q_mean.dims else q_mean.values
    
    # Create figure with subplots
    fig, (ax_i, ax_q) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot I data
    im_i = ax_i.pcolormesh(X, Y, i_plot_data, shading='auto', cmap='viridis')
    ax_i.set_title('I Signal (Average over shots)')
    ax_i.set_xlabel('Charge Gate (V)')
    ax_i.set_ylabel('Amplifier Prefactor')
    cbar_i = plt.colorbar(im_i, ax=ax_i)
    cbar_i.set_label('I Signal')
    
    # Plot Q data
    im_q = ax_q.pcolormesh(X, Y, q_plot_data, shading='auto', cmap='plasma')
    ax_q.set_title('Q Signal (Average over shots)')
    ax_q.set_xlabel('Charge Gate (V)')
    ax_q.set_ylabel('Amplifier Prefactor')
    cbar_q = plt.colorbar(im_q, ax=ax_q)
    cbar_q.set_label('Q Signal')
    
    # Add overall title
    metadata = dataset.attrs
    fig.suptitle(f'Readout Power Analysis - {metadata.get("total_experiments", "N")} Stable Experiments, '
                f'{metadata.get("total_shots", "N")} Total Shots', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Plot saved to: {save_path}")
    else:
        plt.show()
    
    print(f"✅ I/Q colormaps created:")
    print(f"   Charge gate range: {charge_gates.min():.6f} to {charge_gates.max():.6f}")
    print(f"   Amp prefactor range: {amp_prefactors.min():.3f} to {amp_prefactors.max():.3f}")
    print(f"   I signal range: {i_plot_data.min():.3f} to {i_plot_data.max():.3f}")
    print(f"   Q signal range: {q_plot_data.min():.3f} to {q_plot_data.max():.3f}")
    
    return fig

def plot_iq_colormaps_advanced(dataset, amp_prefactor_slice=None, charge_gate_slice=None, 
                             prepared_state_idx=0, qubit_idx=0, save_path=None, 
                             figsize=(15, 6), cmaps=('viridis', 'plasma')):
    """
    Advanced I/Q colormap plotting with slicing and customization options.
    
    Parameters:
    -----------
    dataset : xr.Dataset
        Dataset containing I and Q data variables
    amp_prefactor_slice : slice, optional
        Slice for amp_prefactor dimension (e.g., slice(10, 25))
    charge_gate_slice : slice, optional
        Slice for charge_gate dimension (e.g., slice(50, 150))
    prepared_state_idx : int, default 0
        Index of prepared_state to plot
    qubit_idx : int, default 0
        Index of qubit to plot
    save_path : str, optional
        Path to save the plot
    figsize : tuple, default (15, 6)
        Figure size (width, height)
    cmaps : tuple, default ('viridis', 'plasma')
        Colormaps for I and Q plots
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"📊 Creating advanced I/Q colormaps with slicing...")
    
    # Get I and Q data
    i_data = dataset['I']
    q_data = dataset['Q']
    
    # Apply slicing
    if charge_gate_slice:
        i_data = i_data.isel(charge_gate=charge_gate_slice)
        q_data = q_data.isel(charge_gate=charge_gate_slice)
        print(f"   Applied charge_gate slice: {charge_gate_slice}")
    
    if amp_prefactor_slice and 'amp_prefactor' in i_data.dims:
        i_data = i_data.isel(amp_prefactor=amp_prefactor_slice)
        q_data = q_data.isel(amp_prefactor=amp_prefactor_slice)
        print(f"   Applied amp_prefactor slice: {amp_prefactor_slice}")
    
    # Select specific indices for other dimensions
    if 'qubit' in i_data.dims:
        i_data = i_data.isel(qubit=qubit_idx)
        q_data = q_data.isel(qubit=qubit_idx)
    
    if 'prepared_state' in i_data.dims:
        i_data = i_data.isel(prepared_state=prepared_state_idx)
        q_data = q_data.isel(prepared_state=prepared_state_idx)
    
    # Average over shots
    if 'shot_idx' in i_data.dims:
        i_mean = i_data.mean(dim='shot_idx')
        q_mean = q_data.mean(dim='shot_idx')
        i_std = i_data.std(dim='shot_idx')
        q_std = q_data.std(dim='shot_idx')
    else:
        i_mean = i_data
        q_mean = q_data
        i_std = None
        q_std = None
    
    # Get coordinates
    charge_gates = i_mean.coords['charge_gate'].values
    amp_prefactors = i_mean.coords['amp_prefactor'].values if 'amp_prefactor' in i_mean.coords else np.arange(i_mean.shape[-1])
    
    # Create meshgrid
    X, Y = np.meshgrid(charge_gates, amp_prefactors)
    
    # Prepare data for plotting
    i_plot_data = i_mean.T.values if i_mean.dims[0] == 'charge_gate' else i_mean.values
    q_plot_data = q_mean.T.values if q_mean.dims[0] == 'charge_gate' else q_mean.values
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_i, ax_q = axes
    
    # Plot I data
    im_i = ax_i.pcolormesh(X, Y, i_plot_data, shading='auto', cmap=cmaps[0])
    ax_i.set_title(f'I Signal (μ over {dataset.attrs.get("total_shots", "N")} shots)')
    ax_i.set_xlabel('Charge Gate (V)')
    ax_i.set_ylabel('Amplifier Prefactor')
    cbar_i = plt.colorbar(im_i, ax=ax_i)
    cbar_i.set_label('I Signal')
    ax_i.grid(True, alpha=0.3)
    
    # Plot Q data
    im_q = ax_q.pcolormesh(X, Y, q_plot_data, shading='auto', cmap=cmaps[1])
    ax_q.set_title(f'Q Signal (μ over {dataset.attrs.get("total_shots", "N")} shots)')
    ax_q.set_xlabel('Charge Gate (V)')
    ax_q.set_ylabel('Amplifier Prefactor')
    cbar_q = plt.colorbar(im_q, ax=ax_q)
    cbar_q.set_label('Q Signal')
    ax_q.grid(True, alpha=0.3)
    
    # Add statistics text boxes
    if i_std is not None:
        i_stats_text = f'μ±σ: {i_plot_data.mean():.3f}±{i_std.mean().values:.3f}'
        q_stats_text = f'μ±σ: {q_plot_data.mean():.3f}±{q_std.mean().values:.3f}'
        
        ax_i.text(0.02, 0.98, i_stats_text, transform=ax_i.transAxes, 
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                  verticalalignment='top')
        ax_q.text(0.02, 0.98, q_stats_text, transform=ax_q.transAxes,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                  verticalalignment='top')
    
    # Overall title with metadata
    metadata = dataset.attrs
    title = (f'Stable Ramsey Experiments: {metadata.get("total_experiments", "N")} exp, '
             f'{metadata.get("charge_gate_window_pts", "N")} charge gates, '
             f'{metadata.get("total_shots", "N")} shots')
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Advanced plot saved to: {save_path}")
    
    return fig

# Example usage and testing
if __name__ == "__main__":
    # Test with the stable indices
    root_path = r"d:\data\MIST\20251124\LCH_graph_charge_gate_r_rp\prepare_0"
    
    results = process_stable_readout_data(root_path)
    
    # Test the new charge gate index finding functionality
    print("\n" + "="*60)
    print("CHARGE GATE INDEX ANALYSIS")
    print("="*60)
    
    if results['readout_data'] and results['ramsey_dataset']:
        # Find charge gate indices
        charge_gate_indices = find_charge_gate_indices(results['readout_data'], results['ramsey_dataset'])
        
        # Extract readout data at phase positions
        print("\n" + "-"*40)
        print("Extracting readout data at phase positions...")
        gate_period=0.463
        gate_step=0.005
        extracted_results = extract_readout_data_at_phase(results['readout_data'], results['ramsey_dataset'], charge_gate_window=(gate_period,gate_step))
        
        # Concatenate extracted data
        print("\n" + "-"*40)
        print("Concatenating extracted readout data...")
        concatenated_results = concatenate_extracted_readout_data(extracted_results)
        
        # Create structured dataset
        print("\n" + "-"*40)
        print("Creating structured dataset...")
        if concatenated_results:
            readout_dataset = create_concatenated_dataset(concatenated_results)
            
            # Plot I/Q colormaps
            print("\n" + "-"*40)
            print("Creating I/Q colormap plots...")
            if readout_dataset is not None:
                # Basic plot
                fig1 = plot_iq_colormaps(readout_dataset, save_path=f"{root_path}\\iq_colormaps_basic.png")
                
                # Advanced plot with slicing (middle portion)
                fig2 = plot_iq_colormaps_advanced(
                    readout_dataset,
                    amp_prefactor_slice=slice(10, 25),  # Middle amp_prefactor range
                    save_path=f"{root_path}\\iq_colormaps_advanced.png"
                )
        
        print(f"\nFinal Results Summary:")
        print(f"Stable indices: {results['stable_indices']}")
        print(f"Charge gate indices: {charge_gate_indices}")
        if concatenated_results:
            print(f"Concatenated data shape: I={concatenated_results['I_concatenated'].shape}, Q={concatenated_results['Q_concatenated'].shape}")
            print(f"Total shots across all experiments: {concatenated_results['metadata']['total_shots']}")
            if 'readout_dataset' in locals():
                print(f"Dataset created with dimensions: {dict(readout_dataset.dims)}")
                print(f"Plots saved to: {root_path}")
    else:
        print("No readout data available for charge gate analysis")
        print(f"Available data: readout_data={bool(results['readout_data'])}, ramsey_dataset={bool(results['ramsey_dataset'])}")