# Import required libraries
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
from qcat.analysis.charge_gate_ramsey.analysis import ChargeGateRamseyAnalysis


def plot_phase_values(results_df, output_folder=None, x_axis='index'):
    """
    Plot abscos_phase values multiplied by abscos_frequency
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with analysis results  
    output_folder : str, optional
        Folder to save the plot. If None, plot is not saved.
    x_axis : str, default 'index'
        Choose x-axis: 'index' for folder index or 'run_start' for relative time from first experiment
    """
    
    # Filter successful fits
    successful_results = results_df[results_df['abscos_fit_success'] == True].copy()
    
    if len(successful_results) == 0:
        print("No successful fits to plot.")
        return None
    
    # Sort by folder name to maintain consistent order
    successful_results = successful_results.sort_values('folder_name').reset_index(drop=True)
    
    # Prepare x-axis data
    if x_axis == 'run_start':
        # Convert run_start to datetime if available
        successful_results['run_start_dt'] = pd.to_datetime(successful_results['run_start'], errors='coerce')
        
        # Check if we have valid timestamps
        if successful_results['run_start_dt'].isna().all():
            print("No valid run_start timestamps found, falling back to index")
            x_axis = 'index'
        else:
            # Calculate relative time from the earliest run_start
            earliest_time = successful_results['run_start_dt'].min()
            time_deltas = successful_results['run_start_dt'] - earliest_time
            x_data = time_deltas.dt.total_seconds() / 3600  # Convert to hours
            x_label = 'Relative Time (hours from first experiment)'
            plot_title = 'Phase × Frequency vs Relative Time'
            save_name = 'ramsey_phase_freq_product_vs_relative_time.png'
    
    if x_axis == 'index':
        x_data = np.arange(len(successful_results))
        x_label = 'Folder Index'
        plot_title = 'Phase × Frequency vs Folder Index'
        save_name = 'ramsey_phase_freq_product_vs_index.png'
    
    # Get phases and frequencies, then multiply them
    phases = successful_results['abscos_phase'].values
    frequencies = successful_results['abscos_frequency'].values
    phase_freq_product = phases * frequencies
    
    # Color coding based on fit quality
    colors = ['green' if chi < 1e-10 else 'orange' if chi < 1e-9 else 'red' 
              for chi in successful_results['abscos_redchi']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    ax.scatter(x_data, phase_freq_product, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.plot(x_data, phase_freq_product, 'b--', alpha=0.5, linewidth=1, label='Phase × Frequency trend')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Phase × Frequency (V·Hz)')
    ax.set_title(plot_title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Handle x-tick labels based on x_axis choice
    if x_axis == 'index' and len(successful_results) <= 20:
        folder_labels = [name.replace('LCH_charge_gate_ramsey_', '') for name in successful_results['folder_name']]
        ax.set_xticks(x_data)
        ax.set_xticklabels(folder_labels, rotation=45, ha='right', fontsize=8)
    
    # Add color legend for fit quality
    legend_elements = [
        Patch(facecolor='green', label='Good fit (χ²/dof < 1e-10)'),
        Patch(facecolor='orange', label='Fair fit (1e-10 ≤ χ²/dof < 1e-9)'),
        Patch(facecolor='red', label='Poor fit (χ²/dof ≥ 1e-9)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=9)
    
    plt.tight_layout()
    
    # Save plot if output folder is provided
    if output_folder:
        plot_path = os.path.join(output_folder, save_name)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Phase × Frequency product plot saved to: {plot_path}")
    
    return fig

def plot_frequency_values(results_df, output_folder=None, x_axis='index'):
    """
    Plot abscos_frequency values
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with analysis results
    output_folder : str, optional
        Folder to save the plot. If None, plot is not saved.
    x_axis : str, default 'index'
        Choose x-axis: 'index' for folder index or 'run_start' for relative time from first experiment
    """
    
    # Filter successful fits
    successful_results = results_df[results_df['abscos_fit_success'] == True].copy()
    
    if len(successful_results) == 0:
        print("No successful fits to plot.")
        return None
    
    # Sort by folder name to maintain consistent order
    successful_results = successful_results.sort_values('folder_name').reset_index(drop=True)
    
    # Prepare x-axis data
    if x_axis == 'run_start':
        # Convert run_start to datetime if available
        successful_results['run_start_dt'] = pd.to_datetime(successful_results['run_start'], errors='coerce')
        
        # Check if we have valid timestamps
        if successful_results['run_start_dt'].isna().all():
            print("No valid run_start timestamps found, falling back to index")
            x_axis = 'index'
        else:
            # Calculate relative time from the earliest run_start
            earliest_time = successful_results['run_start_dt'].min()
            time_deltas = successful_results['run_start_dt'] - earliest_time
            x_data = time_deltas.dt.total_seconds() / 3600  # Convert to hours
            x_label = 'Relative Time (hours from first experiment)'
            plot_title = 'AbsCos Frequency vs Relative Time'
            save_name = 'ramsey_frequency_vs_relative_time.png'
    
    if x_axis == 'index':
        x_data = np.arange(len(successful_results))
        x_label = 'Folder Index'
        plot_title = 'AbsCos Frequency vs Folder Index'
        save_name = 'ramsey_frequency_vs_index.png'
    
    frequencies = successful_results['abscos_frequency'].values
    
    # Color coding based on fit quality
    colors = ['green' if chi < 1e-10 else 'orange' if chi < 1e-9 else 'red' 
              for chi in successful_results['abscos_redchi']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    ax.scatter(x_data, frequencies, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.plot(x_data, frequencies, 'r--', alpha=0.5, linewidth=1, label='Frequency trend')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('AbsCos Frequency (Hz)')
    ax.set_title(plot_title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Handle x-tick labels based on x_axis choice
    if x_axis == 'index' and len(successful_results) <= 20:
        folder_labels = [name.replace('LCH_charge_gate_ramsey_', '') for name in successful_results['folder_name']]
        ax.set_xticks(x_data)
        ax.set_xticklabels(folder_labels, rotation=45, ha='right', fontsize=8)
    
    # Add color legend for fit quality
    legend_elements = [
        Patch(facecolor='green', label='Good fit (χ²/dof < 1e-10)'),
        Patch(facecolor='orange', label='Fair fit (1e-10 ≤ χ²/dof < 1e-9)'),
        Patch(facecolor='red', label='Poor fit (χ²/dof ≥ 1e-9)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=9)
    
    plt.tight_layout()
    
    # Save plot if output folder is provided
    if output_folder:
        plot_path = os.path.join(output_folder, save_name)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Frequency values plot saved to: {plot_path}")
    
    return fig

def plot_phase_differences(results_df, output_folder=None, x_axis='index'):
    """
    Plot phase × frequency product differences between consecutive experiments
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with analysis results
    output_folder : str, optional
        Folder to save the plot. If None, plot is not saved.
    x_axis : str, default 'index'
        Choose x-axis: 'index' for folder index or 'run_start' for relative time from first experiment
    """
    
    # Filter successful fits
    successful_results = results_df[results_df['abscos_fit_success'] == True].copy()
    
    if len(successful_results) == 0:
        print("No successful fits to plot.")
        return None
    
    if len(successful_results) < 2:
        print("Need at least 2 successful fits to calculate differences.")
        return None
    
    # Sort by folder name to maintain consistent order
    successful_results = successful_results.sort_values('folder_name').reset_index(drop=True)
    
    # Get phases and frequencies, then multiply them
    phases = successful_results['abscos_phase'].values
    frequencies = successful_results['abscos_frequency'].values
    phase_freq_product = phases * frequencies
    phase_freq_diffs = np.diff(phase_freq_product)  # Calculate i+1th - ith differences
    
    # Prepare x-axis data for difference points (use second point of each pair)
    if x_axis == 'run_start':
        # Convert run_start to datetime if available
        successful_results['run_start_dt'] = pd.to_datetime(successful_results['run_start'], errors='coerce')
        
        # Check if we have valid timestamps
        if successful_results['run_start_dt'].isna().all():
            print("No valid run_start timestamps found, falling back to index")
            x_axis = 'index'
        else:
            # Calculate relative time from the earliest run_start
            earliest_time = successful_results['run_start_dt'].min()
            time_deltas = successful_results['run_start_dt'] - earliest_time
            time_hours = time_deltas.dt.total_seconds() / 3600  # Convert to hours
            x_data = time_hours.iloc[1:]  # Use times from second point of each pair
            x_label = 'Relative Time (hours from first experiment)'
            plot_title = 'Phase × Frequency Difference vs Relative Time'
            save_name = 'ramsey_phase_freq_diff_vs_relative_time.png'
    
    if x_axis == 'index':
        x_data = np.arange(len(phase_freq_diffs))  # Indices for differences (0, 1, 2, ...)
        x_label = 'Difference Index'
        plot_title = 'Phase × Frequency Difference Between Consecutive Experiments'
        save_name = 'ramsey_phase_freq_diff_vs_index.png'
    
    # Color coding based on fit quality (use colors from the second experiment in each pair)
    colors = ['green' if chi < 1e-10 else 'orange' if chi < 1e-9 else 'red' 
              for chi in successful_results['abscos_redchi'].iloc[1:]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    ax.scatter(x_data, phase_freq_diffs, c=colors, s=60, alpha=0.7, 
               edgecolors='black', linewidth=0.5)
    ax.plot(x_data, phase_freq_diffs, 'g--', alpha=0.5, linewidth=1, label='Phase × Frequency difference trend')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)  # Reference line at y=0
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Phase × Frequency Difference (V·Hz)')
    ax.set_title(plot_title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Handle x-tick labels based on x_axis choice
    if x_axis == 'index' and len(successful_results) <= 20:
        # Use labels from the second experiment in each pair
        diff_labels = [name.replace('LCH_charge_gate_ramsey_', '') 
                      for name in successful_results['folder_name'].iloc[1:]]
        ax.set_xticks(x_data)
        ax.set_xticklabels(diff_labels, rotation=45, ha='right', fontsize=8)
    
    # Add color legend for fit quality
    legend_elements = [
        Patch(facecolor='green', label='Good fit (χ²/dof < 1e-10)'),
        Patch(facecolor='orange', label='Fair fit (1e-10 ≤ χ²/dof < 1e-9)'),
        Patch(facecolor='red', label='Poor fit (χ²/dof ≥ 1e-9)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=9)
    
    plt.tight_layout()
    
    # Save plot if output folder is provided
    if output_folder:
        plot_path = os.path.join(output_folder, save_name)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Phase × Frequency differences plot saved to: {plot_path}")
    
    return fig

def plot_2d_frequency_map(results_df, output_folder=None):
    """
    Plot 2D colormap with merged_freqs as z-axis, charge_gates vs folder_index
    
    This function re-analyzes each successful dataset to extract the raw frequency data
    used in the abscos fitting process.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with analysis results
    output_folder : str, optional
        Folder to save the plot. If None, plot is not saved.
    """
    
    # Filter successful fits and sort by folder name
    successful_results = results_df[results_df['abscos_fit_success'] == True].copy()
    successful_results = successful_results.sort_values('folder_name').reset_index(drop=True)
    
    if len(successful_results) == 0:
        print("No successful fits to create 2D frequency map.")
        return None
    
    print(f"Creating 2D frequency map for {len(successful_results)} datasets...")
    
    # Collect all charge gate and frequency data
    all_charge_gates = []
    all_frequencies = []
    folder_indices = []
    
    for idx, row in successful_results.iterrows():
        folder_path = row['folder_path']
        ds_path = os.path.join(folder_path, 'ds_raw.h5')
        
        try:
            # Re-load and analyze to get raw frequency data
            ds = load_xarray_h5(ds_path)
            sep_data = repetition_data(ds, repetition_dim="qubit")
            
            for sqdata in sep_data:
                # Rename signal column
                if "I" in sqdata:
                    sqdata = sqdata.rename({"I": "signal"})
                elif "state" in sqdata:
                    sqdata = sqdata.rename({"state": "signal"})
                else:
                    data_vars = list(sqdata.data_vars.keys())
                    signal_candidates = [var for var in data_vars if var.lower() in ['signal', 'i', 'q', 'state']]
                    if signal_candidates:
                        sqdata = sqdata.rename({signal_candidates[0]: "signal"})
                
                # Perform analysis to get merged frequency data
                analysis = ChargeGateRamseyAnalysis(sqdata)
                analysis.all_ave_freq = 0.25e-3
                analysis.fixed_frequency = 0.545  # Set fixed frequency for consistency
                analysis._get_frequency()
                
                # Extract the same merged data used in _fit_abscos
                charge_gates = analysis.fit_results_dataset.coords['charge_gate'].values
                f1_vals = analysis.fit_results_dataset['f1'].values
                f2_vals = analysis.fit_results_dataset['f2'].values
                
                # Merge f1 and f2 data points (same logic as in _fit_abscos)
                merged_charge_gates = []
                merged_freqs = []
                
                # Add valid f1 points
                valid_f1 = ~np.isnan(f1_vals)
                if np.any(valid_f1):
                    merged_charge_gates.extend(charge_gates[valid_f1])
                    merged_freqs.extend(f1_vals[valid_f1])
                
                # Add valid f2 points
                valid_f2 = ~np.isnan(f2_vals)
                if np.any(valid_f2):
                    merged_charge_gates.extend(charge_gates[valid_f2])
                    merged_freqs.extend(f2_vals[valid_f2])
                
                if len(merged_freqs) > 0:
                    merged_charge_gates = np.array(merged_charge_gates)
                    merged_freqs = np.abs(np.array(merged_freqs) - analysis.all_ave_freq)
                    
                    # Store data with folder index
                    all_charge_gates.extend(merged_charge_gates)
                    all_frequencies.extend(merged_freqs)
                    folder_indices.extend([idx] * len(merged_charge_gates))
                
                break  # Only process first qubit
                
        except Exception as e:
            print(f"Error processing {row['folder_name']} for 2D map: {e}")
            continue
    
    if len(all_frequencies) == 0:
        print("No frequency data available for 2D map.")
        return None
    
    # Convert to numpy arrays
    all_charge_gates = np.array(all_charge_gates)
    all_frequencies = np.array(all_frequencies)
    folder_indices = np.array(folder_indices)
    
    # Create meshgrid for 2D heatmap
    # Determine unique charge gate values and folder indices
    unique_charge_gates = np.unique(all_charge_gates)
    unique_folder_indices = np.unique(folder_indices)
    
    # Create 2D grid
    charge_grid, folder_grid = np.meshgrid(unique_charge_gates, unique_folder_indices)
    freq_grid = np.full_like(charge_grid, np.nan, dtype=float)
    
    # Fill the grid with frequency values
    for i, folder_idx in enumerate(unique_folder_indices):
        for j, charge_gate in enumerate(unique_charge_gates):
            # Find matching data points
            mask = (folder_indices == folder_idx) & (np.abs(all_charge_gates - charge_gate) < 1e-10)
            if np.any(mask):
                # If multiple points exist at same location, take the mean
                freq_grid[i, j] = np.mean(all_frequencies[mask])
    
    # Create single plot with 2D heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot 2D frequency heatmap as background
    im = ax.pcolormesh(charge_grid, folder_grid, freq_grid, shading='auto', cmap='viridis', alpha=0.8)
    
    # Overlay phase values as horizontal lines on the same axis
    # Get phase data for the same successful results
    phases = successful_results['abscos_phase'].values
    indices = np.arange(len(successful_results))
    
    # Plot phase values as horizontal lines across the charge gate range
    charge_range = [charge_grid.min(), charge_grid.max()]
    colors = ['red' if chi < 2 else 'orange' if chi < 5 else 'yellow' 
              for chi in successful_results['abscos_redchi']]
    
    # Scale phase values to fit within charge gate range for visualization
    phase_min, phase_max = phases.min(), phases.max()
    charge_min, charge_max = charge_range[0], charge_range[1]
    charge_span = charge_max - charge_min
    
    # Plot horizontal lines and markers for each folder index
    for i, (original_phase, color) in enumerate(zip(phases, colors)):
        # Add marker at the mapped phase position
        ax.scatter(original_phase, i, c=color, s=80, alpha=1.0, 
                  edgecolors='black', linewidth=2, marker='o', zorder=11)
    
    # Connect phase points with a line
    ax.plot(phases, indices, 'white', alpha=0.8, linewidth=2, zorder=9, 
            linestyle='--', label='Phase trend')
    
    # Add colorbar for frequency data
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|Frequency - Ave_Freq| (Hz)', rotation=270, labelpad=20)
    
    # Customize plot
    ax.set_xlabel('Charge Gate (V)')
    ax.set_ylabel('Folder Index')
    ax.set_title('Ramsey 2D Frequency Map with Phase Overlay')
    ax.grid(True, alpha=0.3)
    
    # Set y-ticks to folder indices
    index_range = np.arange(len(successful_results))
    ax.set_yticks(index_range)
    if len(successful_results) <= 15:
        folder_labels = [name.replace('LCH_charge_gate_ramsey_', '') for name in successful_results['folder_name']]
        ax.set_yticklabels(folder_labels, fontsize=8)
    
    # Add legend for phase quality and phase scale
    legend_elements = [
        Patch(facecolor='red', label='Good fit (χ²/dof < 2)'),
        Patch(facecolor='orange', label='Fair fit (2 ≤ χ²/dof < 5)'),
        Patch(facecolor='yellow', label='Poor fit (χ²/dof ≥ 5)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.99))
    
    # Add text box showing phase scale mapping
    textstr = f'Phase range: {phase_min:.4f} to {phase_max:.4f} V\\nMapped to charge gate positions'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.99, 0.01, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if output folder is provided
    if output_folder:
        plot_path = os.path.join(output_folder, 'ramsey_2d_frequency_map.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"2D frequency map saved to: {plot_path}")
    
    return fig


def plot_phase_histogram(results_df, output_folder=None, bins=20):
    """
    Plot histogram of phase × frequency product values
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with analysis results
    output_folder : str, optional
        Folder to save the plot. If None, plot is not saved.
    bins : int, default 20
        Number of histogram bins
        
    Returns:
    --------
    fig : matplotlib figure
        The histogram figure
    """
    
    # Filter successful fits
    successful_results = results_df[results_df['abscos_fit_success'] == True].copy()
    
    if len(successful_results) == 0:
        print("No successful fits to plot histogram.")
        return None
    
    # Get phase and frequency values, then multiply them
    phases = successful_results['abscos_phase'].values
    frequencies = successful_results['abscos_frequency'].values
    phase_freq_product = phases * frequencies
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create histogram
    n, bins_edges, patches = ax.hist(phase_freq_product, bins=bins, alpha=0.7, color='skyblue', 
                                    edgecolor='black', linewidth=1)
    
    # Color bars based on fit quality
    chi_values = successful_results['abscos_redchi'].values
    
    # Create color mapping for each bin based on the data points it contains
    for i, (product_val, chi_val) in enumerate(zip(phase_freq_product, chi_values)):
        # Find which bin this product value belongs to
        bin_idx = np.digitize(product_val, bins_edges) - 1
        bin_idx = max(0, min(bin_idx, len(patches) - 1))  # Ensure valid index
        
        # Color based on fit quality
        if chi_val < 2:
            color = 'green'
        elif chi_val < 5:
            color = 'orange'  
        else:
            color = 'red'
        
        # Update patch color if it's not already colored or if this is better quality
        current_color = patches[bin_idx].get_facecolor()
        if current_color == (0.529411764705882, 0.8078431372549019, 0.9803921568627451, 0.7):  # skyblue
            patches[bin_idx].set_facecolor(color)
        elif current_color == (1.0, 0.6470588235294118, 0.0, 1.0) and color == 'green':  # orange -> green
            patches[bin_idx].set_facecolor(color)
    
    # Add statistics
    mean_product = phase_freq_product.mean()
    std_product = phase_freq_product.std()
    median_product = np.median(phase_freq_product)
    
    # Customize plot
    ax.set_xlabel('Gate Charge', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Histogram of Gate Charge\\n({len(successful_results)} successful fits)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if output folder is provided
    if output_folder:
        plot_path = os.path.join(output_folder, 'ramsey_phase_freq_product_histogram.png')
        fig.savefig(plot_path, dpi=150)
        print(f"Phase × Frequency histogram saved to: {plot_path}")
    
    return fig