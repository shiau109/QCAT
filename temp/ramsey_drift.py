import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
from qcat.analysis.charge_gate_ramsey.analysis import ChargeGateRamseyAnalysis

def analyze_ramsey_folders(root_folder):
    """
    Analyze all subfolders with 'LCH_charge_gate_ramsey' pattern and extract abscos_phase values.
    
    Parameters:
    -----------
    root_folder : str
        Root folder containing subfolders to analyze
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with folder names and abscos_phase values
    """
    
    # Find all subfolders with the pattern
    pattern = os.path.join(root_folder, "*LCH_charge_gate_ramsey*")
    ramsey_folders = glob.glob(pattern)
    
    print(f"Found {len(ramsey_folders)} folders matching pattern:")
    for folder in ramsey_folders:
        print(f"  - {os.path.basename(folder)}")
    
    results = []
    
    for folder_path in ramsey_folders:
        folder_name = os.path.basename(folder_path)
        ds_path = os.path.join(folder_path, 'ds_raw.h5')
        
        # Check if ds_raw.h5 exists
        if not os.path.exists(ds_path):
            print(f"Warning: ds_raw.h5 not found in {folder_name}")
            results.append({
                'folder_name': folder_name,
                'folder_path': folder_path,
                'abscos_phase': np.nan,
                'abscos_amplitude': np.nan,
                'abscos_frequency': np.nan,
                'abscos_fit_success': False,
                'abscos_redchi': np.nan,
                'status': 'ds_raw.h5 not found'
            })
            continue
        
        try:
            # Load and analyze data
            print(f"\nAnalyzing {folder_name}...")
            ds = load_xarray_h5(ds_path)
            sep_data = repetition_data(ds, repetition_dim="qubit")
            
            for sqdata in sep_data:
                qubit_name = sqdata["qubit"].values.item()
                print(f"  Qubit: {qubit_name}")
                
                # Rename signal column (try different possible names)
                if "I" in sqdata:
                    sqdata = sqdata.rename({"I": "signal"})
                elif "state" in sqdata:
                    sqdata = sqdata.rename({"state": "signal"})
                else:
                    # Look for other possible signal columns
                    data_vars = list(sqdata.data_vars.keys())
                    signal_candidates = [var for var in data_vars if var.lower() in ['signal', 'i', 'q', 'state']]
                    if signal_candidates:
                        sqdata = sqdata.rename({signal_candidates[0]: "signal"})
                    else:
                        raise ValueError(f"No suitable signal column found in data variables: {data_vars}")
                
                # Perform analysis
                analysis = ChargeGateRamseyAnalysis(sqdata)
                analysis.all_ave_freq = 0.25e-3  # Set average frequency as requested
                analysis.fixed_frequency = 0.54 # Set fixed frequency for abscos fitting
                analysis._start_analysis()
                
                # Extract fit results
                fit_success = analysis.fit_results_dataset.attrs.get('abscos_fit_success', False)
                
                result_entry = {
                    'folder_name': folder_name,
                    'folder_path': folder_path,
                    'qubit_name': qubit_name,
                    'abscos_phase': analysis.fit_results_dataset.attrs.get('abscos_phase', np.nan),
                    'abscos_amplitude': analysis.fit_results_dataset.attrs.get('abscos_amplitude', np.nan),
                    'abscos_frequency': analysis.fit_results_dataset.attrs.get('abscos_frequency', np.nan),
                    'abscos_fit_success': fit_success,
                    'abscos_redchi': analysis.fit_results_dataset.attrs.get('abscos_redchi', np.nan),
                    'status': 'success' if fit_success else 'fit_failed'
                }
                
                results.append(result_entry)
                
                print(f"    Fit success: {fit_success}")
                if fit_success:
                    print(f"    Phase: {result_entry['abscos_phase']:.4f}")
                    print(f"    Amplitude: {result_entry['abscos_amplitude']:.6f}")
                    print(f"    Frequency: {result_entry['abscos_frequency']:.4f}")
                    print(f"    Reduced χ²: {result_entry['abscos_redchi']:.4f}")
                
        except Exception as e:
            print(f"Error analyzing {folder_name}: {str(e)}")
            results.append({
                'folder_name': folder_name,
                'folder_path': folder_path,
                'qubit_name': 'unknown',
                'abscos_phase': np.nan,
                'abscos_amplitude': np.nan,
                'abscos_frequency': np.nan,
                'abscos_fit_success': False,
                'abscos_redchi': np.nan,
                'status': f'error: {str(e)}'
            })
    
    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    return results_df

def save_results(results_df, output_path):
    """Save results to CSV file"""
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

def plot_phase_vs_index(results_df, output_folder):
    """Plot abscos_phase as a function of folder index"""
    
    # Filter successful fits
    successful_results = results_df[results_df['abscos_fit_success'] == True].copy()
    
    if len(successful_results) == 0:
        print("No successful fits to plot.")
        return None
    
    # Sort by folder name to maintain consistent order
    successful_results = successful_results.sort_values('folder_name').reset_index(drop=True)
    
    # Create three subplots vertically
    fig, (ax_phase, ax_freq, ax_diff) = plt.subplots(3, 1, figsize=(16, 10))
    
    # Common data
    indices = np.arange(len(successful_results))
    phases = successful_results['abscos_phase'].values
    frequencies = successful_results['abscos_frequency'].values
    
    # Color coding based on fit quality
    colors = ['green' if chi < 1e-10 else 'orange' if chi < 1e-9 else 'red' 
              for chi in successful_results['abscos_redchi']]
    
    # First subplot: Phase values
    ax_phase.scatter(indices, phases, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax_phase.plot(indices, phases, 'b--', alpha=0.5, linewidth=1, label='Phase trend')
    
    ax_phase.set_xlabel('Folder Index')
    ax_phase.set_ylabel('AbsCos Phase (V)')
    ax_phase.set_title('AbsCos Phase vs Folder Index')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend()
    
    # Second subplot: Frequency values
    ax_freq.scatter(indices, frequencies, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax_freq.plot(indices, frequencies, 'r--', alpha=0.5, linewidth=1, label='Frequency trend')
    
    ax_freq.set_xlabel('Folder Index')
    ax_freq.set_ylabel('AbsCos Frequency (Hz)')
    ax_freq.set_title('AbsCos Frequency vs Folder Index')
    ax_freq.grid(True, alpha=0.3)
    ax_freq.legend()
    
    # Third subplot: Phase differences between consecutive experiments
    if len(phases) > 1:
        phase_diffs = np.diff(phases)  # Calculate i+1th - ith differences
        diff_indices = indices[:-1]  # Indices for differences (starting from 1)
        diff_colors = colors[:-1]  # Use colors from the second experiment in each pair
        
        ax_diff.scatter(diff_indices, phase_diffs, c=diff_colors, s=60, alpha=0.7, 
                       edgecolors='black', linewidth=0.5)
        ax_diff.plot(diff_indices, phase_diffs, 'g--', alpha=0.5, linewidth=1, label='Difference trend')
        ax_diff.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)  # Reference line at y=0
        
        ax_diff.set_xlabel('Folder Index')
        ax_diff.set_ylabel('Phase Difference (V)')
        ax_diff.set_title('Phase Difference Between Consecutive Experiments')
        ax_diff.grid(True, alpha=0.3)
        ax_diff.legend()
    else:
        ax_diff.text(0.5, 0.5, 'Not enough data points for differences', 
                    transform=ax_diff.transAxes, ha='center', va='center')
        ax_diff.set_title('Phase Difference Between Consecutive Experiments')
        ax_diff.grid(True, alpha=0.3)
    
    # Add folder names as x-tick labels for all subplots
    if len(successful_results) <= 20:  # Only show labels if not too many
        folder_labels = [name.replace('LCH_charge_gate_ramsey_', '') for name in successful_results['folder_name']]
        
        # Set labels for phase subplot
        ax_phase.set_xticks(indices)
        ax_phase.set_xticklabels(folder_labels, rotation=45, ha='right', fontsize=8)
        
        # Set labels for frequency subplot
        ax_freq.set_xticks(indices)
        ax_freq.set_xticklabels(folder_labels, rotation=45, ha='right', fontsize=8)
        
        # Set labels for difference subplot (only for indices where differences exist)
        if len(phases) > 1:
            diff_labels = folder_labels[1:]  # Labels for difference points
            ax_diff.set_xticks(diff_indices)
            ax_diff.set_xticklabels(diff_labels, rotation=45, ha='right', fontsize=8)
    else:
        # Reduced tick marks for many experiments
        tick_indices = indices[::max(1, len(indices)//10)]
        ax_phase.set_xticks(tick_indices)
        ax_freq.set_xticks(tick_indices)
        if len(phases) > 1:
            diff_tick_indices = diff_indices[::max(1, len(diff_indices)//10)]
            ax_diff.set_xticks(diff_tick_indices)
    
    # Add color legend for fit quality (positioned on the bottom subplot)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Good fit (χ²/dof < 2)'),
        Patch(facecolor='orange', label='Fair fit (2 ≤ χ²/dof < 5)'),
        Patch(facecolor='red', label='Poor fit (χ²/dof ≥ 5)')
    ]
    # Add legend to the bottom subplot
    ax_diff.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_folder, 'ramsey_phase_vs_index.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Phase plot saved to: {plot_path}")
    
    return fig

def plot_2d_frequency_map(results_df, output_folder):
    """
    Plot 2D colormap with merged_freqs as z-axis, charge_gates vs folder_index
    
    This function re-analyzes each successful dataset to extract the raw frequency data
    used in the abscos fitting process.
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
    
    # Map phases to charge gate positions (normalized then scaled)
    if phase_max != phase_min:
        normalized_phases = (phases - phase_min) / (phase_max - phase_min)
        mapped_phases = charge_min + normalized_phases * charge_span * 0.8 + charge_span * 0.1  # Use 80% of range with 10% margins
    else:
        mapped_phases = np.full_like(phases, (charge_min + charge_max) / 2)
    
    # Plot horizontal lines and markers for each folder index
    for i, (mapped_phase, original_phase, color) in enumerate(zip(mapped_phases, phases, colors)):
        # Draw horizontal line spanning a portion of the charge gate range centered on mapped phase
        line_span = charge_span * 0.05  # 5% of charge range
        ax.plot([mapped_phase - line_span/2, mapped_phase + line_span/2], [i, i], 
                color=color, linewidth=4, alpha=0.9, zorder=10)
        # Add marker at the mapped phase position
        ax.scatter(mapped_phase, i, c=color, s=80, alpha=1.0, 
                  edgecolors='black', linewidth=2, marker='o', zorder=11)
    
    # Connect phase points with a line
    ax.plot(mapped_phases, indices, 'white', alpha=0.8, linewidth=2, zorder=9, 
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
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Good fit (χ²/dof < 2)'),
        Patch(facecolor='orange', label='Fair fit (2 ≤ χ²/dof < 5)'),
        Patch(facecolor='yellow', label='Poor fit (χ²/dof ≥ 5)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.99))
    
    # Add text box showing phase scale mapping
    textstr = f'Phase range: {phase_min:.4f} to {phase_max:.4f} V\nMapped to charge gate positions'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.99, 0.01, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_folder, 'ramsey_2d_frequency_map.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"2D frequency map saved to: {plot_path}")
    
    return fig

if __name__ == "__main__":
    # Set the root folder path
    root_folder = r"D:\data\MIST\20251124\LCH_graph_charge_gate_r_rp\prepare_0"
    
    # Analyze all folders
    print("Starting Ramsey analysis for all subfolders...")
    results_df = analyze_ramsey_folders(root_folder)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total folders analyzed: {len(results_df)}")
    print(f"Successful fits: {results_df['abscos_fit_success'].sum()}")
    print(f"Failed fits: {(~results_df['abscos_fit_success']).sum()}")
    
    # Show successful results
    successful_results = results_df[results_df['abscos_fit_success'] == True]
    if len(successful_results) > 0:
        print(f"\nSuccessful fit results:")
        print(f"Phase values (mean ± std): {successful_results['abscos_phase'].mean():.4f} ± {successful_results['abscos_phase'].std():.4f}")
        print(f"Phase range: {successful_results['abscos_phase'].min():.4f} to {successful_results['abscos_phase'].max():.4f}")
        
        print(f"\nDetailed results:")
        for _, row in successful_results.iterrows():
            print(f"  {row['folder_name']}: phase = {row['abscos_phase']:.4f}, χ²/dof = {row['abscos_redchi']:.3f}")
    
    # Save results
    output_path = os.path.join(root_folder, 'ramsey_analysis_results.csv')
    save_results(results_df, output_path)
    
    # Plot phase vs index
    print(f"\nCreating phase vs index plot...")
    fig1 = plot_phase_vs_index(results_df, root_folder)
    
    # Plot 2D frequency map
    print(f"\nCreating 2D frequency map...")
    fig2 = plot_2d_frequency_map(results_df, root_folder)
    
    # Show plots
    if fig1 is not None or fig2 is not None:
        plt.show()
    
    print(f"\nAnalysis complete!")