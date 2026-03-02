import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
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
        node_json_path = os.path.join(folder_path, 'node.json')
        
        # Read node.json metadata
        run_start = None
        run_end = None
        if os.path.exists(node_json_path):
            try:
                with open(node_json_path, 'r') as f:
                    node_data = json.load(f)
                    run_start = node_data.get('metadata', {}).get('run_start')
                    run_end = node_data.get('metadata', {}).get('run_end')
            except Exception as e:
                print(f"Warning: Could not read node.json in {folder_name}: {str(e)}")
        
        # Check if ds_raw.h5 exists
        if not os.path.exists(ds_path):
            print(f"Warning: ds_raw.h5 not found in {folder_name}")
            results.append({
                'folder_name': folder_name,
                'run_start': run_start,
                'run_end': run_end,
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
                # print(f"  Qubit: {qubit_name}")
                
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
                    'qubit_name': qubit_name,
                    'run_start': run_start,
                    'run_end': run_end,
                    'abscos_phase': analysis.fit_results_dataset.attrs.get('abscos_phase', np.nan),
                    'abscos_amplitude': analysis.fit_results_dataset.attrs.get('abscos_amplitude', np.nan),
                    'abscos_frequency': analysis.fit_results_dataset.attrs.get('abscos_frequency', np.nan),
                    'abscos_fit_success': fit_success,
                    'abscos_redchi': analysis.fit_results_dataset.attrs.get('abscos_redchi', np.nan),
                    'status': 'success' if fit_success else 'fit_failed'
                }
                
                results.append(result_entry)
                

                
        except Exception as e:
            print(f"Error analyzing {folder_name}: {str(e)}")
            results.append({
                'folder_name': folder_name,
                'qubit_name': 'unknown',
                'run_start': run_start,
                'run_end': run_end,
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

def save_results(results_df, output_folder, root_folder):
    """Save results to netCDF formats"""

    
    # Convert DataFrame to xarray Dataset and save as netCDF
    ds = xr.Dataset.from_dataframe(results_df)
    # Store root_folder as a global attribute
    ds.attrs['root_folder'] = root_folder
    h5_path = os.path.join(output_folder, 'ramsey_analysis_results.h5')
    ds.to_netcdf(h5_path)
    print(f"Results saved to netCDF: {h5_path}")



if __name__ == "__main__":
    # Set the root folder path
    root_folder = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\MIST\20251201\r_9_150x50_50_s300_ro_005x18_s100_fb\set_5"
    
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
    save_results(results_df, root_folder, root_folder)
    
    print(f"\nAnalysis complete!")