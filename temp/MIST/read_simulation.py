import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_hyper_data(data, data_key, save_path=None, figsize=(10, 8), cmap='viridis', xlim=None, ylim=None):
    """
    Plot Hyper_data (0 or 1) as a 2D colormap.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the pickle data with hyper data, 'n_g', and 'chi_ac_over_f01_values'
    data_key : str
        Key for the hyper data to plot ('Hyper_data_0' or 'Hyper_data_1')
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10, 8).
    cmap : str, optional
        Colormap for the plot. Default is 'viridis'.
    xlim : tuple, optional
        X-axis (n_g) limits as (min, max). If None, uses full data range.
    ylim : tuple, optional
        Y-axis (chi_ac_over_f01_values) limits as (min, max). If None, uses full data range.
    """
    
    if data_key not in data:
        print(f"Error: '{data_key}' not found in data")
        return None
        
    hyper_data = data[data_key]
    # Correct axis interpretation: shape[0] -> chi_ac_over_f01, shape[1] -> n_g
    chi_ac_over_f01 = data.get('chi_ac_over_f01_values', np.arange(hyper_data.shape[0]))
    n_g = data.get('n_g', np.arange(hyper_data.shape[1]))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the extent for proper axis scaling
    extent = [n_g[0], n_g[-1], chi_ac_over_f01[0], chi_ac_over_f01[-1]]
    
    # Plot the 2D data (no transpose needed now) with log scale
    im = ax.imshow(hyper_data, aspect='auto', origin='lower', 
                   extent=extent, cmap=cmap, interpolation='nearest',
                   norm=LogNorm(vmin=1e-6, vmax=hyper_data.max()))
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=f'{data_key} values (log scale)')
    
    # Set labels and title
    ax.set_xlabel('n_g')
    ax.set_ylabel('chi_ac_over_f01_values')
    ax.set_title(f'{data_key} Visualization')
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    fig.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax


def create_mirrored_hyper_data(data, data_key):
    """
    Create a mirrored version of hyper data along the n_g axis and add with original.
    The mirror axis is at the middle of n_g, creating symmetric data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the pickle data
    data_key : str
        Key for the hyper data ('Hyper_data_0' or 'Hyper_data_1')
        
    Returns:
    --------
    dict : Modified data dictionary with mirrored+original hyper data using original coordinates
    """
    if data_key not in data:
        print(f"Error: '{data_key}' not found in data")
        return None
        
    # Get the original data
    original_hyper = data[data_key].copy()
    
    # Create mirrored version along n_g axis (flip along axis=1 which is n_g)
    mirrored_hyper = np.flip(original_hyper, axis=1)
    
    # Add original and mirrored data element-wise
    combined_hyper = original_hyper + mirrored_hyper
    
    # Create new data dictionary with combined data using original coordinates
    new_data = data.copy()
    new_data[f'{data_key}_mirrored'] = combined_hyper
    # Use original n_g coordinates (no extension needed)
    
    return new_data


def plot_mirrored_hyper_data(data, data_key, save_path=None, figsize=(10, 8), cmap='viridis', xlim=None, ylim=None):
    """
    Plot mirrored hyper data using original coordinates.
    The data is the sum of original + mirrored, creating symmetric patterns.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the mirrored data
    data_key : str
        Base key for the hyper data ('Hyper_data_0' or 'Hyper_data_1')
    save_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10, 8).
    cmap : str, optional
        Colormap for the plot. Default is 'viridis'.
    xlim : tuple, optional
        X-axis (n_g) limits as (min, max). If None, uses full data range.
    ylim : tuple, optional
        Y-axis (chi_ac_over_f01_values) limits as (min, max). If None, uses full data range.
    """
    mirrored_key = f'{data_key}_mirrored'
    
    if mirrored_key not in data:
        print(f"Error: '{mirrored_key}' not found in data")
        return None
        
    hyper_data = data[mirrored_key]
    # Use original coordinates (no extension)
    chi_ac_over_f01 = data.get('chi_ac_over_f01_values', np.arange(hyper_data.shape[0]))
    n_g = data.get('n_g', np.arange(hyper_data.shape[1]))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the extent for proper axis scaling
    extent = [n_g[0], n_g[-1], chi_ac_over_f01[0], chi_ac_over_f01[-1]]
    
    # Plot the 2D data (no transpose needed now) with log scale
    im = ax.imshow(hyper_data, aspect='auto', origin='lower', 
                   extent=extent, cmap=cmap, interpolation='nearest',
                   norm=LogNorm(vmin=1e-6, vmax=hyper_data.max()))
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=f'{data_key} + mirrored values (log scale)')
    
    # Set labels and title
    ax.set_xlabel('n_g')
    ax.set_ylabel('chi_ac_over_f01_values')
    ax.set_title(f'{data_key} + Mirrored (Symmetric) Visualization')
    
    # Add vertical line at the middle of n_g to show the mirror axis
    n_g_middle = (n_g[0] + n_g[-1]) / 2
    ax.axvline(x=n_g_middle, color='white', linestyle='--', alpha=0.7, linewidth=2, label='Mirror axis')
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add legend for the mirror axis line
    ax.legend()
    
    # Tight layout
    fig.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Mirrored plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax



if __name__ == "__main__":
# Path to the pickle file
    pickle_file_path = r"d:\data\MIST\20251201\LCH_q1_data.pkl"
    norm_ac_shift = 166. /4912.0  # Example normalization factor shift/f_ro
    print(f"Normalization factor for AC shift: {norm_ac_shift}")
    # Check if file exists
    if os.path.exists(pickle_file_path):
        try:
            # Open and load the pickle file
            with open(pickle_file_path, 'rb') as file:
                data = pickle.load(file)
            
            print(f"Successfully loaded pickle file: {pickle_file_path}")
            print(f"Data type: {type(data)}")
            
            # Print basic information about the loaded data
            if hasattr(data, 'shape'):
                print(f"Data shape: {data.shape}")
            elif hasattr(data, '__len__'):
                print(f"Data length: {len(data)}")
            
            # If it's a dictionary, show keys
            if isinstance(data, dict):
                print(f"Dictionary keys: {list(data.keys())}")
            
            # Display first few elements or structure
            print("\nData preview:")
            if isinstance(data, dict):
                for key, value in list(data.items())[:5]:  # Show first 5 items
                    print(f"  {key}: {type(value)} - {str(value)[:100]}...")
            elif hasattr(data, '__iter__') and not isinstance(data, str):
                try:
                    for i, item in enumerate(data):
                        if i >= 5:  # Show first 5 items
                            print(f"  ... (showing first 5 items)")
                            break
                        print(f"  [{i}]: {type(item)} - {str(item)[:100]}...")
                except:
                    print(f"  {str(data)[:500]}...")
            else:
                print(f"  {str(data)[:500]}...")
            print(data["Ej"])
            print(data["Ec"])
            # Detailed analysis of Hyper_data_0
            if isinstance(data, dict) and 'Hyper_data_0' in data:
                print("\n=== Detailed analysis of Hyper_data_0 ===")
                hyper_data_0 = data['Hyper_data_0']
                print(f"Type: {type(hyper_data_0)}")
                if hasattr(hyper_data_0, 'shape'):
                    print(f"Shape: {hyper_data_0.shape}")
                if hasattr(hyper_data_0, 'dtype'):
                    print(f"Data type: {hyper_data_0.dtype}")
                
                # Check if it's a numpy array and show some statistics
                if isinstance(hyper_data_0, np.ndarray):
                    print(f"Min value: {np.min(hyper_data_0)}")
                    print(f"Max value: {np.max(hyper_data_0)}")
                    print(f"Mean value: {np.mean(hyper_data_0)}")
                    print(f"Array dimensions: {hyper_data_0.ndim}")
                    
                    # Show coordinate arrays if they exist
                    if 'n_g' in data:
                        print(f"n_g shape: {data['n_g'].shape}")
                        print(f"n_g range: {data['n_g'].min()} to {data['n_g'].max()}")
                    if 'chi_ac_over_f01_values' in data:
                        print(f"chi_ac_over_f01_values shape: {data['chi_ac_over_f01_values'].shape}")
                        print(f"chi_ac_over_f01_values range: {data['chi_ac_over_f01_values'].min()} to {data['chi_ac_over_f01_values'].max()}")

            # Call the plotting function for both Hyper_data_0 and Hyper_data_1
            pickle_dir = os.path.dirname(pickle_file_path)
            
            if isinstance(data, dict) and 'Hyper_data_0' in data:
                print("\n=== Plotting Hyper_data_0 ===")
                save_path_0 = os.path.join(pickle_dir, 'hyper_data_0_plot.png')
                plot_hyper_data(data, 'Hyper_data_0', save_path=save_path_0, ylim=(norm_ac_shift*0.1, norm_ac_shift*1.1))

                
                # Create and plot mirrored version
                print("\n=== Creating and Plotting Mirrored Hyper_data_0 ===")
                mirrored_data = create_mirrored_hyper_data(data, 'Hyper_data_0')
                if mirrored_data:
                    save_path_0_mirrored = os.path.join(pickle_dir, 'hyper_data_0_mirrored_plot.png')
                    plot_mirrored_hyper_data(mirrored_data, 'Hyper_data_0', save_path=save_path_0_mirrored, ylim=(norm_ac_shift*0.1, norm_ac_shift*1.1))
                    
                    # Example with custom limits for mirrored plot
                    save_path_0_mirrored_limited = os.path.join(pickle_dir, 'hyper_data_0_mirrored_limited_plot.png')
                    plot_mirrored_hyper_data(mirrored_data, 'Hyper_data_0', save_path=save_path_0_mirrored_limited,
                                           xlim=(0, 0.25), ylim=(norm_ac_shift*0.1, norm_ac_shift*1.1))
                
            if isinstance(data, dict) and 'Hyper_data_1' in data:
                print("\n=== Plotting Hyper_data_1 ===")
                save_path_1 = os.path.join(pickle_dir, 'hyper_data_1_plot.png')
                plot_hyper_data(data, 'Hyper_data_1', save_path=save_path_1, ylim=(norm_ac_shift*0.1, norm_ac_shift*1.1))
                 # Example with custom axis limits
   
                # Create and plot mirrored version
                print("\n=== Creating and Plotting Mirrored Hyper_data_1 ===")
                mirrored_data = create_mirrored_hyper_data(data, 'Hyper_data_1')
                if mirrored_data:
                    save_path_1_mirrored = os.path.join(pickle_dir, 'hyper_data_1_mirrored_plot.png')
                    plot_mirrored_hyper_data(mirrored_data, 'Hyper_data_1', save_path=save_path_1_mirrored, ylim=(norm_ac_shift*0.1, norm_ac_shift*1.1))
                     # Example with custom limits for mirrored plot
                    save_path_1_mirrored_limited = os.path.join(pickle_dir, 'hyper_data_1_mirrored_limited_plot.png')
                    plot_mirrored_hyper_data(mirrored_data, 'Hyper_data_1', save_path=save_path_1_mirrored_limited,
                                           xlim=(0, 0.25), ylim=(norm_ac_shift*0.1, norm_ac_shift*1.1))               
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            print(f"Error type: {type(e).__name__}")
            
    else:
        print(f"File not found: {pickle_file_path}")
        print("Please check the file path and make sure the file exists.")