import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
def plot_raw_2d_colormap( rawdata:xr.DataArray ):
    # Plot raw data as 2D color map (charge_gate vs idle_time)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract coordinates
    idle_time = rawdata.coords['idle_time'].values
    charge_gate = rawdata.coords['charge_gate'].values
    
    # Convert idle_time to microseconds for better readability
    idle_time_us = idle_time / 1000.0  # Assuming idle_time is in ns
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(idle_time_us, charge_gate)
    
    # Create the 2D color map
    im = ax.pcolormesh(X, Y, rawdata.values, 
                      shading='auto', 
                      cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label(f'{rawdata.name}', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('Idle Time (μs)')
    ax.set_ylabel('Charge Gate (V)')
    
    # Get qubit name for title if available
    qubit_name = rawdata.coords.get('qubit', 'Unknown')
    ax.set_title(f' {qubit_name} Charge Gate Ramsey')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Tight layout to prevent label cutoff
    fig.tight_layout()
    
    return fig

def plot_2d_spectrum( fftdata:xr.Dataset, analysis_result=None ):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # Extract spectrum DataArray from the Dataset
    spectrum_da = fftdata['spectrum']
    
    # Extract coordinates
    frequencies = spectrum_da.coords['frequency'].values
    charge_gates = spectrum_da.coords['charge_gate'].values
    
    # Convert frequency to more readable units (MHz if needed)
    # Assuming frequencies are in Hz, convert to MHz for better readability
    freq_mhz = frequencies * 1e3
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(freq_mhz, charge_gates)
    
    # Create the 2D FFT spectrum color map
    im = ax.pcolormesh(X, Y, spectrum_da.values, 
                      shading='auto', 
                      cmap='plasma')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spectrum Amplitude', rotation=270, labelpad=20)
    
    # Overlay fit results if available
    if analysis_result is not None:
        fit_charge_gates = analysis_result.coords['charge_gate'].values
        
        # Plot f1 frequencies
        f1_values = analysis_result['f1'].values
        valid_f1 = ~np.isnan(f1_values)
        if np.any(valid_f1):
            f1_mhz = f1_values[valid_f1] * 1e3  # Convert GHz to MHz
            ax.plot(f1_mhz, fit_charge_gates[valid_f1], 'bo', 
                   markersize=4,  label='f1', alpha=0.8)
        
        # Plot f2 frequencies
        f2_values = analysis_result['f2'].values
        valid_f2 = ~np.isnan(f2_values)
        if np.any(valid_f2):
            f2_mhz = f2_values[valid_f2] * 1e3  # Convert GHz to MHz
            ax.plot(f2_mhz, fit_charge_gates[valid_f2], 'ro', 
                   markersize=4,  label='f2', alpha=0.8)

        # Plot ave_freq frequencies
        ave_freq_values = analysis_result['ave_freq'].values
        valid_ave_freq = ~np.isnan(ave_freq_values)
        if np.any(valid_ave_freq):
            ave_freq_mhz = ave_freq_values[valid_ave_freq] * 1e3  # Convert GHz to MHz
            ax.plot(ave_freq_mhz, fit_charge_gates[valid_ave_freq],'o', color='black', 
                   markersize=4, label='ave_freq', alpha=0.8)
            
    
    # Set labels and title
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Charge Gate (V)')
    
    # Get qubit name for title if available
    qubit_name = fftdata.coords.get('qubit', 'Unknown')
    ax.set_title(f'{qubit_name} FFT Spectrum vs Charge Gate')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits to focus on non-zero frequencies
    if len(freq_mhz) > 1:
        ax.set_xlim(freq_mhz[1], freq_mhz[-1])  # Skip DC component
    
    # Tight layout to prevent label cutoff
    fig.tight_layout()
    
    return fig

def plot_1d_frequencies(analysis_result: xr.Dataset):
    """
    Plot absolute differences |f1-all_ave_freq| and |f2-all_ave_freq| as functions of charge_gate voltage.
    
    Parameters:
    -----------
    analysis_result : xr.Dataset
        Dataset containing f1, f2, and ave_freq with charge_gate coordinate
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    print(analysis_result)

    # Extract charge gate values
    charge_gates = analysis_result.coords['charge_gate'].values
    
    # Get all_ave_freq from dataset attributes
    all_ave_freq = analysis_result.attrs.get('all_ave_freq', 0)
    
    # Plot |f1 - all_ave_freq|
    f1_values = analysis_result['f1'].values
    valid_f1 = ~np.isnan(f1_values)
    if np.any(valid_f1):
        f1_diff_khz = np.abs(f1_values[valid_f1] - all_ave_freq) * 1e3  # Convert GHz to MHz
        ax.scatter(charge_gates[valid_f1], f1_diff_khz, 
                  c='blue', s=50, label='|f1 - all_ave_freq|', alpha=0.8)
    
    # Plot |f2 - all_ave_freq|
    f2_values = analysis_result['f2'].values
    valid_f2 = ~np.isnan(f2_values)
    if np.any(valid_f2):
        f2_diff_khz = np.abs(f2_values[valid_f2] - all_ave_freq) * 1e3  # Convert GHz to MHz
        ax.scatter(charge_gates[valid_f2], f2_diff_khz, 
                  c='red', s=50, label='|f2 - all_ave_freq|', alpha=0.8)
    
    # Plot absolute cosine fit curve if available
    if analysis_result.attrs.get('abscos_fit_success', False):
        # Get fit parameters
        amplitude = analysis_result.attrs.get('abscos_amplitude', 0)
        frequency = analysis_result.attrs.get('abscos_frequency', 0)
        phase = analysis_result.attrs.get('abscos_phase', 0)
        chisqr = analysis_result.attrs.get('abscos_chisqr', 0)
        redchi = analysis_result.attrs.get('abscos_redchi', 0)
        
        # Create fine charge gate array for smooth curve
        charge_fine = np.linspace(charge_gates.min(), charge_gates.max(), 200)
        
        # Calculate fit curve (convert to frequency difference from all_ave_freq)
        fit_curve =  amplitude * np.abs(np.cos(2 * np.pi * frequency * (charge_fine - phase)))
        fit_diff_khz = fit_curve * 1e3  # Convert to kHz difference
        
        # Plot the fit curve
        ax.plot(charge_fine, fit_diff_khz, 'g-', linewidth=2, 
               label='|cos| fit', alpha=0.8)
        
        # Add text box with fit parameters
        textstr = f'|cos| Fit Parameters:\n'
        textstr += f'Amplitude: {amplitude*1e3:.2f} kHz\n'
        textstr += f'Frequency: {frequency:.3f} V⁻¹\n'
        textstr += f'Phase: {phase:.3f} V\n'
        textstr += f'χ²/dof: {redchi:.3f}'
        
        # Position text box in upper right
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Set labels and title
    ax.set_xlabel('Charge Gate (V)')
    ax.set_ylabel('Frequency Difference (kHz)')
    
    # Get qubit name for title if available
    qubit_name = analysis_result.coords.get('qubit', 'Unknown')
    ax.set_title(f'{qubit_name} Ramsey Frequency Differences vs Charge Gate')
    
    # Add legend if any data was plotted
    if np.any(valid_f1) or np.any(valid_f2) or analysis_result.attrs.get('abscos_fit_success', False):
        ax.legend()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Tight layout to prevent label cutoff
    fig.tight_layout()
    
    return fig

