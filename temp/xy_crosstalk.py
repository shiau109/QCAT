
import os
import json
from qcat.parser.qm_reader import load_xarray_h5
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from qcat.utilities.function_fitting.fit_cosine import FitCosine
# Folder to check

root_path = r"d:\data\6SQ_XYtest_2\RF_crsstalk"
folders = [r"#125_04b_power_rabi_111638",
        r"#126_04b_power_rabi_111653",
        r"#127_04b_power_rabi_111710",
        r"#128_04b_power_rabi_111725",
        r"#129_04b_power_rabi_114059",
        r"#130_04b_power_rabi_114116",
        r"#131_04b_power_rabi_121501",
        r"#132_04b_power_rabi_121513"]
folder_path = [os.path.join(root_path, folder) for folder in folders]
# Build paths
fig, ax = plt.subplots(figsize=(10, 6))

fit_results = []  # Store fit results for each dataset

for i, folder in enumerate(folder_path):
    file_path = os.path.join(folder, 'ds_raw.h5')
    json_state_path = os.path.join(folder, 'quam_state\\state.json')

    ds = load_xarray_h5(file_path).sel(nb_of_pulses=1)
    with open(json_state_path, 'r') as f:
        json_dict = json.load(f)

    print(f"Dataset {i+1}: {ds}")
    
    # Get the qubit name (assuming single qubit per dataset)
    qubit_name = ds.coords['qubit'].values[0]
    
    # Extract full_amp and signal for this qubit
    full_amp = ds['full_amp'].sel(qubit=qubit_name).values
    signal = ds['I'].sel(qubit=qubit_name).values
    
    # Create DataArray for fitting (FitCosine expects x coordinate)
    signal_da = xr.DataArray(
        signal,
        coords={'x': full_amp},
        dims=['x'],
        name='signal'
    )
    
    # Fit cosine function
    try:
        fitter = FitCosine(signal_da)
        fit_result = fitter.fit()
        fit_results.append(fit_result)
        
        print(f"\nFit results for Dataset {i+1}:")
        print(f"  Amplitude: {fit_result.params['a'].value:.6f} ± {fit_result.params['a'].stderr:.6f}")
        print(f"  Frequency: {fit_result.params['f'].value:.6f} ± {fit_result.params['f'].stderr:.6f}")
        print(f"  Phase: {fit_result.params['phi'].value:.6f} ± {fit_result.params['phi'].stderr:.6f}")
        print(f"  Offset: {fit_result.params['c'].value:.6f} ± {fit_result.params['c'].stderr:.6f}")
        print(f"  R-squared: {1 - fit_result.residual.var() / np.var(signal):.6f}")
        
        # Generate fitted curve for plotting
        x_fit = np.linspace(full_amp.min(), full_amp.max(), 200)
        y_fit = fit_result.eval(x=x_fit)
        
        # Get frequency for legend
        freq_hz = fit_result.params['f'].value
        
        # Plot data and fit with matching colors
        color = f'C{i}'  # Use matplotlib default color cycle
        label_data = f"{qubit_name} (f={freq_hz:.3f} Hz)"
        ax.plot(full_amp, signal, 'o', markersize=3, label=label_data, alpha=0.7, color=color)
        ax.plot(x_fit, y_fit, '-', linewidth=2, alpha=0.8, color=color)  # No label for fit line
        
    except Exception as e:
        print(f"Fit failed for Dataset {i+1}: {e}")
        # Plot data only if fit fails
        color = f'C{i}'  # Use same color scheme even for failed fits
        label = f"{qubit_name} - No fit"
        ax.plot(full_amp, signal, 'o', markersize=3, label=label, alpha=0.7, color=color)

ax.set_xlabel('Full Amplitude (mV)')
ax.set_ylabel('Signal (I)')
ax.set_title('Signal vs Full Amplitude with Cosine Fits')
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

# Save figure to the parent directory of the first folder
parent_dir = os.path.dirname(folder_path[0])
out_path = os.path.join(parent_dir, 'power_rabi_cosine_fits.png')
try:
    fig.savefig(out_path, dpi=200)
    print(f"Saved power rabi fits plot to: {out_path}")
except Exception as e:
    print(f"Failed to save plot: {e}")
plt.show()
plt.close(fig)

# Print summary of fit results
if fit_results:
    print("\n" + "="*50)
    print("FIT SUMMARY")
    print("="*50)
    for i, result in enumerate(fit_results):
        print(f"Dataset {i+1}:")
        print(f"  Rabi frequency: {result.params['f'].value:.6f} Hz")
        print(f"  Amplitude: {result.params['a'].value:.6f}")
        print(f"  Phase: {result.params['phi'].value:.6f} rad")
        print(f"  Offset: {result.params['c'].value:.6f}")
        print(f"  Fit quality (R²): {1 - result.residual.var() / np.var(result.data):.6f}")
        print()

