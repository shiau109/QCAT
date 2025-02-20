import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
# Load the NetCDF file into an xarray dataset
file_path = "d:\Ramsey_20250122151709.nc"
# Reload the dataset and extract the relevant data again
ds = xr.open_dataset(file_path)
ds = ds.sel(mixer="I")
y = ds["q2"].values
x = ds.isel(repeat=0)["q2_x"].values

# Remove DC component by subtracting the mean from each repeat element
y_detrended = y - np.mean(y, axis=1, keepdims=True)

# Perform FFT on each 'repeat' element after detrending
fft_results_detrended = np.fft.fft(y_detrended, axis=1)
freqs = np.fft.fftfreq(x.shape[-1], x[1] - x[0])

# Convert to magnitude and keep only positive frequencies
fft_magnitude = np.abs(fft_results_detrended[:, : len(freqs) // 2])
freqs_positive = freqs[: len(freqs) // 2]

# Plot 2D colormap
plt.figure(figsize=(10, 6))
plt.imshow(
    fft_magnitude,
    aspect="auto",
    extent=[freqs_positive.min(), freqs_positive.max(), 0, ds.sizes["repeat"]],
    origin="lower",
    cmap="viridis",
)

plt.colorbar(label="Magnitude")
plt.xlabel("Frequency")
plt.ylabel("Repeat Index")
plt.title("FFT Magnitude Spectrum (DC Removed)")
plt.show()

from scipy.signal import find_peaks

# Find the peaks in the FFT magnitude spectrum for each repeat index
peak_indices = [find_peaks(fft_magnitude[i])[0] for i in range(fft_magnitude.shape[0])]

# Extract the first and second highest peaks for each repeat
first_peaks = [freqs_positive[indices[np.argmax(fft_magnitude[i, indices])]] if len(indices) > 0 else None for i, indices in enumerate(peak_indices)]
second_peaks = [freqs_positive[indices[np.argsort(fft_magnitude[i, indices])[-2]]] if len(indices) > 1 else None for i, indices in enumerate(peak_indices)]

# Create a DataFrame to display results
import pandas as pd
df_peaks = pd.DataFrame({"Repeat Index": range(len(first_peaks)), "First Peak (Hz)": first_peaks, "Second Peak (Hz)": second_peaks})

# Convert peak lists to numpy arrays for plotting
first_peaks_array = np.array(first_peaks, dtype=np.float64)
second_peaks_array = np.array(second_peaks, dtype=np.float64)

# Plot 2D colormap
plt.figure(figsize=(10, 6))
plt.imshow(
    fft_magnitude,
    aspect="auto",
    extent=[freqs_positive.min(), freqs_positive.max(), 0, ds.sizes["repeat"]],
    origin="lower",
    cmap="viridis",
)

# Plot the first and second highest peaks
plt.scatter(first_peaks_array, np.arange(len(first_peaks_array)), color='red', marker='o', label="First Peak")
plt.scatter(second_peaks_array, np.arange(len(second_peaks_array)), color='cyan', marker='x', label="Second Peak")

plt.colorbar(label="Magnitude")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Repeat Index")
plt.title("FFT Magnitude Spectrum with Peak Frequencies (DC Removed)")
plt.legend()
plt.show()


from lmfit import Model
param_columns = ["A1", "f1", "phi1", "A2", "f2", "phi2", "offset"]
# Define the sine wave model for lmfit
def two_sine_lmfit(x, A1, f1, phi1, A2, f2, phi2, offset):
    return (
        A1 * np.sin(2 * np.pi * f1 * x + phi1)
        + A2 * np.sin(2 * np.pi * f2 * x + phi2)
        + offset
    )

# Create lmfit model
model = Model(two_sine_lmfit)

# Prepare arrays to store fitted parameters
fitted_params_lmfit = []

# Iterate over each repeat index to fit the model using lmfit
for i in range(ds.sizes["repeat"]):
    # Get raw data for this repeat index
    y_raw = y[i, :]
    
    # Use identified peak frequencies as initial values
    f1_init = first_peaks[i] if first_peaks[i] is not None else 0.1
    f2_init = second_peaks[i] if second_peaks[i] is not None else 0.2
    
    # Initial guesses for amplitudes, phases, and offset
    A1_init, A2_init = np.ptp(y_raw) , np.ptp(y_raw)   # Peak-to-peak amplitude
    phi1_init, phi2_init = 0, 0  # Assume zero initial phase
    offset_init = np.mean(y_raw)  # DC offset
    
    # Define parameter settings for lmfit
    params = model.make_params(
        A1=A1_init, f1=f1_init, phi1=phi1_init,
        A2=A2_init, f2=f2_init, phi2=phi2_init, offset=offset_init
    )
    
    try:
        # Fit the model using lmfit
        result = model.fit(y_raw, params, x=x)
        fitted_params_lmfit.append([result.params[name].value for name in param_columns])
    except Exception as e:
        # If fitting fails, store None
        fitted_params_lmfit.append([None] * 7)

# Convert results to DataFrame for easy review
df_fitted_params_lmfit = pd.DataFrame(fitted_params_lmfit, columns=param_columns)
df_fitted_params_lmfit.insert(0, "Repeat Index", range(len(fitted_params_lmfit)))

# Select one repeat index for raw vs. fitted comparison
repeat_idx = 3  # Change this index for other cases

# Extract fitted parameters for the selected repeat index
if None not in fitted_params_lmfit[repeat_idx]:  # Ensure valid fit
    A1, f1, phi1, A2, f2, phi2, offset = fitted_params_lmfit[repeat_idx]
    
    # Generate fitted curve
    y_fitted = two_sine_lmfit(x, A1, f1, phi1, A2, f2, phi2, offset)

    # Plot raw data vs. fitted curve
    plt.figure(figsize=(8, 5))
    plt.plot(x, y[repeat_idx, :], label="Raw Data", linestyle="dotted")
    plt.plot(x, y_fitted, label="Fitted Curve (lmfit)", linewidth=2)
    plt.xlabel("Time (or X-axis)")
    plt.ylabel("Amplitude")
    plt.title(f"Raw Data vs. Fitted Curve (lmfit) - Repeat Index {repeat_idx}")
    plt.legend()
    plt.show()
else:
    print(f"Fitting failed for repeat index {repeat_idx}, so no plot can be generated.")


# Extract fitted frequency values for plotting
f1_fitted_array = np.array(df_fitted_params_lmfit["f1"], dtype=np.float64)
f2_fitted_array = np.array(df_fitted_params_lmfit["f2"], dtype=np.float64)

# Create a figure with two subplots
plt.figure(figsize=(10, 6))
# Plot 2D colormap for FFT magnitude spectrum
im = plt.imshow(
    fft_magnitude,
    aspect="auto",
    extent=[freqs_positive.min(), freqs_positive.max(), 0, ds.sizes["repeat"]],
    origin="lower",
    cmap="viridis",
)
plt.scatter(f1_fitted_array, np.arange(len(f1_fitted_array)), color='red', marker='o', label="Fitted f1")
plt.scatter(f2_fitted_array, np.arange(len(f2_fitted_array)), color='cyan', marker='x', label="Fitted f2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Repeat Index")
plt.title("FFT Magnitude Spectrum with Fitted Frequencies (lmfit)")
plt.legend()


# Add colorbar to FFT plot
plt.colorbar(im, label="Magnitude")

plt.show()
