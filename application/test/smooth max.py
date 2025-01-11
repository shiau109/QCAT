
import xarray as xr

dataset = xr.open_dataset(r"d:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\20250108_224458_detector_q1_bias0.075V_crosstalk_q6_long_drive_pulse_expectcrosstalk_0.1_0.1mius\data.nc")
print(dataset)
# data = dataset["q2_ro"].values[0]
data = -dataset["q1_ro"].values[0]


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

## Example 2D data with noise
# data = np.array([
#     [1, 2, 3, 4, 5],
#     [1, 2, 3, 4, 5],
#     [1, 2, 3, 4, 5],
#     [1, 2, 3, 4, 5],
#     [1, 2, 3, 4, 5],
# ])

# # Add noise to simulate real-world data
# np.random.seed(0)
# data += np.random.normal(0, 0.5, data.shape)
normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Step 1: Interpolate data to a higher resolution
zoom_factor = 2  # Scale up the resolution (2x in both dimensions)
interpolated_data = zoom(normalized_data, zoom=zoom_factor, order=3)  # Cubic interpolation

# Step 1: Smooth the data to reduce noise
smoothed_data = gaussian_filter(interpolated_data, sigma=2)

# Step 2: Find the maximum values along a specific axis (e.g., rows)
max_indices = np.argmax(smoothed_data, axis=1)
rows = np.arange(smoothed_data.shape[0])  # Row indices
cols = max_indices              # Column indices of maximum values

# Step 3: Apply RANSAC for robust regression
# Reshape data for sklearn (requires 2D arrays)
cols_reshaped = cols.reshape(-1, 1)
ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=2, residual_threshold=1.0, random_state=0)
ransac.fit(cols_reshaped, rows)

# Extract the slope and intercept of the robust line
slope = ransac.estimator_.coef_[0]
intercept = ransac.estimator_.intercept_

# Step 4: Visualize the raw data, smoothed data, and detected maximum values
plt.figure(figsize=(12, 6))

# Raw Data
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='viridis', aspect='auto')
plt.title("Raw Data")
plt.colorbar(label="Value")
plt.scatter(max_indices, rows, color='red', label='Max Values (Raw)')
plt.legend()

# Smoothed Data
plt.subplot(1, 2, 2)
plt.imshow(smoothed_data, cmap='viridis', aspect='auto')
plt.title("Smoothed Data with Detected Line")
plt.colorbar(label="Value")
plt.scatter(max_indices, rows, color='red', label='Max Values (Smoothed)')

# Plot RANSAC Line
predicted_rows = ransac.predict(np.arange(data.shape[1]).reshape(-1, 1))
plt.plot(np.arange(data.shape[1]), predicted_rows, '-k', label='RANSAC Fitted Line')
plt.legend()

plt.tight_layout()
plt.show()

# Output the slope of the fitted line
print(f"Detected slope of the line using RANSAC: {slope}")

