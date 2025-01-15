import xarray as xr

dataset = xr.open_dataset(r"d:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\20250108_224458_detector_q1_bias0.075V_crosstalk_q6_long_drive_pulse_expectcrosstalk_0.1_0.1mius\data.nc")
print(dataset)
data = -dataset["q1_ro"].values[0].transpose()
# data = -dataset["q1_ro"].values[0]
y_axis = dataset.coords["detector_z"]
x_axis = dataset.coords["crosstalk_z"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from sklearn.linear_model import RANSACRegressor, LinearRegression

# # Example 2D raw data with x and y axis values
# x_axis = np.linspace(0, 10, 5)  # Define x-axis values for columns
# y_axis = np.linspace(0, 20, 5)  # Define y-axis values for rows
# data = np.array([
#     [1, 2, 3, 4, 5],
#     [2, 3, 4, 5, 6],
#     [3, 4, 5, 6, 7],
#     [4, 5, 6, 7, 8],
#     [5, 6, 7, 8, 9],
# ])

# # Add noise to simulate real-world data
# np.random.seed(0)
# data += np.random.normal(0, 0.5, data.shape)

# Step 1: Interpolate the data to increase resolution
zoom_factor = 2  # Increase resolution by 2x
interpolated_data = zoom(data, zoom=zoom_factor, order=3)
interpolated_x_axis = np.linspace(x_axis.min(), x_axis.max(), interpolated_data.shape[1])
interpolated_y_axis = np.linspace(y_axis.min(), y_axis.max(), interpolated_data.shape[0])

# Step 2: Apply Gaussian filter to reduce noise
smoothed_data = gaussian_filter(interpolated_data, sigma=1)

# Step 3: Normalize the smoothed data
# Normalize each row to its maximum value
baseline_corrected_data = smoothed_data - np.min(smoothed_data)#np.min(smoothed_data, axis=1, keepdims=True)  # Subtract baseline
normalized_data = baseline_corrected_data / np.max(baseline_corrected_data)# Normalize # np.max(baseline_corrected_data, axis=0, keepdims=True)  

# Step 4: Threshold the data to select top 90% of values
threshold = np.percentile(normalized_data, 95)
binary_map = normalized_data > threshold

# Step 5: Extract positions of non-zero elements
positions = np.argwhere(binary_map)  # List of [row, col] positions
rows, cols = positions[:, 0], positions[:, 1]  # Separate rows and cols

# Map array indices to x and y axis values
x_positions = interpolated_x_axis[cols]
y_positions = interpolated_y_axis[rows]

# Step 6: Fit positions using RANSAC
ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=2, residual_threshold=1.0, random_state=0)
ransac.fit(x_positions.reshape(-1, 1), y_positions)

# Extract slope and intercept
slope = ransac.estimator_.coef_[0]
intercept = ransac.estimator_.intercept_

# Print results
print(f"Detected slope using RANSAC: {slope}")
print(f"Intercept: {intercept}")

# Step 7: Visualization using pcolormesh
plt.figure(figsize=(15, 5))

# Raw Data
plt.subplot(1, 4, 1)
plt.pcolormesh(x_axis, y_axis, data, cmap='viridis', shading='auto')
plt.title("Raw Data")
plt.colorbar(label="Value")

# Smoothed Data
plt.subplot(1, 4, 2)
plt.pcolormesh(interpolated_x_axis, interpolated_y_axis, smoothed_data, cmap='viridis', shading='auto')
plt.title("Smoothed Data")
plt.colorbar(label="Value")

# Normalized Data
plt.subplot(1, 4, 3)
plt.pcolormesh(interpolated_x_axis, interpolated_y_axis, normalized_data, cmap='viridis', shading='auto')
plt.title("Normalized Data")
plt.colorbar(label="Value")

# Binary Map with Fitted Line
plt.subplot(1, 4, 4)
plt.pcolormesh(interpolated_x_axis, interpolated_y_axis, binary_map, cmap='gray', shading='auto')
plt.title("Binary Map with Fitted Line")
plt.scatter(x_positions, y_positions, color='red', label='Selected Points')
predicted_y_positions = ransac.predict(interpolated_x_axis.reshape(-1, 1))
plt.plot(interpolated_x_axis, predicted_y_positions, '-k', label='RANSAC Fitted Line')
plt.legend()

plt.tight_layout()
plt.show()
