
import os
import json
from qcat.parser.qm_reader import load_xarray_h5
import numpy as np
import matplotlib.pyplot as plt
# Folder to check
folder_path = r"d:\\data\\6SQ_XYtest\\bare_resonator\\20251029_PF6FQ_BB"

# Build paths
file_path = os.path.join(folder_path, '20251029_PF6FQ_BB_20251029_182046_0db.nc')


ds = load_xarray_h5(file_path)
print(ds)

# Extract frequency coordinate
frequency = ds.coords['frequency'].values

# Reconstruct complex S21 from real and imaginary parts
s21_real = ds['s21'].sel(s_params='real').values
s21_imag = ds['s21'].sel(s_params='imag').values
s21_complex = s21_real + 1j * s21_imag

# Calculate magnitude |S21|
s21_magnitude = np.log10(np.abs(s21_complex))*10

# Plot |S21| vs frequency
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(frequency / 1e9, s21_magnitude, color='C0', linewidth=1)
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('|S21|')
ax.set_title('S21 Magnitude vs Frequency')
ax.set_xlim(5.75, 6.25)  # Limit to 5.5-6.5 GHz range
ax.grid(True, alpha=0.3)
fig.tight_layout()

# Save figure
out_path = os.path.join(folder_path, 's21_magnitude.png')
try:
    fig.savefig(out_path, dpi=200)
    print(f"Saved S21 magnitude plot to: {out_path}")
except Exception as e:
    print(f"Failed to save plot: {e}")
# plt.close(fig)
