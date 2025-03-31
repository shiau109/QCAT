import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from qcat.common_calculator.qcq_zz_interaction import ZZ_interaction
from qcat.analysis.qubit.zz_interaction import ZZinteractionEcho

# Instantiate the calculator with default values
calc = ZZ_interaction()
coupler_freq = np.linspace(5, 8, 100)
calc.w2 = coupler_freq
coupling = 0.085 *1.0 #*0.63
corr_coupling = coupling*np.sqrt( (coupler_freq)/6.3 )
calc.g23 = corr_coupling
calc.g12 = corr_coupling
calc.g13 = 0.0038 *1.071 *1.1 #*0.63

print(coupler_freq)

# Compute Nanjing's formulas and Tsinghua's formulas
zz2_nanjing, zz3_nanjing, zz4_nanjing = calc.nanjing_formula()
zz2_tsinghua, zz3_tsinghua, zz4_tsinghua = calc.tsinghua_formula()
print(calc.d12,calc.d13)
print(calc.g12)


# Create a figure and axis for the ZZ interaction comparison plot
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(calc.w2, np.abs(zz2_nanjing + zz3_nanjing + zz4_nanjing)*1000, label="Nanjing zz")
# ax1.plot(calc.w2, np.abs(zz2_nanjing+np.zeros(zz3_nanjing.shape) )*1000, label="Nanjing zz2")
# Uncomment the following line to plot Tsinghua's formula as well:
ax1.plot(calc.w2, np.abs(zz2_tsinghua + zz3_tsinghua + zz4_tsinghua)*1000, label="Tsinghua zz", linestyle='--')
ax1.set_xlabel("w2")
ax1.set_ylabel("zz2 values")
ax1.legend()
ax1.set_title("Comparison of zz2 computed from two formulas")

# Open the netCDF dataset with your data.
dataset = xr.open_dataset(r"d:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\20250112_122247_find_ZZfree_q1_q2\find_ZZfree_q1_q2.nc")
dataset = dataset.sel(mixer="I")
print(dataset)

# Extract the desired data variable and update its attributes.
data = dataset["q1_ro"]
data.attrs = dataset.attrs
data.name = "q1_ro"

# Create an instance of the ZZinteractionEcho analysis class and run the analysis.
analysis = ZZinteractionEcho(data)
analysis._start_analysis()

# Build a new DataArray from the analysis result for frequency.
output_dataarray = xr.DataArray(
    data=analysis.statistic_result["frequency"].values,
    dims=["flux"],
    coords=dict(
        flux=data.coords["flux"].values,
    )
)
# print(data.coords["flux"].values)
# Compute x values based on the flux coordinate.
x = np.sqrt(8 * 0.16 *43 * abs(np.cos((data.coords["flux"].values +0.015 +0.10) /0.62 * np.pi))) - 0.16
print(x[0],x[-1])
print(data.coords["flux"].values[0],data.coords["flux"].values[-1])

# Create a new figure and axis for the frequency vs x plot.
ax1.plot(x, analysis.statistic_result["frequency"].values, marker='o', label="data")
ax1.set_yscale("log")
ax1.set_ylim(1e-3,5)


fig0, ax0 = plt.subplots(figsize=(8, 5))
ax0.plot(data.coords["flux"].values, x)
plt.show()
