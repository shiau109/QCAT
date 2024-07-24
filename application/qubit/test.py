import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import netCDF4
import h5netcdf
dataset = xr.open_dataset(r"C:\Users\arthu\20240723_1116_q1_xy_EchoT2.nc")
print(dataset)
time = (dataset.coords["time"].values)/1000

from qcat.visualization.qubit_relaxation import plot_qubit_relaxation
from qcat.analysis.qubit.relaxation import qubit_relaxation_fitting

for ro_name, data in dataset.data_vars.items():
    print(ro_name)
    fit_result = qubit_relaxation_fitting(time, data.values[0])
    print(fit_result.params)
    fig, ax = plt.subplots()
    plot_qubit_relaxation(time, data[0], ax, fit_result)

plt.show()