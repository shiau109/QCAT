import matplotlib.pyplot as plt
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import xarray as xr
from qcat.NCU.Fit_library import QS_fit_analysis
import numpy as np


if __name__ == "__main__":
    from qcat.utilities.simple_visualization import plot_2d_colormap_from_h5
    ds = load_xarray_h5(r"d:\github\ASQMDriver\data\MIST\2025-09-11\#179_08b_readout_power_optimization_093601\ds_raw.h5")
    print(ds)
    sep_data = repetition_data(ds, repetition_dim="qubit")


    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        
        # plot_2d_colormap_from_h5(sq_data, data_var="I", x_dim='charge_gate', y_dim="idle_time")

    
    plt.show()
