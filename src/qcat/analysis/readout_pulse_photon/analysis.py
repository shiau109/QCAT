from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import xarray as xr
from qcat.NCU.Fit_library import QS_fit_analysis
import numpy as np

def Readout_pulse_shaping(Raw_data,Processed_data):
    Collect=[] 
    for i in range(len(Raw_data.second_samples)):
        fit= QS_fit_analysis(Processed_data['data'][0][i],f=Raw_data.first_samples)
        Collect.append(fit)
    analysis_result=dict(data_fit=Collect)
    return analysis_result   



if __name__ == "__main__":
    from qcat.utilities.simple_visualization import plot_2d_colormap_from_h5
    import matplotlib.pyplot as plt

    ds = load_xarray_h5(r"d:\github\ASQMDriver\data\MIST\2025-09-11\#202_LCH_time_dep_rr_photon_215140\ds_raw.h5")
    print(ds)
    sep_data = repetition_data(ds, repetition_dim="qubit")


    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        plot_2d_colormap_from_h5(sq_data, data_var="I", x_dim='detuning', y_dim="delay_time")

        plot_info=dict(P_rescale=False, #normalize contrast to population
               Dis=None,
               linecut=0, 
               readout_qubit_info=True,
               color_bound=False,
               bound_value=[0,1],
               log_scale=False)

        fit_info=dict(Photon_convert=False,
                    Ac_info=None,
                    X_eff=None)
        
        from types import SimpleNamespace
        Raw_data = SimpleNamespace(
            meas_q = qubit_name,
            first_samples=ds.coords["detuning"].values,
            second_samples=ds.coords["delay_time"].values
        )
        Process_data_rpsh = { "data": [sq_data["I"].transpose('delay_time','detuning').values],
                    "first_samples":sq_data.coords["detuning"].values,
                    "second_samples":sq_data.coords["delay_time"].values
        }
        result_rpsh = Readout_pulse_shaping(Raw_data,Process_data_rpsh)

        from qcat.analysis.readout_pulse_photon.visualization import plot_all
        plot_all(Raw_data,Process_data_rpsh,result_rpsh,fit_info,plot_info, Save=False, Save_graph_path=None, id=None)
    
    plt.show()
