from qcat.NCU.Fit_library import QS_fit_analysis
import numpy as np
from lmfit import Model


def Ac_stark_shift(Raw_data,Processed_data):
    Collect=[] 
    for i in range(len(Raw_data.second_samples)):
        fit= QS_fit_analysis(Processed_data['data'][0][i],f=Raw_data.first_samples)
        Collect.append(fit)
    analysis_result=dict(data_fit=Collect)
    return analysis_result  

def Ac_Stark_shift_analysis(voltage:np.ndarray,Analysis_result:dict,X_eff:float,fit_window_data_index:list=None):
    f01=[]
    power_l= voltage**2/50 # unit: W
    
    for i in range(len(voltage)):
        f01.append(Analysis_result['data_fit'][i].attrs['f01_fit'])

    def Starkshift_P(x, A, fa):
        return fa - (2 * x * A + 1) * X_eff

    gmodel = Model(Starkshift_P)
    if fit_window_data_index is None:
        fit_x = power_l
        fit_y = np.array(f01)
    else:
        fit_x = power_l[fit_window_data_index[0]:fit_window_data_index[1]]
        fit_y = np.array(f01)[fit_window_data_index[0]:fit_window_data_index[1]]
    result = gmodel.fit(fit_y, x=fit_x, A=1e5, fa=f01[0])

    coeff = result.best_values['A']
    fa_fit = result.best_values['fa']
    n = coeff * power_l

    def Starkshift_n(n_, fa):
        return fa - (2 * n_ + 1) * X_eff
    para_fit = np.linspace(min(power_l), max(power_l), 20 * len(power_l))
    fitting = Starkshift_P(para_fit, coeff, fa_fit)

    return xr.Dataset(data_vars=dict(data=(['P'], np.array(f01)), fitting=(['para_fit'], fitting)),
                     coords=dict(P=(['P'], power_l), para_fit=(['para_fit'], para_fit)),
                     attrs=dict(exper="Ac-Stark", coeff=coeff, fa_0=fa_fit, X_eff=X_eff))



if __name__ == '__main__':


    
    import matplotlib.pyplot as plt
    from qcat.parser.qm_reader import load_xarray_h5
    from qcat.analysis.ac_stark_shift.visualization import Ac_stark_shift_plot
    # Load the dataset
    file_path = r"d:\data\MIST\20251201\#7937_LCH_qubit_spectroscopy_vs_ROamp_111032\ds_raw.h5"
    ds = load_xarray_h5(file_path)
    from qcat.parser.qm_reader import repetition_data

    sq_data = repetition_data(ds, repetition_dim="qubit")[0]

    print(sq_data)
    import xarray as xr
    # Build Raw_data as xarray Dataset
    from types import SimpleNamespace

    Raw_data = SimpleNamespace(
        first_samples=sq_data.coords["detuning"].values,
        second_samples=sq_data.coords["readout_amp_ratio"].values
    )


    # Create Processed_data as a copy of sq_data with 'I' renamed to 'data'
    
    Processed_data = { "data": [sq_data["I"].transpose('readout_amp_ratio','detuning').values],
                    "first_samples":sq_data.coords["detuning"].values,
                    "second_samples":sq_data.coords["readout_amp_ratio"].values
        }
    print(Processed_data["data"][0].shape)
    print(type(Processed_data["data"][0][0]),Processed_data["data"][0][0].shape)
    print(type(Raw_data.first_samples),Raw_data.first_samples.shape)

    plot_info = dict(P_rescale=False, #normalize contrast to population
                Dis=None,
                linecut=0, 
                readout_qubit_info=True,
                color_bound=False,
                bound_value=[0,1])

    fit_info = dict(fit_window_data_index=[0,21],
                given_factors=dict(kc=1.43*1e6,
                                    ki=0.074*1e6,
                                    g= 91.3*1e6,
                                    X_eff=1.3709*1e6,
                                    f_bare=5.991*1e9,
                                    f_eff_bare=5.99625*1e9,
                                    R_F=None),
                target_average_photon_number=9, #! Use it to predict the wiring attenuation     
                ro_output_att=0)

    # Example usage:
    result_acss = Ac_stark_shift(Raw_data, Processed_data)
    Ac_info = Ac_stark_shift_plot(Raw_data, Processed_data, result_acss, fit_info, plot_info, False, None, None)

    plt.show()
