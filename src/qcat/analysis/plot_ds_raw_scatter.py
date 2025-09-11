import matplotlib.pyplot as plt
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import xarray as xr
from qcat.NCU.Fit_library import QS_fit_analysis
import numpy as np

def Readout_pulse_shaping(Raw_data,Processed_data):
    Collect=[] 
    print(Raw_data.second_samples.shape)
    for i in range(len(Raw_data.second_samples)):
        print(Processed_data['data'][0][i].shape,Raw_data.first_samples.shape)
        fit= QS_fit_analysis(Processed_data['data'][0][i],f=Raw_data.first_samples)
        Collect.append(fit)
    analysis_result=dict(data_fit=Collect)
    return analysis_result   

from qcat.NCU.Visualized_library import Fit_analysis_plot, plot_2D
def plot_all(Raw_data,Process_data,Analysis_result,fit_info,plot_info,Save,Save_graph_path,id):
    linecut = plot_info['linecut']
    fig1= Fit_analysis_plot(Analysis_result['data_fit'][linecut],P_rescale=plot_info['P_rescale'],Dis=plot_info['Dis'])
    fig2= plot_2D(Process_data['second_samples']*1e6,Process_data['first_samples']/1e9,Process_data['data'][0].transpose(),
                    label=[r"$t_{d}\ [\mu$s]",r"$f_{XY}\ $[GHz]"],
                    title="Readout_pulse_shaping_"+Raw_data.meas_q,
                    readout_qubit_info= plot_info['readout_qubit_info'],
                    P_rescale    = plot_info['P_rescale'], Dis = plot_info['Dis'],
                    color_bound   = plot_info['color_bound'],
                    bound_value   = plot_info['bound_value'],
                    plot_linecut = False,
                    linecut      = linecut,)
    fig3= Readout_shaping_plot(Process_data['second_samples']*1e6,Analysis_result['data_fit'],fit_info['Photon_convert'],fit_info['Ac_info'],plot_info['log_scale'])
    if Save:
        if id ==None:
            pass
        else:
            fig1.savefig(Save_graph_path+"Readout_pulse_shaping_1"+'_'+id+'.png', pad_inches=0.05, bbox_inches='tight', format='png',dpi=1000)
            fig2.savefig(Save_graph_path+"Readout_pulse_shaping_2"+'_'+id+'.png', pad_inches=0.05, bbox_inches='tight', format='png',dpi=1000)
            fig3.savefig(Save_graph_path+"Readout_pulse_shaping_3"+'_'+id+'.png', pad_inches=0.05, bbox_inches='tight', format='png',dpi=1000)
    return {"2Dmap": fig2, "fitted_frequency": fig3}

def Readout_shaping_plot(delay,results,Photon_convert,Ac_info,log_scale):
    f01=[]

    for i in range(len(results)):
        f01.append(results[i].attrs['f01_fit'])
    f01 = np.array(f01)
    if Photon_convert:
        ylabel= 'Resonator \n photons'
        n= (Ac_info['f01_bare']-np.array(f01) - Ac_info['X_eff'])/(2*Ac_info['X_eff'])

        # offset correction
        print(np.where(delay < 0)[0].max())
        n= n- np.mean(n[:np.where(delay < 0)[0].max()])
        y_value= n
    else:
        ylabel= r"$f_{01}\ $[GHz]"
        y_value= f01/1e9
    if log_scale:
            #log plot
        fig, ax = plt.subplots(ncols = 1, figsize = (6, 4), dpi = 200)
        ax.plot(delay, y_value, color = "k",marker='o', alpha = 0.5,lw=1)
        ax.set_xlabel(r"$t_{XY}\ [\mu$s]", fontsize = 15)
        ax.set_ylabel(ylabel, fontsize = 15)
        ax.set_yscale("log")
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(ncols = 1, figsize = (6, 4), dpi = 200)
        ax.plot(delay, y_value, color = "k",marker='o', alpha = 0.5,lw=1)
        ax.set_xlabel(r"$t_{XY}\ [\mu$s]", fontsize = 15)
        ax.set_ylabel(ylabel, fontsize = 15)
        fig.tight_layout()

    return fig

if __name__ == "__main__":
    from qcat.utilities.simple_visualization import plot_2d_colormap_from_h5
    ds = load_xarray_h5(r"d:\github\ASQMDriver\data\MIST\2025-09-11\#200_LCH_time_dep_rr_photon_212910\ds_raw.h5")
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
        plot_all(Raw_data,Process_data_rpsh,result_rpsh,fit_info,plot_info, Save=False, Save_graph_path=None, id=None)
    
    plt.show()
