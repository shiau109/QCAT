from qcat.NCU.Visualized_library import Fit_analysis_plot, plot_2D
import matplotlib.pyplot as plt
import numpy as np


def plot_all(Raw_data,Process_data,Analysis_result,fit_info,plot_info,Save,Save_graph_path,id):
    linecut = plot_info['linecut']
    fig2= plot_2D(Process_data['second_samples']*1e6,Process_data['first_samples']/1e9,Process_data['data'][0].transpose(),
                    label=[r"$t_{d}\ [\mu$s]",r"$f_{XY}\ $[GHz]"],
                    title="Readout_pulse_shaping_"+Raw_data.meas_q,
                    readout_qubit_info= plot_info['readout_qubit_info'],
                    P_rescale    = plot_info['P_rescale'], Dis = plot_info['Dis'],
                    color_bound   = plot_info['color_bound'],
                    bound_value   = plot_info['bound_value'],
                    plot_linecut = False,
                    linecut      = linecut,)
    return_dict = {"2Dmap": fig2}
    if Analysis_result is not None:
        fig1= Fit_analysis_plot(Analysis_result['data_fit'][linecut],P_rescale=plot_info['P_rescale'],Dis=plot_info['Dis'])
        fig3= Readout_shaping_plot(Process_data['second_samples']*1e6,Analysis_result['data_fit'],fit_info['Photon_convert'],fit_info['Ac_info'],plot_info['log_scale'])
        return_dict["fitted_frequency"] = fig3
    if Save:
        if id ==None:
            pass
        else:
            fig1.savefig(Save_graph_path+"Readout_pulse_shaping_1"+'_'+id+'.png', pad_inches=0.05, bbox_inches='tight', format='png',dpi=1000)
            fig2.savefig(Save_graph_path+"Readout_pulse_shaping_2"+'_'+id+'.png', pad_inches=0.05, bbox_inches='tight', format='png',dpi=1000)
            fig3.savefig(Save_graph_path+"Readout_pulse_shaping_3"+'_'+id+'.png', pad_inches=0.05, bbox_inches='tight', format='png',dpi=1000)
    return return_dict

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