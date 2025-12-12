
from qcat.NCU.Visualized_library import plot_2D, Fit_analysis_plot,Ac_Stark_shift_analysis

import numpy as np
import matplotlib.pyplot as plt

def Ac_Stark_shift_fit_plot(results:dict,given_factors:dict,target_average_photon_number:float,ro_output_att:float): 
    from qcat.common_calculator.analytical import n_predict    
    Nor_f=1e6
    y_fit= results.data_vars['fitting']/Nor_f
    y= results.data_vars['data']/Nor_f
    # x= results.coords['P']*1000   #unit:mW
    x= np.sqrt(results.coords['P']*50)   #unit:mW

    # x_fit= results.coords['para_fit']*1000  #unit:mW
    x_fit= np.sqrt(results.coords['para_fit']*50)  #unit:mW

    fa=results.attrs['fa_0']/Nor_f
    coeff=results.attrs['coeff']


    
    #attenuation calibration
    test_nbar_list=[]
    test_n_list=[]
    test_range= np.linspace(-145,-105,2001)
    # for P_in in test_range:
    #     ng,ne= n_predict(given_factors,P_in)[0],n_predict(given_factors,P_in)[1]
    #     test_n_list.append([ng,ne])
    #     test_nbar_list.append((ng+ne)/2)
    # idx= np.abs(np.array(test_nbar_list) - target_average_photon_number).argmin()
    # predict_att= 10*np.log10(1000*test_nbar_list[idx]/coeff)-test_range[idx]
    # print('n_bar=',np.around(test_nbar_list[idx],2))
    # print('P_in=',np.around(test_range[idx],2),'dBm')
    # print('n_g=',np.around(test_n_list[idx][0],2))
    # print('n_e=',np.around(test_n_list[idx][1],2))
    # print('predict_wiring_att(from DAC to the sample)=',np.around(predict_att-ro_output_att,2),'dB')


    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    x_label= "Readout output voltage"+" [V]"
    y_label= r"$f_{01}$"+' [MHz]'

    # text_msg = "Fit results\n"
    # text_msg += r"$f_{01}= %.4f $"%(fa) +' GHz\n\n'
    # text_msg += "Given factor\n"
    # text_msg += r"$\chi_{eff}/2\pi= %.4f $"%(given_factors['X_eff']*1e-6) +' MHz'
    
    ax.plot(x,y,'o', color="blue", alpha=0.5, ms=5)
    ax.plot(x_fit,y_fit,'-', color="red", alpha=0.5, lw=2) 
    ax.set_xlabel(x_label,size ='15')
    ax.set_ylabel(y_label,size ='15')
    # plot_textbox(ax,text_msg,fontsize=9)
    # ax2 = ax.secondary_xaxis('top', functions=(VtoN,NtoV))
    # ax2.set_xlabel(r"$\bar n $",size ='25')
    fig.tight_layout()
    

    
    fig1, ax = plt.subplots(nrows =1,figsize =(8,4),dpi =200)
    x_label_1="Readout output voltage"+" [V]"
    x_label_2= "Readout power reaching the sample"+" [dBm]"
    y_label= r"$\bar n $"
    y_fit= results.coords['para_fit']*coeff
    # x_fit= 10*np.log10(results.coords['para_fit']*1000)-predict_att
    # coeff=results.attrs['coeff']
    # def PtoV_2(P):
    #     return np.sqrt(10**((P+predict_att)/10)*1e-3*50)
    # def VtoP_2(V):
    #     return 10*np.log10(V**2/50*1e3)-predict_att
    
    # ax2 = ax.secondary_xaxis('top', functions=(VtoP_2,PtoV_2))
    # ax2.set_xlabel(x_label_2,size ='15')
    # ax.plot(PtoV_2(x_fit),y_fit, color="b", alpha=0.5, lw=3)     
    # ax.set_xlabel(x_label_1,size ='15')
    # ax.set_ylabel(y_label,size ='25')
    # ax.axvline(x=PtoV_2(test_range[idx]),color='r',linestyle='dashed', alpha=0.5,lw=1.5)
    # ax.axhline(y=test_nbar_list[idx],color='r',linestyle='dashed', alpha=0.5,lw=1.5)
    # text_msg = r"$f_{r}= %.5f $"%(given_factors['R_F']*1e-9) +' GHz\n'
    # text_msg += r"$\rm{amp.}= %.3f $"%(PtoV_2(test_range[idx])) +' \n'
    # text_msg += r"$\bar n= %.2f $"%(test_nbar_list[idx]) +' \n' 
    # text_msg += r"$n_{g}= %.2f $"%(test_n_list[idx][0]) +' \n'
    # text_msg += r"$n_{e}= %.2f $"%(test_n_list[idx][1]) +' \n'
    # text_msg += r"$P_{in}= %.2f $"%(test_range[idx]) +' dBm\n'
    # text_msg += r"$\rm{wiring\ att}= %.2f $"%(predict_att-ro_output_att) +' dB\n'
    # text_msg += r"$\rm{DAC\ output\ att}= %.2f $"%(ro_output_att) +' dB'
    # plot_textbox(ax,text_msg,fontsize=12)
    fig1.tight_layout()
    
    return fig,fig1, dict(kc=given_factors['kc'],ki=given_factors['ki'],X_eff=given_factors['X_eff'],f_eff_bare=given_factors['f_eff_bare'],f01_bare=results.attrs['fa_0'])#wiring_att=predict_att,
      

def Ac_stark_shift_plot(Raw_data,Process_data,Analysis_result,fit_info,plot_info,Save=None,Save_graph_path=None,id=None):
    linecut     = plot_info['linecut']
    fit_window_data_index= fit_info['fit_window_data_index']
    given_factors= fit_info['given_factors']
    fig1= Fit_analysis_plot(Analysis_result['data_fit'][linecut],P_rescale=plot_info['P_rescale'],Dis=plot_info['Dis'])
    fig2= plot_2D(Process_data['first_samples']/1e6,Process_data['second_samples']**2,Process_data['data'][0],
                    label=[r"$f_{XY}\ $[MHz]",'Stark power'+' [V^2]'],
                    title="Ac_Stark_shift",
                    readout_qubit_info= plot_info['readout_qubit_info'],
                    P_rescale    = plot_info['P_rescale'], Dis = plot_info['Dis'],
                    color_bound   = plot_info['color_bound'],
                    bound_value   = plot_info['bound_value'],
                    plot_linecut = False,
                    linecut      = linecut,)
    Ac_info= Ac_Stark_shift_analysis(Process_data['second_samples'],Analysis_result,X_eff=given_factors['X_eff'],fit_window_data_index=fit_window_data_index,)
    fig3,fig4,Ac_info = Ac_Stark_shift_fit_plot(Ac_info,
                                                    given_factors=given_factors,
                                                    target_average_photon_number=fit_info['target_average_photon_number'],
                                                    ro_output_att=fit_info['ro_output_att'])

    return {"fitted":fig3, "2D":fig2 }
