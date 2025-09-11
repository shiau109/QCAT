import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from qcat.NCU.Visualized_library import plot_textbox_small, add_headers
from qcat.NCU.Fit_library import gauss_func, SQ_threshold_acquisition_statistic

def Single_shot_Rawdata_plot(data:dict):
    Ig_data,Qg_data,Ie_data,Qe_data= 1000*np.array(data['g'][0]), 1000*np.array(data['g'][1]) ,1000*np.array(data['e'][0]) , 1000*np.array(data['e'][1])
    I,Q= np.hstack([Ig_data,Ie_data]), np.hstack([Qg_data,Qe_data])
    
    fig, ax = plt.subplots(ncols =2,figsize =(6,3),dpi =200)
    ax[0].scatter(Ig_data,Qg_data, color="blue", alpha=0.5, s=1)   
    ax[1].scatter(Ie_data,Qe_data, color="red", alpha=0.5, s=1)      
    ax[0].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[0].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[1].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[1].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[0].set_title('Prepare |g>')
    ax[1].set_title('Prepare |e>')
    ax[0].set_xlim(np.mean(I)-5*np.std(I),np.mean(I)+5*np.std(I))
    ax[0].set_ylim(np.mean(Q)-5*np.std(I),np.mean(Q)+5*np.std(I))
    ax[1].set_xlim(np.mean(I)-5*np.std(I),np.mean(I)+5*np.std(I))
    ax[1].set_ylim(np.mean(Q)-5*np.std(I),np.mean(Q)+5*np.std(I))
    ax[0].axes.set_aspect('equal')
    ax[1].axes.set_aspect('equal')
    fig.tight_layout()
    return fig

def Qubit_state_single_shot_plot(results:dict):
    Pgg,Pee,OE= results['error_pack']['Pgg'],results['error_pack']['Pee'],results['error_pack']['overlap']
    Peg,Pge=1-Pgg,1-Pee
    ce_I,ce_Q,sig=1000*results['fit_pack'][0][0],1000*results['fit_pack'][0][1],1000*results['fit_pack'][5]
    Inte_g_data,Inte_e_data= results['fit_pack'][6],results['fit_pack'][7]
    Ig,Qg= results['rot_IQdata'][0][0],results['rot_IQdata'][0][1]
    Ie,Qe= results['rot_IQdata'][1][0],results['rot_IQdata'][1][1]
    I,Q= 1000*np.hstack([Ig,Ie]), 1000*np.hstack([Qg,Qe])
    I_ro,I_fit= 1000*results['I_ro'],1000*results['I_fit']
    Mgg= gauss_func(I_fit,0,sig,results['fit_pack'][1])
    Meg= gauss_func(I_fit,ce_I,sig,results['fit_pack'][2])
    Mge= gauss_func(I_fit,0,sig,results['fit_pack'][3])
    Mee= gauss_func(I_fit,ce_I,sig,results['fit_pack'][4])
    
    fig, ax = plt.subplots(ncols =2,figsize =(6,3),dpi =200)
    fig1, ax1 = plt.subplots(nrows=1,ncols =2,figsize =(7,3.5),dpi =200)

    Outlier_event_g,Inner_event_g= results['Outlier_g_info']['Outlier_event'],results['Outlier_g_info']['Inner_event']
    Outlier_event_e,Inner_event_e= results['Outlier_e_info']['Outlier_event'],results['Outlier_e_info']['Inner_event']
    Outlier_P_g, Outlier_P_e= results['Outlier_g_info']['Outlier_P'],results['Outlier_e_info']['Outlier_P']
    ax[0].scatter(1000*Inner_event_g[0], 1000*Inner_event_g[1], color="blue", alpha=0.5, s=0.5)
    ax[1].scatter(1000*Inner_event_e[0], 1000*Inner_event_e[1], color="red", alpha=0.5, s=0.5)
    ax[0].scatter(1000*Outlier_event_g[0], 1000*Outlier_event_g[1], color="grey", alpha=0.5, s=0.5)
    ax[1].scatter(1000*Outlier_event_e[0], 1000*Outlier_event_e[1], color="grey", alpha=0.5, s=0.5)
    text_msg1=''
    text_msg1 += r"$\rm{Outlier}= %.1f $"%(Outlier_P_g*100)+'%'
    plot_textbox_small(ax[0],text_msg1,x=0.47,y=0.93,fontsize=10)
    text_msg2=''
    text_msg2 += r"$\rm{Outlier}= %.1f $"%(Outlier_P_e*100)+'%'
    plot_textbox_small(ax[1],text_msg2,x=0.47,y=0.93,fontsize=10)

    Inte_data=[Inte_g_data,Inte_e_data]
    Mg=[Mgg,Mge]
    Me=[Meg,Mee]
    Pg=[Pgg,Pge]
    Pe=[Peg,Pee]

    ax1[0].plot(I_ro, Inte_data[0],'o',color='b',alpha=0.3,ms=3)
    ax1[1].plot(I_ro, Inte_data[1],'o',color='r',alpha=0.3,ms=3)
  
    for j in range(2):
        ax1[j].plot(I_fit, Mg[j],'--b',alpha=0.8,lw=1)
        ax1[j].plot(I_fit, Me[j],'--r',alpha=0.8,lw=1)
        ax1[j].set_ylim(10**(int(np.log10(np.max(Inte_data[0])))-3),10**(int(np.log10(np.max(Inte_data[0])))+1))
        ax1[j].set_xlim(ce_I/2-14*sig,ce_I/2+14*sig)
        ax1[j].axvline(x=ce_I/2,color='grey',linestyle='dashed',alpha=0.5,lw=1)
        ax1[j].set_yscale('log')
        text_msg=''
        text_msg += r"$P_{g}= %.1f $"%(Pg[j]*100)+'%'+'\n'
        text_msg += r"$P_{e}= %.1f $"%(Pe[j]*100)+'%'+'\n'
        text_msg += r"$\varepsilon_{o}=%.1f $"%(OE*100)+'%'
        plot_textbox_small(ax1[j],text_msg,fontsize=10)
        ax1[j].set_xlabel(r"$I^{'}\ $[mV]",size ='15')

    ax[0].scatter(0,0,c='k',s=15)
    ax[0].scatter(ce_I,ce_Q,c='k',s=15)
    ax[1].scatter(0,0,c='k',s=15)
    ax[1].scatter(ce_I,ce_Q,c='k',s=15)
    ax[0].add_patch(Ellipse(xy=[0,0],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[0].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[1].add_patch(Ellipse(xy=[0,0],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[1].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[0].add_patch(Ellipse(xy=[0,0],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[0].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[1].add_patch(Ellipse(xy=[0,0],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[1].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[0].add_patch(Ellipse(xy=[0,0],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[0].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[1].add_patch(Ellipse(xy=[0,0],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax[1].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))

    ax[0].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[0].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[1].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[1].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[0].set_xlim(np.minimum(min(I),min(Q))-sig*2,np.maximum(max(I),max(Q))+sig*2)
    ax[0].set_ylim(np.minimum(min(I),min(Q))-sig*2,np.maximum(max(I),max(Q))+sig*2)
    ax[1].set_xlim(np.minimum(min(I),min(Q))-sig*2,np.maximum(max(I),max(Q))+sig*2)
    ax[1].set_ylim(np.minimum(min(I),min(Q))-sig*2,np.maximum(max(I),max(Q))+sig*2)
    ax[0].axes.set_aspect('equal')
    ax[1].axes.set_aspect('equal')
    ax[0].set_title('Prepare |g>')
    ax[1].set_title('Prepare |e>')
    fig.tight_layout()

    font_kwargs = dict(fontweight="bold", fontsize='12',color='k',alpha=0.9)
    row_headers = [""]
    col_headers = ['Prepare |g>','Prepare |e>']
    add_headers(fig1, col_headers=col_headers, row_headers=row_headers, **font_kwargs)
    ax1[0].set_ylabel(r'$PDF$',size ='15')
    fig1.tight_layout()
    return fig, fig1

def Single_shot_2D_hist_plot(data):
    I,Q= data[0],data[1]
    bins=101
    I_=np.linspace(I.min(),I.max(),bins)  
    Q_=np.linspace(Q.min(),Q.max(),bins)
    hist, xedges, yedges = np.histogram2d(I,Q, bins=(bins,bins), density=True)
    X,Y= np.meshgrid(I_,Q_)
    fig, ax = plt.subplots(nrows =1,figsize =(5,4),dpi =200)
    cmap = plt.get_cmap('jet')
    vmax= np.max(hist)
    pcm = ax.pcolormesh(1000*X,1000*Y, hist.transpose(), vmin=0,vmax=vmax, cmap=cmap,shading='auto')
    cbar =fig.colorbar(pcm, ax=ax, extend='both', orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel(r"$I\ $[mV]",size ='15')
    ax.set_ylabel(r"$Q\ $[mV]",size ='15')
    cbar.set_label(r'$PDF$',size ='10')
    ax.set_title('Single shot data histogram')
    ax.axes.set_aspect('equal')
    fig.tight_layout()
    
    return fig 

def Qubit_state_single_shot_1Q(Processed_data,Analysis_result,anal_info,Save,Save_graph_path,id):
    # fig1= Single_shot_Rawdata_plot(Processed_data)
    fig2,fig3= Qubit_state_single_shot_plot(Analysis_result)
    # Direct_g=SQ_threshold_acquisition_statistic(data=Processed_data['g'],SQ_thres=dict(Ig=Analysis_result['Ig'],Qg=Analysis_result['Qg'],Ie=Analysis_result['Ie'],Qe=Analysis_result['Qe']))
    # Direct_e=SQ_threshold_acquisition_statistic(data=Processed_data['e'],SQ_thres=dict(Ig=Analysis_result['Ig'],Qg=Analysis_result['Qg'],Ie=Analysis_result['Ie'],Qe=Analysis_result['Qe']))
    # M_direct=np.array([[Direct_g['Pg'],Direct_g['Pe']],[Direct_e['Pg'],Direct_e['Pe']]])
    # fig4= Plot_assignment_matrix_SQ(Analysis_result['M'],Q=Raw_data_collect['g'].meas_q,title='Single shot fitting')
    # fig5= Plot_assignment_matrix_SQ(M_direct,Q=Raw_data_collect['g'].meas_q,title='Direct counting')
    # fig6= Single_shot_2D_hist_plot(Processed_data['g']) 
    # fig7= Single_shot_2D_hist_plot(Processed_data['e']) 
    # show_args(Analysis_result['error_pack'],title='Readout analysis (error budget)')
    return {'scatter': fig2, 'hist':fig3 }
