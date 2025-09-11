# from Hardware_setting import*
from .Fit_library import*
# from .Pulse_schedule_library import *
import matplotlib.pyplot as plt

#%% plot
def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
    ):

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )
            
    
def Two_hist_plot(data_1,data_2,xlabel,legend_1,legend_2):
    fig, ax = plt.subplots(nrows =1,figsize =(5,4),dpi =200) 
    ax.hist(np.array(data_1), bins='auto', density=False,color='b',alpha=0.5, label=legend_1)
    ax.axvline(np.mean(np.array(data_1)),label='Mean= %.2f'%np.mean(np.array(data_1)), color = "b", ls = "--",lw=1)
    ax.hist(np.array(data_2), bins='auto', density=False,color='r',alpha=0.5, label=legend_2)
    ax.axvline(np.mean(np.array(data_2)),label='Mean= %.2f'%np.mean(np.array(data_2)), color = "r", ls = "--",lw=1)
    ax.set_xlabel(xlabel,size ='14')
    ax.set_ylabel('Counts',size ='12')
    ax.legend(fontsize=10)
    fig.tight_layout()

def hist_plot(data,xlabel):
    times=len(data)
    #Double filtering
    data= data_filter(data,sigma_factor=3)
    data= data_filter(data,sigma_factor=2)
    fig, ax = plt.subplots(nrows =1,figsize =(2.5,2),dpi =200) 
    m, bins, patches = ax.hist(np.array(data), bins='auto', density=False)
    ax.axvline(np.mean(np.array(data)),label='Mean= %.2f'%np.mean(np.array(data))+"\n"+'Std= %.2f'%np.std(np.array(data)) +"\n"+"#%.0f"%(times), color = "k", ls = "--",lw=1)
    ax.set_xlabel(xlabel,size ='10')
    ax.set_ylabel('Counts',size ='10')
    ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def f01_hist_plot(data):
    fig, ax = plt.subplots(nrows =1,figsize =(2.5,2),dpi =200)
    data= np.array(data)
    times=len(data)
    data_tr=(data-np.mean(data))*1e-3 
    m, bins, patches = ax.hist(data_tr, bins='auto', density=False)
    std= np.std(data_tr)
    ax.set_xlabel(r"$f_{01}-\overline{f_{01}}\  $[kHz]",size ='10')
    ax.set_ylabel('Counts',size ='10')
    ax.set_ylim(0,max(m)*1.2)
    ax.legend(['Std= %.0f'%std+'kHz'+"\n"+"#%.0f"%(times)],fontsize=6,)
    fig.tight_layout()
    return fig

def cdf_plot(data,xlabel):
    data= data_filter(data)
    samples= np.sort(np.array(data))
    cdf= np.arange(1,len(samples)+1,1)/float(len(samples))
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.axvline(median(samples), color = "r",label='Median= %.2f'%median(samples), ls = "--",lw=1)
    ax.plot(samples,cdf,'-', color="blue", alpha=0.8, lw=3)
    ax.set_xlabel(xlabel,size ='18')
    ax.set_ylabel('CDF',size ='18')
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig
    
def Timeflow_plot(data,ylabel,Realtime,total_exp_time):
    samples= np.array(data)
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(samples))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(samples),len(samples))
        xlabel='Time flow'+' [times]'
    flow,samples= data_flow_filter(flow,samples)
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow,samples,'bo', alpha=0.8, ms=4)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(ylabel,size ='15')
    fig.tight_layout()    
    return fig

def Gamma_p_timeflow_plot(data,Delta_f01,min_Delta_f01,Delta_f01_thres,Realtime,total_exp_time):
    samples= np.array(data)
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(samples))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(samples),len(samples))
        xlabel='Time flow'+' [times]'
    flow_1=[]    
    samples_1=[]
    # filter by delta_f01
    for i in range(0,len(samples)-1):
        if np.abs(Delta_f01[i+1]-Delta_f01[i])<Delta_f01_thres and Delta_f01[i]>=min_Delta_f01:
            samples_1.append(samples[i])
            flow_1.append(flow[i])
        else:
            pass     

    flow_2,samples_2= data_flow_filter(flow_1,samples_1)
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow_2,samples_2/1000,'bo', alpha=0.8, ms=4)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(r"$\Gamma_{p}\ $[1/ms]",size ='15')
    fig.tight_layout()    
    return fig
    
def f01_Timeflow_plot(f01_data,Realtime,total_exp_time):
    samples= np.array(f01_data)*1e-3
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(samples))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(samples),len(samples))
        xlabel='Time flow'+' [times]'
    flow,samples= data_flow_filter(flow,samples)
    ylabel=r"$f_{01}-\overline{f_{01}}\  $[kHz]"
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow,samples-np.mean(samples),'bo', alpha=0.8, ms=4)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(ylabel,size ='15')
    fig.tight_layout() 
    return fig

def Detuning_Timeflow_plot(f01_data,Realtime,total_exp_time):
    samples= np.array(f01_data)*1e-6
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(samples))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(samples),len(samples))
        xlabel='Time flow'+' [times]'
    ylabel=r"$\delta f_{01}\  $[MHz]"
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow,samples,'bo', alpha=0.8, ms=4)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(ylabel,size ='15')
    fig.tight_layout() 
    return fig

def Detuning_threshold_data_Timeflow_plot(f01_data,Delta_f01_thres,min_Delta_f01,Realtime,total_exp_time):
    samples= np.array(f01_data)*1e-6
    used_data,non_used_data,flow_used, flow_non_used= [],[],[],[]
    used_data_index = []
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(samples))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(samples),len(samples))
        xlabel='Time flow'+' [times]'
    ylabel=r"$\delta f_{01}\  $[MHz]"
    for i in range(0,len(samples)-1):
        if np.abs(f01_data[i+1]-f01_data[i])<Delta_f01_thres and f01_data[i]>=min_Delta_f01:
            used_data_index.append(i)
            used_data.append(samples[i])
            flow_used.append(flow[i])
        else:
            non_used_data.append(samples[i])
            flow_non_used.append(flow[i])
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow_used,used_data,'bo',label= 'used data', alpha=0.8, ms=4)
    ax.plot(flow_non_used,non_used_data,'ro',label= 'non-used data', alpha=0.8, ms=4)
    ax.axhline(min_Delta_f01*1e-6, color = "k", ls = "--",lw=3)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(ylabel,size ='15')
    ax.legend(fontsize=10)
    fig.tight_layout() 
    return fig, used_data_index

def f01_Charge_parity_timeflow_plot(f01_data1,f01_data2,filter_on,ng_precise_normalize,precise_max_delta_f01,Realtime,total_exp_time):
    samples_1= np.array(f01_data1)*1e-6 #f1
    samples_2= np.array(f01_data2)*1e-6 #f2
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(samples_1))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(samples_1),len(samples_1))
        xlabel='Time flow'+' [times]'
    if filter_on:
        # f1<f2 by definition
        Ave_f01=np.mean((samples_1+samples_2)/2)
        #filter once
        flow_,samples_1_,samples_2_=[],[],[]
        for i in range(len(samples_1)):
            if samples_1[i]>Ave_f01 or samples_2[i]<Ave_f01:
                pass
            else:
                flow_.append(flow[i])
                samples_1_.append(samples_1[i])
                samples_2_.append(samples_2[i])
        samples_1_,samples_2_=np.array(samples_1_),np.array(samples_2_)
        Ave_f01_filter=np.mean((samples_1_+samples_2_)/2)  
        #filter twice  
        flow_2,samples_1_2,samples_2_2=[],[],[]
        for i in range(len(samples_1_)):
            if np.abs((samples_1_[i]+samples_2_[i])/2-Ave_f01_filter)/(max(samples_2_)-min(samples_1_))<0.05:
                flow_2.append(flow_[i])
                samples_1_2.append(samples_1_[i])
                samples_2_2.append(samples_2_[i])
        samples_1_2,samples_2_2=np.array(samples_1_2),np.array(samples_2_2)
        Ave_f01_filter=np.mean((samples_1_2+samples_2_2)/2)
        
    else:
        flow_2=flow
        samples_1_2,samples_2_2=np.array(samples_1),np.array(samples_2)
        Ave_f01_filter=np.mean((samples_1_2+samples_2_2)/2)
    ylabel='Detuning [MHz]'
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow_2,samples_1_2,'b',marker='o', alpha=0.6, lw=3, ms=8)
    ax.plot(flow_2,samples_2_2,'r',marker='o', alpha=0.6, lw=3, ms=8)
    ax.plot(flow_2,(samples_1_2+samples_2_2)/2,marker='x',markerfacecolor='None',markeredgecolor='k',linestyle='None', alpha=0.6, ms=8)
    ax.axhline(Ave_f01_filter, color = "k", ls = "--",lw=3)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(ylabel,size ='15')
    print('Average detuning=',Ave_f01_filter)
    fig.tight_layout() 

    ylabel= r"$\delta f_{01}\  $[MHz]"
    if ng_precise_normalize:
        max_delta_f01= precise_max_delta_f01*1e-6
    else:
        max_delta_f01= max((samples_2_2-samples_1_2)/2)

    def delta_f01_to_ng(delta_f01):
        return  np.abs(np.arccos(delta_f01/max_delta_f01)/2/np.pi)
 
    def ng_to_delta_f01(ng):
        return max_delta_f01*np.cos(2*np.pi*np.abs(ng)) 
    fig1, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow_2,(samples_2_2-samples_1_2)/2,'g',marker='o', alpha=0.6, lw=3, ms=8)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(ylabel,size ='15')
    ax.set_ylim(0,max_delta_f01)
    ax2 = ax.secondary_yaxis('right', functions=(delta_f01_to_ng,ng_to_delta_f01))
    ax2.set_yticks([0.25,0.15,0])
    ax2.set_ylabel(r"$n_{g}\  $",size ='15')
    fig1.tight_layout() 

    return fig,fig1,Ave_f01_filter, (samples_1_2+samples_2_2)/2*1e6

def charge_spec_plot(f1,f2,result):
    f1,f2= np.array(f1)*1e-6,np.array(f2)*1e-6
    A_fit, C_g_fit, f01_bar_fit = result['fitting']['A_fit'],result['fitting']['C_g_fit'],result['fitting']['f01_bar_fit']
    phi1_fit= result['fitting']['phi_fit']
    phi2_fit= result['fitting']['phi_fit']+np.pi
    Vg_data= result['fitting']['Vg']
    Vg_fit = result['fitting']['Vg_fit']
    f1_fit = Detuning_ng_func(Vg_fit, A_fit, C_g_fit, phi1_fit, f01_bar_fit)
    f2_fit = Detuning_ng_func(Vg_fit, A_fit, C_g_fit, phi2_fit, f01_bar_fit)
    Vg_0= result['Vg_0']
    delta_f01= result['delta_f01']
    f01_bar= result['f01_bar']
    x_data= result['x_data']
    y_data= result['y_data']
    final_labels=result['final_labels']
    def Vg_to_ng(Vg):
        return  C_g_fit*Vg-phi1_fit/2/np.pi

    def ng_to_Vg(ng):
        return (ng+phi1_fit/2/np.pi)/C_g_fit
    
    
    ng_fit= Vg_to_ng(Vg_fit)
    ng= Vg_to_ng(Vg_data)
    
    text_msg = "Fit results\n"
    text_msg+= r"$2\cdot \delta f_{01}=%.2f\ $[MHz]"%(2*delta_f01)+'\n'
    text_msg+= r"$V_{g}(n_{g}=0)=%.3f\ $[V]"%(Vg_0)+'\n'
    text_msg+= r"$\overline{f_{01}}=%.4f\ $[GHz]"%(f01_bar*1e-3)
    
    idx_ng_0= find_nearest(ng_fit,0)
    if f1_fit[idx_ng_0]>f2_fit[idx_ng_0]:
        label1='even'
        label2='odd'
    else:
        label1='odd'
        label2='even'
        
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    #ax.plot(ng,(f1-f01_bar_fit),'bo', alpha=0.5, ms=5)
    #ax.plot(ng,(f2-f01_bar_fit),'bo', alpha=0.5, ms=5)
    ax.plot(Vg_to_ng(x_data[final_labels == 0]),(y_data[final_labels == 0]-f01_bar_fit),'bo', alpha=0.5, ms=5)
    ax.plot(Vg_to_ng(x_data[final_labels == 1]),(y_data[final_labels == 1]-f01_bar_fit),'ro', alpha=0.5, ms=5)
    ax.plot(ng_fit, (f1_fit-f01_bar_fit), "b--", label=label1,lw=1.5)
    ax.plot(ng_fit, (f2_fit-f01_bar_fit), "r--", label=label2,lw=1.5)
    ax.set_xlabel(r"$n_{g}\  [2e]$",size ='15')
    ax.set_ylabel(r"$f_{01}\  -\overline{f_{01}}\ $[MHz]",size ='15')
    ax2 = ax.secondary_xaxis('top', functions=(ng_to_Vg,Vg_to_ng))
    ax2.set_xlabel(r"$V_{g}\  $[V]",size ='15')
    ax.legend(loc='center left',fontsize=10,bbox_to_anchor=(1, 0.5))
    plot_textbox(ax,text_msg,fontsize=10)
    fig.tight_layout() 
    return fig
    
def T2_Charge_parity_timeflow_plot(T2_data1,T2_data2,Realtime,total_exp_time):
    samples_1= np.array(T2_data1)
    samples_2= np.array(T2_data2)
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(samples_1))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(samples_1),len(samples_1))
        xlabel='Time flow'+' [times]'
    ylabel=r"$T_{2}\  [\mu$s]"
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow,samples_1,'b',marker='o', alpha=0.6, lw=3, ms=8)
    ax.plot(flow,samples_2,'r',marker='o', alpha=0.6, lw=3, ms=8)
    ax.plot(flow,(samples_1+samples_2)/2,marker='x',markerfacecolor='None',markeredgecolor='k',linestyle='None', alpha=0.6, ms=8)
    ax.axhline(np.mean((samples_1+samples_2)/2), color = "k", ls = "--",lw=3)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(ylabel,size ='15')
    fig.tight_layout() 
    return fig

def Two_P_th_hist_plot(data_1,data_2,legend_1,legend_2,f01):
    hbar = 1.054571800*1e-34
    kB = 1.38e-23    
    Wa= 2*np.pi*f01 #assume same f01 
    
    def PetoT(Pe):
        Pe=Pe/100
        Pg= 1-Pe
        T= (-hbar*Wa)/(kB*np.log(Pe/Pg))*1000
        return T  
 
    def TtoPe(T):
        T=T/1000
        Pe= 1/(1+np.exp(hbar*Wa/T/kB))*100
        return Pe
    fig, ax = plt.subplots(nrows =1,figsize =(5,4),dpi =200) 
    P_1= np.mean(np.array(data_1))*100
    P_2= np.mean(np.array(data_2))*100
    ax.hist(np.array(data_1)*100, bins='auto', density=False,color='b',alpha=0.5, label=legend_1)
    ax.axvline(P_1,label='Mean= %.2f'%P_1+'%,'+ "\n"+r'$T_{q}=%.2f\ $[mK]'%PetoT(P_1), color = "b", ls = "--",lw=1)
    ax.hist(np.array(data_2)*100, bins='auto', density=False,color='r',alpha=0.5, label=legend_2)
    ax.axvline(P_2,label='Mean= %.2f'%P_2+'%,'+ "\n"+r'$T_{q}=%.2f\ $[mK]'%PetoT(P_2), color = "r", ls = "--",lw=1)
    ax2 = ax.secondary_xaxis('top', functions=(PetoT,TtoPe))
    ax.set_xlabel(r"$P_{1}\  $(%)",size ='12')
    ax2.set_xlabel(r'$T_{q}\ $[mK]',size ='12')
    ax.set_ylabel('Counts',size ='12')
    ax.legend(fontsize=10)
    fig.tight_layout()

    
def P_th_hist_plot(data,f01):
    hbar = 1.054571800*1e-34
    kB = 1.38e-23    
    Wa= 2*np.pi*f01
    times=len(data)
    def PetoT(Pe):
        Pe=np.clip(np.asarray(Pe)/100,1e-6,1-1e-6)
        Pg= 1-Pe
        T= (-hbar*Wa)/(kB*np.log(Pe/Pg))*1000
        return T  
 
    def TtoPe(T):
        T=T/1000
        Pe= 1/(1+np.exp(hbar*Wa/T/kB))*100
        return Pe
    if len(data)>=2: 
       data= data_filter(data,sigma_factor=2)
    fig, ax = plt.subplots(nrows =1,figsize =(2.5,2),dpi =200) 
    m, bins, patches = ax.hist(np.array(data)*100, bins='auto', density=False)
    P= np.mean(np.array(data))*100
    ax.axvline(P,label='Mean= %.2f'%P+'%,'+ "\n"+r'$T_{q}=%.2f\ $[mK]'%PetoT(P)+"\n"+"#%.0f"%(times), color = "k", ls = "--",lw=1)
    ax.set_xlabel(r"$P_{1}\  $(%)",size ='10')
    ax.set_ylabel('Counts',size ='10')
    ax.legend(fontsize=5,loc='upper right')
    ax.set_xlim(min(bins)*0.92,max(bins)*1.1)
    ax.set_ylim(0,1.1*max(m))
    ax2 = ax.secondary_xaxis('top', functions=(PetoT,TtoPe))
    ax2.set_xlabel(r'$T_{q}\ $[mK]',size ='10')
    fig.tight_layout()
    return fig
    
def P_th_cdf_plot(data,f01):
    hbar = 1.054571800*1e-34
    kB = 1.38e-23    
    Wa= 2*np.pi*f01
    
    def PetoT(Pe):
        Pe=np.clip(np.asarray(Pe)/100,1e-6,1-1e-6)
        Pg= 1-Pe
        T= (-hbar*Wa)/(kB*np.log(Pe/Pg))*1000
        return T  
    
    def TtoPe(T):
        T=T/1000
        Pe= 1/(1+np.exp(hbar*Wa/T/kB))*100
        return Pe
    data= data_filter(data,sigma_factor=2)
    samples= np.sort(np.array(data))*100
    cdf= np.arange(1,len(samples)+1,1)/float(len(samples))
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.axvline(median(samples), color = "r",label=r"$median$", ls = "--",lw=2)
    ax.plot(samples,cdf,'-', color="blue", alpha=0.8, lw=3)
    ax2 = ax.secondary_xaxis('top', functions=(PetoT,TtoPe))
    ax2.set_xlabel(r'$T_{q}\ $[mK]',size ='15')
    ax.set_xlabel(r"$P_{1}\  $(%)",size ='15')
    ax.set_ylabel('CDF',size ='15')
    ax.legend()
    fig.tight_layout()
    return fig

def P_th_Timeflow_plot(data,f01,Realtime,total_exp_time):
    samples= np.array(data)*100
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(samples))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(samples),len(samples))
        xlabel='Time flow'+' [times]'
    hbar = 1.054571800*1e-34
    kB = 1.38e-23    
    Wa= 2*np.pi*f01
    
    def PetoT(Pe):
        Pe=np.clip(np.asarray(Pe)/100,1e-6,1-1e-6)
        Pg= 1-Pe
        T= (-hbar*Wa)/(kB*np.log(Pe/Pg))*1000
        return T  
    
    def TtoPe(T):
        T=T/1000
        Pe= 1/(1+np.exp(hbar*Wa/T/kB))*100
        return Pe
    flow,samples= data_flow_filter(flow,samples,sigma_factor=2)
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow,samples,'bo', alpha=0.8, ms=5)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(r"$P_{1}\  $(%)",size ='15')
    ax2 = ax.secondary_yaxis('right', functions=(PetoT,TtoPe))
    ax2.set_ylabel(r'$T_{q}\ $[mK]',size ='15')
    fig.tight_layout() 
    return fig
    
def T1_Timeflow_from_single_shot_plot(data,Realtime,total_exp_time,Reset_time,IQ_g,IQ_e,R_integration):
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(data))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(data),len(data))
        xlabel='Time flow'+' [times]'
    if IQ_g[0]!=IQ_e[0] or IQ_g[1]!=IQ_e[1]:
        cg_I,cg_Q,ce_I,ce_Q= IQ_g[0],IQ_g[1],IQ_e[0],IQ_e[1]
        angle= np.angle(ce_I-cg_I+(ce_Q-cg_Q)*1j)
        ce_I_new= rot(ce_I-cg_I,ce_Q-cg_Q,angle)[0]
    T1_avg,Stay_g_avg=[],[]
    for k in range(len(data)):
        I,Q = np.array(data[k][0]), np.array(data[k][1])
        I_new, Q_new= rot(I-cg_I,Q-cg_Q,angle)
        #Threshold filtering
        thres= ce_I_new/2
        Nor=[]
        for i in range(len(data[0][0])):
            if I_new[i]>thres:
                Nor.append(1)
            else:
                Nor.append(0)
        T1_us,Stay_g= T1_from_single_shot_analysis(Nor,Reset_time,R_integration)
        T1_avg.append(np.mean(T1_us))
        Stay_g_avg.append(np.mean(Stay_g))

    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow,T1_avg,'bo', alpha=0.8, ms=5)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(r"$T_{1}^{s}\ [\mu$s]",size ='15')
    fig.tight_layout()

    def t_to_Gamma(t):
        return 1/t  #kHz 

    def Gamma_to_t(G):
        return 1/G
    fig1, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow,np.array(Stay_g_avg)/1000,'bo', alpha=0.8, ms=5)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(r"$t_{|g>}\ $[ms]",size ='15')
    ax2 = ax.secondary_yaxis('right', functions=(t_to_Gamma,Gamma_to_t))
    ax2.set_ylabel(r'$\Gamma_{01}\ $[kHz]',size ='15')
    fig1.tight_layout() 

    return fig,fig1,T1_avg

def SQ_state_Timeflow_single_shot_ref_plot(data,Reset_time,IQ_g,IQ_e,R_integration):
    if IQ_g[0]!=IQ_e[0] or IQ_g[1]!=IQ_e[1]:
        cg_I,cg_Q,ce_I,ce_Q= IQ_g[0],IQ_g[1],IQ_e[0],IQ_e[1]
        I,Q = np.array(data[0]), np.array(data[1])
        flow=np.linspace(0,(R_integration+Reset_time)*(len(I)-1),len(I))
        angle= np.angle(ce_I-cg_I+(ce_Q-cg_Q)*1j)
        ce_I_new= rot(ce_I-cg_I,ce_Q-cg_Q,angle)[0]
        I_new, Q_new= rot(I-cg_I,Q-cg_Q,angle)

        fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
        ax.plot(flow,I_new*1e3,'ko', alpha=0.5, ms=3)
        ax.set_xlabel('Time flow [s]',size ='15')
        ax.set_ylabel('Voltage [mV] (rotated axis)',size ='15')
        ax.axhline(y=0,label=r'$|g>\ $',color='b',linestyle='dashed', alpha=1,lw=2)
        ax.axhline(y=ce_I_new*1e3,label=r'$|e>\ $',color='r',linestyle='dashed', alpha=1,lw=2)
        ax.legend(fontsize=12,loc='upper left')
        ax.set_ylim(None,1.5*max(I_new*1e3))
        fig.tight_layout()

        #Threshold filtering
        thres= ce_I_new/2
        Nor=[]
        for i in range(len(I_new)):
            if I_new[i]>thres:
                Nor.append(1)
            else:
                Nor.append(0)

        fig1, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
        ax.plot(flow,Nor,'ko', alpha=0.5, ms=3)
        ax.set_xlabel('Time flow [s]',size ='15')
        ax.set_ylabel('Population',size ='15')
        #ax.axhline(y=0,label=r'$|g>\ $',color='b',linestyle='dashed', alpha=1,lw=2)
        #ax.axhline(y=1,label=r'$|e>\ $',color='r',linestyle='dashed', alpha=1,lw=2)
        #ax.legend(fontsize=12,loc='upper left')
        ax.set_ylim(-0.25,1.5)
        fig1.tight_layout() 

        T1_us,Stay_g= T1_from_single_shot_analysis(Nor,Reset_time,R_integration)

        fig2, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
        ax.plot(np.linspace(min(flow),max(flow),len(T1_us)),T1_us,'ko', alpha=0.5, ms=3)
        ax.set_xlabel('Time flow [s]',size ='15')
        ax.set_ylabel(r"$T_{1}\ [\mu$s]",size ='15')
        fig2.tight_layout()

        fig3, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
        ax.plot(np.linspace(min(flow),max(flow),len(Stay_g)),Stay_g,'ko', alpha=0.5, ms=3)
        ax.set_xlabel('Time flow [s]',size ='15')
        ax.set_ylabel(r"$t_{|g>}\ [\mu$s]",size ='15')
        fig3.tight_layout()
        return fig, fig1, fig2, fig3
    else:
        return None



def SQ_state_Timeflow_plot(result,single_exp_time):
    I_rot=result['I_rot']
    ce_I_rot=result['ce_I_rot']
    label=result['label']
    flow=1000*np.linspace(0,(single_exp_time)*(len(I_rot)-1),len(I_rot))

    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow,I_rot*1e3,'ko', alpha=0.5, ms=3)
    ax.set_xlabel('Time flow [ms]',size ='15')
    ax.set_ylabel('Voltage [mV] (rotated axis)',size ='15')
    ax.axhline(y=0,label=r'$|g>\ $',color='b',linestyle='dashed', alpha=1,lw=2)
    ax.axhline(y=ce_I_rot*1e3,label=r'$|e>\ $',color='r',linestyle='dashed', alpha=1,lw=2)
    ax.legend(fontsize=12,loc='upper left')
    ax.set_ylim(None,1.5*max(I_rot*1e3))
    fig.tight_layout()

    fig1, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(flow,label,'ko', alpha=0.5, ms=3)
    ax.set_xlabel('Time flow [ms]',size ='15')
    ax.set_ylabel('Population',size ='15')
    #ax.axhline(y=0,label=r'$|g>\ $',color='b',linestyle='dashed', alpha=1,lw=2)
    #ax.axhline(y=1,label=r'$|e>\ $',color='r',linestyle='dashed', alpha=1,lw=2)
    #ax.legend(fontsize=12,loc='upper left')
    ax.set_ylim(-0.25,1.5)
    fig1.tight_layout() 

    return fig, fig1
 

def Cross_correlation_plot(result): 
    Pth=np.array(result['Thermal']['Thermal_P'])*100
    T1=np.array(result['T1']['T1_us'])
    T1_s=np.array(result['T1_s'])
    T2=np.array(result['Ramsey']['T2_us'])
    f01=np.array(result['Ramsey']['Real_detune'])*1e-3
    f01_tr= np.abs(f01-np.mean(f01))
    Data_list=[Pth,T1_s,T1,T2,f01_tr]
    label_list= [r"$P_{th}$",r"$T^{s}_{1}$",r"$T_{1}$",r"$T_{2}$",r"$\vert\Delta f_{01}\vert\  $"]
    Cross_data=[]
    for i in range(len(Data_list)):
        for j in range(len(Data_list)):
            value= Cross_correlation(Data_list[i],Data_list[j])
            Cross_data.append(np.mean(value))
    xlabs=ylabs= label_list 
    M= np.reshape(Cross_data,(len(Data_list),len(Data_list)))    
    fig, ax = plt.subplots(nrows=1,figsize=(5,5),dpi=150)
    ax.set_xticks(np.arange(len(xlabs)), labels = xlabs)
    ax.set_yticks(np.arange(len(ylabs)), labels = ylabs)
    im= ax.imshow(M, cmap='Greens', vmin=-1, vmax=1)
    im_ratio = M.shape[0]/M.shape[1]
    cbar = ax.figure.colorbar(im, ax = ax,fraction=0.0453*im_ratio)

    for i in range(len(xlabs)):
        for j in range(len(ylabs)):
            if i!=j:
                text = ax.text(j, i, round(M[i, j], 2),
                               ha = "center", va = "center", color = "k",size=15)
            else:
                text = ax.text(j, i, round(M[i, j], 2),
                               ha = "center", va = "center", color = "w",size=15)
                
    cbar.ax.set_ylabel("Correlation", rotation = -90, va = "bottom",size=15)
    cbar.set_ticks([-1,-0.5,0,0.5,1])
    cbar.ax.tick_params(labelsize=18)
    ax.tick_params(labelsize='18')
    fig.tight_layout()

    fig1, ax = plt.subplots(nrows=5,ncols=5,figsize=(15,15),dpi=200)
    ax[1][0].scatter(Pth,T1_s,c='b',alpha=0.5,s=15)
    ax[2][0].scatter(Pth,T1,c='b',alpha=0.5,s=15)
    ax[3][0].scatter(Pth,T2,c='b',alpha=0.5,s=15)
    ax[4][0].scatter(Pth,f01_tr,c='b',alpha=0.5,s=15)
    ax[2][1].scatter(T1_s,T1,c='b',alpha=0.5,s=15)
    ax[3][1].scatter(T1_s,T2,c='b',alpha=0.5,s=15)
    ax[4][1].scatter(T1_s,f01_tr,c='b',alpha=0.5,s=15)
    ax[3][2].scatter(T1,T2,c='b',alpha=0.5,s=15)
    ax[4][2].scatter(T1,f01_tr,c='b',alpha=0.5,s=15)
    ax[4][3].scatter(T2,f01_tr,c='b',alpha=0.5,s=15)
    ax[1][0].set_ylabel(r"$T_{1}^{s}\ [\mu$s]",size=18)
    ax[2][0].set_ylabel(r"$T_{1}\ [\mu$s]",size=18)
    ax[3][0].set_ylabel(r"$T_{2}\ [\mu$s]",size=18)
    ax[4][0].set_ylabel(r"$\vert\Delta f_{01}\vert\  $[kHz]",size=18)
    ax[4][0].set_xlabel(r"$P_{th}\  $(%)",size=18)
    ax[4][1].set_xlabel(r"$T_{1}^{s}\ [\mu$s]",size=18)
    ax[4][2].set_xlabel(r"$T_{1}\ [\mu$s]",size=18)
    ax[4][3].set_xlabel(r"$T_{2}\ [\mu$s]",size=18)
    for i in range(5):
        for j in range(5):
            ax[i][j].tick_params(labelsize='12')
            if i==j or j>i:
                ax[i][j].axis('off')
            else:
                pass
    fig1.tight_layout()
    return fig,fig1


def Cross_correlation_plot_2Q(result): 
    meas_q= list(result.keys())
    Q1,Q2= meas_q[0],meas_q[1]
    Pth_1=np.array(result[Q1]['Thermal']['Thermal_P'])*100
    T1_1=np.array(result[Q1]['T1']['T1_us'])
    T1_s_1=np.array(result[Q1]['T1_s'])
    T2_1=np.array(result[Q1]['Ramsey']['T2_us'])
    f01_1=np.array(result[Q1]['Ramsey']['Real_detune'])*1e-3
    f01_tr_1= np.abs(f01_1-np.mean(f01_1))
    Pth_2=np.array(result[Q2]['Thermal']['Thermal_P'])*100
    T1_2=np.array(result[Q2]['T1']['T1_us'])
    T1_s_2=np.array(result[Q2]['T1_s'])
    T2_2=np.array(result[Q2]['Ramsey']['T2_us'])
    f01_2=np.array(result[Q2]['Ramsey']['Real_detune'])*1e-3
    f01_tr_2= np.abs(f01_2-np.mean(f01_2))
    Data_list=[Pth_1,Pth_2,T1_s_1,T1_s_2,T1_1,T1_2,T2_1,T2_2,f01_tr_1,f01_tr_2]
    label_list= [r"$P_{th}(Q1)$",r"$P_{th}(Q2)$",
                 r"$T^{s}_{1}(Q1)$",r"$T^{s}_{1}(Q2)$",
                 r"$T_{1}(Q1)$",r"$T_{1}(Q2)$",
                 r"$T_{2}(Q1)$",r"$T_{2}(Q2)$",
                 r"$\vert\Delta f_{01}\vert(Q1)$",r"$\vert\Delta f_{01}\vert(Q2)$"]
    Cross_data=[]
    for i in range(len(Data_list)):
        for j in range(len(Data_list)):
            value= Cross_correlation(Data_list[i],Data_list[j])
            Cross_data.append(np.mean(value))
    xlabs=ylabs= label_list 
    M= np.reshape(Cross_data,(len(Data_list),len(Data_list)))    

    fig, ax = plt.subplots(nrows=1,figsize=(25,25),dpi=200)
    ax.set_xticks(np.arange(len(xlabs)), labels = xlabs)
    ax.set_yticks(np.arange(len(ylabs)), labels = ylabs)
    im= ax.imshow(M, cmap='Greens', vmin=-1, vmax=1)
    im_ratio = M.shape[0]/M.shape[1]
    cbar = ax.figure.colorbar(im, ax = ax,fraction=0.0453*im_ratio)

    for i in range(len(xlabs)):
        for j in range(len(ylabs)):
            if i!=j:
                text = ax.text(j, i, round(M[i, j], 2),
                               ha = "center", va = "center", color = "k",size=28)
            else:
                text = ax.text(j, i, round(M[i, j], 2),
                               ha = "center", va = "center", color = "w",size=28)
                
    cbar.ax.set_ylabel("Correlation", rotation = -90, va = "bottom",size=28)
    cbar.set_ticks([-1,-0.5,0,0.5,1])
    cbar.ax.tick_params(labelsize=28)
    ax.tick_params(labelsize='28')
    fig.tight_layout()
    
    Data_list_1=[Pth_1,T1_s_1,T1_1,T2_1,f01_tr_1]
    Data_list_2=[Pth_2,T1_s_2,T1_2,T2_2,f01_tr_2]
    label_list=[r"$P_{th}\  $(%)",
                 r"$T_{1}^{s}\ [\mu$s]",
                 r"$T_{1}\ [\mu$s]",
                 r"$T_{2}\ [\mu$s]",
                 r"$\vert\Delta f_{01}\vert\  $[kHz]"]
    fig1, ax = plt.subplots(nrows=len(Data_list_1),ncols=len(Data_list_2),figsize=(15,15),dpi=200)
    for i in range(len(Data_list_1)):
        for j in range(len(Data_list_2)):
            ax[i][j].scatter(Data_list_1[j],Data_list_2[i],c='b',alpha=0.5,s=15)
            ax[-1][i].set_xlabel(label_list[i],size=18)
            ax[i][0].set_ylabel(label_list[i],size=18)
            ax[i][j].tick_params(labelsize='12')

    fig1.supxlabel('Q1',size=22)
    fig1.supylabel('Q2',size=22)
    fig1.tight_layout()
    return fig,fig1


def Coherence_monitor_plot(result: dict, f01, Realtime, total_exp_time):
    num = len(result['Thermal']['Thermal_P'])
    if Realtime:
        flow = np.linspace(0, total_exp_time, num)
        xlabel = 'Time flow [hours]'
    else:
        flow = np.linspace(1, num, num)
        xlabel = 'Time flow [times]'

    hbar = 1.054571800e-34
    kB = 1.38e-23
    Wa = 2*np.pi*f01

    def PetoT(Pe):
        Pe=np.clip(np.asarray(Pe)/100,1e-6,1-1e-6)
        Pg=1-Pe
        T=(-hbar*Wa)/(kB*np.log(Pe/Pg))*1000
        T[~np.isfinite(T)]=np.nan
        return T

    def TtoPe(T):
        T=np.asarray(T)/1000
        Pe=1/(1+np.exp(hbar*Wa/(T*kB)))*100
        Pe[~np.isfinite(Pe)]=np.nan
        return Pe

    T1 = np.array(result['T1']['T1_us'])
    T2 = np.array(result['Ramsey']['T2_us'])
    T_phi = 1/(1/T2-1/(2*T1))
    f01 = np.array(result['Ramsey']['Real_detune']) * 1e-3
    Pth = np.array(result['Thermal']['Thermal_P']) * 100


    fig, ax = plt.subplots(nrows=3, figsize=(9, 6), dpi=200)
    ax[0].plot(flow, Pth, 'bo', alpha=0.8, ms=3)
    ax[1].plot(flow, T1, label=r"$T_{1}$", marker='x', markerfacecolor='None',
               markeredgecolor='orange', linestyle='None', alpha=0.8, ms=3)
    ax[1].plot(flow, T2, label=r"$T_{2}$", marker='x', markerfacecolor='None',
               markeredgecolor='k', linestyle='None', alpha=0.8, ms=3)
    # ax[1].plot(flow, T_phi, label=r"$T_{\phi}$", marker='x', markerfacecolor='None',
    #            markeredgecolor='r', linestyle='None', alpha=0.8, ms=3)
    ax[2].plot(flow, f01 - np.mean(f01), marker='x', markerfacecolor='None',
               markeredgecolor='k', linestyle='None', alpha=0.8, ms=3)

    ax[0].set_ylabel(r"$P_{th}\  $(%)", size='15')
    ax[1].set_ylabel(r"$\rm{Lifetime}\  [\mu s]$", size='15')
    ax[2].set_ylabel(r"$\Delta f_{01}\  $[kHz]", size='15')
    ax2 = ax[0].secondary_yaxis('right', functions=(PetoT, TtoPe))
    ax2.set_ylabel(r'$T_{q}\ $[mK]', size='15')
    ax[1].legend(fontsize=12, loc='lower left')
    ax[1].set_yscale('log')
    ax[2].set_xlabel(xlabel, size='15')
    fig.tight_layout()
    return fig


def Coherence_monitor_plot_2Q(result:dict,f01,Realtime,total_exp_time):
    meas_q= list(result.keys())
    Q1,Q2= meas_q[0],meas_q[1]
    f01_1,f01_2= f01[Q1],f01[Q2] 
    num=len(result[Q1]['Thermal']['Thermal_P'])
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,num)
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,num,num)
        xlabel='Time flow'+' [times]'
    hbar = 1.054571800*1e-34
    kB = 1.38e-23    
    Wa_1= 2*np.pi*f01_1
    Wa_2= 2*np.pi*f01_2
    
    def PetoT(Pe):
        Pe=np.clip(np.asarray(Pe)/100,1e-6,1-1e-6)
        Pg=1-Pe
        T=(-hbar*Wa_1)/(kB*np.log(Pe/Pg))*1000  
        T[~np.isfinite(T)]=np.nan
        return T

    def TtoPe(T):
        T=np.asarray(T)/1000
        Pe=1/(1+np.exp(hbar*Wa_1/(T*kB)))*100
        Pe[~np.isfinite(Pe)]=np.nan
        return Pe
    
    def PetoT_2(Pe):
        Pe=np.clip(np.asarray(Pe)/100,1e-6,1-1e-6)
        Pg=1-Pe
        T=(-hbar*Wa_2)/(kB*np.log(Pe/Pg))*1000
        T[~np.isfinite(T)]=np.nan
        return T

    def TtoPe_2(T):
        T=np.asarray(T)/1000
        Pe=1/(1+np.exp(hbar*Wa_2/(T*kB)))*100
        Pe[~np.isfinite(Pe)]=np.nan
        return Pe
    
    T1_1=np.array(result[Q1]['T1']['T1_us'])
    T2_1=np.array(result[Q1]['Ramsey']['T2_us'])
    T_phi_1=1/(1/T2_1-1/(2*T1_1))
    f01_1=np.array(result[Q1]['Ramsey']['Real_detune'])*1e-3
    Pth_1=np.array(result[Q1]['Thermal']['Thermal_P'])*100
    T1_2=np.array(result[Q2]['T1']['T1_us'])
    T2_2=np.array(result[Q2]['Ramsey']['T2_us'])
    T_phi_2=1/(1/T2_2-1/(2*T1_2))
    f01_2=np.array(result[Q2]['Ramsey']['Real_detune'])*1e-3
    Pth_2=np.array(result[Q2]['Thermal']['Thermal_P'])*100
    valid = (np.isfinite(T1_1) & np.isfinite(T2_1) & np.isfinite(f01_1) & np.isfinite(Pth_1) &
        np.isfinite(T1_2) & np.isfinite(T2_2) & np.isfinite(f01_2) & np.isfinite(Pth_2)
            )
    T1_1, T2_1, T_phi_1, f01_1, Pth_1 = T1_1[valid], T2_1[valid], T_phi_1[valid], f01_1[valid], Pth_1[valid]
    T1_2, T2_2, T_phi_2, f01_2, Pth_2 = T1_2[valid], T2_2[valid], T_phi_2[valid], f01_2[valid], Pth_2[valid]
    flow = flow[valid]

    fig, ax = plt.subplots(nrows =6,figsize =(9,12),dpi =200) 
    ax[0].plot(flow,Pth_1,'bo', alpha=0.8, ms=3)
    ax[1].plot(flow,Pth_2,'bo', alpha=0.8, ms=3)
    #ax[1].plot(flow,T_phi_1,label=r"$T_{\phi}$",marker='x',markerfacecolor='None',markeredgecolor='r',linestyle='None', alpha=0.8, ms=3)
    ax[2].plot(flow,T1_1,label=r"$T_{1}$",marker='x',markerfacecolor='None',markeredgecolor='orange',linestyle='None', alpha=0.8, ms=3)
    ax[2].plot(flow,T2_1,label=r"$T_{2}$",marker='x',markerfacecolor='None',markeredgecolor='k',linestyle='None', alpha=0.8, ms=3)
    ax[3].plot(flow,T1_2,label=r"$T_{1}$",marker='x',markerfacecolor='None',markeredgecolor='orange',linestyle='None', alpha=0.8, ms=3)
    ax[3].plot(flow,T2_2,label=r"$T_{2}$",marker='x',markerfacecolor='None',markeredgecolor='k',linestyle='None', alpha=0.8, ms=3)
    ax[4].plot(flow,f01_1-np.mean(f01_1),marker='x',markerfacecolor='None',markeredgecolor='k',linestyle='None', alpha=0.8, ms=3)
    ax[5].plot(flow,f01_2-np.mean(f01_2),marker='x',markerfacecolor='None',markeredgecolor='k',linestyle='None', alpha=0.8, ms=3)
    ax[0].set_ylabel(r"$P_{th}\  $(%)",size ='15')
    ax[1].set_ylabel(r"$P_{th}\  $(%)",size ='15')
    ax[2].set_ylabel(r"$\rm{Lifetime}\  [\mu s]$",size ='15')
    ax[3].set_ylabel(r"$\rm{Lifetime}\  [\mu s]$",size ='15')
    ax[4].set_ylabel(r"$f_{01}-\overline{f_{01}}\  $[kHz]",size ='15')
    ax[5].set_ylabel(r"$f_{01}-\overline{f_{01}}\  $[kHz]",size ='15')
    ax2_1 = ax[0].secondary_yaxis('right', functions=(PetoT,TtoPe))
    ax2_1.set_ylabel(r'$T_{q}\ $[mK]',size ='15')
    ax2_2 = ax[1].secondary_yaxis('right', functions=(PetoT_2,TtoPe_2))
    ax2_2.set_ylabel(r'$T_{q}\ $[mK]',size ='15')
    ax[2].legend(fontsize=12,loc='lower left')
    ax[2].set_yscale('log')
    ax[3].legend(fontsize=12,loc='lower left')
    ax[3].set_yscale('log')
    ax[-1].set_xlabel(xlabel,size ='15')
    fig.tight_layout() 
    return fig

def Z_bias_error_bar_plot(times,Z_bias,data,xlabel:str,ylabel:str,inverse_ylabel:str):
    T1_array= np.array(data).reshape(times, len(Z_bias))
    mean, sigma= T1_array.mean(axis=0),T1_array.std(axis=0)
    fig, ax = plt.subplots(nrows =2,figsize =(6,4),dpi =200) 
    for i in range(times):
            ax[0].plot(Z_bias,T1_array[i],'o', color="blue", alpha=0.5, ms=2)
            ax[1].plot(Z_bias,1/T1_array[i],'o', color="blue", alpha=0.5, ms=2)
    ax[0].plot(Z_bias,mean,'-', color="r", alpha=0.6,lw=1.5)
    ax[1].plot(Z_bias,1/mean,'-', color="r", alpha=0.6,lw=1.5)
    ax[1].set_ylabel(inverse_ylabel,size ='15')
    #ax.errorbar(Z_bias, mean, yerr=2*sigma, fmt='o', color='blue',
              #ecolor='blue', elinewidth=2, capsize=5)
    ax[0].set_ylabel(ylabel,size ='15')
    ax[1].set_xlabel(xlabel,size ='15')
    ax[0].set_title(r"$times= %.0f $" %(times),size ='15')
    fig.tight_layout()
    return fig


def Z_bias_cdf_plot(times,Z_bias,data,xlabel:str):
    T1_array= np.array(data).reshape(times, len(Z_bias))

    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =250) 
    for i in range(len(Z_bias)):
        samples= np.sort(T1_array.transpose()[i])
        cdf= np.arange(1,len(samples)+1,1)/float(len(samples))
       # ax.axvline(median(samples), ls = "--",lw=1)
        ax.plot(samples,cdf,'-',label=r"$Z bias= %.3f $" %(Z_bias[i]), alpha=1, lw=1)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel("CDF",size ='15')
    ax.set_title(r"$times= %.0f $" %(times),size ='15')
    #ax.legend(fontsize=6,loc='center right',bbox_to_anchor=(1,0.4),ncol=2,fancybox=True)
    fig.tight_layout()



def Ramsey_F_Z_bias_error_bar_plot(times,Z_bias,data,xlabel:str):
    Ramsey_F_array= np.array(data).reshape(times, len(Z_bias))
    Ramsey_F_mean= Ramsey_F_array.mean(axis=0)
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =250) 
    ax.plot(Z_bias,Ramsey_F_mean*1e-6,'o', color="blue", alpha=0.5, ms=10)
    ax.set_ylabel(r"$detuning\ $[MHz]",size ='15')
    ax.set_xlabel(xlabel,size ='15')
    ax.set_title(r"$times= %.0f $" %(times),size ='15')
    fig.tight_layout()
    
    return fig
    
def dataset_to_array(dataset:xr.core.dataset.Dataset,dims:float):
    if dims==1:
        I= dataset.y0.data.transpose()
        Q= dataset.y1.data.transpose()
    elif dims==2:
        gridded_dataset = dh.to_gridded_dataset(dataset)
        I=gridded_dataset.y0.data.transpose()
        Q=gridded_dataset.y1.data.transpose()
    else: raise KeyError ('dims is not 1 or 2')  
    
    return I,Q

def Multi_dataset_to_array(dataset:xr.core.dataset.Dataset,dims:float,Q:list):
    I_data={}
    Q_data={}
    if dims==1:
        for i in range(len(Q)):
            I_data[Q[i]]= dataset['y'+str(i)].data.transpose()
            Q_data[Q[i]]= dataset['y'+str(1+i)].data.transpose()
    elif dims==2:
        gridded_dataset = dh.to_gridded_dataset(dataset)
        for i in range(len(Q)):
            I_data[Q[i]]= gridded_dataset['y'+str(i)].data.transpose()
            Q_data[Q[i]]= gridded_dataset['y'+str(1+i)].data.transpose()        
        
    else: raise KeyError ('dims is not 1 or 2')  
    
    return I_data,Q_data

def plot_cavity_spectrum(f,y1,y2):
    Nor_f=1/1000
    z1_label= 'Amp'+' [mV]'
    z2_label= 'Phase'+' [deg]'
    xlabel='Frequency [GHz]'
    fig,ax= plt.subplots(nrows=1,ncols =2,figsize =(8,3),dpi =200)
    ax[0].plot(f/1e9, y1/Nor_f,'b',alpha=0.8,lw=2)
    ax[0].set_xlabel(xlabel,size ='15')
    ax[0].set_ylabel(z1_label,size ='15')
    ax[1].plot(f/1e9, y2,'r',alpha=0.8,lw=2)
    ax[1].set_xlabel(xlabel,size ='15')
    ax[1].set_ylabel(z2_label,size ='15')
    fig.tight_layout()
    return fig

def plot_2D_cavityfluxdep(x:np.ndarray,y:np.ndarray,data:np.ndarray,label:list,title:str,S21_normalize:bool,data_type:str,color_bound:bool,bound_value:list,plot_linecut:bool,linecut:float,fit_plot:bool,fit_info:list):
    if S21_normalize:
        if data_type =='Amplitude':
            data1=data.transpose()
            for i in range(len(data1)):
                data1[i]=data1[i]/np.mean(data1[i][:10])

            Nor_f= 1
            z_label= r'$\vert S_{21}\vert\ $'
            data=data1.transpose()

        elif data_type =='Phase':
            data[data<0]+= 360
            Nor_f= 1
            z_label= r'$\angle S_{21}\ [deg]$'
            
        else: raise TypeError ('data_type is not Amplitude or Phase') 
    else:
        if data_type =='Amplitude':
            Nor_f= 1/1000
            z_label= 'Amp'+' [mV]'

        elif data_type =='Phase':
            data[data<0]+= 360
            Nor_f= 1
            z_label= r'$\angle S_{21}\ [deg]$'
            
        else: raise TypeError ('data_type is not Amplitude or Phase') 
    X,Y=np.meshgrid(x,y)
    z= data
    cmap = plt.get_cmap('RdBu_r')
    fig, ax0 = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    if color_bound:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,vmin=bound_value[0],vmax=bound_value[1],shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        cbar.set_ticks([bound_value[0],bound_value[1]])
    else:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
    if fit_plot ==True:    
        ax0.plot(fit_info[0][0],fit_info[0][1],'--r',alpha=1,lw=1.5)
        for i in range(len(fit_info[1])):
            ax0.axvline(x=fit_info[1][i],linestyle='dashed',c='r', alpha=1,lw=1.5)
    else:
        pass         
    ax0.set_xlabel(label[0],size ='15')
    ax0.set_ylabel(label[1],size ='15')
    ax0.set_title(title,size ='15')
    cbar.set_label(z_label,size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()
    
    if plot_linecut is True:
        fig1,ax= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
        ax.plot(x, z[linecut]/Nor_f,'b',alpha=0.8,lw=2)
        ax.set_xlabel(label[0],size ='15')
        ax.set_ylabel(z_label,size ='15')
        ax.set_title(title+'_Linecut_'+label[1]+' '+str(np.around(y[linecut],4)))
        fig1.tight_layout()
    return fig
        
def plot_2D_cavitypowerdep(x:np.ndarray,y:np.ndarray,data:np.ndarray,label:list,title:str,S21_normalize:bool,data_type:str,color_bound:bool,bound_value:list,plot_linecut:bool,linecut:float):
    if S21_normalize:
        if data_type =='Amplitude':
            for i in range(len(data)):
                dip_idx=list(data[i]).index(min(data[i]))
                if dip_idx>len(data)/2:
                    data[i]=data[i]/np.mean(data[i][:10])
                else:
                    data[i]=data[i]/np.mean(data[i][-10:])
            Nor_f= 1
            z_label= r'$\vert S_{21}\vert\ $'

        elif data_type =='Phase':
            data[data<0]+= 360
            Nor_f= 1
            z_label= r'$\angle S_{21}\ [deg]$'
            
        else: raise TypeError ('data_type is not Amplitude or Phase') 
    else:
        if data_type =='Amplitude':
            Nor_f= 1/1000
            z_label= 'Amp'+' [mV]'

        elif data_type =='Phase':
            data[data<0]+= 360
            Nor_f= 1
            z_label= r'$\angle S_{21}\ [deg]$'
            
        else: raise TypeError ('data_type is not Amplitude or Phase') 
    
    X,Y=np.meshgrid(x,y)
    z= data
    cmap = plt.get_cmap('RdBu_r')
    fig, ax0 = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    if color_bound:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f,cmap=cmap,vmin=bound_value[0],vmax=bound_value[1],shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        cbar.set_ticks([bound_value[0],bound_value[1]])
    else:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
         
    ax0.set_xlabel(label[0],size ='15')
    ax0.set_ylabel(label[1],size ='15')
    ax0.set_xticks([min(x),max(x)])
    ax0.set_title(title,size ='15')
    cbar.set_label(z_label,size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()
    
    if plot_linecut is True:
        fig,ax= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
        ax.plot(x, z[linecut]/Nor_f,'b',alpha=0.8,lw=2)
        ax.set_xlabel(label[0],size ='15')
        ax.set_ylabel(z_label,size ='15')
        ax.set_title(title+'_Linecut_'+label[1]+' '+str(np.around(y[linecut],4)))
        fig.tight_layout()
    return fig    
        
def plot_2D(x:np.ndarray,y:np.ndarray,data:np.ndarray,label:list,title:str,readout_qubit_info:bool, P_rescale:bool, Dis:any,color_bound:bool,bound_value:list,plot_linecut:bool,linecut:float):
    if readout_qubit_info:
        if P_rescale is not True:
            Nor_f=1/1000
            z_label= 'Contrast'+' [mV]'
        else:
            Nor_f= Dis
            z_label= r"$P_{1}\ $"
    else: 
        Nor_f=1/1000
        z_label= 'Amp'+' [mV]'
    
    X,Y=np.meshgrid(x,y)
    z= data
    cmap = plt.get_cmap('RdBu_r')
    fig, ax0 = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    if color_bound:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f,cmap=cmap,vmin=bound_value[0],vmax=bound_value[1],shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        cbar.set_ticks([bound_value[0],bound_value[1]])
    else:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        
    ax0.set_xlabel(label[0],size ='15')
    ax0.set_ylabel(label[1],size ='15')
    ax0.set_title(title,size ='15')
    cbar.set_label(z_label,size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()
    
    if plot_linecut is True:
        fig1,ax1= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
        ax1.plot(x, z[linecut]/Nor_f,'b',alpha=0.8,lw=2)
        ax1.set_xlabel(label[0],size ='15')
        ax1.set_ylabel(z_label,size ='15')
        ax1.set_title(title+'_Linecut_'+label[1]+'='+str(np.around(y[linecut],4)))
        fig1.tight_layout()
    return fig

def plot_2D_2Q_population(x:np.ndarray,y:np.ndarray,data:np.ndarray,label:list,title:str):
    X,Y=np.meshgrid(x,y)
    z= data
    cmap = plt.get_cmap('RdBu_r')
    fig, ax0 = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    pcm = ax0.pcolormesh(X, Y, z,cmap=cmap,vmin=0,vmax=1,shading='auto')
    cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
    cbar.set_ticks([0,0.5,1])
    ax0.set_xlabel(label[0],size ='15')
    ax0.set_ylabel(label[1],size ='15')
    ax0.set_title(title,size ='15')
    cbar.set_label(label[2],size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()
    return fig

def plot_flux_crosstalk(x:np.ndarray,y:np.ndarray,data:np.ndarray,result:dict,label:list,title:str,readout_qubit_info:bool, P_rescale:bool, Dis:any,color_bound:bool,bound_value:list,plot_linecut:bool,linecut:float):
    max_data_cor= result['max_data_cor']
    V_target, V_meas= max_data_cor['V_target'],max_data_cor['V_meas']
    para_fit= result['fit']['para_fit']
    fitting= result['fit']['fitting']
    if readout_qubit_info:
        if P_rescale is not True:
            Nor_f=1/1000
            z_label= 'Contrast'+' [mV]'
        else:
            Nor_f= Dis
            z_label= r"$P_{1}\ $"
    else: 
        Nor_f=1/1000
        z_label= 'Amp'+' [mV]'
    
    X,Y=np.meshgrid(x,y)
    z= data
    cmap = plt.get_cmap('hot_r')
    fig, ax0 = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    if color_bound:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f,cmap=cmap,vmin=bound_value[0],vmax=bound_value[1],shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        cbar.set_ticks([bound_value[0],bound_value[1]])
    else:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
    #ax0.plot(V_target, V_meas,'bo',alpha=0.4, ms=4)  
    ax0.plot(para_fit, fitting,color='k',linestyle='dashed', alpha=0.8,lw=1.5)  
    ax0.set_xlabel(label[0],size ='15')
    ax0.set_ylabel(label[1],size ='15')
    ax0.set_title(title,size ='15')
    cbar.set_label(z_label,size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()
    
    return fig

def plot_2D_with_lines(x:np.ndarray,y:np.ndarray,data:np.ndarray,label:list,title:str,readout_qubit_info:bool, P_rescale:bool, Dis:any,
                       color_bound:bool,bound_value:list,plot_linecut:bool,linecut:float,vertical:bool,horizontal:bool,vertical_X_value:float,horizontal_Y_value:float):
    if readout_qubit_info:
        if P_rescale is not True:
            Nor_f=1/1000
            z_label= 'Contrast'+' [mV]'
        else:
            Nor_f= Dis
            z_label= r"$P_{1}\ $"
    else: 
        Nor_f=1/1000
        z_label= 'Amp'+' [mV]'
    
    X,Y=np.meshgrid(x,y)
    z= data
    cmap = plt.get_cmap('hot_r')
    fig, ax0 = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    if color_bound:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f,cmap=cmap,vmin=bound_value[0],vmax=bound_value[1],shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        cbar.set_ticks([bound_value[0],bound_value[1]])
    else:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
    if vertical:
        ax0.axvline(x=vertical_X_value,color='k',linestyle='dashed', alpha=0.8,lw=1.5)
    if horizontal:
        ax0.axhline(y=horizontal_Y_value,color='k',linestyle='dashed', alpha=0.8,lw=1.5)

    ax0.set_xlabel(label[0],size ='15')
    ax0.set_ylabel(label[1],size ='15')
    ax0.set_title(title,size ='15')
    cbar.set_label(z_label,size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()
    
    if plot_linecut is True:
        fig1,ax1= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
        ax1.plot(x, z[linecut]/Nor_f,'b',alpha=0.8,lw=2)
        ax1.set_xlabel(label[0],size ='15')
        ax1.set_ylabel(z_label,size ='15')
        ax1.set_title(title+'_Linecut_'+label[1]+'='+str(np.around(y[linecut],4)))
        fig1.tight_layout()
    return fig



def T1_plot_2D(x:np.ndarray,y:any,data:np.ndarray,title:str,readout_qubit_info:bool, P_rescale:bool, Dis:any,color_bound:bool,bound_value:list,plot_linecut:bool,linecut:float,Realtime:False,total_exp_time:float):
    
    xlabel=r"$t_{f}$"+r"$\ [\mu$s]"
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,y)
        ylabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,y,y)
        ylabel='Time flow'+' [times]'
    if readout_qubit_info:
        if P_rescale is not True:
            Nor_f=1/1000
            z_label= 'Contrast'+' [mV]'
        else:
            Nor_f= Dis
            z_label= r"$P_{1}\ $"
    else: 
        Nor_f=1/1000
        z_label= 'Amp'+' [mV]'
    
    X,Y=np.meshgrid(x,flow)
    z= data
    cmap = plt.get_cmap('hot_r')
    fig, ax0 = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    if color_bound:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f,cmap=cmap,vmin=bound_value[0],vmax=bound_value[1],shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        cbar.set_ticks([bound_value[0],bound_value[1]])
    else:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        
    ax0.set_xlabel(xlabel,size ='15')
    ax0.set_ylabel(ylabel,size ='15')
    ax0.set_title(title,size ='15')
    cbar.set_label(z_label,size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()
    
    if plot_linecut is True:
        fig,ax= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
        ax.plot(x, z[linecut]/Nor_f,'b',alpha=0.8,lw=2)
        ax.set_xlabel(label[0],size ='15')
        ax.set_ylabel(z_label,size ='15')
        ax.set_title(title+'_Linecut_'+label[1]+'='+str(np.around(y[linecut],4)))
        fig.tight_layout()





def plot_2D_Zgate_two_tone_combined(data:list,label:list,title:str,readout_qubit_info:bool, P_rescale:bool, Dis:any,color_bound:bool,bound_value:list):
    if readout_qubit_info:
        if P_rescale is not True:
            Nor_f=1/1000
            z_label= 'Contrast'+' [mV]'
        else:
            Nor_f= Dis
            z_label= r"$P_{1}\ $"
    else: 
        Nor_f=1/1000
        z_label= 'Amp'+' [mV]'
    cmap = plt.get_cmap('RdBu_r')
    fig, ax0 = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    if color_bound:
        for i in range(len(data)):
            X,Y=np.meshgrid(data[i]['first_samples']/1e9,data[i]['second_samples'])
            z= data[i]['data'][0]
            if i ==0:
               pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,vmin=bound_value[0],vmax=bound_value[1],shading='auto')
            else:
                ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,vmin=bound_value[0],vmax=bound_value[1],shading='auto')
                
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        cbar.set_ticks([bound_value[0],bound_value[1]])
    else:
        for i in range(len(data)):
            X,Y=np.meshgrid(data[i]['first_samples']/1e9,data[i]['second_samples'])
            z= data[i]['data'][0]
            if i ==0:
               pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,shading='auto')
            else:
                ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,shading='auto')
                
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
    ax0.set_xlabel(label[0],size ='15')
    ax0.set_ylabel(label[1],size ='15')
    ax0.set_title(title,size ='15')
    cbar.set_label(z_label,size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()
    return fig
    


def plot_textbox(ax,text, **kw):
    box_props = dict(boxstyle="round", pad=0.4, facecolor="white", alpha=0.5)
    new_kw_with_defaults = dict(
        x=1.05,
        y=0.95,
        transform=ax.transAxes,
        bbox=box_props,
        verticalalignment="top",
        s=text,
    )
    new_kw_with_defaults.update(kw)
    t_obj = ax.text(**new_kw_with_defaults)
    return t_obj

def plot_textbox_small(ax,text, **kw):
    box_props = dict(boxstyle="round", pad=0.4, facecolor="white", alpha=0.5)
    new_kw_with_defaults = dict(
        x=0.68,
        y=0.95,
        transform=ax.transAxes,
        bbox=box_props,
        verticalalignment="top",
        s=text,
        fontsize=6
    )
    new_kw_with_defaults.update(kw)
    t_obj = ax.text(**new_kw_with_defaults)
    return t_obj

def Raw_avg_IQ_plot(I:np.ndarray,Q:np.ndarray,SSI:np.ndarray,SSQ:np.ndarray):
    I,Q= I*1000,Q*1000
    SSI,SSQ= SSI*1000,SSQ*1000
    fig, ax = plt.subplots(ncols =1,figsize =(3,3),dpi =200)
    ax.scatter(SSI, SSQ, color="blue", alpha=0.5, s=2)
    ax.scatter(I, Q, color="k", alpha=0.5, s=2)
    ax.set_xlabel(r"$I\ $[mV]",size ='15')
    ax.set_ylabel(r"$Q\ $[mV]",size ='15')
    ax.axes.set_aspect('equal')
    fig.tight_layout()


def Thermal_population_single_shot_plot(data:list,results:dict,y_scale:str):
    I, Q= np.array(data[0]), np.array(data[1])
    fig, ax = plt.subplots(nrows =1,figsize =(3,3),dpi =200)
    ax.scatter(1000*I, 1000*Q, color="blue", alpha=0.5, s=5)   
    ax.set_xlabel(r"$I\ $[mV]",size ='15')
    ax.set_ylabel(r"$Q\ $[mV]",size ='15')
    ax.set_title('Single shot raw data')
    ax.axes.set_aspect('equal')
    fig.tight_layout()
    ce_I,ce_Q,sig=1000*results['fit_pack'][0][0],1000*results['fit_pack'][0][1],1000*results['fit_pack'][3]
    Inte_g_data= results['fit_pack'][4]
    Ig,Qg= results['rot_IQdata'][0][0],results['rot_IQdata'][0][1]
    I_ro,I_fit= 1000*results['I_ro'],1000*results['I_fit']
    Mgg= gauss_func(I_fit,0,sig,results['fit_pack'][1])
    Meg= gauss_func(I_fit,ce_I,sig,results['fit_pack'][2])
    
    fig, ax = plt.subplots(nrows =1,figsize =(3,3),dpi =200)
    fig1,ax1= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    ax.scatter(1000*Ig, 1000*Qg, color="blue", alpha=0.5, s=5)  
    ax1.plot(I_ro, Inte_g_data,'bo',alpha=0.5,ms=8)
    ax1.plot(I_fit, Mgg,'b',alpha=0.8,lw=2)
    ax1.plot(I_fit, Meg,'--b',alpha=0.8,lw=2)
    ax1.plot(I_fit, Mgg+Meg,'--k',alpha=1,lw=1)

    if y_scale=='log':
        ax1.set_yscale('log')
        ax1.set_ylim(np.max(Inte_g_data)*1e-3,np.max(Inte_g_data)*10)
    elif y_scale=='linear': 
        pass
    else: raise KeyError ('Incorrect statement of y_scale')
    ax.scatter(0,0,c='k',s=15)
    ax.scatter(ce_I,ce_Q,c='k',s=15)
    ax.add_patch(Ellipse(xy=[0,0],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax.add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax.set_xlim(1000*np.minimum(min(Ig),min(Qg)),1000*np.maximum(max(Ig),max(Qg)))
    ax.set_ylim(1000*np.minimum(min(Ig),min(Qg)),1000*np.maximum(max(Ig),max(Qg)))
    ax.set_xlabel(r"$I\ $[mV]",size ='15')
    ax.set_ylabel(r"$Q\ $[mV]",size ='15')
    ax.axes.set_aspect('equal')
    ax.set_title('Single shot rotated data',size ='12')
    fig.tight_layout()
    ax1.set_xlabel(r"$I^{'}\ $[mV]",size ='15')
    ax1.set_ylabel(r'$PDF$',size ='15')
    ax1.set_title('Single shot rotated data')
    ax1.set_xlim(-4*sig,ce_I+4*sig)
    fig1.tight_layout()
    return fig,fig1


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
    # red_g= np.where(1000*Ig>ce_I/2)[0]  
    # blue_g= np.where(1000*Ig<ce_I/2)[0]
    # red_e= np.where(1000*Ie>ce_I/2)[0]  
    # blue_e= np.where(1000*Ie<ce_I/2)[0]
    
    # ax[0].scatter(1000*Ig[blue_g], 1000*Qg[blue_g], color="blue", alpha=0.5, s=1)
    # ax[0].scatter(1000*Ig[red_g], 1000*Qg[red_g], color="red", alpha=0.5, s=1)
    # ax[1].scatter(1000*Ie[blue_e], 1000*Qe[blue_e], color="blue", alpha=0.5, s=1)
    # ax[1].scatter(1000*Ie[red_e], 1000*Qe[red_e], color="red", alpha=0.5, s=1)
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

def PA_alignment_qubit_state_single_shot_plot_same_Fs(data_off:dict,data_on:dict,PAoff_results:dict,PAon_results:dict,tau_off:float,tau_on:float,Target_Fs_value:float):
    Ig_data_off,Qg_data_off,Ie_data_off,Qe_data_off= 1000*np.array(data_off['g'][0]), 1000*np.array(data_off['g'][1]) ,1000*np.array(data_off['e'][0]) , 1000*np.array(data_off['e'][1])
    I_off,Q_off= np.hstack([Ig_data_off,Ie_data_off]), np.hstack([Qg_data_off,Qe_data_off])
    Ig_data_on,Qg_data_on,Ie_data_on,Qe_data_on= 1000*np.array(data_on['g'][0]), 1000*np.array(data_on['g'][1]) ,1000*np.array(data_on['e'][0]) , 1000*np.array(data_on['e'][1])
    I_on,Q_on= np.hstack([Ig_data_on,Ie_data_on]), np.hstack([Qg_data_on,Qe_data_on])
    cg_I_off,cg_Q_off= 1000*PAoff_results['Ig'],1000*PAoff_results['Qg']
    ce_I_off,ce_Q_off= 1000*PAoff_results['Ie'],1000*PAoff_results['Qe']
    cg_I_on,cg_Q_on= 1000*PAon_results['Ig'],1000*PAon_results['Qg']
    ce_I_on,ce_Q_on= 1000*PAon_results['Ie'],1000*PAon_results['Qe']
    D_off,sig_off= 1000*PAoff_results['error_pack']['D'],1000*PAoff_results['error_pack']['sigma']
    D_on,  sig_on= 1000*PAon_results['error_pack']['D'],1000*PAon_results['error_pack']['sigma']
    x1_cross,y1_cross, slope1 = get_line_eq(cg_I_off,cg_Q_off,ce_I_off,ce_Q_off)
    x2_cross,y2_cross, slope2 = get_line_eq(cg_I_on,cg_Q_on,ce_I_on,ce_Q_on)
    
    red_I,red_Q=[],[]
    blue_I,blue_Q=[],[]
    
    for i in range(len(I_off)):
        if Q_off[i]-line_equ(I_off[i],slope1,x1_cross,y1_cross) <= 0 and cg_Q_off-line_equ(cg_I_off,slope1,x1_cross,y1_cross) < 0:
            blue_I.append(I_off[i])
            blue_Q.append(Q_off[i])
        elif Q_off[i]-line_equ(I_off[i],slope1,x1_cross,y1_cross) > 0 and cg_Q_off-line_equ(cg_I_off,slope1,x1_cross,y1_cross) < 0:
            red_I.append(I_off[i])
            red_Q.append(Q_off[i])
        elif Q_off[i]-line_equ(I_off[i],slope1,x1_cross,y1_cross) > 0 and cg_Q_off-line_equ(cg_I_off,slope1,x1_cross,y1_cross) > 0:
            blue_I.append(I_off[i])
            blue_Q.append(Q_off[i])
        elif Q_off[i]-line_equ(I_off[i],slope1,x1_cross,y1_cross) <= 0 and cg_Q_off-line_equ(cg_I_off,slope1,x1_cross,y1_cross) > 0:
            red_I.append(I_off[i])
            red_Q.append(Q_off[i])    
    for i in range(len(I_on)):
        if Q_on[i]-line_equ(I_on[i],slope2,x2_cross,y2_cross) <= 0 and cg_Q_on-line_equ(cg_I_on,slope2,x2_cross,y2_cross) < 0:
            blue_I.append(I_on[i])
            blue_Q.append(Q_on[i])
        elif Q_on[i]-line_equ(I_on[i],slope2,x2_cross,y2_cross) > 0 and cg_Q_on-line_equ(cg_I_on,slope2,x2_cross,y2_cross) < 0:
            red_I.append(I_on[i])
            red_Q.append(Q_on[i])
        elif Q_on[i]-line_equ(I_on[i],slope2,x2_cross,y2_cross) > 0 and cg_Q_on-line_equ(cg_I_on,slope2,x2_cross,y2_cross) > 0:
            blue_I.append(I_on[i])
            blue_Q.append(Q_on[i])
        elif Q_on[i]-line_equ(I_on[i],slope2,x2_cross,y2_cross) <= 0 and cg_Q_on-line_equ(cg_I_on,slope2,x2_cross,y2_cross) > 0:
            red_I.append(I_on[i])
            red_Q.append(Q_on[i])        
    
    fig, ax = plt.subplots(ncols =1,figsize =(6,6),dpi =300)
    ax.scatter(blue_I,blue_Q, color="blue", alpha=0.3, s=2)
    ax.scatter(red_I,red_Q, color="red", alpha=0.3, s=2)
    ax.scatter(cg_I_off,cg_Q_off,c='k',s=25)
    ax.scatter(ce_I_off,ce_Q_off,c='k',s=25)
    ax.scatter(cg_I_on,cg_Q_on,c='k',s=25)
    ax.scatter(ce_I_on,ce_Q_on,c='k',s=25)

    ax.add_patch(Ellipse(xy=[cg_I_off,cg_Q_off],width=sig_off*4,height=sig_off*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=2, linestyle='--',angle=0))
    ax.add_patch(Ellipse(xy=[ce_I_off,ce_Q_off],width=sig_off*4,height=sig_off*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=2, linestyle='--',angle=0))
    ax.add_patch(Ellipse(xy=[cg_I_on,cg_Q_on],width=sig_on*4,height=sig_on*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=2, linestyle='--',angle=0))
    ax.add_patch(Ellipse(xy=[ce_I_on,ce_Q_on],width=sig_on*4,height=sig_on*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=2, linestyle='--',angle=0))
    ax.plot([cg_I_off,ce_I_off],[cg_Q_off,ce_Q_off],'--k',alpha=0.8,lw=2)
    ax.plot([cg_I_on,ce_I_on],[cg_Q_on,ce_Q_on],'--k',alpha=0.8,lw=2)
    ax.set_xlabel(r"$I\ $[mV]",size ='15')
    ax.set_ylabel(r"$Q\ $[mV]",size ='15')
    mean_I=(cg_I_off+cg_I_on+ce_I_off+ce_I_on)/4
    mean_Q=(cg_Q_off+cg_Q_on+ce_Q_off+ce_Q_on)/4
    ax.set_xlim(mean_I-6*sig_on,mean_I+8*sig_on)
    ax.set_ylim(mean_Q-8*sig_on,mean_Q+7*sig_on)
    Power_gain=10*np.log10((D_on/D_off)**2)
    Noise_diff=10*np.log10((sig_on/sig_off)**2)
    SNR_impro= Power_gain-Noise_diff
    text_msg=''
    text_msg += r"$F_{s}\approx %.1f\ $"%(Target_Fs_value*100) +'%\n'
    text_msg += r"$\tau_{off}= %.0f\ $"%(tau_off) +'ns\n'
    text_msg += r"$\tau_{on}= %.0f\ $"%(tau_on) +'ns'

    plot_textbox(ax,text_msg,x=0.65,y=0.2,fontsize=15)
    ax.axes.set_aspect('equal')
    fig.tight_layout() 
    
    
def PA_alignment_qubit_state_single_shot_plot(data_off:dict,data_on:dict,PAoff_results:dict,PAon_results:dict):
    Ig_data_off,Qg_data_off,Ie_data_off,Qe_data_off= 1000*np.array(data_off['g'][0]), 1000*np.array(data_off['g'][1]) ,1000*np.array(data_off['e'][0]) , 1000*np.array(data_off['e'][1])
    I_off,Q_off= np.hstack([Ig_data_off,Ie_data_off]), np.hstack([Qg_data_off,Qe_data_off])
    Ig_data_on,Qg_data_on,Ie_data_on,Qe_data_on= 1000*np.array(data_on['g'][0]), 1000*np.array(data_on['g'][1]) ,1000*np.array(data_on['e'][0]) , 1000*np.array(data_on['e'][1])
    I_on,Q_on= np.hstack([Ig_data_on,Ie_data_on]), np.hstack([Qg_data_on,Qe_data_on])
    cg_I_off,cg_Q_off= 1000*PAoff_results['Ig'],1000*PAoff_results['Qg']
    ce_I_off,ce_Q_off= 1000*PAoff_results['Ie'],1000*PAoff_results['Qe']
    cg_I_on,cg_Q_on= 1000*PAon_results['Ig'],1000*PAon_results['Qg']
    ce_I_on,ce_Q_on= 1000*PAon_results['Ie'],1000*PAon_results['Qe']
    D_off,sig_off= 1000*PAoff_results['error_pack']['D'],1000*PAoff_results['error_pack']['sigma']
    D_on,  sig_on= 1000*PAon_results['error_pack']['D'],1000*PAon_results['error_pack']['sigma']
    x1_cross,y1_cross, slope1 = get_line_eq(cg_I_off,cg_Q_off,ce_I_off,ce_Q_off)
    x2_cross,y2_cross, slope2 = get_line_eq(cg_I_on,cg_Q_on,ce_I_on,ce_Q_on)
    
    red_I,red_Q=[],[]
    blue_I,blue_Q=[],[]
    
    for i in range(len(I_off)):
        if Q_off[i]-line_equ(I_off[i],slope1,x1_cross,y1_cross) <= 0 and cg_Q_off-line_equ(cg_I_off,slope1,x1_cross,y1_cross) < 0:
            blue_I.append(I_off[i])
            blue_Q.append(Q_off[i])
        elif Q_off[i]-line_equ(I_off[i],slope1,x1_cross,y1_cross) > 0 and cg_Q_off-line_equ(cg_I_off,slope1,x1_cross,y1_cross) < 0:
            red_I.append(I_off[i])
            red_Q.append(Q_off[i])
        elif Q_off[i]-line_equ(I_off[i],slope1,x1_cross,y1_cross) > 0 and cg_Q_off-line_equ(cg_I_off,slope1,x1_cross,y1_cross) > 0:
            blue_I.append(I_off[i])
            blue_Q.append(Q_off[i])
        elif Q_off[i]-line_equ(I_off[i],slope1,x1_cross,y1_cross) <= 0 and cg_Q_off-line_equ(cg_I_off,slope1,x1_cross,y1_cross) > 0:
            red_I.append(I_off[i])
            red_Q.append(Q_off[i])    
    for i in range(len(I_on)):
        if Q_on[i]-line_equ(I_on[i],slope2,x2_cross,y2_cross) <= 0 and cg_Q_on-line_equ(cg_I_on,slope2,x2_cross,y2_cross) < 0:
            blue_I.append(I_on[i])
            blue_Q.append(Q_on[i])
        elif Q_on[i]-line_equ(I_on[i],slope2,x2_cross,y2_cross) > 0 and cg_Q_on-line_equ(cg_I_on,slope2,x2_cross,y2_cross) < 0:
            red_I.append(I_on[i])
            red_Q.append(Q_on[i])
        elif Q_on[i]-line_equ(I_on[i],slope2,x2_cross,y2_cross) > 0 and cg_Q_on-line_equ(cg_I_on,slope2,x2_cross,y2_cross) > 0:
            blue_I.append(I_on[i])
            blue_Q.append(Q_on[i])
        elif Q_on[i]-line_equ(I_on[i],slope2,x2_cross,y2_cross) <= 0 and cg_Q_on-line_equ(cg_I_on,slope2,x2_cross,y2_cross) > 0:
            red_I.append(I_on[i])
            red_Q.append(Q_on[i])        
    
    fig, ax = plt.subplots(ncols =1,figsize =(6,6),dpi =300)
    ax.scatter(blue_I,blue_Q, color="blue", alpha=0.3, s=2)
    ax.scatter(red_I,red_Q, color="red", alpha=0.3, s=2)
    ax.scatter(cg_I_off,cg_Q_off,c='k',s=25)
    ax.scatter(ce_I_off,ce_Q_off,c='k',s=25)
    ax.scatter(cg_I_on,cg_Q_on,c='k',s=25)
    ax.scatter(ce_I_on,ce_Q_on,c='k',s=25)

    ax.add_patch(Ellipse(xy=[cg_I_off,cg_Q_off],width=sig_off*4,height=sig_off*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=2, linestyle='--',angle=0))
    ax.add_patch(Ellipse(xy=[ce_I_off,ce_Q_off],width=sig_off*4,height=sig_off*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=2, linestyle='--',angle=0))
    ax.add_patch(Ellipse(xy=[cg_I_on,cg_Q_on],width=sig_on*4,height=sig_on*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=2, linestyle='--',angle=0))
    ax.add_patch(Ellipse(xy=[ce_I_on,ce_Q_on],width=sig_on*4,height=sig_on*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=2, linestyle='--',angle=0))
    ax.plot([cg_I_off,ce_I_off],[cg_Q_off,ce_Q_off],'--k',alpha=0.8,lw=2)
    ax.plot([cg_I_on,ce_I_on],[cg_Q_on,ce_Q_on],'--k',alpha=0.8,lw=2)
    ax.set_xlabel(r"$I\ $[mV]",size ='15')
    ax.set_ylabel(r"$Q\ $[mV]",size ='15')
    mean_I=(cg_I_off+cg_I_on+ce_I_off+ce_I_on)/4
    mean_Q=(cg_Q_off+cg_Q_on+ce_Q_off+ce_Q_on)/4
    ax.set_xlim(mean_I-10*sig_on,mean_I+10*sig_on)
    ax.set_ylim(mean_Q-10*sig_on,mean_Q+10*sig_on)
    Power_gain=10*np.log10((D_on/D_off)**2)
    Noise_diff=10*np.log10((sig_on/sig_off)**2)
    SNR_impro= Power_gain-Noise_diff
    text_msg = r"$D_{off}= %.2f $"%(D_off) +' mV\n'
    text_msg += r"$D_{on}= %.2f $"%(D_on) +' mV\n' 
    text_msg += r"$\sigma_{off}= %.2f $"%(sig_off) +' mV\n'
    text_msg += r"$\sigma_{on}= %.2f $"%(sig_on) +' mV\n'
    text_msg += r"$\rm{SNR_{off}}= %.2f $"%(D_off/sig_off) +' \n'
    text_msg += r"$\rm{SNR_{on}}= %.2f $"%(D_on/sig_on) +' \n'+'\n'
    text_msg += 'Power Gain:' +'\n'
    text_msg += r"$G_{s}= %.1f $"%(Power_gain) +' dB\n'
    text_msg += r"$G_{n}= %.1f $"%(Noise_diff) +' dB\n'
    text_msg += r"$\rm{SNR\ impro.}= %.1f $"%(SNR_impro) +' dB'
    plot_textbox(ax,text_msg,x=0.7,y=0.4,fontsize=10)
    ax.axes.set_aspect('equal')
    fig.tight_layout()
    

    
    
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
    
def Multi_2Q_single_shot_Rawdata_plot(data:dict):
    I1_00,Q1_00,I2_00,Q2_00= 1000*data[str(['g','g'])][0], 1000*data[str(['g','g'])][1],1000*data[str(['g','g'])][2],1000*data[str(['g','g'])][3]
    I1_01,Q1_01,I2_01,Q2_01= 1000*data[str(['g','e'])][0], 1000*data[str(['g','e'])][1],1000*data[str(['g','e'])][2],1000*data[str(['g','e'])][3]
    I1_10,Q1_10,I2_10,Q2_10= 1000*data[str(['e','g'])][0], 1000*data[str(['e','g'])][1],1000*data[str(['e','g'])][2],1000*data[str(['e','g'])][3]
    I1_11,Q1_11,I2_11,Q2_11= 1000*data[str(['e','e'])][0], 1000*data[str(['e','e'])][1],1000*data[str(['e','e'])][2],1000*data[str(['e','e'])][3]
    I1,Q1= np.hstack([I1_00,I1_01,I1_10,I1_11]), np.hstack([Q1_00,Q1_01,Q1_10,Q1_11])
    I2,Q2= np.hstack([I2_00,I2_01,I2_10,I2_11]), np.hstack([Q2_00,Q2_01,Q2_10,Q2_11])
    I1_data,Q1_data=[I1_00,I1_01,I1_10,I1_11],[Q1_00,Q1_01,Q1_10,Q1_11]
    I2_data,Q2_data=[I2_00,I2_01,I2_10,I2_11],[Q2_00,Q2_01,Q2_10,Q2_11]
    state=['Prepare '+'00','Prepare '+'01','Prepare '+'10','Prepare '+'11']
    
    fig, ax = plt.subplots(nrows=2,ncols =4,figsize =(10,5),dpi =200)         
    row_headers = ["Q1", "Q2"]
    col_headers = state
    for i in range(2):
        ax[i][0].set_ylabel(r"$Q\ $[mV]",size ='15')
        for j in range(4):
            if i==0:
                ax[i][j].set_xlim(np.minimum(min(I1),min(Q1)),np.maximum(max(I1),max(Q1)))
                ax[i][j].set_ylim(np.minimum(min(I1),min(Q1)),np.maximum(max(I1),max(Q1)))
                ax[i][j].scatter(I1_data[j],Q1_data[j], color="b", alpha=0.5, s=1)
                ax[i][j].axes.set_aspect('equal')
            else:
                ax[i][j].set_xlim(np.minimum(min(I2),min(Q2)),np.maximum(max(I2),max(Q2)))
                ax[i][j].set_ylim(np.minimum(min(I2),min(Q2)),np.maximum(max(I2),max(Q2)))
                ax[i][j].scatter(I2_data[j],Q2_data[j], color="b", alpha=0.5, s=1)
                ax[i][j].set_xlabel(r"$I\ $[mV]",size ='15')
                ax[i][j].axes.set_aspect('equal')
    font_kwargs = dict(fontweight="bold", fontsize='10',color='r',alpha=0.9)
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)
    fig.tight_layout()
    return fig
    
 
def Multi_2Q_single_shot_plot(results:dict):
    sig_Q1,sig_Q2=1000*results['Q1_pre_00']['fit_pack'][3],1000*results['Q2_pre_00']['fit_pack'][3]
    ce_I1_00,ce_Q1_00,Inte_data1_00=1000*results['Q1_pre_00']['fit_pack'][0][0],1000*results['Q1_pre_00']['fit_pack'][0][1],results['Q1_pre_00']['fit_pack'][4]
    ce_I1_01,ce_Q1_01,Inte_data1_01=1000*results['Q1_pre_0p']['fit_pack'][0][0],1000*results['Q1_pre_0p']['fit_pack'][0][1],results['Q1_pre_0p']['fit_pack'][4]
    ce_I1_10,ce_Q1_10,Inte_data1_10=1000*results['Q1_pre_p0']['fit_pack'][0][0],1000*results['Q1_pre_p0']['fit_pack'][0][1],results['Q1_pre_p0']['fit_pack'][4]
    ce_I1_11,ce_Q1_11,Inte_data1_11=1000*results['Q1_pre_pp']['fit_pack'][0][0],1000*results['Q1_pre_pp']['fit_pack'][0][1],results['Q1_pre_pp']['fit_pack'][4]
    ce_I2_00,ce_Q2_00,Inte_data2_00=1000*results['Q2_pre_00']['fit_pack'][0][0],1000*results['Q2_pre_00']['fit_pack'][0][1],results['Q2_pre_00']['fit_pack'][4]
    ce_I2_01,ce_Q2_01,Inte_data2_01=1000*results['Q2_pre_0p']['fit_pack'][0][0],1000*results['Q2_pre_0p']['fit_pack'][0][1],results['Q2_pre_0p']['fit_pack'][4]
    ce_I2_10,ce_Q2_10,Inte_data2_10=1000*results['Q2_pre_p0']['fit_pack'][0][0],1000*results['Q2_pre_p0']['fit_pack'][0][1],results['Q2_pre_p0']['fit_pack'][4]
    ce_I2_11,ce_Q2_11,Inte_data2_11=1000*results['Q2_pre_pp']['fit_pack'][0][0],1000*results['Q2_pre_pp']['fit_pack'][0][1],results['Q2_pre_pp']['fit_pack'][4]
    
    I1_ro_00,I1_fit_00= 1000*results['Q1_pre_00']['I_ro'],1000*results['Q1_pre_00']['I_fit']
    I1_ro_01,I1_fit_01= 1000*results['Q1_pre_0p']['I_ro'],1000*results['Q1_pre_0p']['I_fit']
    I1_ro_10,I1_fit_10= 1000*results['Q1_pre_p0']['I_ro'],1000*results['Q1_pre_p0']['I_fit']
    I1_ro_11,I1_fit_11= 1000*results['Q1_pre_pp']['I_ro'],1000*results['Q1_pre_pp']['I_fit']
    I2_ro_00,I2_fit_00= 1000*results['Q2_pre_00']['I_ro'],1000*results['Q2_pre_00']['I_fit']
    I2_ro_01,I2_fit_01= 1000*results['Q2_pre_0p']['I_ro'],1000*results['Q2_pre_0p']['I_fit']
    I2_ro_10,I2_fit_10= 1000*results['Q2_pre_p0']['I_ro'],1000*results['Q2_pre_p0']['I_fit']
    I2_ro_11,I2_fit_11= 1000*results['Q2_pre_pp']['I_ro'],1000*results['Q2_pre_pp']['I_fit']
    
    Mg1_00,Me1_00= gauss_func(I1_fit_00,0,sig_Q1,results['Q1_pre_00']['fit_pack'][1]),gauss_func(I1_fit_00,ce_I1_00,sig_Q1,results['Q1_pre_00']['fit_pack'][2])
    Mg1_01,Me1_01= gauss_func(I1_fit_01,0,sig_Q1,results['Q1_pre_0p']['fit_pack'][1]),gauss_func(I1_fit_01,ce_I1_01,sig_Q1,results['Q1_pre_0p']['fit_pack'][2])
    Mg1_10,Me1_10= gauss_func(I1_fit_10,0,sig_Q1,results['Q1_pre_p0']['fit_pack'][1]),gauss_func(I1_fit_10,ce_I1_10,sig_Q1,results['Q1_pre_p0']['fit_pack'][2])
    Mg1_11,Me1_11= gauss_func(I1_fit_11,0,sig_Q1,results['Q1_pre_pp']['fit_pack'][1]),gauss_func(I1_fit_11,ce_I1_11,sig_Q1,results['Q1_pre_pp']['fit_pack'][2])
    Mg2_00,Me2_00= gauss_func(I2_fit_00,0,sig_Q2,results['Q2_pre_00']['fit_pack'][1]),gauss_func(I2_fit_00,ce_I2_00,sig_Q2,results['Q2_pre_00']['fit_pack'][2])
    Mg2_01,Me2_01= gauss_func(I2_fit_01,0,sig_Q2,results['Q2_pre_0p']['fit_pack'][1]),gauss_func(I2_fit_01,ce_I2_01,sig_Q2,results['Q2_pre_0p']['fit_pack'][2])
    Mg2_10,Me2_10= gauss_func(I2_fit_10,0,sig_Q2,results['Q2_pre_p0']['fit_pack'][1]),gauss_func(I2_fit_10,ce_I2_10,sig_Q2,results['Q2_pre_p0']['fit_pack'][2])
    Mg2_11,Me2_11= gauss_func(I2_fit_11,0,sig_Q2,results['Q2_pre_pp']['fit_pack'][1]),gauss_func(I2_fit_11,ce_I2_11,sig_Q2,results['Q2_pre_pp']['fit_pack'][2])
    
    Pg1_00,Pg1_01,Pg1_10,Pg1_11= results['Q1_pre_00']['Pg'],results['Q1_pre_0p']['Pg'],results['Q1_pre_p0']['Pg'],results['Q1_pre_pp']['Pg']
    Pg2_00,Pg2_01,Pg2_10,Pg2_11= results['Q2_pre_00']['Pg'],results['Q2_pre_0p']['Pg'],results['Q2_pre_p0']['Pg'],results['Q2_pre_pp']['Pg']
    OE1_00,OE1_01,OE1_10,OE1_11= results['Q1_pre_00']['overlap'],results['Q1_pre_0p']['overlap'],results['Q1_pre_p0']['overlap'],results['Q1_pre_pp']['overlap']
    OE2_00,OE2_01,OE2_10,OE2_11= results['Q2_pre_00']['overlap'],results['Q2_pre_0p']['overlap'],results['Q2_pre_p0']['overlap'],results['Q2_pre_pp']['overlap']
    
    I1_00,Q1_00,I2_00,Q2_00= 1000*results['Q1_pre_00']['rot_IQdata'][0][0], 1000*results['Q1_pre_00']['rot_IQdata'][0][1],1000*results['Q2_pre_00']['rot_IQdata'][0][0], 1000*results['Q2_pre_00']['rot_IQdata'][0][1]
    I1_01,Q1_01,I2_01,Q2_01= 1000*results['Q1_pre_0p']['rot_IQdata'][0][0], 1000*results['Q1_pre_0p']['rot_IQdata'][0][1],1000*results['Q2_pre_0p']['rot_IQdata'][0][0], 1000*results['Q2_pre_0p']['rot_IQdata'][0][1]
    I1_10,Q1_10,I2_10,Q2_10= 1000*results['Q1_pre_p0']['rot_IQdata'][0][0], 1000*results['Q1_pre_p0']['rot_IQdata'][0][1],1000*results['Q2_pre_p0']['rot_IQdata'][0][0], 1000*results['Q2_pre_p0']['rot_IQdata'][0][1]
    I1_11,Q1_11,I2_11,Q2_11= 1000*results['Q1_pre_pp']['rot_IQdata'][0][0], 1000*results['Q1_pre_pp']['rot_IQdata'][0][1],1000*results['Q2_pre_pp']['rot_IQdata'][0][0], 1000*results['Q2_pre_pp']['rot_IQdata'][0][1]
    I1,Q1= np.hstack([I1_00,I1_01,I1_10,I1_11]), np.hstack([Q1_00,Q1_01,Q1_10,Q1_11])
    I2,Q2= np.hstack([I2_00,I2_01,I2_10,I2_11]), np.hstack([Q2_00,Q2_01,Q2_10,Q2_11])
    I1_data,Q1_data=[I1_00,I1_01,I1_10,I1_11],[Q1_00,Q1_01,Q1_10,Q1_11]
    I2_data,Q2_data=[I2_00,I2_01,I2_10,I2_11],[Q2_00,Q2_01,Q2_10,Q2_11]
    ce_I1=[ce_I1_00,ce_I1_01,ce_I1_10,ce_I1_11]
    ce_I2=[ce_I2_00,ce_I2_01,ce_I2_10,ce_I2_11]
    ce_Q1=[ce_Q1_00,ce_Q1_01,ce_Q1_10,ce_Q1_11]
    ce_Q2=[ce_Q2_00,ce_Q2_01,ce_Q2_10,ce_Q2_11]
    Mg1,Mg2=[Mg1_00,Mg1_01,Mg1_10,Mg1_11],[Mg2_00,Mg2_01,Mg2_10,Mg2_11]
    Me1,Me2=[Me1_00,Me1_01,Me1_10,Me1_11],[Me2_00,Me2_01,Me2_10,Me2_11]
    I1_ro=[I1_ro_00,I1_ro_01,I1_ro_10,I1_ro_11]
    I2_ro=[I2_ro_00,I2_ro_01,I2_ro_10,I2_ro_11]
    I1_fit=[I1_fit_00,I1_fit_01,I1_fit_10,I1_fit_11]
    I2_fit=[I2_fit_00,I2_fit_01,I2_fit_10,I2_fit_11]
    Inte_data1=[Inte_data1_00,Inte_data1_01,Inte_data1_10,Inte_data1_11]
    Inte_data2=[Inte_data2_00,Inte_data2_01,Inte_data2_10,Inte_data2_11]
    Pg1=[Pg1_00,Pg1_01,Pg1_10,Pg1_11]
    Pg2=[Pg2_00,Pg2_01,Pg2_10,Pg2_11]
    OE1=[OE1_00,OE1_01,OE1_10,OE1_11]
    OE2=[OE2_00,OE2_01,OE2_10,OE2_11]
    state=['Prepare '+'00','Prepare '+'01','Prepare '+'10','Prepare '+'11']
    
    fig, ax = plt.subplots(nrows=2,ncols =4,figsize =(10,5),dpi =200)     
    fig1, ax1 = plt.subplots(nrows=2,ncols =4,figsize =(10,5),dpi =200)      
    row_headers = ["Q1", "Q2"]
    col_headers = state
    for i in range(2):
        ax[i][0].set_ylabel(r"$Q\ $[mV]",size ='15')
        ax1[i][0].set_ylabel(r'$PDF$',size ='15')
        for j in range(4):
            if i==0:
                red= np.where(I1_data[j]>ce_I1[j]/2)[0]  
                blue= np.where(I1_data[j]<ce_I1[j]/2)[0]
                ax[i][j].set_xlim(np.minimum(min(I1),min(Q1)),np.maximum(max(I1),max(Q1)))
                ax[i][j].set_ylim(np.minimum(min(I1),min(Q1)),np.maximum(max(I1),max(Q1)))
                ax[i][j].scatter(I1_data[j][red],Q1_data[j][red], color="r", alpha=0.5, s=1)
                ax[i][j].scatter(I1_data[j][blue],Q1_data[j][blue], color="b", alpha=0.5, s=1)
                ax[i][j].scatter(0,0,c='k',s=15)
                ax[i][j].scatter(ce_I1[j],ce_Q1[j],c='k',s=15)
                ax[i][j].add_patch(Ellipse(xy=[0,0],width=sig_Q1*4,height=sig_Q1*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
                ax[i][j].add_patch(Ellipse(xy=[ce_I1[j],ce_Q1[j]],width=sig_Q1*4,height=sig_Q1*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
                ax[i][j].axes.set_aspect('equal')
                red1= np.where(I1_ro[j]>ce_I1[j]/2)[0]  
                blue1= np.where(I1_ro[j]<ce_I1[j]/2)[0]
                ax1[i][j].plot(I1_ro[j][red1], Inte_data1[j][red1],'o',color='r',alpha=0.3,ms=3)
                ax1[i][j].plot(I1_ro[j][blue1], Inte_data1[j][blue1],'o',color='b',alpha=0.3,ms=3)
                ax1[i][j].plot(I1_fit[j], Mg1[j],'--b',alpha=0.8,lw=1)
                ax1[i][j].plot(I1_fit[j], Me1[j],'--r',alpha=0.8,lw=1)
                ax1[i][j].set_ylim(10**(int(np.log10(np.max(Inte_data1[0])))-3),10**(int(np.log10(np.max(Inte_data1[0])))+1))
                ax1[i][j].set_xlim(ce_I1[j]/2-14*sig_Q1,ce_I1[j]/2+14*sig_Q1)
                ax1[i][j].axvline(x=ce_I1[j]/2,color='grey',linestyle='dashed',alpha=0.5,lw=1)
                ax1[i][j].set_yscale('log')
                text_msg=''
                text_msg += r"$P_{g}= %.1f $"%(Pg1[j]*100)+'%'+'\n'
                text_msg += r"$P_{e}= %.1f $"%(100-Pg1[j]*100)+'%'+'\n'
                text_msg += r"$\varepsilon_{o}=%.1f $"%(OE1[j]*100)+'%'
                plot_textbox_small(ax1[i][j],text_msg)
                
            else:
                red= np.where(I2_data[j]>ce_I2[j]/2)[0]  
                blue= np.where(I2_data[j]<ce_I2[j]/2)[0]
                ax[i][j].set_xlim(np.minimum(min(I2),min(Q2)),np.maximum(max(I2),max(Q2)))
                ax[i][j].set_ylim(np.minimum(min(I2),min(Q2)),np.maximum(max(I2),max(Q2)))
                ax[i][j].scatter(I2_data[j][red],Q2_data[j][red], color="r", alpha=0.5, s=1)
                ax[i][j].scatter(I2_data[j][blue],Q2_data[j][blue], color="b", alpha=0.5, s=1)
                ax[i][j].scatter(0,0,c='k',s=15)
                ax[i][j].scatter(ce_I2[j],ce_Q2[j],c='k',s=15)
                ax[i][j].add_patch(Ellipse(xy=[0,0],width=sig_Q2*4,height=sig_Q2*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
                ax[i][j].add_patch(Ellipse(xy=[ce_I2[j],ce_Q2[j]],width=sig_Q2*4,height=sig_Q2*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
                ax[i][j].set_xlabel(r"$I\ $[mV]",size ='15')
                ax[i][j].axes.set_aspect('equal')
                red1= np.where(I1_ro[j]>ce_I1[j]/2)[0]  
                blue1= np.where(I1_ro[j]<ce_I1[j]/2)[0]
                ax1[i][j].plot(I2_ro[j][red1], Inte_data2[j][red1],'o',color='r',alpha=0.3,ms=3)
                ax1[i][j].plot(I2_ro[j][blue1], Inte_data2[j][blue1],'o',color='b',alpha=0.3,ms=3)
                ax1[i][j].plot(I2_fit[j], Mg2[j],'--b',alpha=0.8,lw=1)
                ax1[i][j].plot(I2_fit[j], Me2[j],'--r',alpha=0.8,lw=1)
                ax1[i][j].set_yscale('log')
                ax1[i][j].axvline(x=ce_I2[j]/2,color='grey',linestyle='dashed',alpha=0.5,lw=1)
                ax1[i][j].set_ylim(10**(int(np.log10(np.max(Inte_data1[0])))-3),10**(int(np.log10(np.max(Inte_data2[0])))+1))
                ax1[i][j].set_xlim(ce_I2[j]/2-14*sig_Q1,ce_I2[j]/2+14*sig_Q1)
                ax1[i][j].set_xlabel(r"$I^{'}\ $[mV]",size ='15')
                text_msg=''
                text_msg += r"$P_{g}= %.1f $"%(Pg2[j]*100)+'%'+'\n'
                text_msg += r"$P_{e}= %.1f $"%(100-Pg2[j]*100)+'%'+'\n'
                text_msg += r"$\varepsilon_{o}=%.1f $"%(OE2[j]*100)+'%'
                plot_textbox_small(ax1[i][j],text_msg)
                
    font_kwargs = dict(fontweight="bold", fontsize='10',color='r',alpha=0.9)
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)
    add_headers(fig1, col_headers=col_headers, row_headers=row_headers, **font_kwargs)
    fig.tight_layout()
    fig1.tight_layout()
    return fig,fig1
    
    
def Single_shot_fit_plot(results:dict):
    c_I,c_Q,sig=1000*results['fit_pack'][0],1000*results['fit_pack'][1],1000*results['fit_pack'][2]
    I,Q= results['data'][0],results['data'][1]
    
    fig1, ax = plt.subplots(nrows =1,figsize =(3,3),dpi =200)
    ax.scatter(1000*I, 1000*Q, color="blue", alpha=0.5, s=5)       
    ax.scatter(c_I,c_Q,c='k',s=15)
    ax.add_patch(Ellipse(xy=[c_I,c_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    ax.set_xlabel(r"$I\ $[mV]",size ='15')
    ax.set_ylabel(r"$Q\ $[mV]",size ='15')
    ax.set_title('Single shot raw data')
    ax.set_xlim(c_I-8*sig,c_I+8*sig)
    ax.set_ylim(c_Q-8*sig,c_Q+8*sig)
    ax.axes.set_aspect('equal')
    fig1.tight_layout()
    
    fig2, ax = plt.subplots(nrows =1,figsize =(5,4),dpi =200)
    cmap = plt.get_cmap('jet')
    vmax= np.max(results['data_hist'])
    pcm = ax.pcolormesh(1000*results['coords'][0],1000*results['coords'][1], results['data_hist'].transpose(), vmin=0,vmax=vmax, cmap=cmap,shading='auto')
    cbar =fig2.colorbar(pcm, ax=ax, extend='both', orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel(r"$I\ $[mV]",size ='15')
    ax.set_ylabel(r"$Q\ $[mV]",size ='15')
    cbar.set_label(r'$PDF$',size ='10')
    ax.set_title('Single shot data histogram')
    ax.scatter(c_I,c_Q,c='k',s=15)
    ax.add_patch(Ellipse(xy=[c_I,c_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="w", linewidth=0.8, linestyle='--',angle=0))
    ax.axes.set_aspect('equal')
    fig2.tight_layout()
    
    fig3, ax = plt.subplots(nrows =1,figsize =(5,4),dpi =200)
    cmap = plt.get_cmap('jet')
    vmax= np.max(results['fitting'])
    bounds = np.linspace(0,vmax,256)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    pcm = ax.pcolormesh(1000*results['coords'][0],1000*results['coords'][1], results['fitting'], vmin=0,vmax=vmax, cmap=cmap,shading='auto')
    cbar =fig3.colorbar(pcm, ax=ax, extend='both', orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel(r"$I\ $[mV]",size ='15')
    ax.set_ylabel(r"$Q\ $[mV]",size ='15')
    cbar.set_label(r'$PDF$',size ='10')
    ax.set_title('Single shot fitting')
    ax.scatter(c_I,c_Q,c='k',s=15)
    ax.add_patch(Ellipse(xy=[c_I,c_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="w", linewidth=0.8, linestyle='--',angle=0))
    ax.axes.set_aspect('equal')
    fig3.tight_layout()
    return fig1,fig2,fig3
    
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

def Qubit_state_Avgtimetrace_plot(results:dict,IF:float,trace_recordlength:float):
    time_array= np.linspace(0,trace_recordlength,int(trace_recordlength*1e9))
    offset_Ig,offset_Qg= np.mean(results['g'][0][-100:-1]), np.mean(results['g'][1][-100:-1])
    offset_Ie,offset_Qe= np.mean(results['e'][0][-100:-1]), np.mean(results['e'][1][-100:-1])
    raw_Ig,raw_Qg= results['g'][0]-offset_Ig, results['g'][1]-offset_Qg
    raw_Ie,raw_Qe= results['e'][0]-offset_Ie, results['e'][1]-offset_Qe
    raw_Ig,raw_Qg= Digital_down_convert(raw_Ig,raw_Qg,IF,time_array)
    Ig,Qg= Trace_filtering(raw_Ig,fc=0.5*IF), Trace_filtering(raw_Qg,fc=0.5*IF)
    raw_Ie,raw_Qe= Digital_down_convert(raw_Ie,raw_Qe,IF,time_array)
    Ie,Qe= Trace_filtering(raw_Ie,fc=0.5*IF), Trace_filtering(raw_Qe,fc=0.5*IF)
    trace= np.linspace(0,trace_recordlength*1e9,int(trace_recordlength*1e9))
    
    fig,ax= plt.subplots(nrows =3,figsize =(6,4),dpi =200)
    ax[0].plot(trace/1000, Ig*1000,'b',alpha=0.8,lw=2)
    ax[0].plot(trace/1000, Ie*1000,'r',alpha=0.8,lw=2)
    ax[0].set_ylabel(r"$I\ $[mV]",size ='15')
    ax[0].set_title('Avg_IQ_timetrace')

    ax[1].plot(trace/1000, Qg*1000,'b',alpha=0.8,lw=2)
    ax[1].plot(trace/1000, Qe*1000,'r',alpha=0.8,lw=2)
    ax[1].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[2].plot(trace/1000,1000* np.sqrt(Ig**2+Qg**2),'b',alpha=0.8,lw=2)
    ax[2].plot(trace/1000,1000* np.sqrt(Ie**2+Qe**2),'r',alpha=0.8,lw=2)
    ax[2].set_xlabel(r"$t\ [\mu$s]",size ='15')
    ax[2].set_ylabel(r"$Amp\ $[mV]",size ='15')
    fig.tight_layout()
    return fig
    
def Avgtimetrace_plot(results:dict,IF:float,BW:float):
    trace_recordlength= results['trace_recordlength']
    time_array= np.linspace(0,trace_recordlength,int(trace_recordlength*1e9))
    def Data_processing(results):
        offset_Ig,offset_Qg= np.mean(results['g'][0][-100:-1]), np.mean(results['g'][1][-100:-1])
        raw_Ig,raw_Qg= results['g'][0]-offset_Ig, results['g'][1]-offset_Qg
        raw_Ig,raw_Qg= Digital_down_convert(raw_Ig,raw_Qg,IF,time_array)
        Ig,Qg= Trace_filtering(raw_Ig,fc=BW), Trace_filtering(raw_Qg,fc=BW)
        return Ig[10:-10],Qg[10:-10]
    Ig,Qg= Data_processing(results)
    trace= 1e9*time_array[10:-10]
    fig,ax= plt.subplots(nrows =3,figsize =(6,4),dpi =200)
    ax[0].plot(trace/1000, Ig*1000,'b',alpha=0.5,lw=2)
    ax[0].set_ylabel(r"$I\ $[mV]",size ='15')
    ax[0].set_title('Avg_IQ_timetrace')

    ax[1].plot(trace/1000, Qg*1000,'b',alpha=0.5,lw=2)
    ax[1].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[2].plot(trace/1000,1000* np.sqrt(Ig**2+Qg**2),'b',alpha=0.5,lw=2)
    ax[2].set_ylabel(r"$Amp\ $[mV]",size ='15')
    ax[2].set_xlabel(r"$t\ [\mu$s]",size ='15')
    fig.tight_layout()
    
    fig,ax= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    ax.plot(trace/1000, (Ig*1000)**2+(Qg*1000)**2,'b',alpha=0.5,lw=2)
    ax.set_ylabel(r'$I^{2}+Q^{2}\ \rm{[mV^{2}]}$',size ='15')
    ax.set_xlabel(r"$t\ [\mu$s]",size ='15')
    fig.tight_layout()
    
def Avgtimetrace_plot_with_PA(results_off:dict,results_on:dict,IF:float,BW:float):
    trace_recordlength= results_off['trace_recordlength']
    time_array= np.linspace(0,trace_recordlength,int(trace_recordlength*1e9))
    def Data_processing(results):
        offset_Ig,offset_Qg= np.mean(results['g'][0][-100:-1]), np.mean(results['g'][1][-100:-1])
        raw_Ig,raw_Qg= results['g'][0]-offset_Ig, results['g'][1]-offset_Qg
        raw_Ig,raw_Qg= Digital_down_convert(raw_Ig,raw_Qg,IF,time_array)
        Ig,Qg= Trace_filtering(raw_Ig,fc=BW), Trace_filtering(raw_Qg,fc=BW)
        return Ig[10:-10],Qg[10:-10]
    Ig_off,Qg_off= Data_processing(results_off)
    Ig_on,Qg_on= Data_processing(results_on)
    trace= 1e9*time_array[10:-10]
    fig,ax= plt.subplots(nrows =2,figsize =(6,4),dpi =200)
    ax[0].plot(trace/1000, Ig_off*1000,'b',alpha=0.5,lw=2)
    ax[0].plot(trace/1000, Ig_on*1000,'r',alpha=0.5,lw=2)
    ax[0].set_ylabel(r"$I\ $[mV]",size ='15')
    ax[0].set_title('Avg_IQ_timetrace')

    ax[1].plot(trace/1000, Qg_off*1000,'b',label='PA off',alpha=0.5,lw=2)
    ax[1].plot(trace/1000, Qg_on*1000,'r',label='PA on',alpha=0.5,lw=2)
    ax[1].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[1].set_xlabel(r"$t\ [\mu$s]",size ='15')
    ax[1].legend(fontsize=10)
    fig.tight_layout()
    
    fig,ax= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    ax.plot(trace/1000,1000**2* (Ig_off**2+Qg_off**2),'b',label='PA off',alpha=0.5,lw=2)
    ax.plot(trace/1000,1000**2* (Ig_on**2+Qg_on**2),'r',label='PA on',alpha=0.5,lw=2)
    ax.set_ylabel(r'$I^{2}+Q^{2}\ \rm{[mV^{2}]}$',size ='15')
    ax.set_xlabel(r"$t\ [\mu$s]",size ='15')
    ax.legend(fontsize=15)
    fig.tight_layout()
    

    

def Fit_analysis_plot(results:xr.core.dataset.Dataset, P_rescale:bool, Dis:any):
    if P_rescale is not True:
        Nor_f=1/1000
        y_label= 'Contrast'+' [mV]'
    elif P_rescale is True:
        Nor_f= Dis
        y_label= r"$P_{1}\ $"
    else: raise KeyError ('P_rescale is not bool') 
    
    y_fit= results.data_vars['fitting']/Nor_f
    y= results.data_vars['data']/Nor_f
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    text_msg = "Fit results\n"
    if results.attrs['exper'] == 'QS':
        title= 'Two tone spectroscopy'
        x_label= 'Frequency'+' [GHz]'
        x= results.coords['f']*1e-9
        x_fit= results.coords['para_fit']*1e-9
        text_msg += r"$f_{01}= %.4f $"%(results.attrs['f01_fit']*1e-9) +' GHz\n'
        text_msg += r"$BW= %.2f $"%(results.attrs['bandwidth']*1e-6) +' MHz\n'
        
    elif results.attrs['exper'] == 'T1':  
        title= 'T1 relaxation'
        x_label= r"$t_{f}$"+r"$\ [\mu$s]" 
        x= results.coords['freeDu']*1e6
        x_fit= results.coords['para_fit']*1e6
        text_msg += r"$T_{1}= %.3f $"%(results.attrs['T1_fit']*1e6) +r"$\ [\mu$s]"+'\n'
        
    elif results.attrs['exper'] == 'T2':  
        title= 'Ramsey'
        x_label= r"$t_{f}$"+r"$\ [\mu$s]" 
        x= results.coords['freeDu']*1e6
        x_fit= results.coords['para_fit']*1e6
        text_msg += r"$T_{2}= %.3f $"%(results.attrs['T2_fit']*1e6) +r"$\ [\mu$s]"+'\n'        
        text_msg += r"$detuning= %.3f $"%(results.attrs['f']*1e-6) +' MHz\n'

    elif results.attrs['exper'] == 'Ramsey_charge_parity':  
        title= 'Ramsey charge parity'
        x_label= r"$t_{f}$"+r"$\ [\mu$s]" 
        x= results.coords['freeDu']*1e6
        x_fit= results.coords['para_fit']*1e6      
        text_msg += r"$detuning(1)= %.3f $"%(results.attrs['f1']*1e-6) +' MHz\n'      
        text_msg += r"$detuning(2)= %.3f $"%(results.attrs['f2']*1e-6) +' MHz\n'
        text_msg += r"$T_{2}(1)= %.3f $"%(results.attrs['T2_1']*1e6) +r"$\ [\mu$s]"+'\n'       
        text_msg += r"$T_{2}(2)= %.3f $"%(results.attrs['T2_2']*1e6) +r"$\ [\mu$s]" 
    
    elif results.attrs['exper'] == 'Swap':  
        title= 'Swap'
        x_label= results.attrs['xlabel']
        x= results.coords['samples']
        x_fit= results.coords['para_fit']      
        text_msg += r"$f= %.3f $"%(results.attrs['f']) +' \n'

    elif results.attrs['exper'] == 'iSwap':  
        title= 'iSwap'
        x_label=r"$t_{f}$"+r"$\ [\mu$s]"
        x= results.coords['freeDu']*1e6 
        x_fit= results.coords['para_fit']*1e6     
        text_msg += r"$f= %.1f $"%(results.attrs['f']*1e-6) +" [MHz]" +' \n'
       
    elif results.attrs['exper'] == "Zgate_twotone":  
        title= "Zgate twotone"
        q= results.attrs['q']
        x_label= " Z"+q[1]+' gate voltage'+" [V]" 
        y_label= r"$f_{01}$"+' [GHz]'
        Nor_f=1e9
        y_fit= results.data_vars['fitting']/Nor_f
        y= results.data_vars['data']/Nor_f
        x= results.coords['Z']
        x_fit= results.coords['para_fit']
        Ec=results.attrs['Ec']
        Ejmax=results.attrs['Ejmax_fit']   
        m_fit=results.attrs['m_fit']
        d_fit=results.attrs['d_fit']
        phi_offset=results.attrs['phi_offset_fit']
        bias_to_Ej_Ec= results.attrs['bias_to_Ej_Ec']
        text_msg += r"$E_{c}/h= %.3f $"%(Ec*1e-9) +' GHz\n'        
        text_msg += r"$E_{j, max}/h= %.3f $"%(Ejmax*1e-9) +' GHz\n'
        text_msg += r"$f_{01, max}= %.3f $"%((Transmon(0,m_fit,Ejmax,Ec,phi_offset,d_fit))*1e-9) +' GHz\n'
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i in range(len(bias_to_Ej_Ec)):
            ax.axhline(y=Transmon(bias_to_Ej_Ec[i],m_fit,Ejmax,Ec,phi_offset,d_fit)/1e9,linestyle='dashed',c=colors[i],label=r"$E_{j}/E_{c}= %.0f $"%(Ej_transmon(bias_to_Ej_Ec[i],m_fit,Ejmax,phi_offset,d_fit)/Ec), alpha=0.8,lw=1)
        ax.legend(fontsize=8)   
        
    elif results.attrs['exper'] == 'Rabi': 
        title='PowerRabi'
        pi_2= results.attrs['pi_2']
        pi_Du= results.attrs['pi_Du']
        x_label= 'XY amp'+r"$\ [V]$"
        x= results.coords['samples']
        x_fit= results.coords['para_fit']
        text_msg += r"$\pi$" +'\n'
        text_msg += r"$amp= %.3f $"%(pi_2) +r"$\ [V]$"+'\n'  
        text_msg += r"$Du= %.0f $"%(pi_Du*1e9) +' [ns]\n'
        ax.axvline(x=pi_2, color='r',linestyle='dashed', alpha=0.8,lw=1)
        
    ax.plot(x,y,'o', color="blue", alpha=0.5, ms=5)
    ax.plot(x_fit,y_fit,'-', color="red", alpha=0.5, lw=1.5)     
    ax.set_xlabel(x_label,size ='15')
    ax.set_title(title,size ='15')
    ax.set_ylabel(y_label,size ='15')
    if P_rescale is True:
        ax.set_ylim(0,1)
    plot_textbox(ax,text_msg,fontsize=10)
    fig.tight_layout()
    return fig

def T1_comparison_with_PA(results_off,results_on,P_rescale,Dis):
    if P_rescale is not True:
        Nor_f_off=Nor_f_on=1/1000
        y_label= 'Contrast'+' [mV]'
    elif P_rescale is True:
        Nor_f_off= Dis[0]
        Nor_f_on= Dis[1]
        y_label= r"$P_{1}\ $"
    else: raise KeyError ('P_rescale is not bool') 
    
    y_fit_off= results_off.data_vars['fitting']/Nor_f_off
    y_off= results_off.data_vars['data']/Nor_f_off
    y_fit_on= results_on.data_vars['fitting']/Nor_f_on
    y_on= results_on.data_vars['data']/Nor_f_on
    
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    text_msg = "Fit results\n"
    title= 'T1 relaxation'
    x_label= r"$t_{f}$"+r"$\ [\mu$s]" 
    x= results_off.coords['freeDu']*1e6
    x_fit= results_off.coords['para_fit']*1e6
    text_msg += r"PA off$\ T_{1}= %.3f $"%(results_off.attrs['T1_fit']*1e6) +r"$\ [\mu$s]"+'\n'
    text_msg += r"PA on$\ T_{1}= %.3f $"%(results_on.attrs['T1_fit']*1e6) +r"$\ [\mu$s]"+'\n'

        
    ax.plot(x,y_off,'o', color="blue",label=r"$\rm{PA\ off} $", alpha=0.5, ms=5)
    ax.plot(x,y_on,'o', color="red",label=r"$\rm{PA\ on} $", alpha=0.5, ms=5)
    ax.plot(x_fit,y_fit_off,'-', color="k", alpha=1, lw=1)
    ax.plot(x_fit,y_fit_on,'-', color="k", alpha=1, lw=1)     
    ax.set_xlabel(x_label,size ='15')
    ax.set_title(title,size ='15')
    ax.set_ylabel(y_label,size ='15')
    if P_rescale is True:
        ax.set_ylim(0,1)
    plot_textbox(ax,text_msg,fontsize=12)
    ax.legend(fontsize=15)
    fig.tight_layout()
    plt.show()
    

def SQ_RB_analysis_plot(RB_method:str,SRB_results:xr.core.dataset.Dataset,IRB_results:xr.core.dataset.Dataset,interleaved_gate:str, P_rescale:bool, Dis:any):
    if P_rescale is not True:
        Nor_f=1/1000
        y_label= 'Contrast'+' [mV]'
    elif P_rescale is True:
        Nor_f= Dis
        y_label= r"$P_{1}\ $"
    else: raise KeyError ('P_rescale is not bool') 
    
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    x_label="Number of Cliffords"
    text_msg = "Fit results\n"
    if RB_method=='Standard':
        title= 'Standard RB'
        x= SRB_results.coords['N']
        x_fit= SRB_results.coords['para_fit']
        y_fit= SRB_results.data_vars['fitting']/Nor_f
        y= SRB_results.data_vars['data']/Nor_f
        raw= SRB_results.attrs['raw']
        if P_rescale is True:
            y_label = r"$F_{s}$"
            y = 1 - y
            y_fit = 1 - y_fit
            for i in range(len(x)):  
                y_mean = 1 - np.mean(raw.T[i]) / Nor_f   
                y_err = np.std(raw.T[i]) / Nor_f         
                ax.errorbar(x[i], y_mean,
                    yerr=y_err,
                    fmt='o',        
                    color="b",
                    alpha=0.5,
                    ms=5,
                    capsize=2
                )
        else:
            for i in range(len(raw.T[0])):
                ax.plot(x,raw[i]/Nor_f,'o', color="k", alpha=0.3, ms=5)
        text_msg += 'EPC= %.3f'%(((1-SRB_results.attrs['p_fit'])/2)*100)+'%'+' \n' 
        text_msg += r"$F_{g}=%.2f$"%((1-(1-SRB_results.attrs['p_fit'])/2)*100)+'%'
        
        
    elif RB_method=='Interleaved':
        title= 'Interleaved RB'
        x= IRB_results.coords['N']
        x_fit= IRB_results.coords['para_fit']
        y_fit= IRB_results.data_vars['fitting']/Nor_f
        y= IRB_results.data_vars['data']/Nor_f
        raw= IRB_results.attrs['raw']
        if P_rescale is True:
            y_label=r"$F_{s}$"
            y=1-y
            y_fit=1-y_fit
            for i in range(len(x)):  
                y_mean = 1 - np.mean(raw.T[i]) / Nor_f   
                y_err = np.std(raw.T[i]) / Nor_f         
                ax.errorbar(x[i], y_mean,
                    yerr=y_err,
                    fmt='o',        
                    color="b",
                    alpha=0.5,
                    ms=5,
                    capsize=2
                )
        else:
            for i in range(len(raw.transpose()[0])):
                ax.plot(x,raw[i]/Nor_f,'o', color="k", alpha=0.3, ms=5)
        text_msg += interleaved_gate +' \n'
        p_ratio= IRB_results.attrs['p_fit']/SRB_results.attrs['p_fit']
        print(SRB_results.attrs['p_fit'])
        print(IRB_results.attrs['p_fit'])
        text_msg += 'EPC= %.2f'%(((1-p_ratio)/2)*100)+'%'+'\n' 
        text_msg += r"$F_{g}=%.2f$"%((1-(1-p_ratio)/2)*100)+'%'
        

    ax.plot(x_fit,y_fit,'-', color="red", alpha=0.8, lw=2)     
    ax.set_xlabel(x_label,size ='15')
    ax.set_title(title,size ='15')
    ax.set_ylabel(y_label,size ='15')
    plot_textbox(ax,text_msg,fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig
    
def SQ_PB_analysis_plot(result:xr.core.dataset.Dataset,SRB_results:xr.core.dataset.Dataset):
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    x_label="Number of Cliffords"
    y_label="Sequence Purity"
    text_msg = "Fit results\n"
    title= 'Purity Benchmarking'
    x= result.coords['N']
    x_fit= result.coords['para_fit']
    y_fit= result.data_vars['fitting']
    y= result.data_vars['data']
    raw= result.attrs['raw']
    for i in range(len(x)):  
            y_mean = np.mean(raw.T[i])  
            y_err = np.std(raw.T[i]) 
            ax.errorbar(x[i], y_mean,
                yerr=y_err,
                fmt='o',        
                color="b",
                alpha=0.5,
                ms=5,
                capsize=2
            )
    EPC_t= (1-SRB_results.attrs['p_fit'])/2
    EPC_i= (1-np.sqrt(result.attrs['p_fit']))/2
    text_msg += 'coherent EPC= %.2f'%((EPC_t-EPC_i)*100)+'%'+ '\n' 
    text_msg += 'incoherent EPC= %.2f'%((EPC_i)*100)+'%'+' \n' 
    ax.plot(x_fit,y_fit,'-', color="red", alpha=0.8, lw=2)     
    ax.set_xlabel(x_label,size ='15')
    ax.set_title(title,size ='15')
    ax.set_ylabel(y_label,size ='15')
    plot_textbox(ax,text_msg,fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig


def Resonator_analysis_plot(results:xr.core.dataset.Dataset):
    
    y1_label= r'$\vert S_{21}\vert\ $'
    y2_label= r'$\angle S_{21}\ [deg]$'
    fig, ax = plt.subplots(nrows=2,ncols =2,figsize =(6,4),dpi =200)     
    text_msg = "Fit results\n"
    title= 'Resonator spectroscopy'
    x_label= 'Frequency'+' [GHz]'
    x= results.coords['f']*1e-9
    x_fit= results.coords['para_fit']*1e-9
    fr= results.attrs['fr']
    text_msg += r"$f_{r}= %.5f $"%(fr*1e-9) +' GHz\n'
    #text_msg += r"$\frac{\kappa_{l}}{2\pi}= %.3f $"%(fr/results.attrs['Ql']*1e-6) +'MHz   '+r"$Q_{l}= %.0f $"%(results.attrs['Ql']) +'\n'
    text_msg += r"$\frac{\kappa_{i}}{2\pi}= %.3f $"%(fr/results.attrs['Qi']*1e-6) +'MHz   '+r"$Q_{i}= %.0f $"%(results.attrs['Qi']) +'\n'
    text_msg += r"$\frac{\kappa_{c}}{2\pi}= %.3f $"%(fr/results.attrs['Qc']*1e-6) +'MHz   '+r"$Q_{c}= %.0f $"%(results.attrs['Qc']) +'\n'

    
    amp=results.data_vars['amp']
    pha=results.data_vars['pha']
    pha[pha<0]+= 360
    S21=results.data_vars['S21']
    fit_pha=results.data_vars['pha_fitting']
    fit_pha[fit_pha<0]+= 360
    ax[0][0].plot(x,amp,'o', color="b",label=r"$data$", alpha=0.5, ms=4)
    ax[0][0].plot(x_fit,results.data_vars['amp_fitting'],'-', color="k",label=r"$fit$", alpha=0.8, lw=1) 
    ax[1][0].plot(x,pha,'o', color="r",label=r"$data$", alpha=0.5, ms=4)
    ax[1][0].plot(x_fit,fit_pha,'-', color="k",label=r"$fit$", alpha=0.8, lw=1)
    ax[1][0].set_xlabel(x_label,size ='12')
    fig.suptitle(title,size ='15')
    ax[0][0].set_ylabel(y1_label,size ='12')
    ax[1][0].set_ylabel(y2_label,size ='12')
    plot_textbox(ax[1][1],text_msg,x=0,y=0.8,fontsize=10)

    data_real= amp*np.cos(2*np.pi*pha/360)
    data_imag= amp*np.sin(2*np.pi*pha/360)
    cmap = plt.get_cmap('jet')
    ax[0][1].set_xlabel(r"$Re(S_{21}) $",size ='12')
    ax[0][1].set_ylabel(r'$Im(S_{21}) $',size ='12')
    m=ax[0][1].scatter(data_real,data_imag,c=x, vmin=x[0], vmax=x[-1], s=20, cmap=cmap,edgecolors=None, label=r"$data$")
    ax[0][1].plot(np.real(S21),np.imag(S21),'-r',lw=1,alpha=0.7, label=r"$fit$")
    cbar=fig.colorbar(m,ax=ax[0][1])
    xscale=[min(x),max(x)]
    cbar.set_ticks(xscale)
    ax[1][1].axis('off')
    fig.tight_layout()
    plt.show()
    return fig
    
    



def Readout_opt_resonator_analysis_plot(results_g:xr.core.dataset.Dataset,results_e:xr.core.dataset.Dataset):
    
    y1_label= r'$\vert S_{21}\vert\ $'
    y2_label= r'$\angle S_{21}\ [deg]$'
    fig, ax = plt.subplots(nrows=2,ncols =2,figsize =(6,4),dpi =200)     
    text_msg = "Fit results\n"
    title= 'Rebuilt resonator spectrum by the single shot'
    x_label= 'Frequency'+' [GHz]'
    x= results_g.coords['f']*1e-9
    x_fit= results_g.coords['para_fit']*1e-9
    fr_g= results_g.attrs['fr']
    fr_e= results_e.attrs['fr']
    
    text_msg += r"$f_{r}(|g>)= %.5f $"%(fr_g*1e-9) +' GHz\n'
    text_msg += r"$f_{r}(|e>)= %.5f $"%(fr_e*1e-9) +' GHz\n'
    text_msg += r"$f_{r}(eff.bare)= %.5f $"%((fr_g+fr_e)/2*1e-9) +' GHz\n'
    text_msg += r"$\chi_{eff}/2\pi= %.5f $"%((fr_g-fr_e)/2*1e-6) +' MHz'
    print('f_{r}(|g>)=',fr_g*1e-9,'GHz')
    print('f_{r}(|e>)=',fr_e*1e-9,'GHz')
    print('f_{r}(eff.bare)=',(fr_g+fr_e)/2*1e-9,'GHz')
    print('X_{eff}/2pi=',(fr_g-fr_e)/2*1e-6,'MHz')
    
    amp_g=results_g.data_vars['amp']
    pha_g=results_g.data_vars['pha']
    pha_g[pha_g<0]+= 360
    fit_pha_g= results_g.data_vars['pha_fitting']
    fit_pha_g[fit_pha_g<0]+= 360
    S21_g=results_g.data_vars['S21']
    amp_e=results_e.data_vars['amp']
    pha_e=results_e.data_vars['pha']
    S21_e=results_e.data_vars['S21']
    pha_e[pha_e<0]+= 360
    fit_pha_e= results_e.data_vars['pha_fitting']
    fit_pha_e[fit_pha_e<0]+= 360
    
    ax[0][0].plot(x,amp_g,'o', color="b",label=r"$data$", alpha=0.5, ms=4)
    ax[0][0].plot(x_fit,results_g.data_vars['amp_fitting'],'-', color="b",label=r"$fit$", alpha=0.8, lw=1) 
    ax[1][0].plot(x,pha_g,'o', color="b",label=r"$data$", alpha=0.5, ms=4)
    ax[1][0].plot(x_fit,results_g.data_vars['pha_fitting'],'-', color="b",label=r"$fit$", alpha=0.8, lw=1)
    ax[0][0].plot(x,amp_e,'o', color="r",label=r"$data$", alpha=0.5, ms=4)
    ax[0][0].plot(x_fit,results_e.data_vars['amp_fitting'],'-', color="r",label=r"$fit$", alpha=0.8, lw=1) 
    ax[1][0].plot(x,pha_e,'o', color="r",label=r"$data$", alpha=0.5, ms=4)
    ax[1][0].plot(x_fit,results_e.data_vars['pha_fitting'],'-', color="r",label=r"$fit$", alpha=0.8, lw=1)
    ax[1][0].set_xlabel(x_label,size ='12')
    fig.suptitle(title,size ='15')
    ax[0][0].set_ylabel(y1_label,size ='12')
    ax[1][0].set_ylabel(y2_label,size ='12')
    plot_textbox(ax[1][1],text_msg,x=0,y=0.8,fontsize=10)

    data_real_g= amp_g*np.cos(2*np.pi*pha_g/360)
    data_imag_g= amp_g*np.sin(2*np.pi*pha_g/360)
    data_real_e= amp_e*np.cos(2*np.pi*pha_e/360)
    data_imag_e= amp_e*np.sin(2*np.pi*pha_e/360)
    cmap = plt.get_cmap('jet')
    ax[0][1].set_xlabel(r"$Re(S_{21}) $",size ='12')
    ax[0][1].set_ylabel(r'$Im(S_{21}) $',size ='12')
    m=ax[0][1].scatter(data_real_g,data_imag_g,c='b',alpha=0.7, s=20,edgecolors=None, label=r"$data$")
    m=ax[0][1].scatter(data_real_e,data_imag_e,c='r',alpha=0.7, s=20,edgecolors=None, label=r"$data$")
    
    ax[0][1].plot(np.real(S21_g),np.imag(S21_g),'-b',lw=1,alpha=0.7, label=r"$fit$")
    ax[0][1].plot(np.real(S21_e),np.imag(S21_e),'-r',lw=1,alpha=0.7, label=r"$fit$")
    #cbar=fig.colorbar(m,ax=ax[0][1])
    #xscale=[min(x),max(x)]
    #cbar.set_ticks(xscale)
    #cbar.ax.set_title(x_label)
    ax[1][1].axis('off')
    fig.tight_layout()
    plt.show()
    return fig


def Readout_F_amp_opt_resonator_analysis_plot(f_samples:any,amp_samples:any,result:list,target_Ql:float,electric_delay:float,plot_idx:list,fit_idx:list,f_bare:float):
    f_g,f_e,f_eff_bare,eff_chi=[],[],[],[]
    start_idx= fit_idx[0]
    if fit_idx[1]==-1:
        end_idx= len(amp_samples)+1
    else:
        end_idx= fit_idx[1]+1
    if plot_idx[1]==-1:
        fin_idx= len(amp_samples)
    else:
        fin_idx= plot_idx[1]
    for i in range(len(amp_samples)):
        Fit_result,fb= Readout_F_opt_Fit(result[i],f_samples,target_Ql=target_Ql,electric_delay=electric_delay)
        f_g.append(Fit_result['g_fit'].attrs['fr']*1e-9)
        f_e.append(Fit_result['e_fit'].attrs['fr']*1e-9)
        eff_chi.append((Fit_result['g_fit'].attrs['fr']-Fit_result['e_fit'].attrs['fr'])/2*1e-9)
        f_eff_bare.append(fb*1e-9)    
    f_eff_bare_model.set_param_hint('f_bare',value=f_bare, vary=False)
    result = f_eff_bare_model.fit(np.array(f_eff_bare)[start_idx:end_idx]*1e9,amp=amp_samples[start_idx:end_idx],a=1e9*max(f_eff_bare)-f_bare,b=Parameter(name='b', value= 0.1, min=0))
    a_fit= result.best_values['a']
    b_fit= result.best_values['b']   
    print('a=',a_fit)
    print('b=',b_fit)
    f_bare_fit= result.best_values['f_bare']

    f_r_state_func_model.set_param_hint('f_bare',value=f_bare, vary=False)
    f_r_state_func_model.set_param_hint('b',value=b_fit, vary=False)
    result_g = f_r_state_func_model.fit(np.array(f_g)[start_idx:end_idx]*1e9,amp=amp_samples[start_idx:end_idx],a=1e9*max(f_g)-f_bare)
    a_fit_g= result_g.best_values['a']

    amp_fit=np.linspace(min(amp_samples),max(amp_samples),10*len(amp_samples))
    f_bare_amp_fit=f_eff_bare_func(amp_fit,a_fit,b_fit,f_bare_fit)
    f_g_amp_fit=f_r_state_func(amp_fit,a_fit_g,b_fit,f_bare_fit)
    f_e_amp_fit= f_bare_amp_fit-(f_g_amp_fit-f_bare_amp_fit)

    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =300)
    ax.plot(f_g[plot_idx[0]:fin_idx+1],amp_samples[plot_idx[0]:fin_idx+1], "bo",label=r"$|g>$", alpha=0.5, ms=12)
    ax.plot(f_e[plot_idx[0]:fin_idx+1],amp_samples[plot_idx[0]:fin_idx+1], "ro",label=r"$|e>$", alpha=0.5, ms=12)
    ax.plot(f_eff_bare[plot_idx[0]:fin_idx+1],amp_samples[plot_idx[0]:fin_idx+1],"X", color="orange",label=r"$eff.bare$", alpha=0.5, ms=12)
    ax.plot(f_bare_amp_fit/1e9,amp_fit,'--', color="k",label=r"$fit$", alpha=0.8, lw=1.5)
    ax.plot(f_g_amp_fit/1e9,amp_fit,'--', color="b", alpha=0.8, lw=1.5)
    ax.plot(f_e_amp_fit/1e9,amp_fit,'--', color="r", alpha=0.8, lw=1.5)
    ax.set_ylabel('Readout amplitude'+' [V]',size ='15')
    ax.set_xlabel('Frequency'+' [GHz]',size ='15')
    ax.legend(loc='center left',fontsize=10, bbox_to_anchor=(1, 0.5))
    ax.set_xticks([min(f_e[plot_idx[0]:fin_idx+1]),max(f_g[plot_idx[0]:fin_idx+1])])
    fig.tight_layout()
    return fig, dict(a=a_fit,b=b_fit,f_bare=f_bare_fit) 
    
def Readout_opt_amp_plot(amp_samples:np.ndarray,results:list):
    Ig,Qg,Ie,Qe,D,F_s,F_g,F_e,F,Outlier_g,Outlier_e,SNR= [],[],[],[],[],[],[],[],[],[],[],[]
    for i in range(len(results)):
        Ig.append(results[i]['Ig'])   
        Qg.append(results[i]['Qg'])
        Ie.append(results[i]['Ie'])
        Qe.append(results[i]['Qe'])
        D.append(results[i]['error_pack']['D'])
        F_g.append(results[i]['error_pack']['F_g'])
        F_e.append(results[i]['error_pack']['F_e'])
        F_s.append(results[i]['error_pack']['F_s'])
        F.append(results[i]['error_pack']['F'])
        SNR.append(results[i]['error_pack']['SNR'])
        Outlier_g.append(results[i]['error_pack']['pre_g_outlier'])
        Outlier_e.append(results[i]['error_pack']['pre_e_outlier'])
    Ig,Qg,Ie,Qe,D,F_g,F_e,F_s,F,Outlier_g,Outlier_e= np.array(Ig),np.array(Qg),np.array(Ie),np.array(Qe),np.array(D),np.array(F_g),np.array(F_e),np.array(F_s),np.array(F),np.array(Outlier_g),np.array(Outlier_e)
    
    y1_label= r'$\rm{Outlier}\ $(%)'
    y2_label= r'$F$'
    fig, ax = plt.subplots(nrows=2,ncols =2,figsize =(9,6),dpi =200)     
    
    title= 'Readout amplitude optimization'
    x_label= 'Readout amplitude'+' [V]'
    x= amp_samples
    not_allow_amp_safe,not_allow_amp_best_SNR= [],[]
    Outlier_threshold_safe= 0.007
    Outlier_threshold_best_SNR= 0.015
    for i in range(len(x)):
        if Outlier_g[i]>=Outlier_threshold_safe or Outlier_e[i]>=Outlier_threshold_safe:
            not_allow_amp_safe.append(x[i])
        else:
            pass
        if Outlier_g[i]>=Outlier_threshold_best_SNR or Outlier_e[i]>=Outlier_threshold_best_SNR:
            not_allow_amp_best_SNR.append(x[i])
        else:
            pass    

    idx_safe= list(x).index(min(not_allow_amp_safe))
    idx_best= list(x).index(min(not_allow_amp_best_SNR))
    if idx_safe!=0:
        idx_safe-=1
    else:
        pass
    if idx_best!=0:
        idx_best-=1
    else:
        pass

    text_msg = r"$\rm{Safe/Best\ SNR\ amp.}$" +'\n'
    text_msg += r"$ %.3f/%.3f$"%(x[idx_safe],x[idx_best]) +' [V]\n\n'
    text_msg += r"$SNR= %.2f/\ %.2f\ $"%(SNR[idx_safe],SNR[idx_best]) +'\n'
    text_msg += r"$F_{g}= %.2f/\ %.2f\ $"%(100*F_g[idx_safe],100*F_g[idx_best]) +'%\n'
    text_msg += r"$F_{e}= %.2f/\ %.2f\ $"%(100*F_e[idx_safe],100*F_e[idx_best]) +'%\n'
    text_msg += r"$F_{s}= %.2f/\ %.2f\ $"%(100*F_s[idx_safe],100*F_s[idx_best]) +'%\n'
    text_msg += r"$F= %.2f/\ %.2f\ $"%(100*F[idx_safe],100*F[idx_best]) +'%'
    
    plot_textbox(ax[1][1],text_msg,x=0.2,y=0.9,fontsize=16)
    
    ax[0][0].plot(x,Outlier_g*100,'o', color="b", alpha=0.5, ms=8)
    ax[0][0].plot(x,Outlier_e*100,'o', color="r", alpha=0.5, ms=8)
    ax[0][0].axhline(y=Outlier_threshold_safe*100,linestyle='dashed',c='orange',label='Safe', alpha=0.8,lw=2)
    ax[0][0].axhline(y=Outlier_threshold_best_SNR*100,linestyle='dashed',c='purple',label='Best SNR', alpha=0.8,lw=2)
    ax[1][0].plot(x,F_g,'o', color="b",label=r"$F_{g}$", alpha=0.5, ms=8)
    ax[1][0].plot(x,F_e,'o', color="r",label=r"$F_{e}$", alpha=0.5, ms=8)
    ax[1][0].plot(x,F_s,'o', color="g",label=r"$F_{s}$", alpha=0.5, ms=8)
    ax[1][0].plot(x,F,'o', color="grey",label=r"$F$", alpha=0.5, ms=8)
    ax[1][0].set_xlabel(x_label,size ='18')
    fig.suptitle(title,size ='18')
    ax[0][0].set_ylabel(y1_label,size ='18')
    ax[1][0].set_ylabel(y2_label,size ='18')
    ax[0][0].legend(fontsize=16)
    ax[1][0].legend(fontsize=18,loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0][1].set_xlabel(x_label,size ='18')
    ax[0][1].set_ylabel('SNR',size ='18')
    ax[0][1].axvline(x=x[idx_safe],linestyle='dashed',c='orange',label='Safe', alpha=0.8,lw=2)
    ax[0][1].axvline(x=x[idx_best],linestyle='dashed',c='purple',label='Best SNR', alpha=0.8,lw=2)
    ax[0][1].axhline(y=SNR[idx_safe],linestyle='dashed',c='orange',label='Safe', alpha=0.8,lw=2)
    ax[0][1].axhline(y=SNR[idx_best],linestyle='dashed',c='purple',label='Best SNR', alpha=0.8,lw=2)
    ax[0][1].plot(x,SNR,'o', color="r",label=r"$F_{e}$", alpha=0.5, ms=8)
    ax[0][0].tick_params(labelsize='15')
    ax[1][0].tick_params(labelsize='15')
    ax[0][1].tick_params(labelsize='15')
    ax[1][1].axis('off')
    fig.tight_layout()
    

    idx= [idx_safe,idx_best]
    fig1, ax = plt.subplots(nrows=2,ncols =2,figsize =(6,6),dpi =200)
    for i in range(2):
        ce_I,ce_Q,sig=1000*results[idx[i]]['fit_pack'][0][0],1000*results[idx[i]]['fit_pack'][0][1],1000*results[idx[i]]['fit_pack'][5]
        Ig,Qg= results[idx[i]]['rot_IQdata'][0][0],results[idx[i]]['rot_IQdata'][0][1]
        Ie,Qe= results[idx[i]]['rot_IQdata'][1][0],results[idx[i]]['rot_IQdata'][1][1]
        I,Q= 1000*np.hstack([Ig,Ie]), 1000*np.hstack([Qg,Qe])
        Outlier_event_g,Inner_event_g= results[idx[i]]['Outlier_g_info']['Outlier_event'],results[idx[i]]['Outlier_g_info']['Inner_event']

        Outlier_event_e,Inner_event_e= results[idx[i]]['Outlier_e_info']['Outlier_event'],results[idx[i]]['Outlier_e_info']['Inner_event']
        Outlier_P_g, Outlier_P_e= results[idx[i]]['Outlier_g_info']['Outlier_P'],results[idx[i]]['Outlier_e_info']['Outlier_P']
        ax[i][0].scatter(1000*Inner_event_g[0], 1000*Inner_event_g[1], color="blue", alpha=0.5, s=0.5)
        ax[i][1].scatter(1000*Inner_event_e[0], 1000*Inner_event_e[1], color="red", alpha=0.5, s=0.5)
        ax[i][0].scatter(1000*Outlier_event_g[0], 1000*Outlier_event_g[1], color="grey", alpha=0.5, s=0.5)
        ax[i][1].scatter(1000*Outlier_event_e[0], 1000*Outlier_event_e[1], color="grey", alpha=0.5, s=0.5)
        text_msg1=''
        text_msg1 += r"$\rm{Outlier}= %.1f $"%(Outlier_P_g*100)+'%'
        plot_textbox_small(ax[i][0],text_msg1,x=0.47,y=0.93,fontsize=10)
        text_msg2=''
        text_msg2 += r"$\rm{Outlier}= %.1f $"%(Outlier_P_e*100)+'%'
        plot_textbox_small(ax[i][1],text_msg2,x=0.47,y=0.93,fontsize=10)

        ax[i][0].scatter(0,0,c='k',s=15)
        ax[i][0].scatter(ce_I,ce_Q,c='k',s=15)
        ax[i][1].scatter(0,0,c='k',s=15)
        ax[i][1].scatter(ce_I,ce_Q,c='k',s=15)
        ax[i][0].add_patch(Ellipse(xy=[0,0],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][0].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][1].add_patch(Ellipse(xy=[0,0],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][1].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][0].add_patch(Ellipse(xy=[0,0],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][0].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][1].add_patch(Ellipse(xy=[0,0],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][1].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][0].add_patch(Ellipse(xy=[0,0],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][0].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][1].add_patch(Ellipse(xy=[0,0],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][1].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
        ax[i][0].set_xlim(np.minimum(min(I),min(Q)),np.maximum(max(I),max(Q)))
        ax[i][0].set_ylim(np.minimum(min(I),min(Q)),np.maximum(max(I),max(Q)))
        ax[i][1].set_xlim(np.minimum(min(I),min(Q)),np.maximum(max(I),max(Q)))
        ax[i][1].set_ylim(np.minimum(min(I),min(Q)),np.maximum(max(I),max(Q)))
        ax[i][0].axes.set_aspect('equal')
        ax[i][1].axes.set_aspect('equal')
    ax[1][0].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[1][1].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[0][0].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[1][0].set_ylabel(r"$Q\ $[mV]",size ='15')
    state=['Prepare |g>','Prepare |e>']     
    row_headers = ["Safe %.1f"%(Outlier_threshold_safe*100)+'(%)', "Best SNR %.1f"%(Outlier_threshold_best_SNR*100)+'(%)']
    col_headers = state
    font_kwargs = dict(fontweight="bold", fontsize='14',color='b',alpha=0.7)
    add_headers(fig1, col_headers=col_headers, row_headers=row_headers, **font_kwargs)
    fig.tight_layout()
    return fig,fig1, x[idx_safe],x[idx_best]
    

def Readout_opt_inte_plot(results:dict,inte_samples:float,Target_Fs_value:float):
    
    Ig,Qg,Ie,Qe,D,F_s,F_g,F_e,F= [],[],[],[],[],[],[],[],[]
    for i in range(len(results)):
        Ig.append(results[i]['Ig'])   
        Qg.append(results[i]['Qg'])
        Ie.append(results[i]['Ie'])
        Qe.append(results[i]['Qe'])
        D.append(results[i]['error_pack']['D'])
        F_g.append(results[i]['error_pack']['F_g'])
        F_e.append(results[i]['error_pack']['F_e'])
        F_s.append(results[i]['error_pack']['F_s'])
        F.append(results[i]['error_pack']['F'])
    i=complex(0,1)
    Ig,Qg,Ie,Qe,D,F_g,F_e,F_s,F= np.array(Ig),np.array(Qg),np.array(Ie),np.array(Qe),np.array(D),np.array(F_g),np.array(F_e),np.array(F_s),np.array(F)
    y1_label= r'$D\ $[mV]'
    y2_label= r'$F$'
    fig, ax = plt.subplots(nrows=2,ncols =2,figsize =(6,4),dpi =200)     
    title= 'Readout integration optimization'
    x_label= r'$\tau $(ns)'
    x=inte_samples*1e9
    idx= find_nearest(F_s, Target_Fs_value)
    if F_s[idx]<Target_Fs_value:
        if idx== len(x)-1:
            pass
        else:
            idx+=1
    
    text_msg = ''
    text_msg += r"$\tau= %.0f\ $"%(x[idx]) +'ns\n'
    text_msg += r"$F_{g}= %.1f\ $"%(100*F_g[idx]) +'%\n'
    text_msg += r"$F_{e}= %.1f\ $"%(100*F_e[idx]) +'%\n'
    text_msg += r"$F_{s}= %.1f\ $"%(100*F_s[idx]) +'%\n'
    text_msg += r"$F= %.1f\ $"%(100*F[idx]) +'%'
    

    #ax[0][0].fill_between(forbidden_amp,0,max(D)*1000,facecolor='r',alpha=0.2)
    ax[0][0].plot(x,D*1000,'o', color="b",label=r"$data$", alpha=0.5, ms=4)
    ax[1][0].plot(x,F_g,'o', color="b",label=r"$F_{g}$", alpha=0.5, ms=4)
    ax[1][0].plot(x,F_e,'o', color="r",label=r"$F_{e}$", alpha=0.5, ms=4)
    ax[1][0].plot(x,F_s,'o', color="g",label=r"$F_{s}$", alpha=0.5, ms=4)
    ax[1][0].plot(x,F,'o', color="grey",label=r"$F$", alpha=0.5, ms=4)
    ax[1][0].set_xlabel(x_label,size ='15')
    fig.suptitle(title,size ='15')
    ax[0][0].set_ylabel(y1_label,size ='15')
    ax[1][0].set_ylabel(y2_label,size ='15')
    ax[1][0].legend(fontsize=8)
    plot_textbox(ax[1][1],text_msg,x=0,y=0.8,fontsize=12)

    cmap = plt.get_cmap('jet')
    ax[0][1].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[0][1].set_ylabel(r'$Q\ $[mV]',size ='15')
    m=ax[0][1].scatter(Ig*1000,Qg*1000,c=x, vmin=x[0], vmax=x[-1],alpha=0.7, s=20, cmap=cmap,edgecolors=None, label=r"$data$")
    m=ax[0][1].scatter(Ie*1000,Qe*1000,c=x, vmin=x[0], vmax=x[-1],alpha=0.7, s=20, cmap=cmap,edgecolors=None, label=r"$data$")
    
    cbar=fig.colorbar(m,ax=ax[0][1])
    xscale=[min(x),max(x)]
    cbar.set_ticks(xscale)
    #cbar.ax.set_title(x_label)
    
    ax[1][1].axis('off')
    fig.tight_layout()
    return idx,x[idx],fig
    
    
def Resonator_Fluxdep_fit_parameters_plot(results:list,bias_array:np.ndarray,fit_window_data_index:list,bias_label:str):
    pass_index=[]
    Qc,Qi,kc,ki,fr,bias=[],[],[],[],[],[]
    Qc_,Qi_,kc_,ki_,fr_,bias_=[],[],[],[],[],[]
    for i in range(len(results)):
        l=results[i]
        Qi.append(l.attrs['Qi'])
        bias.append(bias_array[i])
        fr.append(l.attrs['fr']*1e-9)
        Qc.append(l.attrs['Qc'])
        kc.append(l.attrs['fr']*1e-6/l.attrs['Qc'])
        ki.append(l.attrs['fr']*1e-6/l.attrs['Qi'])
        
    for j in range(len(Qi)):   
        if Qi[j]> 300000 or Qi[j]< 100 or Qc[j]> 50000 or Qc[j]< 100 or np.abs(fr[j])>7 or fr[j]<0:
            pass
        else:
            pass_index.append(j)
            Qi_.append(Qi[j])
            bias_.append(bias[j])
            fr_.append(fr[j])
            Qc_.append(Qc[j])
            kc_.append(kc[j])
            ki_.append(ki[j])
    print('Pass data flux index=',pass_index)
    print('Total pass data number=',len(pass_index))        
    bias,data_fit=[],[]
    start1,end1= fit_window_data_index[0][0],fit_window_data_index[0][1]
    start2,end2= fit_window_data_index[1][0],fit_window_data_index[1][1]
    window_index=[[start1,end1],[start2,end2]]
    for i in range(start1,end1+1):
        bias.append(bias_[i])
        data_fit.append(fr_[i])        
    for i in range(start2,end2+1):
        bias.append(bias_[i])
        data_fit.append(fr_[i])  

    f_guess, phase_guess=fft_oscillation_guess(data=np.array(data_fit), t=np.array(bias))
    result = Cavity_flux_model.fit(np.array(data_fit),phi_ex=np.array(bias),m=f_guess*2*np.pi,phi_offset=phase_guess,f_bare=np.mean(np.array(data_fit)),c=1)
    m_fit= result.best_values['m']
    phi_offset_fit= result.best_values['phi_offset']
    f_bare_fit= result.best_values['f_bare']
    c_fit= result.best_values['c']
    bias_fit=np.linspace(min(bias_array),max(bias_array),20*len(bias_array))
    fit=Cavity_flux(bias_fit,m_fit,phi_offset_fit,f_bare_fit,c_fit)

    def Sweet_search(n,phi_offset,m):
        return (2*n*np.pi-phi_offset)/m
    sweet_choice=[]
    sweet_RF=[]
    sweet_0= Sweet_search(0,phi_offset_fit,m_fit)
    for n in range(1,5+1):
        sweet_p= Sweet_search(n/2,phi_offset_fit,m_fit)
        sweet_m= Sweet_search(-n/2,phi_offset_fit,m_fit)

        if min(bias_array)<sweet_p<max(bias_array):
            sweet_choice.append(sweet_p)
            sweet_RF.append(Cavity_flux(sweet_p,m_fit,phi_offset_fit,f_bare_fit,c_fit))
        if min(bias_array)<sweet_m<max(bias_array):    
            sweet_choice.append(sweet_m)   
            sweet_RF.append(Cavity_flux(sweet_m,m_fit,phi_offset_fit,f_bare_fit,c_fit))
        if min(bias_array)<sweet_0<max(bias_array):    
            sweet_choice.append(sweet_0)   
            sweet_RF.append(Cavity_flux(sweet_0,m_fit,phi_offset_fit,f_bare_fit,c_fit))
    # print('-4pi',(-4*np.pi-phi_offset_fit)/m_fit)
    # print('-2pi',(-2*np.pi-phi_offset_fit)/m_fit)
    # print('2pi',(2*np.pi-phi_offset_fit)/m_fit)
    # print('4pi',(4*np.pi-phi_offset_fit)/m_fit)
    print('Voltage range involves period numbers',np.around((max(bias_array)-min(bias_array))*m_fit/2/np.pi,3))
    fig, ax = plt.subplots(nrows =3,figsize =(6,5),dpi =200)
    color=['r','b']
    for i in range(len(window_index)):
        ax[0].axvline(x=bias_[window_index[i][0]],linestyle='dashed',c=color[i], alpha=0.8,lw=1)
        ax[0].axvline(x=bias_[window_index[i][1]],linestyle='dashed',c=color[i], alpha=0.8,lw=1)
    ax[0].plot(bias_fit,fit,'--', color="red",label=r"$fit$", alpha=0.5, lw=1.5)
    ax[0].plot(bias_,fr_,'o', color="blue",label=r"$data$", alpha=0.5, ms=4)
    ax[1].plot(bias_,Qc_,marker='s', color="blue",label=r"$Q_{c}$", alpha=0.5, lw=1.5)
    ax[1].plot(bias_,Qi_,marker='X', color="red",label=r"$Q_{i}$", alpha=0.5, lw=1.5)
    ax[2].plot(bias_,kc_,marker='s', color="blue",label=r"$\kappa_{c}$", alpha=0.5, lw=1.5)
    ax[2].plot(bias_,ki_,marker='X', color="red",label=r"$\kappa_{i}$", alpha=0.5, lw=1.5)
    ax[0].set_ylabel("Frequency [GHz]",size ='12')
    ax[1].set_ylabel("Quality factor",size ='12')
    ax[2].set_ylabel(r"$\kappa/2\pi\ $[MHz]",size ='12')
    ax[0].legend(fontsize=8,loc='lower right')
    ax[1].legend(fontsize=8,loc='lower right')
    ax[2].legend(fontsize=8,loc='lower right')
    ax[2].set_xlabel(bias_label,size ='12')
    fig.tight_layout()

    return [bias_fit,fit], sweet_choice, sweet_RF,fig 
    
    
    
def show_above_threshold(data:list,threshold_low:float,threshold_high:float):   
    found_data=[]
    for i in range(len(data)):
        if threshold_high>= data[i] >= threshold_low:
            found_data.append([i,data[i]])
        else: 
            pass
    return found_data        

def Plot_assignment_matrix_SQ(M:np.ndarray,Q:str,title:str):
    
    xlabs = np.linspace(0, M.shape[1]-1, M.shape[1], dtype = int) # if M is 3x3 matrix, xlabs = [0, 1, 2]
    ylabs = np.linspace(0, M.shape[0]-1, M.shape[0], dtype = int)
             
    fig, ax = plt.subplots(nrows=1,figsize=(5,5),dpi=150)
    ax.set_xticks(np.arange(len(xlabs)), labels = xlabs)
    ax.set_yticks(np.arange(len(ylabs)), labels = ylabs)
    ax.set_xlabel("Assigned state "+Q,size ='15')
    ax.set_ylabel("Prepared state "+Q,size ='15')
    ax.set_title(title,size ='15')
    im= ax.imshow(M*100, cmap='Greens', vmin=0, vmax=100) #scale: %
    im_ratio = M.shape[0]/M.shape[1]
    cbar = ax.figure.colorbar(im, ax = ax,fraction=0.0453*im_ratio)

    for i in range(len(xlabs)):
        for j in range(len(ylabs)):
            if i!=j:
                text = ax.text(i, j, round(M[j, i]*100, 1),
                               ha = "center", va = "center", color = "k",size=18)
            else:
                text = ax.text(i, j, round(M[j, i]*100, 1),
                               ha = "center", va = "center", color = "w",size=18)
                
    cbar.ax.set_ylabel("Population (%)", rotation = -90, va = "bottom",size=18)
    cbar.set_ticks([0,20,40,60,80,100])
    cbar.ax.tick_params(labelsize=18)
    ax.tick_params(labelsize='18')
    fig.tight_layout()
    return fig


def Plot_QND_matrix_SQ(M:np.ndarray,Q:str,Pre_state:str):
    
    xlabs = ["0", "1"]
    ylabs = ["0", "1"]
             
    fig, ax = plt.subplots(nrows=1,figsize=(5,5),dpi=150)
    ax.set_xticks(np.arange(len(xlabs)), labels = xlabs)
    ax.set_yticks(np.arange(len(ylabs)), labels = ylabs)
    ax.set_xlabel("1st readout ",size ='15')
    ax.set_ylabel("2nd readout ",size ='15')
    ax.set_title('Prepare '+Q+' at '+Pre_state,size ='15')
    im= ax.imshow(M*100, cmap='Greens', vmin=0, vmax=100) #scale: %
    im_ratio = M.shape[0]/M.shape[1]
    cbar = ax.figure.colorbar(im, ax = ax,fraction=0.0453*im_ratio)

    for i in range(len(xlabs)):
        for j in range(len(ylabs)):
            if M[i, j]<0.5:
                text = ax.text(j, i, round(M[i, j]*100, 1),
                               ha = "center", va = "center", color = "k",size=18)
            else:
                text = ax.text(j, i, round(M[i, j]*100, 1),
                               ha = "center", va = "center", color = "w",size=18)
                
    cbar.ax.set_ylabel("Population (%)", rotation = -90, va = "bottom",size=18)
    cbar.set_ticks([0,20,40,60,80,100])
    cbar.ax.tick_params(labelsize=18)
    ax.tick_params(labelsize='18')
    fig.tight_layout()
    return fig
    
    
    
def Plot_assignment_matrix_2Q(M:np.ndarray,Q1Q2:list):
    
    xlabs = ["00", "01", "10", "11"]
    ylabs = ["00", "01", "10", "11"]
             
    fig, ax = plt.subplots(nrows=1,figsize=(5,5),dpi=150)
    ax.set_xticks(np.arange(len(xlabs)), labels = xlabs)
    ax.set_yticks(np.arange(len(ylabs)), labels = ylabs)
    ax.set_xlabel("Assigned state "+Q1Q2[0]+Q1Q2[1],size ='15')
    ax.set_ylabel("Prepared state "+Q1Q2[0]+Q1Q2[1],size ='15')
    ax.set_title('Direct counting',size ='15')
    im= ax.imshow(M*100, cmap='Greens', vmin=0, vmax=100) #scale: %
    im_ratio = M.shape[0]/M.shape[1]
    cbar = ax.figure.colorbar(im, ax = ax,fraction=0.0453*im_ratio)

    for i in range(len(xlabs)):
        for j in range(len(ylabs)):
            if i!=j:
                text = ax.text(j, i, round(M[i, j]*100, 1),
                               ha = "center", va = "center", color = "k",size=18)
            else:
                text = ax.text(j, i, round(M[i, j]*100, 1),
                               ha = "center", va = "center", color = "w",size=18)
                
    cbar.ax.set_ylabel("Population (%)", rotation = -90, va = "bottom",size=18)
    cbar.set_ticks([0,20,40,60,80,100])
    cbar.ax.tick_params(labelsize=18)
    ax.tick_params(labelsize='18')
    fig.tight_layout()
    return fig
    
def Z_bias_f01_T1_plot(times,Z_bias,data,f01_spec,bias_to_Ej_Ec):
    T1_array= np.array(data).reshape(times, len(Z_bias))
    mean, sigma= T1_array.mean(axis=0),T1_array.std(axis=0)
    #Plot of T1 and Gamma1 and f01 in function of Z bias
    fig, ax = plt.subplots(nrows =3,figsize =(6,5),dpi =200) 
    for i in range(times):
            ax[0].plot(Z_bias,T1_array[i],'o', color="blue", alpha=0.5, ms=2)
            ax[1].plot(Z_bias,1/T1_array[i],'o', color="blue", alpha=0.5, ms=2)
    x_label= " Z"+q[1]+' gate voltage'+" [V]" 
    y_label= r"$f_{01}$"+' [GHz]'        
    ax[0].set_ylabel(r"$T_{1}\  [\mu$s]",size ='15')
    ax[0].set_title(r"$times= %.0f $" %(times),size ='15')
    ax[0].plot(Z_bias,mean,'-', color="r", alpha=0.6,lw=1.5)
    ax[1].plot(Z_bias,1/mean,'-', color="r", alpha=0.6,lw=1.5)
    ax[1].set_ylabel(r"$\Gamma_{1}\  [1/\mu$s]",size ='15')
    Z=np.linspace(min(Z_bias),max(Z_bias),100*len(Z_bias))
    Ec         = f01_spec['Ec']
    Ejmax      = f01_spec['Ejmax']   
    m_fit      = f01_spec['m']
    phi_offset = f01_spec['phi_offset']
    d_fit      = f01_spec['d']
    text_msg=''
    text_msg += r"$E_{c}/h= %.3f $"%(Ec*1e-9) +' GHz\n'        
    text_msg += r"$E_{j, max}/h= %.3f $"%(Ejmax*1e-9) +' GHz\n'
    text_msg += r"$f_{01, max}= %.3f $"%((Transmon(0,m_fit,Ejmax,Ec,phi_offset,d_fit))*1e-9) +' GHz'
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(len(bias_to_Ej_Ec)):
        ax[2].axhline(y=Transmon(bias_to_Ej_Ec[i],m_fit,Ejmax,Ec,phi_offset,d_fit)/1e9,linestyle='dashed',c=colors[i],label=r"$E_{j}/E_{c}= %.0f $"%(Ej_transmon(bias_to_Ej_Ec[i],m_fit,Ejmax,phi_offset,d_fit)/Ec), alpha=0.8,lw=1)
    
    ax[2].legend(fontsize=5,loc='lower right')  
    ax[2].plot(Z,Transmon(Z,m_fit,Ejmax,Ec,phi_offset,d_fit)/1e9,'-', color="red", alpha=0.8, lw=1)
    ax[2].plot(f01_spec['Z_bias'],f01_spec['fit_data'] /1e9,'o', color="blue", alpha=0.5, ms=4)
    ax[2].set_xlabel(x_label,size ='15')
    ax[2].set_ylabel(y_label,size ='15')
    plot_textbox(ax[2],text_msg,fontsize=8)
    fig.tight_layout()
    Trans_Z=Transmon(Z_bias,m_fit,Ejmax,Ec,phi_offset,d_fit)
    # x-axis is turned into f01 
    leng= len(Trans_Z)
    idx_max= np.argmax(Trans_Z[int(leng/4):int(3*leng/4)]/1e9)+int(leng/4)+1
    idx_min= int(np.argmin(Trans_Z[:int(leng/2)]/1e9))

    #Plot of T1 and Gamma1 in function of frequency
    fig1, ax = plt.subplots(nrows =2,figsize =(6,5),dpi =200) 

    for i in range(times):
            ax[0].plot(Trans_Z[idx_min:idx_max]/1e9,T1_array[i][idx_min:idx_max],'o', color="blue", alpha=0.5, ms=2)
            ax[1].plot(Trans_Z[idx_min:idx_max]/1e9,1/T1_array[i][idx_min:idx_max],'o', color="blue", alpha=0.5, ms=2)
    x_label= r"$f_{01}$"+' [GHz]'        
    ax[0].set_ylabel(r"$T_{1}\  [\mu$s]",size ='15')
    ax[0].set_title(r"$times= %.0f $" %(times),size ='15')
    ax[0].plot(Trans_Z[idx_min:idx_max]/1e9,mean[idx_min:idx_max],'-', color="r", alpha=0.6,lw=1.5)
    ax[1].plot(Trans_Z[idx_min:idx_max]/1e9,1/mean[idx_min:idx_max],'-', color="r", alpha=0.6,lw=1.5)
    ax[1].set_ylabel(r"$\Gamma_{1}\  [1/\mu$s]",size ='15')
    ax[1].set_xlabel(x_label,size ='15')
    fig1.tight_layout()
    return fig,fig1



def Zgate_T1_Swap_plot(times, Z_bias,convert_bias_to_f01,Realtime,total_exp_time,data,f01_spec,xlabel):
    """ Change the z gate T1 plot into the z gate v.s times

    Args:
        convert_bias_to_f01 (bool): as title
        Realtime (bool): convert times into realtime
        total_exp_time (float): use this with "Readtime" to identify the experiment time in hours.
        data (list): T1 data with respect to bias and times
        f01_spec: xr.Dataset(data_vars=dict(data=(['Z'],f01_),
                                     fitting=(['para_fit'],fitting)),
                                     coords=dict(Z=(['Z'],bias_),
                                     para_fit=(['para_fit'],para_fit)),
                                     attrs=dict(exper="Zgate_twotone",
                                                      q=q,
                                                      Ec=Ec,
                                                      Ejmax_fit=Ejmax_fit,
                                                      phi_offset_fit=phi_offset_fit,
                                                      m_fit=m_fit,
                                                      bias_to_Ej_Ec=bias_to_Ej_Ec,
                                                      d_fit=d_fit))

    """
    T1_array= np.array(data).reshape(times, len(Z_bias))
    
    if Realtime == True:
        y_times=np.linspace(0,total_exp_time,times)
        ylabel='Time flow'+' [hours]'
    else: 
        y_times= np.linspace(1,times,times)
        ylabel='Time flow'+' [times]'
    
    T1_2D_array = []
    if convert_bias_to_f01 == True:
        Ec         = f01_spec['Ec']
        Ejmax      = f01_spec['Ejmax']   
        m_fit      = f01_spec['m']
        phi_offset = f01_spec['phi_offset']
        d_fit      = f01_spec['d']

        Trans_Z=Transmon(Z_bias,m_fit,Ejmax,Ec,phi_offset,d_fit)
        # x-axis is turned into f01 
        leng= len(Trans_Z)
        idx_max= np.argmax(Trans_Z[int(leng/4):int(3*leng/4)]/1e9)+int(leng/4)+1
        idx_min= int(np.argmin(Trans_Z[:int(leng/2)]/1e9))
        X,Y=np.meshgrid(Trans_Z[idx_min:idx_max]/1e9,y_times)
        for i in range(times):
            T1_2D_array.append(T1_array[i][idx_min:idx_max].tolist())
        xlabel = r"$f_{01}$[GHz]"
        
    else:
        X,Y=np.meshgrid(Z_bias,y_times)
        for i in range(times):
            T1_2D_array.append(T1_array[i].tolist())
        xlabel = xlabel
    
    cmap = plt.get_cmap('plasma')
    fig, ax0 = plt.subplots(nrows = 1,figsize =(6,4),dpi =200)
    pcm = ax0.pcolormesh(X, Y, T1_2D_array, cmap=cmap)
    cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
    ax0.set_title("Zgate_T1 v.s. times",size ='15')
    ax0.set_xlabel(xlabel, size = "15")
    ax0.set_ylabel(ylabel, size = "15")
    cbar.set_label(r"$T_{1}\  [\mu$s]",size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()
    
    
def Readout_speed_RF_dep_analysis(results:list,ti:float,color_bound:bool,bound_value:list,plot_linecut:bool,linecut:float):
    f=results[0]
    ti=int(ti*1e9)
    trace_recordlength= results[1]
    time_array= np.linspace(0,trace_recordlength,int(trace_recordlength*1e9))[ti:]
    raw=results[-1]
    I,Q,Amp,fit,k=[],[],[],[],[]
    for i in range(len(f)):
        offset_I,offset_Q= np.mean(raw[i][0][-100:-1]), np.mean(raw[i][1][-100:-1])
        raw_I,raw_Q= raw[i][0]-offset_I, raw[i][1]-offset_Q
        amp=np.sqrt(raw_I[ti:]**2+raw_Q[ti:]**2)
        I.append(raw_I[ti:]) 
        Q.append(raw_Q[ti:])
        #Amp.append(Trace_filtering(data=amp,fc=50*1e6))
        Amp.append(amp)
 
    I,Q,Amp= np.array(I), np.array(Q), np.array(Amp)      
    x,y=time_array*1e9,f/1e9
    X,Y=np.meshgrid(x,y)
    z= Amp.reshape(len(f),len(time_array))
    cmap = plt.get_cmap('RdBu_r')
    Nor_f=1/1000
    
    fig, ax0 = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    if color_bound:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f,cmap=cmap,vmin=bound_value[0],vmax=bound_value[1],shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
        cbar.set_ticks([bound_value[0],bound_value[1]])
    else:
        pcm = ax0.pcolormesh(X, Y, z/Nor_f, cmap=cmap,shading='auto')
        cbar =fig.colorbar(pcm, ax=ax0, extend='both', orientation='vertical')
         
    ax0.set_xlabel("Time [ns]",size ='15')
    ax0.set_ylabel("Frequency [GHz]",size ='15')
    ax0.set_title("Time response",size ='15')
    cbar.set_label('Amp [mV]',size ='15')
    cbar.ax.tick_params(labelsize=10)
    ax0.tick_params(labelsize='10')
    fig.tight_layout()  
    
    if plot_linecut is True:
        result = Photon_field_model.fit(z[linecut],t=time_array,kappa=2*np.pi*1*1e6,w=2*np.pi*1e6,A=np.max(amp),B=amp[-1],phi=0,t_=time_array[0])
        k_fit= result.best_values['kappa']
        fit=result.best_fit
        text_msg=''
        text_msg += r"$\kappa/2\pi= %.3f $"%(k_fit*1e-6/2/np.pi) +' MHz'
        
        fig,ax= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
        plot_textbox(ax,text_msg,x=0.5,y=0.8,fontsize=15)
        ax.plot(x, z[linecut]/Nor_f,'b',alpha=0.8,lw=2)
        ax.plot(x, fit/Nor_f,'--r',alpha=0.8,lw=2)
        ax.set_xlabel("Time [ns]",size ='15')
        ax.set_ylabel('Amp [mV]',size ='15')
        ax.set_title("Time response"+'_Linecut_'+"Frequency [GHz]"+' '+str(np.around(y[linecut],4)))
        fig.tight_layout()

def System_noise_analysis(results:list,IF:float,ti:float,para_info:dict, Ac_info:dict,n_avg:int,decay_window:list,floor_window:list):
    hbar = 1.054571800*1e-34
    kB = 1.38e-23 
    power_l= (para_info['A_rf'])**2/50
    Wrf= 2*np.pi*para_info['frf']
    kc= 2*np.pi*para_info['kc']
    k= 2*np.pi*para_info['k']
    BW= para_info['BW']
    coeff=Ac_info.attrs['coeff']
    n=coeff*power_l
    ti=int(ti*1e9)
    trace_recordlength= results['trace_recordlength']
    t=np.linspace(0,trace_recordlength,int(trace_recordlength*1e9))
    offset_Ig,offset_Qg= np.mean(results['g'][0][-100:-1]), np.mean(results['g'][1][-100:-1])
    raw_Ig,raw_Qg= results['g'][0]-offset_Ig, results['g'][1]-offset_Qg
    raw_Ig,raw_Qg= Digital_down_convert(raw_Ig,raw_Qg,IF,t)
    Ig,Qg= Trace_filtering(raw_Ig,fc=BW), Trace_filtering(raw_Qg,fc=BW)
    time_array= t[ti:]
    A= Ig**2+Qg**2
    k_guess= Parameter(name='kappa',value=k, min=0.9*k,max=1.1*k) 
    result = Power_photon_field_model.fit(A[ti:],t=time_array,kappa=k_guess,A=np.max(A[ti:][0:50]),B=np.mean(A[floor_window[0]:floor_window[1]]),t_=time_array[0])
    k_fit= result.best_values['kappa'] #power data
    A_fit= result.best_values['A']
    B_fit= result.best_values['B']
    t_fit= result.best_values['t_']
    

    
    plot_time_array= 1e9*t[decay_window[0]:]
    fit= Power_photon_field_decay_func(t[decay_window[0]:],k_fit,A_fit,B_fit,t_fit)
    cal= Power_photon_field_decay_func(t[decay_window[0]:],k_fit,A_fit,B_fit,t_fit)
    floor= np.mean(cal[(floor_window[0]-decay_window[0]):(floor_window[1]-decay_window[0])])
    decay= np.sum(cal[0:(decay_window[1]-decay_window[0])]-floor)*1e-9
    Ts=n*hbar*Wrf*(kc/k*0.909)*floor*n_avg/(kB*BW*2*decay)
    
    print('n=',n)
    print('n*hbar*Wrf*kc/k=',n*hbar*Wrf*(kc/k))
    print('decay_sum=',decay)
    print('BW=',BW*2)
    print('floor=',floor)
    print('Tn=',hbar*Wrf/kB,'K')
    print('Ts=',Ts,'K')
    print('sigma**2=',np.std(A[floor_window[0]:floor_window[1]]))
    
    text_msg="Analysis results\n"
    text_msg += r"$\kappa_{guess}/2\pi= %.2f $"%(k*1e-6/2/np.pi) +' MHz\n'
    text_msg += r"$\kappa_{fit}/2\pi= %.2f $"%(k_fit*1e-6/2/np.pi) +' MHz\n'
    text_msg += r"$T_{s}= %.3f $"%(Ts) +' K\n'
    text_msg += r"$\Delta n= %.2f $"%(Ts/(hbar*Wrf/kB)) 
    
    fig,ax= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    #plot_textbox(ax,text_msg,x=0.65,y=0.8,fontsize=10)
    ax.plot(plot_time_array[:-10], A[decay_window[0]:][:-10]*1000**2,'b',alpha=0.8,lw=2)
    ax.plot(plot_time_array[:-10], fit[:-10]*1000**2,'--r',alpha=0.8,lw=2)
    ax.axvline(x=1e9*t[floor_window[0]],linestyle='dashed',c='r', alpha=0.8,lw=1)
    ax.axvline(x=1e9*t[floor_window[1]],linestyle='dashed',c='r', alpha=0.8,lw=1)
    ax.axvline(x=1e9*t[decay_window[0]],linestyle='dashed',c='b', alpha=0.8,lw=1)
    ax.axvline(x=1e9*t[decay_window[1]],linestyle='dashed',c='b', alpha=0.8,lw=1)
    ax.set_xlabel("Time [ns]",size ='15')
    ax.set_ylabel(r'$I^{2}+Q^{2}\ \rm{[mV^{2}]}$',size ='15')
    fig.tight_layout()

def System_noise_analysis_with_PA(Off_results:list,On_results:list,IF:float,ti:list,para_info:dict, Ac_info:dict,n_avg:int,decay_window:list,floor_window:list,signal_gain_detect:list,ini_power:float):
    hbar = 1.054571800*1e-34
    kB = 1.38e-23 
    power_l= (para_info['A_rf'])**2/50
    Wrf= 2*np.pi*para_info['frf']
    kc= 2*np.pi*para_info['kc']
    k= 2*np.pi*para_info['k']
    BW= para_info['BW']
    coeff=Ac_info.attrs['coeff']
    n=coeff*power_l
    print(n)
    trace_recordlength= Off_results['trace_recordlength']
    tt=np.linspace(0,trace_recordlength,int(trace_recordlength*1e9))
    def Data_processing(results,ti):
        ti=int(ti*1e9)
        offset_Ig,offset_Qg= np.mean(results['g'][0][-100:-1]), np.mean(results['g'][1][-100:-1])
        raw_Ig,raw_Qg= results['g'][0]-offset_Ig, results['g'][1]-offset_Qg
        raw_Ig,raw_Qg= Digital_down_convert(raw_Ig,raw_Qg,IF,tt)
        Ig,Qg= Trace_filtering(raw_Ig,fc=BW), Trace_filtering(raw_Qg,fc=BW)
        time_array= tt[ti:]
        A= Ig**2+Qg**2
        return ti,time_array,A
    
    ti_off,time_array_off,A_off= Data_processing(Off_results,ti[0])
    ti_on,time_array_on,A_on= Data_processing(On_results,ti[1])
    decay_window_off,decay_window_on= decay_window[0],decay_window[1]
    floor_window_off,floor_window_on= floor_window[0],floor_window[1]
    f_off,f_on= np.mean(A_off[floor_window_off[0]:floor_window_off[1]]),np.mean(A_on[floor_window_on[0]:floor_window_on[1]])
    time_array=[time_array_off[:-5],time_array_on[:-5]]
    Gs= ((np.mean(A_on[signal_gain_detect[0]:signal_gain_detect[1]])-f_on)/(np.mean(A_off[signal_gain_detect[0]:signal_gain_detect[1]])-f_off))

    def Decay_func_off(t,kappa,A): 
        t_= decay_window_off[0]*1e-9
        t1=(t-t_)
        B= f_off
        return np.abs(A*np.exp(-kappa*t1)+B)
    def Decay_func_on(t,kappa,A): 
        t_= decay_window_on[0]*1e-9
        t1=(t-t_)
        B= f_on
        return np.abs(A*Gs*np.exp(-kappa*t1)+B)

    def func_dataset(params, i, t):
        kappa = params[f'kappa_{i+1}']
        A = params[f'A_{i+1}']
        if i==0:
            return Decay_func_off(t,kappa,A)
        else:
            return Decay_func_on(t,kappa,A)
    
    weight_off=10/np.std(A_off[ti_off:])
    weight_on=1/np.std(A_on[ti_on:])
    def objective(params, t, data):
        resid1 = (data[0]-func_dataset(params, 0, t[0]))*weight_off
        resid2 = (data[1]-func_dataset(params, 1, t[1]))*weight_on
        return np.concatenate((resid1, resid2))
    
    data=[A_off[ti_off:-5],A_on[ti_on:-5]]
    fit_params = Parameters()
    for iy, y in enumerate(data):
        if iy==0:
            D=A_off[ti_off:]
        else:
            D=A_on[ti_on:]
        fit_params.add(f'kappa_{iy+1}',value=k, min=0.9*k,max=1.1*k)
        fit_params.add(f'A_{iy+1}', value=ini_power/1000**2)
    fit_params['kappa_2'].expr = 'kappa_1'
    fit_params['A_2'].expr = 'A_1'
    

    out = minimize(objective, fit_params, args=(time_array, data),method='nelder', options={'xatol':4e-4})
    report_fit(out.params)
    k_fit= out.params['kappa_1']
    A_fit= out.params['A_1']
    
    plot_time_array_off= 1e9*tt[decay_window_off[0]:]
    cal_off=  Decay_func_off(tt[decay_window_off[0]:],k_fit,A_fit)
    floor_off= np.mean(cal_off[(floor_window_off[0]-decay_window_off[0]):(floor_window_off[1]-decay_window_off[0])])
    decay_off= np.sum(cal_off[0:(decay_window_off[1]-decay_window_off[0])]-floor_off)*1e-9
    Ts_off=n*hbar*Wrf*(kc/k*0.909)*floor_off*n_avg/(kB*BW*2*decay_off)
    

    
    
    plot_time_array_on= 1e9*tt[decay_window_on[0]:]
    cal_on= Decay_func_on(tt[decay_window_on[0]:],k_fit,A_fit)
    floor_on= np.mean(cal_on[(floor_window_on[0]-decay_window_on[0]):(floor_window_on[1]-decay_window_on[0])])
    decay_on= np.sum(cal_on[0:(decay_window_on[1]-decay_window_on[0])]-floor_on)*1e-9
    Ts_on=n*hbar*Wrf*(kc/k*0.909)*floor_on*n_avg/(kB*BW*2*decay_on)
    
    print('decay_sum_off=',decay_off)
    print('floor_off=',floor_off)
    print('Tn=',hbar*Wrf/kB,'K')
    print('Ts_off=',Ts_off,'K')
    print('decay_sum_on=',decay_on)
    print('floor_on=',floor_on)
    print('Tn=',hbar*Wrf/kB,'K')
    print('Ts_on=',Ts_on,'K')
    print('Gs(given)=',10*np.log10(Gs),'dB')
    print('Gs(fit_check)=',10*np.log10((cal_on[0]-floor_on)/(cal_off[0]-floor_off)),'dB')
    print('Gn(fit)=',10*np.log10(floor_on/floor_off),'dB')
    print('SNR impro.=',10*np.log10(Ts_off/Ts_on),'dB')
    
    text_msg="Analysis results\n"
    text_msg += r"$\kappa_{guess}/2\pi= %.2f $"%(k*1e-6/2/np.pi) +' MHz\n'
    text_msg += r"$\kappa_{fit}/2\pi= %.2f $"%(k_fit*1e-6/2/np.pi) +' MHz\n'
    text_msg += r"$T_{s}= %.3f $"%(Ts_off) +' K\n'
    text_msg += r"$\Delta n= %.2f $"%(Ts_off/(hbar*Wrf/kB)) 
    
    fig,ax= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    plot_textbox(ax,text_msg,x=0.62,y=0.8,fontsize=12)
    ax.plot(plot_time_array_off[:-10], A_off[decay_window_off[0]:][:-10]*1000**2,'b',alpha=0.5,lw=2)
    ax.plot(plot_time_array_off[:-10], cal_off[:-10]*1000**2,'--k',alpha=0.8,lw=1.5)
    ax.axvline(x=1e9*tt[floor_window_off[0]],linestyle='dashed',c='r', alpha=0.8,lw=1)
    ax.axvline(x=1e9*tt[floor_window_off[1]],linestyle='dashed',c='r', alpha=0.8,lw=1)
    ax.axvline(x=1e9*tt[decay_window_off[0]],linestyle='dashed',c='b', alpha=0.8,lw=1)
    ax.axvline(x=1e9*tt[decay_window_off[1]],linestyle='dashed',c='b', alpha=0.8,lw=1)
    ax.set_xlabel("Time [ns]",size ='15')
    ax.set_ylabel(r'$I^{2}+Q^{2}\ \rm{[mV^{2}]}$',size ='15')
    fig.tight_layout()

    text_msg="Analysis results\n"
    text_msg += r"$\kappa_{guess}/2\pi= %.2f $"%(k*1e-6/2/np.pi) +' MHz\n'
    text_msg += r"$\kappa_{fit}/2\pi= %.2f $"%(k_fit*1e-6/2/np.pi) +' MHz\n'
    text_msg += r"$T_{s}= %.3f $"%(Ts_on) +' K\n'
    text_msg += r"$\Delta n= %.2f $"%(Ts_on/(hbar*Wrf/kB)) 
    
    fig,ax= plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    plot_textbox(ax,text_msg,x=0.62,y=0.8,fontsize=12)
    ax.plot(plot_time_array_on[:-10], A_on[decay_window_on[0]:][:-10]*1000**2,'r',alpha=0.5,lw=2)
    ax.plot(plot_time_array_on[:-10], cal_on[:-10]*1000**2,'--k',alpha=0.8,lw=1.5)
    ax.axvline(x=1e9*tt[floor_window_on[0]],linestyle='dashed',c='r', alpha=0.8,lw=1)
    ax.axvline(x=1e9*tt[floor_window_on[1]],linestyle='dashed',c='r', alpha=0.8,lw=1)
    ax.axvline(x=1e9*tt[decay_window_on[0]],linestyle='dashed',c='b', alpha=0.8,lw=1)
    ax.axvline(x=1e9*tt[decay_window_on[1]],linestyle='dashed',c='b', alpha=0.8,lw=1)
    ax.set_xlabel("Time [ns]",size ='15')
    ax.set_ylabel(r'$I^{2}+Q^{2}\ \rm{[mV^{2}]}$',size ='15')
    fig.tight_layout()

def Ramsey_spec_fit_plot(results:dict): 
    central=results['central']
    data=results['data']
    n=results['n']
    fxy=results['fxy']/1e9
    fitting=results['fitting']
    text_msg=''
    text_msg += r"$f_{01}= %.5f $"%(central/1e9) +' GHz'
        
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    plot_textbox(ax,text_msg,x=0.6,y=0.9,fontsize=15)
    ax.set_xlabel(r"$f_{XY}\ $[GHz]",size ='15')
    ax.set_ylabel(r'$f_{Ramsey}\ $[MHz]',size ='15')
    ax.plot(fxy,data/1e6,'bo',alpha=0.6,ms=10) 
    ax.plot(fxy,fitting/1e6,'--r',alpha=0.6,lw=3)
    ax.plot(central/1e9,n/1e6,'k',marker='X',alpha=0.5,ms=15)
    ax.tick_params(labelsize='10')
    fig.tight_layout()
    return fig

def Swap_f_fit_plot(results:dict,xlabel): 
    central=results['central']
    data=results['data']
    m=results['m']
    n=results['n']
    fxy=results['fxy']
    fitting=results['fitting']
    text_msg=''
    text_msg += (xlabel+ r"$= %.5f $")%(central) 
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    plot_textbox(ax,text_msg,x=0.2,y=0.9,fontsize=12)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(r'$f\ $',size ='15')
    ax.plot(fxy,data,'bo',alpha=0.6,ms=10) 
    ax.plot(fxy,fitting,'--r',alpha=0.6,lw=3)
    ax.plot(central,Rabi_f(central,central,m,n),'k',marker='X',alpha=0.5,ms=15)
    ax.tick_params(labelsize='10')
    fig.tight_layout()
    return fig

def Avoid_crossing_fit_plot(Z_array,results,xlabel,Upper_branch_window_index,Lower_branch_window_index): 
    f01_fit1,f01_fit2=[],[]
    win1,win2= Lower_branch_window_index, Upper_branch_window_index
    Z_win1,Z_win2= Z_array[win1[0]:win1[1]],Z_array[win2[0]:win2[1]]
    for i in range(len(results['data_fit1'])):
        f01_fit1.append(results['data_fit1'][i].attrs['f01_fit']/1e9)
    for i in range(len(results['data_fit2'])):
        f01_fit2.append(results['data_fit2'][i].attrs['f01_fit']/1e9)
    f01_fit1= f01_fit1[win1[0]:win1[1]]
    f01_fit2= f01_fit2[win2[0]:win2[1]]
    # def Upper_hybrid_func(phi, f1_0, df1, f2_0, df2, J):
    #     f1 = f1_0+df1*phi
    #     f2 = f2_0+df2*phi
    #     avg = (f1+f2)/2
    #     delta = (f1-f2)/2
    #     f_plus = avg + np.sqrt(delta**2+J**2)
    #     return f_plus
    # def Lower_hybrid_func(phi, f1_0, df1, f2_0, df2, J):
    #     f1 = f1_0+df1*phi
    #     f2 = f2_0+df2*phi
    #     avg = (f1+f2)/2
    #     delta = (f1-f2)/2
    #     f_minus = avg - np.sqrt(delta**2+J**2)
    #     return f_minus
    # params = Parameters()
    # params.add('f1_0', value=np.mean(f01_fit1))
    # params.add('df1', value=2)
    # params.add('f2_0', value=np.mean(f01_fit2))
    # params.add('df2', value=-0.5)
    # params.add('J', value=10e6, min=0)  

    # Lower_hybrid_model= Model(Lower_hybrid_func)
    # Upper_hybrid_model= Model(Upper_hybrid_func)
    # result1 = Lower_hybrid_model.fit(f01_fit1,phi=Z_win1,params=params)
    # result2 = Upper_hybrid_model.fit(f01_fit2,phi=Z_win2,params=params)
    # fit_result1= Lower_hybrid_func(Z_array, **result1.params.valuesdict())
    # fit_result2= Upper_hybrid_func(Z_array, **result2.params.valuesdict())

    # params1 = Parameters()
    # params1.add('f1_0', value=np.mean(f01_fit1))   
    # params1.add('df1', value=-10)
    # params1.add('f2_0', value=np.mean(f01_fit2))
    # params1.add('df2', value=10)
    # params1.add('J', value=0.01, min=0.005)  
    # def hybrid_model(phi, f1_0, df1, f2_0, df2, J):
    #     f1 = f1_0+df1*phi
    #     f2 = f2_0+df2*phi
    #     avg = (f1+f2)/2
    #     delta = (f1-f2)/2
    #     f_plus = avg + np.sqrt(delta**2+J**2)
    #     f_minus = avg - np.sqrt(delta**2+J**2)
    #     return f_plus, f_minus

    # def residual(params, phi, f_plus_data, f_minus_data):
    #     f1_0 = params['f1_0']
    #     df1 = params['df1']
    #     f2_0 = params['f2_0']
    #     df2 = params['df2']
    #     J = params['J']
    #     f_plus_model, f_minus_model = hybrid_model(phi, f1_0, df1, f2_0, df2, J)
    #     return np.concatenate([f_plus_model - f_plus_data,f_minus_model - f_minus_data])
    
    # result = minimize(residual,params1,args=(Z_array, fit_result1, fit_result2))
    # best_vals = result.params.valuesdict()
    # f_plus_fit, f_minus_fit = hybrid_model(Z_array, **best_vals)
    
    # text_msg=''
    # text_msg += (xlabel+ r"$= %.5f $")%(central) 
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    #plot_textbox(ax,text_msg,x=0.2,y=0.9,fontsize=12)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(r"$f_{01}\ $[GHz]",size ='15')
    ax.plot(Z_win1,f01_fit1,'bo',alpha=0.6,ms=8) 
    ax.plot(Z_win2,f01_fit2,'ro',alpha=0.6,ms=8) 
    # ax.plot(Z_array,fit_result1,'--b',alpha=0.6,lw=1.5) 
    # ax.plot(Z_array,fit_result2,'--r',alpha=0.6,lw=1.5)
    # ax.plot(Z_array,f_plus_fit,'--k',alpha=0.6,lw=1.5) 
    # ax.plot(Z_array,f_minus_fit,'--k',alpha=0.6,lw=1.5) 
    ax.tick_params(labelsize='10')
    fig.tight_layout()
    return fig
    
def Ac_Stark_shift_fit_plot(results:dict,given_factors:dict,target_average_photon_number:float,ro_output_att:float): 
    def n_predict(para,P_in): # input power for an unit [dBm]
        hbar = 1.054571800*1e-34 
        i=complex(0,1)
        kc=2*np.pi*para['kc']
        ki=2*np.pi*para['ki']
        keff= kc+ki
        Wc= 2*np.pi*para['f_eff_bare']
        Wrf= 2*np.pi*para['R_F']
        delta= Wc-Wrf
        X= -2*np.pi*para['X_eff']
        amp= np.sqrt(10**(P_in/10)/1000/hbar/Wrf)
        n_g= np.abs(amp*np.sqrt(kc/2)/(-i*(delta-X)-keff/2))**2
        n_e= np.abs(amp*np.sqrt(kc/2)/(-i*(delta+X)-keff/2))**2
        return n_g,n_e
    Nor_f=1e9
    y_fit= results.data_vars['fitting']/Nor_f
    y= results.data_vars['data']/Nor_f
    x= results.coords['P']*1000   #unit:mW
    x_fit= results.coords['para_fit']*1000  #unit:mW
    fa=results.attrs['fa_0']/Nor_f
    coeff=results.attrs['coeff']

    def PtoV(P):
        return np.sqrt(P*1e-3*50)

    def VtoN(V):
        return V**2/50*coeff
    
    def NtoV(N):
        return np.sqrt(N/coeff*50)
    
    #attenuation calibration
    test_nbar_list=[]
    test_n_list=[]
    test_range= np.linspace(-145,-105,2001)
    for P_in in test_range:
        ng,ne= n_predict(given_factors,P_in)[0],n_predict(given_factors,P_in)[1]
        test_n_list.append([ng,ne])
        test_nbar_list.append((ng+ne)/2)
    idx= np.abs(np.array(test_nbar_list) - target_average_photon_number).argmin()
    predict_att= 10*np.log10(1000*test_nbar_list[idx]/coeff)-test_range[idx]
    print('n_bar=',np.around(test_nbar_list[idx],2))
    print('P_in=',np.around(test_range[idx],2),'dBm')
    print('n_g=',np.around(test_n_list[idx][0],2))
    print('n_e=',np.around(test_n_list[idx][1],2))
    print('predict_wiring_att(from DAC to the sample)=',np.around(predict_att-ro_output_att,2),'dB')


    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
    x_label= "Readout output voltage"+" [V]"
    y_label= r"$f_{01}$"+' [GHz]'

    text_msg = "Fit results\n"
    text_msg += r"$f_{01}= %.4f $"%(fa) +' GHz\n\n'
    text_msg += "Given factor\n"
    text_msg += r"$\chi_{eff}/2\pi= %.4f $"%(given_factors['X_eff']*1e-6) +' MHz'
    
    ax.plot(PtoV(x),y,'o', color="blue", alpha=0.5, ms=5)
    ax.plot(PtoV(x_fit),y_fit,'-', color="red", alpha=0.5, lw=2) 
    ax.set_xlabel(x_label,size ='15')
    ax.set_ylabel(y_label,size ='15')
    plot_textbox(ax,text_msg,fontsize=9)
    ax2 = ax.secondary_xaxis('top', functions=(VtoN,NtoV))
    ax2.set_xlabel(r"$\bar n $",size ='25')
    fig.tight_layout()
    

    
    fig1, ax = plt.subplots(nrows =1,figsize =(8,4),dpi =200)
    x_label_1="Readout output voltage"+" [V]"
    x_label_2= "Readout power reaching the sample"+" [dBm]"
    y_label= r"$\bar n $"
    y_fit= results.coords['para_fit']*coeff
    x_fit= 10*np.log10(results.coords['para_fit']*1000)-predict_att
    coeff=results.attrs['coeff']
    def PtoV_2(P):
        return np.sqrt(10**((P+predict_att)/10)*1e-3*50)
    def VtoP_2(V):
        return 10*np.log10(V**2/50*1e3)-predict_att
    
    ax2 = ax.secondary_xaxis('top', functions=(VtoP_2,PtoV_2))
    ax2.set_xlabel(x_label_2,size ='15')
    ax.plot(PtoV_2(x_fit),y_fit, color="b", alpha=0.5, lw=3)     
    ax.set_xlabel(x_label_1,size ='15')
    ax.set_ylabel(y_label,size ='25')
    ax.axvline(x=PtoV_2(test_range[idx]),color='r',linestyle='dashed', alpha=0.5,lw=1.5)
    ax.axhline(y=test_nbar_list[idx],color='r',linestyle='dashed', alpha=0.5,lw=1.5)
    text_msg = r"$f_{r}= %.5f $"%(given_factors['R_F']*1e-9) +' GHz\n'
    text_msg += r"$\rm{amp.}= %.3f $"%(PtoV_2(test_range[idx])) +' \n'
    text_msg += r"$\bar n= %.2f $"%(test_nbar_list[idx]) +' \n' 
    text_msg += r"$n_{g}= %.2f $"%(test_n_list[idx][0]) +' \n'
    text_msg += r"$n_{e}= %.2f $"%(test_n_list[idx][1]) +' \n'
    text_msg += r"$P_{in}= %.2f $"%(test_range[idx]) +' dBm\n'
    text_msg += r"$\rm{wiring\ att}= %.2f $"%(predict_att-ro_output_att) +' dB\n'
    text_msg += r"$\rm{DAC\ output\ att}= %.2f $"%(ro_output_att) +' dB'
    plot_textbox(ax,text_msg,fontsize=12)
    fig1.tight_layout()
    
    return fig,fig1, dict(kc=given_factors['kc'],ki=given_factors['ki'],X_eff=given_factors['X_eff'],f_eff_bare=given_factors['f_eff_bare'],wiring_att=predict_att,f01_bare=results.attrs['fa_0'])
        
        
def ADC_saturation_plot(q:str,data_bank:list,ro_output_att:float,fit_window_data_index:list):
    given_power= 10*np.log10(data_bank[0]**2/50*1000)-ro_output_att
    meas_power= 10*np.log10(data_bank[1]**2/50*1000)
    start,end= fit_window_data_index[0],fit_window_data_index[1]
    def Line_func(x,A):
        return x+A
    Line_model = Model(Line_func)
    result = Line_model.fit(meas_power[start:end],x=given_power[start:end],A=0)
    A_fit=result.best_values['A']
    Amp_fit= Line_func(given_power,A_fit)
    P1dB=[]
    for i in range(len(given_power)):
        if (Amp_fit-meas_power)[i]>1:
            P1dB.append(given_power[i])
            
    text_msg='Gain='+ r"$ %.2f $"%(A_fit) +' dB\n'  
    text_msg+= r"$P_{1dB}= %.2f $"%(P1dB[0]) +' dBm'  
            
    xlabel='Given power'+' [dBm]'
    ylabel='Measured power'+' [dBm]'
    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(given_power,meas_power,'bo', alpha=0.8, ms=4)
    ax.plot(given_power,Amp_fit,'r', alpha=0.8, lw=2)
    ax.grid()
    ax.axvline(P1dB[0], color = "r", ls = "--",lw=1)
    ax.set_xlabel(xlabel,size ='15')
    ax.set_ylabel(ylabel,size ='15')
    plot_textbox(ax,text_msg,x=0.1,y=0.9,fontsize=15)
    fig.tight_layout()
    
def Welch_PSD_analysis(data,ylabel, analysis_type,Realtime,total_exp_time):
    samples= np.array(data)
    if Realtime is True:
        f, PSD = welch(samples, fs=1/(total_exp_time*3600/len(samples)), nperseg=500)
        # f[1:] remove DC part
        bare_guess= Parameter(name='f', value=np.min(PSD) , min=np.min(PSD), max=np.max(PSD))
        result = Loren_noise_model.fit(PSD[1:],f=f[1:],fc=100e-6,A=np.max(PSD),base=bare_guess)
        fc_fit= result.best_values['fc']
        A_fit= result.best_values['A']
        base_fit= result.best_values['base']
        fmin,fmax= np.min(f),np.max(f)
        para_fit= np.linspace(fmin,fmax,50*len(f))[1:]
        fitting= Loren_noise_func(para_fit,fc_fit,A_fit,base_fit)
        text_msg="Fit results\n"
        if analysis_type == 'Time':
            text_msg+= r"$A_{L}=%.2f\ [\mu s]$"%(A_fit) +'\n'
            text_msg+= r"$A_{w}=%.2f\ [\mu s^{2}$/Hz]"%(base_fit) +'\n'
            text_msg+= r"$f_{c}=%.2f\ [\mu$Hz]"%(fc_fit*1e6) 
            
        elif analysis_type == 'Frequency':
            text_msg+= r"$A_{L}=%.2f\ [\rm{mHz}]$"%(A_fit*1e-3) +'\n'
            text_msg+= r"$A_{w}=%.2f\ [\rm{kHz^{2}}$/Hz]"%(base_fit*(1e-3)**2) +'\n'
            text_msg+= r"$f_{c}=%.2f\ [\rm{mHz}]$"%(fc_fit*1e6*1e-3) 
        else:
            raise KeyError ('analysis_type is not correct')
        
        
        fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
        ax.scatter(f[1:], PSD[1:], s=70, facecolors='none', edgecolors='blue',alpha=0.7)
        ax.plot(para_fit, fitting,'r',label='Total', lw=3,alpha=0.9)
        ax.plot(para_fit, fitting-base_fit,'b',linestyle='dotted',label='Lorentzian', lw=3,alpha=0.9)
        ax.axhline(y=base_fit, xmin=0, xmax=1,color='k',linestyle='dashed',label='White', alpha=0.9,lw=2.5)
        ax.set_xlabel("Frequency [Hz]",size ='15')
        ax.set_ylabel(ylabel,size ='15')
        ax.set_xlim(10**(int(np.log10(min(f[1:])))-1),fmax)
        ax.set_ylim(10**(int(np.log10(min(PSD[1:])))-1),10**(int(np.log10(max(PSD[1:])))+1))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=12)
        plot_textbox(ax,text_msg,fontsize=12)
        ax.tick_params('both',labelsize='12',labelbottom=True)
        fig.tight_layout()
        return fig
         
    else: 
        pass 



def Parity_switch_Welch_PSD_analysis(data,ylabel,single_exp_time,plot):
    def RTS_func(f,Gamma_p,A,B):     #random telegraph signal
        return A**2*(4*Gamma_p)/((2*Gamma_p)**2+(2*np.pi*f)**2)+B
    RTS_model = Model(RTS_func)
    samples= np.array(data)
    f, PSD = welch(samples, fs=1/(single_exp_time), nperseg=1000)
    # f[1:] remove DC part
    A_guess= max(PSD[1:-1])
    B_guess= Parameter(name='B', value=min(PSD[1:-1]), min=min(PSD[1:-1]), max=max(PSD[1:-1]))    
    Gamma_p_guess= Parameter(name='Gamma_p', value=1e3, min=1e2, max=1e5)
    result = RTS_model.fit(PSD[1:-1],f=f[1:-1],Gamma_p=Gamma_p_guess,A=A_guess,B=B_guess)
    Gamma_p_fit= result.best_values['Gamma_p']
    A_fit= result.best_values['A']
    B_fit= result.best_values['B']
    fmin,fmax= np.min(f),np.max(f)
    para_fit= np.linspace(fmin,fmax,50*len(f))[1:]
    fitting= RTS_func(para_fit,Gamma_p_fit,A_fit,B_fit)
    text_msg="Fit results\n"
    text_msg+= r"$\Gamma_{p}=%.2f\ $[1/ms]"%(Gamma_p_fit*1e-3)
    if plot:     
        fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
        ax.plot(f[:-1], PSD[:-1],alpha=0.8,lw=4)
        ax.set_title('Single measurement')
        ax.plot(para_fit, fitting,'--r',label='Total', lw=2,alpha=0.9)
        ax.plot(para_fit, fitting-B_fit,'b',linestyle='dotted',label='Lorentzian', lw=2,alpha=0.9)
        ax.axhline(y=B_fit, xmin=0, xmax=1,color='k',linestyle='dashed',label='White', alpha=0.9,lw=1.5)
        ax.set_xlabel("Frequency [Hz]",size ='15')
        ax.set_ylabel(ylabel,size ='15')
        ax.set_xlim(10**(int(np.log10(min(f[1:])))),fmax)
        ax.set_ylim(10**(int(np.log10(min(PSD[1:])))-1),10**(int(np.log10(max(PSD[1:])))))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=12)
        plot_textbox(ax,text_msg,fontsize=12)
        ax.tick_params('both',labelsize='12',labelbottom=True)
        fig.tight_layout()
    else:
        fig=None
    return fig, dict(Gamma_p=Gamma_p_fit,f=f,SD=PSD)
                  
def Parity_switch_Welch_PSD_analysis_average(f,SD,Delta_f01,min_Delta_f01,Delta_f01_thres,Gamma_p,single_exp_time):
    def RTS_func(f,Gamma_p,A,B):     #random telegraph signal
        return A**2*(4*Gamma_p)/((2*Gamma_p)**2+(2*np.pi*f)**2)+B
    RTS_model = Model(RTS_func)
    SD_filter=[]
    Gamma_p_filter=[]
    # filter by delta_f01
    for i in range(0,len(SD)-1):
        if np.abs(Delta_f01[i+1]-Delta_f01[i])<Delta_f01_thres and Delta_f01[i]>=min_Delta_f01:
            SD_filter.append(SD[i])
            Gamma_p_filter.append(Gamma_p[i])
        else:
            pass     
    # filter by three sigmas
    SD_filter,_= data_flow_filter(SD_filter,Gamma_p_filter)     

    SD_avg= np.mean(np.array(SD_filter),axis=0)
    A_guess= max(SD_avg[1:-1])
    B_guess= Parameter(name='B', value=min(SD_avg[1:-1]), min=min(SD_avg[1:-1]), max=max(SD_avg[1:-1]))
    Gamma_p_guess= Parameter(name='Gamma_p', value=1e3, min=1e2, max=1e5)
    result = RTS_model.fit(SD_avg[1:-1],f=f[1:-1],Gamma_p=Gamma_p_guess,A=A_guess,B=B_guess)
    Gamma_p_fit= result.best_values['Gamma_p']
    A_fit= result.best_values['A']
    B_fit= result.best_values['B']
    fmin,fmax= np.min(f),np.max(f)
    para_fit= np.linspace(fmin,fmax,50*len(f))[1:]
    fitting= RTS_func(para_fit,Gamma_p_fit,A_fit,B_fit)
    text_msg="Fit results\n"
    text_msg+= r"$\Gamma_{p}=%.2f\ $[1/ms]"%(Gamma_p_fit*1e-3)

    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(f[:-1], SD_avg[:-1],alpha=0.8,lw=4)
    ax.set_title('Average '+str(len(SD_filter))+' times')
    ax.plot(para_fit, fitting,'--r',label='Total', lw=2,alpha=0.9)
    ax.plot(para_fit, fitting-B_fit,'b',linestyle='dotted',label='Lorentzian', lw=2,alpha=0.9)
    ax.axhline(y=B_fit, xmin=0, xmax=1,color='k',linestyle='dashed',label='White', alpha=0.9,lw=1.5)
    ax.set_xlabel("Frequency [Hz]",size ='15')
    ax.set_ylabel(r' $S_{P}\ $[1/Hz]',size ='15')
    ax.set_xlim(10**(int(np.log10(min(f[1:])))),fmax)
    ax.set_ylim(10**(int(np.log10(min(SD_avg[1:])))-1),10**(int(np.log10(max(SD_avg[1:])))))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    plot_textbox(ax,text_msg,fontsize=12)
    ax.tick_params('both',labelsize='12',labelbottom=True)
    fig.tight_layout()
    return fig,Gamma_p_filter


def Parity_switch_HMM_Welch_PSD_analysis(data,ylabel,single_exp_time,plot):
    def RTS_func(f,Gamma_p,A):     #random telegraph signal
        return A**2*(4*Gamma_p)/((2*Gamma_p)**2+(2*np.pi*f)**2)
    RTS_model = Model(RTS_func)
    samples= np.array(data)
    f, PSD = welch(samples, fs=1/(single_exp_time), nperseg=1000)
    # f[1:] remove DC part
    A_guess= max(PSD[1:-1])
    Gamma_p_guess= Parameter(name='Gamma_p', value=1e3, min=1e2, max=1e5)
    result = RTS_model.fit(PSD[1:-1],f=f[1:-1],Gamma_p=Gamma_p_guess,A=A_guess)
    Gamma_p_fit= result.best_values['Gamma_p']
    A_fit= result.best_values['A']
    fmin,fmax= np.min(f),np.max(f)
    para_fit= np.linspace(fmin,fmax,50*len(f))[1:]
    fitting= RTS_func(para_fit,Gamma_p_fit,A_fit)
    text_msg="Fit results\n"
    text_msg+= r"$\Gamma_{p}=%.2f\ $[1/ms]"%(Gamma_p_fit*1e-3)
    if plot:     
        fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
        ax.plot(f[:-1], PSD[:-1],alpha=0.8,lw=4)
        ax.set_title('Single measurement')
        ax.plot(para_fit, fitting,'--r',label='Fit', lw=2,alpha=0.9)
        ax.set_xlabel("Frequency [Hz]",size ='15')
        ax.set_ylabel(ylabel,size ='15')
        ax.set_xlim(10**(int(np.log10(min(f[1:])))),fmax)
        ax.set_ylim(10**(int(np.log10(min(PSD[1:])))-1),10**(int(np.log10(max(PSD[1:])))))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=12)
        plot_textbox(ax,text_msg,fontsize=12)
        ax.tick_params('both',labelsize='12',labelbottom=True)
        fig.tight_layout()
    else:
        fig=None
    return fig, dict(Gamma_p=Gamma_p_fit,f=f,SD=PSD)


def Parity_switch_HMM_Welch_PSD_analysis_average(f,SD,Delta_f01,min_Delta_f01,Delta_f01_thres,Gamma_p,single_exp_time):
    def RTS_func(f,Gamma_p,A):     #random telegraph signal
        return A**2*(4*Gamma_p)/((2*Gamma_p)**2+(2*np.pi*f)**2)
    RTS_model = Model(RTS_func)
    SD_filter=[]
    Gamma_p_filter=[]
    # filter by delta_f01
    for i in range(0,len(SD)-1):
        if np.abs(Delta_f01[i+1]-Delta_f01[i])<Delta_f01_thres and Delta_f01[i]>=min_Delta_f01:
            SD_filter.append(SD[i])
            Gamma_p_filter.append(Gamma_p[i])
        else:
            pass     
    # filter by three sigmas
    SD_filter,_= data_flow_filter(SD_filter,Gamma_p_filter)     

    SD_avg= np.mean(np.array(SD_filter),axis=0)
    A_guess= max(SD_avg[1:-1])
    Gamma_p_guess= Parameter(name='Gamma_p', value=1e3, min=1e2, max=1e5)
    result = RTS_model.fit(SD_avg[1:-1],f=f[1:-1],Gamma_p=Gamma_p_guess,A=A_guess)
    Gamma_p_fit= result.best_values['Gamma_p']
    A_fit= result.best_values['A']
    fmin,fmax= np.min(f),np.max(f)
    para_fit= np.linspace(fmin,fmax,50*len(f))[1:]
    fitting= RTS_func(para_fit,Gamma_p_fit,A_fit)
    text_msg="Fit results\n"
    text_msg+= r"$\Gamma_{p}=%.2f\ $[1/ms]"%(Gamma_p_fit*1e-3)

    fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200) 
    ax.plot(f[:-1], SD_avg[:-1],alpha=0.8,lw=4)
    ax.set_title('Average '+str(len(SD_filter))+' times')
    ax.plot(para_fit, fitting,'--r',label='Fit', lw=2,alpha=0.9)
    ax.set_xlabel("Frequency [Hz]",size ='15')
    ax.set_ylabel(r' $S_{P}\ $[1/Hz]',size ='15')
    ax.set_xlim(10**(int(np.log10(min(f[1:])))),fmax)
    ax.set_ylim(10**(int(np.log10(min(SD_avg[1:])))-1),10**(int(np.log10(max(SD_avg[1:])))))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    plot_textbox(ax,text_msg,fontsize=12)
    ax.tick_params('both',labelsize='12',labelbottom=True)
    fig.tight_layout()
    return fig,Gamma_p_filter


def Parity_HMM_analysis(parity_sequence, single_exp_time,plot=False):
    #print(parity_sequence)
    #print(single_exp_time)
    single_exp_time = single_exp_time * 1e6 # microsecond
    observations = np.where(parity_sequence == -1, 0, 1)
    observations = observations.reshape(-1, 1)

    model = hmm.CategoricalHMM(
        n_components=2,
        n_iter=int(1e3),
        tol=1e-9,
        random_state=42,
        init_params="",
        params="ste"
    )

    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.95, 0.05], 
                                [0.05, 0.95]])
    model.emissionprob_ = np.array([
                                [0.8, 0.2], 
                                [0.2, 0.8],])

    model.fit(observations)

    print("Fitted startprob:", model.startprob_)
    print("Fitted transmat:\n", model.transmat_)
    print("Fitted emissionprob\n:", model.emissionprob_)

    logprob, hidden_states = model.decode(observations, algorithm="viterbi")
    even_number = np.count_nonzero(hidden_states)
    odd_number = len(hidden_states)-even_number
    print("odd nmber:", odd_number)
    print("even number:", even_number)
    mapped_states = np.where(hidden_states == 0, -1, 1)
    time = np.arange(0, single_exp_time * len(parity_sequence), single_exp_time)
    if plot:
        fig, ax = plt.subplots(nrows =1,figsize =(6,4),dpi =200)
        ax.plot(time, parity_sequence, label="Parity measurement", linestyle = "-", marker = "o", alpha = 0.3
        )
        ax.plot(time,
                mapped_states,
                label="Inferred Hidden State",
                drawstyle="steps-post",
                color="black",
                )
        ax.set_xlabel(r"$\text{Time}\ [\mu$s]", fontsize = 12)
        ax.set_ylabel("Inferred State", fontsize = 12)
        ax.set_xlim(-5 * single_exp_time, 300 * single_exp_time)
        ax.set_ylim(-1.1, 1.55)
        ax.set_yticks([-1,1])
        ax.legend(loc = "upper right")
        fig.suptitle("HMM fit", fontsize = 12)
        fig.tight_layout()
    else:
        fig=None
    return fig, mapped_states


def Raw_data_hist_plot(data): 
    X,Y= data['I_fit'], data['Q_fit']
    Ig_data,Qg_data,Ie_data,Qe_data= data['IQdata'][0][0],data['IQdata'][0][1],data['IQdata'][1][0],data['IQdata'][1][1]
    if len(Ig_data)<10000:
        bins=51
    elif 10000<len(Ig_data)<20000:
        bins=101
    else:
        bins=201
    hist_g, edges_Ig, edges_Qg = np.histogram2d(Ig_data, Qg_data, bins = (bins,bins), density = True)
    hist_e, edges_Ie, edges_Qe = np.histogram2d(Ie_data, Qe_data, bins = (bins,bins), density = True)
    fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (6, 6), dpi = 250)
    ax[0][0].pcolormesh(X * 1000, Y * 1000, hist_g.transpose())
    ax[0][0].set_xlabel("I [mV]")
    ax[0][0].set_ylabel("Q [mV]")
    ax[0][0].set_title("ground state 2D Histogram")
    ax[0][0].set_aspect("equal")
    ax[0][1].scatter(Ig_data * 1000, Qg_data * 1000, color = "blue", alpha = 0.5, s = 1)
    ax[0][1].set_xlabel("I [mV]")
    ax[0][1].set_ylabel("Q [mV]")
    ax[0][1].set_title("Single shot raw data")
    ax[0][1].set_aspect("equal")
    ax[1][0].pcolormesh(X * 1000, Y * 1000, hist_e.transpose())
    ax[1][0].set_xlabel("I [mV]")
    ax[1][0].set_ylabel("Q [mV]")
    ax[1][0].set_title("excited state 2D Histogram")
    ax[1][0].set_aspect("equal")
    ax[1][1].scatter(Ie_data * 1000, Qe_data * 1000, color = "red" , alpha = 0.5, s = 1)
    ax[1][1].set_xlabel("I [mV]")
    ax[1][1].set_ylabel("Q [mV]")
    ax[1][1].set_title("Single shot raw data")
    ax[1][1].set_aspect("equal")
    fig.tight_layout()
    plt.show()
    return fig

def Four_states_single_shot_plot_pre_ge(data:dict): # Input analysis_result
    Cent = data["Cent"]
    g_data = data['IQdata'][0]
    e_data = data['IQdata'][1]
    Ig_data, Qg_data = g_data[0], g_data[1]
    Ie_data, Qe_data = e_data[0], e_data[1]

    Cg        = data['fit_pack'][0]
    Ce        = data['fit_pack'][1]
    Cf        = data['fit_pack'][2]
    Cd        = data['fit_pack'][3]
    sigma_fit = data['fit_pack'][8]

    r = np.array([np.linalg.norm(Cg - Cent), np.linalg.norm(Ce - Cent), np.linalg.norm(Cf- Cent), np.linalg.norm(Cd- Cent)])
    r = np.sum(r)/len(r)

    fig, ax = plt.subplots(ncols = 2, figsize = (6, 3), dpi = 200)
    ax[0].scatter(Ig_data * 1000, Qg_data * 1000, color = "black", s = 1, alpha = 0.3)
    ax[0].set_xlabel("I [mV]", fontsize = 12)
    ax[0].set_ylabel("Q [mV]", fontsize = 12)
    ax[0].set_title("Prepared |0>", fontsize = 12)
    ax[0].add_patch(Ellipse(xy=[Cg[0]*1000,Cg[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Ce[0]*1000,Ce[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cf[0]*1000,Cf[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cd[0]*1000,Cd[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cent[0]*1000,Cent[1]*1000],width=r*2000,height=r*2000,fill=False, facecolor= None, edgecolor="#824D92FF", linewidth=1.5, linestyle='--',angle=0))
    ax[0].axes.set_aspect('equal')

    ax[1].scatter(Ie_data * 1000, Qe_data * 1000, color = "black", s = 1, alpha = 0.3)
    ax[1].set_xlabel("I [mV]", fontsize = 12)
    ax[1].set_ylabel("Q [mV]", fontsize = 12)
    ax[1].set_title("Prepared |1>", fontsize = 12)
    ax[1].add_patch(Ellipse(xy=[Cg[0]*1000,Cg[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Ce[0]*1000,Ce[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cf[0]*1000,Cf[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cd[0]*1000,Cd[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cent[0]*1000,Cent[1]*1000],width=r*2000,height=r*2000,fill=False, facecolor= None, edgecolor="#824D92FF", linewidth=1.5, linestyle='--',angle=0))
    ax[1].axes.set_aspect('equal')
    fig.tight_layout()
    return fig

def Four_states_single_shot_plot_pre_gef(data:dict): # Input analysis_result
    Cent = data["Cent"]
    g_data = data['IQdata'][0]
    e_data = data['IQdata'][1]
    f_data = data['IQdata'][2]
    Ig_data, Qg_data = g_data[0], g_data[1]
    Ie_data, Qe_data = e_data[0], e_data[1]
    If_data, Qf_data = f_data[0], f_data[1]

    Cg        = data['fit_pack'][0]
    Ce        = data['fit_pack'][1]
    Cf        = data['fit_pack'][2]
    Cd        = data['fit_pack'][3]
    sigma_fit = data['fit_pack'][8]

    r = np.array([np.linalg.norm(Cg - Cent), np.linalg.norm(Ce - Cent), np.linalg.norm(Cf- Cent), np.linalg.norm(Cd- Cent)])
    r = np.sum(r)/len(r)

    fig, ax = plt.subplots(ncols = 3, figsize = (9, 3), dpi = 200)
    ax[0].scatter(Ig_data * 1000, Qg_data * 1000, color = "black", s = 1, alpha = 0.3)
    ax[0].set_xlabel("I [mV]", fontsize = 12)
    ax[0].set_ylabel("Q [mV]", fontsize = 12)
    ax[0].set_title("Prepared |0>", fontsize = 12)
    ax[0].add_patch(Ellipse(xy=[Cg[0]*1000,Cg[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Ce[0]*1000,Ce[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cf[0]*1000,Cf[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cd[0]*1000,Cd[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cent[0]*1000,Cent[1]*1000],width=r*2000,height=r*2000,fill=False, facecolor= None, edgecolor="#824D92FF", linewidth=1.5, linestyle='--',angle=0))
    ax[0].axes.set_aspect('equal')

    ax[1].scatter(Ie_data * 1000, Qe_data * 1000, color = "black", s = 1, alpha = 0.3)
    ax[1].set_xlabel("I [mV]", fontsize = 12)
    ax[1].set_ylabel("Q [mV]", fontsize = 12)
    ax[1].set_title("Prepared |1>", fontsize = 12)
    ax[1].add_patch(Ellipse(xy=[Cg[0]*1000,Cg[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Ce[0]*1000,Ce[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cf[0]*1000,Cf[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cd[0]*1000,Cd[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cent[0]*1000,Cent[1]*1000],width=r*2000,height=r*2000,fill=False, facecolor= None, edgecolor="#824D92FF", linewidth=1.5, linestyle='--',angle=0))
    ax[1].axes.set_aspect('equal')
    fig.tight_layout()

    ax[2].scatter(If_data * 1000, Qf_data * 1000, color = "black", s = 1, alpha = 0.3)
    ax[2].set_xlabel("I [mV]", fontsize = 12)
    ax[2].set_ylabel("Q [mV]", fontsize = 12)
    ax[2].set_title("Prepared |2>", fontsize = 12)
    ax[2].add_patch(Ellipse(xy=[Cg[0]*1000,Cg[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Ce[0]*1000,Ce[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Cf[0]*1000,Cf[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Cd[0]*1000,Cd[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Cent[0]*1000,Cent[1]*1000],width=r*2000,height=r*2000,fill=False, facecolor= None, edgecolor="#824D92FF", linewidth=1.5, linestyle='--',angle=0))
    ax[2].axes.set_aspect('equal')
    fig.tight_layout()
    return fig

def Four_states_threshold_plot_pre_ge(data:dict,R_integration:float,Reset_time:float): # Input analysis_result
    Cent = data["Cent"]
    g_data = data['IQdata'][0]
    e_data = data['IQdata'][1]
    flow=np.linspace(0,(R_integration+Reset_time)*(len(g_data[0])-1),len(g_data[0]))
    thresholds = data["thresholds"]
    Cg        = data['fit_pack'][0]
    Ce        = data['fit_pack'][1]
    Cf        = data['fit_pack'][2]
    Cd        = data['fit_pack'][3]
    sigma_fit = data['fit_pack'][8]

    Cg_phase = np.angle((Cg-Cent)[0] + 1j*(Cg-Cent)[1])
    Ce_phase = np.angle((Ce-Cent)[0] + 1j*(Ce-Cent)[1])
    Cf_phase = np.angle((Cf-Cent)[0] + 1j*(Cf-Cent)[1])
    Cd_phase = np.angle((Cd-Cent)[0] + 1j*(Cd-Cent)[1])
    cloud_phase = np.array([Cg_phase, Ce_phase, Cf_phase, Cd_phase])
    cloud_phase_deg = np.rad2deg(cloud_phase)

    Ig_data, Qg_data = g_data[0], g_data[1]
    Ie_data, Qe_data = e_data[0], e_data[1]
    Ig_data = Ig_data - Cent[0]
    Qg_data = Qg_data - Cent[1]
    Ie_data = Ie_data - Cent[0]
    Qe_data = Qe_data - Cent[1]

    data_phase_g = np.angle(Ig_data + 1j*Qg_data)
    data_phase_e = np.angle(Ie_data + 1j*Qe_data)

    data_abs_g = np.abs(Ig_data + 1j*Qg_data)
    data_abs_e = np.abs(Ie_data + 1j*Qe_data)
    
    I_data, Q_data = np.hstack([Ig_data,Ie_data]), np.hstack([Qg_data,Qe_data])

    def sortingByThreshold(labels, phase, r, ratio_list):
        n = len(labels)
        for i in range(n):
            th1 = thresholds[i]
            th2 = thresholds[(i + 1) % n]
            if th1 < th2:
                if phase < th1 or phase > th2:
                    ratio_list[i] += 1
                    return (i, [r, phase])
            else:
                if th1 > phase > th2:
                    ratio_list[i] += 1
                    return (i, [r, phase])
                
    labels = [0, 1, 2, 3]
    ratio_g = np.array([0, 0, 0, 0])
    ratio_e = np.array([0, 0, 0, 0])

    sorted_gdata = [[], [], [], []]
    sorted_edata = [[], [], [], []]
    sequence_glabel = np.array([])
    sequence_gphase = np.array([])
    sequence_elabel = np.array([])
    sequence_ephase = np.array([])
    for i in range(len(data_phase_g)):
        label, pos = sortingByThreshold(labels, data_phase_g[i], data_abs_g[i], ratio_g)
        sorted_gdata[label].append(pos)
        sequence_gphase = np.append(sequence_gphase, pos[1])
        if label <= 1:
            sequence_glabel = np.append(sequence_glabel, label)
            
    for i in range(len(data_phase_e)):
        label, pos = sortingByThreshold(labels, data_phase_e[i], data_abs_e[i], ratio_e)
        sorted_edata[label].append(pos)
        sequence_ephase = np.append(sequence_ephase, pos[1])
        if label <= 1:
            sequence_elabel = np.append(sequence_elabel, label)
    
    sequence_gphase_deg = np.rad2deg(sequence_gphase)
    sequence_ephase_deg = np.rad2deg(sequence_ephase)
    #* ========= for the plot of threshold ============
    def pol2cart(data):
        r = data[0]
        theta = data[1]
        x = r * np.cos(theta) + Cent[0]
        y = r * np.sin(theta) + Cent[1]
        return(x, y)
    sorted_Igg, sorted_Qgg = pol2cart(np.column_stack(sorted_gdata[0]))
    sorted_Ige, sorted_Qge = pol2cart(np.column_stack(sorted_gdata[1]))
    sorted_Igf, sorted_Qgf = pol2cart(np.column_stack(sorted_gdata[2]))
    sorted_Igd, sorted_Qgd = pol2cart(np.column_stack(sorted_gdata[3]))

    sorted_Ieg, sorted_Qeg = pol2cart(np.column_stack(sorted_edata[0]))
    sorted_Iee, sorted_Qee = pol2cart(np.column_stack(sorted_edata[1]))
    sorted_Ief, sorted_Qef = pol2cart(np.column_stack(sorted_edata[2]))
    sorted_Ied, sorted_Qed = pol2cart(np.column_stack(sorted_edata[3]))

    pos_dg = pol2cart([1, thresholds[0]])
    pos_ge = pol2cart([1, thresholds[1]])
    pos_ef = pol2cart([1, thresholds[2]])
    pos_fd = pol2cart([1, thresholds[3]])
    X_dg = np.linspace(Cent[0], pos_dg[0] + (pos_dg[0] - Cent[0])*5, 1000)
    Y_dg = np.linspace(Cent[1], pos_dg[1] + (pos_dg[1] - Cent[1])*5, 1000)
    X_ge = np.linspace(Cent[0], pos_ge[0] + (pos_ge[0] - Cent[0])*5, 1000)
    Y_ge = np.linspace(Cent[1], pos_ge[1] + (pos_ge[1] - Cent[1])*5, 1000)
    X_ef = np.linspace(Cent[0], pos_ef[0] + (pos_ef[0] - Cent[0])*5, 1000)
    Y_ef = np.linspace(Cent[1], pos_ef[1] + (pos_ef[1] - Cent[1])*5, 1000)
    X_fd = np.linspace(Cent[0], pos_fd[0] + (pos_fd[0] - Cent[0])*5, 1000)
    Y_fd = np.linspace(Cent[1], pos_fd[1] + (pos_fd[1] - Cent[1])*5, 1000)

    #* =========== threshold plot ==============
    fig1, ax = plt.subplots(ncols = 2, figsize = (6, 3), dpi = 200)
    ax[0].scatter(sorted_Igg * 1000, sorted_Qgg * 1000, s = 1, color = "#4242BDFF", label = "|0>")
    ax[0].scatter(sorted_Ige * 1000, sorted_Qge * 1000, s = 1, color = "#CC5454FF", label = "|1>")
    ax[0].scatter(sorted_Igf * 1000, sorted_Qgf * 1000, s = 1, color = "#4AB64AFF", label = "|2>")
    ax[0].scatter(sorted_Igd * 1000, sorted_Qgd * 1000, s = 1, color = "#C2BC66FF", label = "|3>")
    ax[0].add_patch(Ellipse(xy=[Cg[0] * 1000,Cg[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Ce[0] * 1000,Ce[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cf[0] * 1000,Cf[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cd[0] * 1000,Cd[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].plot(X_dg * 1000, Y_dg * 1000, color = "black")
    ax[0].plot(X_ge * 1000, Y_ge * 1000, color = "black")
    ax[0].plot(X_ef * 1000, Y_ef * 1000, color = "black")
    ax[0].plot(X_fd * 1000, Y_fd * 1000, color = "black")
    ax[0].set_xlabel("I [mV]", fontsize = 12)
    ax[0].set_ylabel("Q [mV]", fontsize = 12)
    ax[0].set_title("Prepared |0>", fontsize = 12)
    ax[0].set_xlim((np.min(g_data[0]) - sigma_fit) * 1000, (np.max(g_data[0]) + sigma_fit) * 1000)
    ax[0].set_ylim((np.min(g_data[1]) - sigma_fit) * 1000, (np.max(g_data[1]) + sigma_fit) * 1000)
    ax[0].legend(fontsize = 7, edgecolor = "black", markerscale = 3, framealpha = 0.1, ncols = 2, columnspacing = 0.5)
    ax[0].axes.set_aspect('equal')

    ax[1].scatter(sorted_Ieg * 1000, sorted_Qeg * 1000, s = 1, color = "#4242BDFF", label = "|0>")
    ax[1].scatter(sorted_Iee * 1000, sorted_Qee * 1000, s = 1, color = "#CC5454FF", label = "|1>")
    ax[1].scatter(sorted_Ief * 1000, sorted_Qef * 1000, s = 1, color = "#4AB64AFF", label = "|2>")
    ax[1].scatter(sorted_Ied * 1000, sorted_Qed * 1000, s = 1, color = "#C2BC66FF", label = "|3>")
    ax[1].add_patch(Ellipse(xy=[Cg[0] * 1000,Cg[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Ce[0] * 1000,Ce[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cf[0] * 1000,Cf[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cd[0] * 1000,Cd[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].plot(X_dg * 1000, Y_dg * 1000, color = "black")
    ax[1].plot(X_ge * 1000, Y_ge * 1000, color = "black")
    ax[1].plot(X_ef * 1000, Y_ef * 1000, color = "black")
    ax[1].plot(X_fd * 1000, Y_fd * 1000, color = "black")
    ax[1].set_xlabel("I [mV]", fontsize = 12)
    ax[1].set_ylabel("Q [mV]", fontsize = 12)
    ax[1].set_title("Prepared |1>", fontsize = 12)
    ax[1].set_xlim((np.min(g_data[0]) - sigma_fit) * 1000, (np.max(g_data[0]) + sigma_fit) * 1000)
    ax[1].set_ylim((np.min(g_data[1]) - sigma_fit) * 1000, (np.max(g_data[1]) + sigma_fit) * 1000)    
    ax[1].legend(fontsize = 7, edgecolor = "black", markerscale = 3, framealpha = 0.1, ncols = 2, columnspacing = 0.5)
    ax[1].axes.set_aspect('equal')
    fig1.tight_layout()

    #* =========== phase time flow ==============
    fig2, ax2 = plt.subplots(ncols = 2, figsize = (9, 3), dpi = 200)
    ax2[0].scatter(flow, sequence_gphase_deg, s=1, alpha = 0.3, color="black")
    ax2[0].hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", linewidth = 2, label = "|0>")
    ax2[0].hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", linewidth = 2, label = "|1>")
    ax2[0].hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", linewidth = 2, label = "|2>")
    ax2[0].hlines(cloud_phase_deg[3], np.min(flow), np.max(flow), color = "#C2BC66FF", linestyle = "-", linewidth = 2, label = "|3>")
    ax2[0].legend(ncols = 1, bbox_to_anchor = (1, 1))
    ax2[0].set_xlabel('Time flow [s]', fontsize = 12)
    ax2[0].set_ylabel("Phase [deg]", fontsize = 12)
    ax2[0].set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
    ax2[0].set_title("Prepared |0>", fontsize = 12)

    ax2[1].scatter(flow, sequence_ephase_deg, s=1, alpha = 0.3, color="black")
    ax2[1].hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", linewidth = 2, label = "|0>")
    ax2[1].hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", linewidth = 2, label = "|1>")
    ax2[1].hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", linewidth = 2, label = "|2>")
    ax2[1].hlines(cloud_phase_deg[3], np.min(flow), np.max(flow), color = "#C2BC66FF", linestyle = "-", linewidth = 2, label = "|3>")
    ax2[1].legend(ncols = 1, bbox_to_anchor = (1, 1))
    ax2[1].set_xlabel('Time flow [s]', fontsize = 12)
    ax2[1].set_ylabel("Phase [deg]", fontsize = 12)
    ax2[1].set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
    ax2[1].set_title("Prepared |1>", fontsize = 12)
    fig2.tight_layout()

    return (fig1, fig2)

def Four_states_threshold_plot_pre_gef(data:dict,R_integration:float,Reset_time:float): # Input analysis_result
    Cent = data["Cent"]
    g_data = data['IQdata'][0]
    e_data = data['IQdata'][1]
    f_data = data['IQdata'][2]

    flow=np.linspace(0,(R_integration+Reset_time)*(len(g_data[0])-1),len(g_data[0]))
    thresholds = data["thresholds"]
    Cg        = data['fit_pack'][0]
    Ce        = data['fit_pack'][1]
    Cf        = data['fit_pack'][2]
    Cd        = data['fit_pack'][3]
    sigma_fit = data['fit_pack'][8]

    Cg_phase = np.angle((Cg-Cent)[0] + 1j*(Cg-Cent)[1])
    Ce_phase = np.angle((Ce-Cent)[0] + 1j*(Ce-Cent)[1])
    Cf_phase = np.angle((Cf-Cent)[0] + 1j*(Cf-Cent)[1])
    Cd_phase = np.angle((Cd-Cent)[0] + 1j*(Cd-Cent)[1])
    cloud_phase = np.array([Cg_phase, Ce_phase, Cf_phase, Cd_phase])
    cloud_phase_deg = np.rad2deg(cloud_phase)

    Ig_data, Qg_data = g_data[0], g_data[1]
    Ie_data, Qe_data = e_data[0], e_data[1]
    If_data, Qf_data = f_data[0], f_data[1]
    Ig_data = Ig_data - Cent[0]
    Qg_data = Qg_data - Cent[1]
    Ie_data = Ie_data - Cent[0]
    Qe_data = Qe_data - Cent[1]
    If_data = If_data - Cent[0]
    Qf_data = Qf_data - Cent[1]

    data_phase_g = np.angle(Ig_data + 1j*Qg_data)
    data_phase_e = np.angle(Ie_data + 1j*Qe_data)
    data_phase_f = np.angle(If_data + 1j*Qf_data)

    data_abs_g = np.abs(Ig_data + 1j*Qg_data)
    data_abs_e = np.abs(Ie_data + 1j*Qe_data)
    data_abs_f = np.abs(If_data + 1j*Qf_data)
    
    def sortingByThreshold(labels, phase, r, ratio_list):
        n = len(labels)
        for i in range(n):
            th1 = thresholds[i]
            th2 = thresholds[(i + 1) % n]
            if th1 < th2:
                if phase < th1 or phase > th2:
                    ratio_list[i] += 1
                    return (i, [r, phase])
            else:
                if th1 > phase > th2:
                    ratio_list[i] += 1
                    return (i, [r, phase])
                
    labels = [0, 1, 2, 3]
    ratio_g = np.array([0, 0, 0, 0])
    ratio_e = np.array([0, 0, 0, 0])
    ratio_f = np.array([0, 0, 0, 0])

    sorted_gdata = [[], [], [], []]
    sorted_edata = [[], [], [], []]
    sorted_fdata = [[], [], [], []]
    sequence_glabel = np.array([])
    sequence_gphase = np.array([])
    sequence_elabel = np.array([])
    sequence_ephase = np.array([])
    sequence_flabel = np.array([])
    sequence_fphase = np.array([])

    for i in range(len(data_phase_g)):
        label, pos = sortingByThreshold(labels, data_phase_g[i], data_abs_g[i], ratio_g)
        sorted_gdata[label].append(pos)
        sequence_gphase = np.append(sequence_gphase, pos[1])
        if label <= 1:
            sequence_glabel = np.append(sequence_glabel, label)
            
    for i in range(len(data_phase_e)):
        label, pos = sortingByThreshold(labels, data_phase_e[i], data_abs_e[i], ratio_e)
        sorted_edata[label].append(pos)
        sequence_ephase = np.append(sequence_ephase, pos[1])
        if label <= 1:
            sequence_elabel = np.append(sequence_elabel, label)
    
    for i in range(len(data_phase_f)):
        label, pos = sortingByThreshold(labels, data_phase_f[i], data_abs_f[i], ratio_f)
        sorted_fdata[label].append(pos)
        sequence_fphase = np.append(sequence_fphase, pos[1])
        if label <= 1:
            sequence_elabel = np.append(sequence_elabel, label)
    sequence_gphase_deg = np.rad2deg(sequence_gphase)
    sequence_ephase_deg = np.rad2deg(sequence_ephase)
    sequence_fphase_deg = np.rad2deg(sequence_fphase)
    #* ========= for the plot of threshold ============
    def pol2cart(data):
        r = data[0]
        theta = data[1]
        x = r * np.cos(theta) + Cent[0]
        y = r * np.sin(theta) + Cent[1]
        return(x, y)
    sorted_Igg, sorted_Qgg = pol2cart(np.column_stack(sorted_gdata[0]))
    sorted_Ige, sorted_Qge = pol2cart(np.column_stack(sorted_gdata[1]))
    sorted_Igf, sorted_Qgf = pol2cart(np.column_stack(sorted_gdata[2]))
    sorted_Igd, sorted_Qgd = pol2cart(np.column_stack(sorted_gdata[3]))

    sorted_Ieg, sorted_Qeg = pol2cart(np.column_stack(sorted_edata[0]))
    sorted_Iee, sorted_Qee = pol2cart(np.column_stack(sorted_edata[1]))
    sorted_Ief, sorted_Qef = pol2cart(np.column_stack(sorted_edata[2]))
    sorted_Ied, sorted_Qed = pol2cart(np.column_stack(sorted_edata[3]))

    sorted_Ifg, sorted_Qfg = pol2cart(np.column_stack(sorted_fdata[0]))
    sorted_Ife, sorted_Qfe = pol2cart(np.column_stack(sorted_fdata[1]))
    sorted_Iff, sorted_Qff = pol2cart(np.column_stack(sorted_fdata[2]))
    sorted_Ifd, sorted_Qfd = pol2cart(np.column_stack(sorted_fdata[3]))

    pos_dg = pol2cart([1, thresholds[0]])
    pos_ge = pol2cart([1, thresholds[1]])
    pos_ef = pol2cart([1, thresholds[2]])
    pos_fd = pol2cart([1, thresholds[3]])
    X_dg = np.linspace(Cent[0], pos_dg[0] + (pos_dg[0] - Cent[0])*5, 1000)
    Y_dg = np.linspace(Cent[1], pos_dg[1] + (pos_dg[1] - Cent[1])*5, 1000)
    X_ge = np.linspace(Cent[0], pos_ge[0] + (pos_ge[0] - Cent[0])*5, 1000)
    Y_ge = np.linspace(Cent[1], pos_ge[1] + (pos_ge[1] - Cent[1])*5, 1000)
    X_ef = np.linspace(Cent[0], pos_ef[0] + (pos_ef[0] - Cent[0])*5, 1000)
    Y_ef = np.linspace(Cent[1], pos_ef[1] + (pos_ef[1] - Cent[1])*5, 1000)
    X_fd = np.linspace(Cent[0], pos_fd[0] + (pos_fd[0] - Cent[0])*5, 1000)
    Y_fd = np.linspace(Cent[1], pos_fd[1] + (pos_fd[1] - Cent[1])*5, 1000)

    #* =========== threshold plot ==============
    fig1, ax = plt.subplots(ncols = 3, figsize = (9, 3), dpi = 200)
    ax[0].scatter(sorted_Igg * 1000, sorted_Qgg * 1000, s = 1, color = "#4242BDFF", label = "|0>")
    ax[0].scatter(sorted_Ige * 1000, sorted_Qge * 1000, s = 1, color = "#CC5454FF", label = "|1>")
    ax[0].scatter(sorted_Igf * 1000, sorted_Qgf * 1000, s = 1, color = "#4AB64AFF", label = "|2>")
    ax[0].scatter(sorted_Igd * 1000, sorted_Qgd * 1000, s = 1, color = "#C2BC66FF", label = "|3>")
    ax[0].add_patch(Ellipse(xy=[Cg[0] * 1000,Cg[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Ce[0] * 1000,Ce[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cf[0] * 1000,Cf[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cd[0] * 1000,Cd[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].plot(X_dg * 1000, Y_dg * 1000, color = "black")
    ax[0].plot(X_ge * 1000, Y_ge * 1000, color = "black")
    ax[0].plot(X_ef * 1000, Y_ef * 1000, color = "black")
    ax[0].plot(X_fd * 1000, Y_fd * 1000, color = "black")
    ax[0].set_xlabel("I [mV]", fontsize = 12)
    ax[0].set_ylabel("Q [mV]", fontsize = 12)
    ax[0].set_title("Prepared |0>", fontsize = 12)
    ax[0].set_xlim((np.min(g_data[0]) - sigma_fit) * 1000, (np.max(g_data[0]) + sigma_fit) * 1000)
    ax[0].set_ylim((np.min(g_data[1]) - sigma_fit) * 1000, (np.max(g_data[1]) + sigma_fit) * 1000)
    ax[0].legend(fontsize = 7, edgecolor = "black", markerscale = 3, framealpha = 0.1, ncols = 2, columnspacing = 0.5)
    ax[0].axes.set_aspect('equal')

    ax[1].scatter(sorted_Ieg * 1000, sorted_Qeg * 1000, s = 1, color = "#4242BDFF", label = "|0>")
    ax[1].scatter(sorted_Iee * 1000, sorted_Qee * 1000, s = 1, color = "#CC5454FF", label = "|1>")
    ax[1].scatter(sorted_Ief * 1000, sorted_Qef * 1000, s = 1, color = "#4AB64AFF", label = "|2>")
    ax[1].scatter(sorted_Ied * 1000, sorted_Qed * 1000, s = 1, color = "#C2BC66FF", label = "|3>")
    ax[1].add_patch(Ellipse(xy=[Cg[0] * 1000,Cg[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Ce[0] * 1000,Ce[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cf[0] * 1000,Cf[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cd[0] * 1000,Cd[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].plot(X_dg * 1000, Y_dg * 1000, color = "black")
    ax[1].plot(X_ge * 1000, Y_ge * 1000, color = "black")
    ax[1].plot(X_ef * 1000, Y_ef * 1000, color = "black")
    ax[1].plot(X_fd * 1000, Y_fd * 1000, color = "black")
    ax[1].set_xlabel("I [mV]", fontsize = 12)
    ax[1].set_ylabel("Q [mV]", fontsize = 12)
    ax[1].set_title("Prepared |1>", fontsize = 12)
    ax[1].set_xlim((np.min(g_data[0]) - sigma_fit) * 1000, (np.max(g_data[0]) + sigma_fit) * 1000)
    ax[1].set_ylim((np.min(g_data[1]) - sigma_fit) * 1000, (np.max(g_data[1]) + sigma_fit) * 1000)    
    ax[1].legend(fontsize = 7, edgecolor = "black", markerscale = 3, framealpha = 0.1, ncols = 2, columnspacing = 0.5)
    ax[1].axes.set_aspect('equal')

    ax[2].scatter(sorted_Ifg * 1000, sorted_Qfg * 1000, s = 1, color = "#4242BDFF", label = "|0>")
    ax[2].scatter(sorted_Ife * 1000, sorted_Qfe * 1000, s = 1, color = "#CC5454FF", label = "|1>")
    ax[2].scatter(sorted_Iff * 1000, sorted_Qff * 1000, s = 1, color = "#4AB64AFF", label = "|2>")
    ax[2].scatter(sorted_Ifd * 1000, sorted_Qfd * 1000, s = 1, color = "#C2BC66FF", label = "|3>")
    ax[2].add_patch(Ellipse(xy=[Cg[0] * 1000,Cg[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Ce[0] * 1000,Ce[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Cf[0] * 1000,Cf[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Cd[0] * 1000,Cd[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#C2BC66FF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].plot(X_dg * 1000, Y_dg * 1000, color = "black")
    ax[2].plot(X_ge * 1000, Y_ge * 1000, color = "black")
    ax[2].plot(X_ef * 1000, Y_ef * 1000, color = "black")
    ax[2].plot(X_fd * 1000, Y_fd * 1000, color = "black")
    ax[2].set_xlabel("I [mV]", fontsize = 12)
    ax[2].set_ylabel("Q [mV]", fontsize = 12)
    ax[2].set_title("Prepared |2>", fontsize = 12)
    ax[2].set_xlim((np.min(g_data[0]) - sigma_fit) * 1000, (np.max(g_data[0]) + sigma_fit) * 1000)
    ax[2].set_ylim((np.min(g_data[1]) - sigma_fit) * 1000, (np.max(g_data[1]) + sigma_fit) * 1000)    
    ax[2].legend(fontsize = 7, edgecolor = "black", markerscale = 3, framealpha = 0.1, ncols = 2, columnspacing = 0.5)
    ax[2].axes.set_aspect('equal')
    fig1.tight_layout()

    #* =========== phase time flow ==============
    fig2, ax2 = plt.subplots(ncols = 3, figsize = (13.5, 3), dpi = 200)
    ax2[0].scatter(flow, sequence_gphase_deg, s=1, alpha = 0.3, color="black")
    ax2[0].hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", linewidth = 2, label = "|0>")
    ax2[0].hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", linewidth = 2, label = "|1>")
    ax2[0].hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", linewidth = 2, label = "|2>")
    ax2[0].hlines(cloud_phase_deg[3], np.min(flow), np.max(flow), color = "#C2BC66FF", linestyle = "-", linewidth = 2, label = "|3>")
    ax2[0].legend(ncols = 1, bbox_to_anchor = (1, 1))
    ax2[0].set_xlabel('Time flow [s]', fontsize = 12)
    ax2[0].set_ylabel("Phase [deg]", fontsize = 12)
    ax2[0].set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
    ax2[0].set_title("Prepared |0>", fontsize = 12)

    ax2[1].scatter(flow, sequence_ephase_deg, s=1, alpha = 0.3, color="black")
    ax2[1].hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", linewidth = 2, label = "|0>")
    ax2[1].hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", linewidth = 2, label = "|1>")
    ax2[1].hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", linewidth = 2, label = "|2>")
    ax2[1].hlines(cloud_phase_deg[3], np.min(flow), np.max(flow), color = "#C2BC66FF", linestyle = "-", linewidth = 2, label = "|3>")
    ax2[1].legend(ncols = 1, bbox_to_anchor = (1, 1))
    ax2[1].set_xlabel('Time flow [s]', fontsize = 12)
    ax2[1].set_ylabel("Phase [deg]", fontsize = 12)
    ax2[1].set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
    ax2[1].set_title("Prepared |1>", fontsize = 12)

    ax2[2].scatter(flow, sequence_fphase_deg, s=1, alpha = 0.3, color="black")
    ax2[2].hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", linewidth = 2, label = "|0>")
    ax2[2].hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", linewidth = 2, label = "|1>")
    ax2[2].hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", linewidth = 2, label = "|2>")
    ax2[2].hlines(cloud_phase_deg[3], np.min(flow), np.max(flow), color = "#C2BC66FF", linestyle = "-", linewidth = 2, label = "|3>")
    ax2[2].legend(ncols = 1, bbox_to_anchor = (1, 1))
    ax2[2].set_xlabel('Time flow [s]', fontsize = 12)
    ax2[2].set_ylabel("Phase [deg]", fontsize = 12)
    ax2[2].set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
    ax2[2].set_title("Prepared |2>", fontsize = 12)
    fig2.tight_layout()

    return (fig1, fig2)

def Phase_Timeflow_plot(data,cloud_phase_deg,sequence_phase_deg,sequence_label,single_exp_time,analysis_cloud_number): # IQ_g should be np.array
    I_data, Q_data = data[0], data[1]
    flow=1000*np.linspace(0,(single_exp_time)*(len(I_data)-1),len(I_data))

    #* =========== phase time flow ==============
    if analysis_cloud_number==4:
        fig, ax = plt.subplots(ncols = 1, figsize = (6, 4), dpi = 200)
        ax.scatter(flow, sequence_phase_deg, color = "black", s = 10, alpha = 0.3)
        ax.hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", label = "|0>")
        ax.hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", label = "|1>")
        ax.hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", label = "|2>")
        ax.hlines(cloud_phase_deg[3], np.min(flow), np.max(flow), color = "#C2BC66FF", linestyle = "-", label = "|3>")
        ax.set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
        ax.set_xlabel('Time flow [ms]', fontsize = 12)
        ax.set_ylabel("Phase [deg]", fontsize = 12)
        ax.set_title("Phase time flow", fontsize = 12)
        ax.legend(ncols = 1, bbox_to_anchor = (1, 1))
        fig.tight_layout()

        sequence_label_deg=[]
        for i in sequence_label:
            sequence_label_deg.append(cloud_phase_deg[int(i)])

        fig1, ax = plt.subplots(ncols = 1, figsize = (6, 4), dpi = 200)
        ax.plot(flow, sequence_label_deg, color = "k",marker='o', alpha = 0.5,lw=1)
        ax.hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", label = "|0>")
        ax.hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", label = "|1>")
        ax.hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", label = "|2>")
        ax.hlines(cloud_phase_deg[3], np.min(flow), np.max(flow), color = "#C2BC66FF", linestyle = "-", label = "|3>")
        ax.set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
        ax.set_xlabel('Time flow [ms]', fontsize = 12)
        ax.set_ylabel("Phase [deg]", fontsize = 12)
        ax.set_title("Phase time flow", fontsize = 12)
        ax.legend(ncols = 1, bbox_to_anchor = (1, 1))
        fig1.tight_layout()
    return fig,fig1

def Parity_Timeflow_plot(flow,P):
    total_count= len(P)
    odd= np.count_nonzero(np.array(P) == -1)
    even= total_count-odd
    flow= flow*1000
    fig, ax = plt.subplots(ncols = 1, figsize = (6, 4), dpi = 200)
    ax.plot(flow, P, color = "k",marker='o', alpha = 0.5,lw=1)
    ax.hlines(-1,xmin=min(flow),xmax=max(flow), color = "#4242BDFF", linestyle = "--", label = "odd")
    ax.hlines(1,xmin=min(flow),xmax=max(flow), color = "#CC5454FF", linestyle = "--", label = "even")
    ax.set_ylim(-1.5,1.5)
    #ax.set_xlim(0,1)
    ax.set_xlabel('Time flow [ms]', fontsize = 12)
    ax.set_ylabel("Parity", fontsize = 12)
    ax.set_title("Parity time flow", fontsize = 12)
    ax.legend(ncols = 1, bbox_to_anchor = (1, 1), fontsize = 12)
    text_msg = "Count results\n\n"
    text_msg+= "odd= %.0f "%(odd)+'\n'
    text_msg+= "even= %.0f "%(even)
    plot_textbox(ax,text_msg,fontsize=10)
    fig.tight_layout()
    return fig

def Parity_Timeflow_2Q_plot(flow,P_Q1,P_Q2):
    total_count= len(P_Q1)
    odd_1= np.count_nonzero(np.array(P_Q1) == -1)
    odd_2= np.count_nonzero(np.array(P_Q2) == -1)
    even_1= total_count-odd_1
    even_2= total_count-odd_2
    flow= flow*1000
    fig, ax = plt.subplots(nrows=2, ncols = 1, dpi = 200, sharex=True)
    fig.subplots_adjust(hspace=0)
    ax[0].plot(flow, P_Q1, color = "b",marker='o', alpha = 0.5,lw=1)
    ax[1].plot(flow, P_Q2, color = "r",marker='o', alpha = 0.5,lw=1)
    ax[0].set_ylim(-1.2,1.2)
    ax[1].set_ylim(-1.2,1.2)
    #ax.set_xlim(0,1)
    ax[1].set_xlabel('Time flow [ms]', fontsize = 12)
    ax[0].set_ylabel("Parity (Q1)", fontsize = 12)
    ax[1].set_ylabel("Parity (Q2)", fontsize = 12)
    ax[0].set_title("Parity time flow", fontsize = 12)
    value= np.mean(Cross_correlation(P_Q1,P_Q2))
    text_msg = "Count results\n\n"
    text_msg+= "Q1"+'\n'
    text_msg+= "odd= %.0f "%(odd_1)+'\n'
    text_msg+= "even= %.0f "%(even_1)+'\n\n'
    text_msg+= "Q2"+'\n'
    text_msg+= "odd= %.0f "%(odd_2)+'\n'
    text_msg+= "even= %.0f "%(even_2)

    text_msg1 = "Cross correlation=%.2f"%(value)
    plot_textbox(ax[0],text_msg,fontsize=10)
    plot_textbox(ax[1],text_msg1,fontsize=10)
    fig.tight_layout()
    return fig


def Gamma_p_timeflow_2Q_plot(data_Q1,data_Q2,
                             Delta_f01_Q1,min_Delta_f01_Q1,
                             Delta_f01_Q2,min_Delta_f01_Q2,
                             Delta_f01_thres,filter_on,
                             Realtime,total_exp_time):
    samples_Q1= np.array(data_Q1)
    samples_Q2= np.array(data_Q2)
    if Realtime is True:
        flow=np.linspace(0,total_exp_time,len(samples_Q1))
        xlabel='Time flow'+' [hours]'
    else: 
        flow= np.linspace(1,len(samples_Q1),len(samples_Q1))
        xlabel='Time flow'+' [times]'
    if filter_on:
        flow_1=[]    
        samples_Q1_1,samples_Q2_1=[],[]
        # filter by delta_f01
        for i in range(0,len(samples_Q1)-1):
            if np.abs(Delta_f01_Q1[i+1]-Delta_f01_Q1[i])<Delta_f01_thres and Delta_f01_Q1[i]>=min_Delta_f01_Q1 and np.abs(Delta_f01_Q2[i+1]-Delta_f01_Q2[i])<Delta_f01_thres and Delta_f01_Q2[i]>=min_Delta_f01_Q2:
                samples_Q1_1.append(samples_Q1[i])
                samples_Q2_1.append(samples_Q2[i])
                flow_1.append(flow[i])
            else:
                pass     
    else:
        flow_1, samples_Q1_1,samples_Q2_1= flow,samples_Q1,samples_Q2

    fig, ax = plt.subplots(nrows=2, ncols = 1, dpi = 200, sharex=True)
    fig.subplots_adjust(hspace=0)
    ax[0].plot(flow_1, np.array(samples_Q1_1)/1000, color = "b",marker='o', alpha = 0.5,lw=1)
    ax[1].plot(flow_1, np.array(samples_Q2_1)/1000, color = "r",marker='o', alpha = 0.5,lw=1)
    ax[1].set_xlabel(xlabel, fontsize = 12)
    ax[0].set_ylabel(r"$\Gamma_{p}\ $[kHz](Q1)" , fontsize = 12)
    ax[1].set_ylabel(r"$\Gamma_{p}\ $[kHz](Q2)" , fontsize = 12)
    value= np.mean(Cross_correlation(samples_Q1_1,samples_Q2_1))
    text_msg = "Cross correlation=%.2f"%(value)
    plot_textbox(ax[0],text_msg,fontsize=10)
    fig.tight_layout() 
    return fig


def Three_states_single_shot_Rawdata_plot(data:dict): # Input processed data
    Ig_data,Qg_data = 1000*np.array(data['g'][0]), 1000*np.array(data['g'][1]) 
    Ie_data,Qe_data = 1000*np.array(data['e'][0]), 1000*np.array(data['e'][1])
    If_data,Qf_data = 1000*np.array(data['f'][0]), 1000*np.array(data['f'][1])
    I,Q= np.hstack([Ig_data,Ie_data,If_data]), np.hstack([Qg_data,Qe_data,Qf_data])
    
    fig, ax = plt.subplots(ncols = 3,figsize =(9,3),dpi =200)
    ax[0].scatter(Ig_data,Qg_data, color="blue"  , alpha=0.5, s=1)   
    ax[1].scatter(Ie_data,Qe_data, color="red"   , alpha=0.5, s=1) 
    ax[2].scatter(If_data,Qf_data, color="orange", alpha=0.5, s=1)      
    ax[0].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[0].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[1].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[1].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[2].set_xlabel(r"$I\ $[mV]",size ='15')
    ax[2].set_ylabel(r"$Q\ $[mV]",size ='15')
    ax[0].set_title('Prepare |0>')
    ax[1].set_title('Prepare |1>')
    ax[2].set_title('Prepare |2>')
    ax[0].axes.set_aspect('equal')
    ax[1].axes.set_aspect('equal')
    ax[2].axes.set_aspect('equal')
    fig.tight_layout()
    return fig

def Three_states_single_shot_plot(data:dict): # Input analysis_result
    Cent = data["Cent"]
    g_data = data['IQdata'][0]
    e_data = data['IQdata'][1]
    f_data = data['IQdata'][2]
    Ig_data, Qg_data = g_data[0], g_data[1]
    Ie_data, Qe_data = e_data[0], e_data[1]
    If_data, Qf_data = f_data[0], f_data[1]

    Cg        = data['fit_pack'][0]
    Ce        = data['fit_pack'][1]
    Cf        = data['fit_pack'][2]
    sigma_fit = data['fit_pack'][6]

    r = np.array([np.linalg.norm(Cg - Cent), np.linalg.norm(Ce - Cent), np.linalg.norm(Cf- Cent)])
    r = np.sum(r)/len(r)

    fig, ax = plt.subplots(ncols = 3, figsize = (9, 3), dpi = 200)
    ax[0].scatter(Ig_data * 1000, Qg_data * 1000, color = "black", s = 1, alpha = 0.3)
    ax[0].set_xlabel("I [mV]", fontsize = 12)
    ax[0].set_ylabel("Q [mV]", fontsize = 12)
    ax[0].set_title("Prepared |0>", fontsize = 12)
    ax[0].add_patch(Ellipse(xy=[Cg[0]*1000,Cg[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Ce[0]*1000,Ce[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cf[0]*1000,Cf[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cent[0]*1000,Cent[1]*1000],width=r*2000,height=r*2000,fill=False, facecolor= None, edgecolor="#824D92FF", linewidth=1.5, linestyle='--',angle=0))
    ax[0].axes.set_aspect('equal')

    ax[1].scatter(Ie_data * 1000, Qe_data * 1000, color = "black", s = 1, alpha = 0.3)
    ax[1].set_xlabel("I [mV]", fontsize = 12)
    ax[1].set_ylabel("Q [mV]", fontsize = 12)
    ax[1].set_title("Prepared |1>", fontsize = 12)
    ax[1].add_patch(Ellipse(xy=[Cg[0]*1000,Cg[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Ce[0]*1000,Ce[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cf[0]*1000,Cf[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cent[0]*1000,Cent[1]*1000],width=r*2000,height=r*2000,fill=False, facecolor= None, edgecolor="#824D92FF", linewidth=1.5, linestyle='--',angle=0))
    ax[1].axes.set_aspect('equal')
    fig.tight_layout()

    ax[2].scatter(If_data * 1000, Qf_data * 1000, color = "black", s = 1, alpha = 0.3)
    ax[2].set_xlabel("I [mV]", fontsize = 12)
    ax[2].set_ylabel("Q [mV]", fontsize = 12)
    ax[2].set_title("Prepared |2>", fontsize = 12)
    ax[2].add_patch(Ellipse(xy=[Cg[0]*1000,Cg[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Ce[0]*1000,Ce[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Cf[0]*1000,Cf[1]*1000],width=sigma_fit*7000,height=sigma_fit*7000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Cent[0]*1000,Cent[1]*1000],width=r*2000,height=r*2000,fill=False, facecolor= None, edgecolor="#824D92FF", linewidth=1.5, linestyle='--',angle=0))
    ax[2].axes.set_aspect('equal')
    fig.tight_layout()
    return fig

def Three_states_threshold_plot(data:dict,R_integration:float,Reset_time:float): # Input analysis_result
    Cent = data["Cent"]
    g_data = data['IQdata'][0]
    e_data = data['IQdata'][1]
    f_data = data['IQdata'][2]

    flow=np.linspace(0,(R_integration+Reset_time)*(len(g_data[0])-1),len(g_data[0]))

    thresholds = data["thresholds"]
    Cg        = data['fit_pack'][0]
    Ce        = data['fit_pack'][1]
    Cf        = data['fit_pack'][2]
    sigma_fit = data['fit_pack'][6]

    Cg_phase = np.angle((Cg-Cent)[0] + 1j*(Cg-Cent)[1])
    Ce_phase = np.angle((Ce-Cent)[0] + 1j*(Ce-Cent)[1])
    Cf_phase = np.angle((Cf-Cent)[0] + 1j*(Cf-Cent)[1])
    cloud_phase = np.array([Cg_phase, Ce_phase, Cf_phase])
    cloud_phase_deg = np.rad2deg(cloud_phase)

    Ig_data, Qg_data = g_data[0], g_data[1]
    Ie_data, Qe_data = e_data[0], e_data[1]
    If_data, Qf_data = f_data[0], f_data[1]
    Ig_data = Ig_data - Cent[0]
    Qg_data = Qg_data - Cent[1]
    Ie_data = Ie_data - Cent[0]
    Qe_data = Qe_data - Cent[1]
    If_data = If_data - Cent[0]
    Qf_data = Qf_data - Cent[1]

    data_phase_g = np.angle(Ig_data + 1j*Qg_data)
    data_phase_e = np.angle(Ie_data + 1j*Qe_data)
    data_phase_f = np.angle(If_data + 1j*Qf_data)

    data_abs_g = np.abs(Ig_data + 1j*Qg_data)
    data_abs_e = np.abs(Ie_data + 1j*Qe_data)
    data_abs_f = np.abs(If_data + 1j*Qf_data)
    
    I_data, Q_data = np.hstack([Ig_data,Ie_data,If_data]), np.hstack([Qg_data,Qe_data,Qf_data])
    I = np.linspace(np.min(I_data), np.max(I_data), 1000)
    Q = np.linspace(np.min(Q_data), np.max(Q_data), 1000)

    def sortingByThreshold(labels, phase, r, ratio_list):
        n = len(labels)
        for i in range(n):
            th1 = thresholds[i]
            th2 = thresholds[(i + 1) % n]
            if th1 < th2:
                if phase < th1 or phase > th2:
                    ratio_list[i] += 1
                    return (i, [r, phase])
            else:
                if th1 > phase > th2:
                    ratio_list[i] += 1
                    return (i, [r, phase])
                
    labels = [0, 1, 2]
    ratio_g = np.array([0, 0, 0])
    ratio_e = np.array([0, 0, 0])
    ratio_f = np.array([0, 0, 0])

    sorted_gdata = [[], [], [], []]
    sorted_edata = [[], [], [], []]
    sorted_fdata = [[], [], [], []]
    sequence_glabel = np.array([])
    sequence_gphase = np.array([])
    sequence_elabel = np.array([])
    sequence_ephase = np.array([])
    sequence_flabel = np.array([])
    sequence_fphase = np.array([])

    for i in range(len(data_phase_g)):
        label, pos = sortingByThreshold(labels, data_phase_g[i], data_abs_g[i], ratio_g)
        sorted_gdata[label].append(pos)
        sequence_gphase = np.append(sequence_gphase, pos[1])
        if label <= 1:
            sequence_glabel = np.append(sequence_glabel, label)
            
    for i in range(len(data_phase_e)):
        label, pos = sortingByThreshold(labels, data_phase_e[i], data_abs_e[i], ratio_e)
        sorted_edata[label].append(pos)
        sequence_ephase = np.append(sequence_ephase, pos[1])
        if label <= 1:
            sequence_elabel = np.append(sequence_elabel, label)

    for i in range(len(data_phase_f)):
        label, pos = sortingByThreshold(labels, data_phase_f[i], data_abs_f[i], ratio_f)
        sorted_fdata[label].append(pos)
        sequence_fphase = np.append(sequence_fphase, pos[1])
        if label <= 1:
            sequence_flabel = np.append(sequence_flabel, label)
    
    sequence_gphase_deg = np.rad2deg(sequence_gphase)
    sequence_ephase_deg = np.rad2deg(sequence_ephase)
    sequence_fphase_deg = np.rad2deg(sequence_fphase)
    #* ========= for the plot of threshold ============
    def pol2cart(data):
        r = data[0]
        theta = data[1]
        x = r * np.cos(theta) + Cent[0]
        y = r * np.sin(theta) + Cent[1]
        return(x, y)
    
    sorted_Igg, sorted_Qgg = pol2cart(np.column_stack(sorted_gdata[0]))
    sorted_Ige, sorted_Qge = pol2cart(np.column_stack(sorted_gdata[1]))
    sorted_Igf, sorted_Qgf = pol2cart(np.column_stack(sorted_gdata[2]))

    sorted_Ieg, sorted_Qeg = pol2cart(np.column_stack(sorted_edata[0]))
    sorted_Iee, sorted_Qee = pol2cart(np.column_stack(sorted_edata[1]))
    sorted_Ief, sorted_Qef = pol2cart(np.column_stack(sorted_edata[2]))

    sorted_Ifg, sorted_Qfg = pol2cart(np.column_stack(sorted_fdata[0]))
    sorted_Ife, sorted_Qfe = pol2cart(np.column_stack(sorted_fdata[1]))
    sorted_Iff, sorted_Qff = pol2cart(np.column_stack(sorted_fdata[2]))

    pos_fg = pol2cart([1, thresholds[0]])
    pos_ge = pol2cart([1, thresholds[1]])
    pos_ef = pol2cart([1, thresholds[2]])
    X_fg = np.linspace(Cent[0], pos_fg[0] + (pos_fg[0] - Cent[0])*5, 1000)
    Y_fg = np.linspace(Cent[1], pos_fg[1] + (pos_fg[1] - Cent[1])*5, 1000)
    X_ge = np.linspace(Cent[0], pos_ge[0] + (pos_ge[0] - Cent[0])*5, 1000)
    Y_ge = np.linspace(Cent[1], pos_ge[1] + (pos_ge[1] - Cent[1])*5, 1000)
    X_ef = np.linspace(Cent[0], pos_ef[0] + (pos_ef[0] - Cent[0])*5, 1000)
    Y_ef = np.linspace(Cent[1], pos_ef[1] + (pos_ef[1] - Cent[1])*5, 1000)

    #* =========== for the state time flow plot ===============
    g_times = np.linspace(1, len(sequence_gphase), len(sequence_gphase))
    e_times = np.linspace(1, len(sequence_ephase), len(sequence_ephase))

    #* =========== threshold plot ==============
    fig1, ax = plt.subplots(ncols = 3, figsize = (9, 3), dpi = 200)
    ax[0].scatter(sorted_Igg * 1000, sorted_Qgg * 1000, s = 1, color = "#4242BDFF", label = "|0>")
    ax[0].scatter(sorted_Ige * 1000, sorted_Qge * 1000, s = 1, color = "#CC5454FF", label = "|1>")
    ax[0].scatter(sorted_Igf * 1000, sorted_Qgf * 1000, s = 1, color = "#4AB64AFF", label = "|2>")
    ax[0].add_patch(Ellipse(xy=[Cg[0] * 1000,Cg[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Ce[0] * 1000,Ce[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].add_patch(Ellipse(xy=[Cf[0] * 1000,Cf[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[0].plot(X_fg * 1000, Y_fg * 1000, color = "black")
    ax[0].plot(X_ge * 1000, Y_ge * 1000, color = "black")
    ax[0].plot(X_ef * 1000, Y_ef * 1000, color = "black")
    ax[0].set_xlabel("I [mV]", fontsize = 12)
    ax[0].set_ylabel("Q [mV]", fontsize = 12)
    ax[0].set_title("Prepared |0>", fontsize = 12)
    ax[0].set_xlim((np.min(g_data[0]) - sigma_fit) * 1000, (np.max(g_data[0]) + sigma_fit) * 1000)
    ax[0].set_ylim((np.min(g_data[1]) - sigma_fit) * 1000, (np.max(g_data[1]) + sigma_fit) * 1000)
    ax[0].legend(fontsize = 7, edgecolor = "black", markerscale = 3, framealpha = 0.1, ncols = 2, columnspacing = 0.5)
    ax[0].axes.set_aspect('equal')

    ax[1].scatter(sorted_Ieg * 1000, sorted_Qeg * 1000, s = 1, color = "#4242BDFF", label = "|0>")
    ax[1].scatter(sorted_Iee * 1000, sorted_Qee * 1000, s = 1, color = "#CC5454FF", label = "|1>")
    ax[1].scatter(sorted_Ief * 1000, sorted_Qef * 1000, s = 1, color = "#4AB64AFF", label = "|2>")
    ax[1].add_patch(Ellipse(xy=[Cg[0] * 1000,Cg[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Ce[0] * 1000,Ce[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].add_patch(Ellipse(xy=[Cf[0] * 1000,Cf[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[1].plot(X_fg * 1000, Y_fg * 1000, color = "black")
    ax[1].plot(X_ge * 1000, Y_ge * 1000, color = "black")
    ax[1].plot(X_ef * 1000, Y_ef * 1000, color = "black")
    ax[1].set_xlabel("I [mV]", fontsize = 12)
    ax[1].set_ylabel("Q [mV]", fontsize = 12)
    ax[1].set_title("Prepared |1>", fontsize = 12)
    ax[1].set_xlim((np.min(g_data[0]) - sigma_fit) * 1000, (np.max(g_data[0]) + sigma_fit) * 1000)
    ax[1].set_ylim((np.min(g_data[1]) - sigma_fit) * 1000, (np.max(g_data[1]) + sigma_fit) * 1000)    
    ax[1].legend(fontsize = 7, edgecolor = "black", markerscale = 3, framealpha = 0.1, ncols = 2, columnspacing = 0.5)
    ax[1].axes.set_aspect('equal')

    ax[2].scatter(sorted_Ifg * 1000, sorted_Qfg * 1000, s = 1, color = "#4242BDFF", label = "|0>")
    ax[2].scatter(sorted_Ife * 1000, sorted_Qfe * 1000, s = 1, color = "#CC5454FF", label = "|1>")
    ax[2].scatter(sorted_Iff * 1000, sorted_Qff * 1000, s = 1, color = "#4AB64AFF", label = "|2>")
    ax[2].add_patch(Ellipse(xy=[Cg[0] * 1000,Cg[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4242BDFF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Ce[0] * 1000,Ce[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#CC5454FF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].add_patch(Ellipse(xy=[Cf[0] * 1000,Cf[1] * 1000],width=sigma_fit*7 * 1000,height=sigma_fit*7 * 1000,fill=False, facecolor= None, edgecolor="#4AB64AFF", linewidth=1.5, linestyle='-',angle=0))
    ax[2].plot(X_fg * 1000, Y_fg * 1000, color = "black")
    ax[2].plot(X_ge * 1000, Y_ge * 1000, color = "black")
    ax[2].plot(X_ef * 1000, Y_ef * 1000, color = "black")
    ax[2].set_xlabel("I [mV]", fontsize = 12)
    ax[2].set_ylabel("Q [mV]", fontsize = 12)
    ax[2].set_title("Prepared |2>", fontsize = 12)
    ax[2].set_xlim((np.min(g_data[0]) - sigma_fit) * 1000, (np.max(g_data[0]) + sigma_fit) * 1000)
    ax[2].set_ylim((np.min(g_data[1]) - sigma_fit) * 1000, (np.max(g_data[1]) + sigma_fit) * 1000)    
    ax[2].legend(fontsize = 7, edgecolor = "black", markerscale = 3, framealpha = 0.1, ncols = 2, columnspacing = 0.5)
    ax[2].axes.set_aspect('equal')
    fig1.tight_layout()

    #* =========== phase time flow ==============
    fig2, ax2 = plt.subplots(ncols = 3, figsize = (13.5, 3), dpi = 200)
    ax2[0].scatter(flow, sequence_gphase_deg, s=1, alpha = 0.3, color="black")
    ax2[0].hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", linewidth = 2, label = "|0>")
    ax2[0].hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", linewidth = 2, label = "|1>")
    ax2[0].hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", linewidth = 2, label = "|2>")
    ax2[0].legend(ncols = 1, bbox_to_anchor = (1, 1))
    ax2[0].set_xlabel('Time flow [s]', fontsize = 12)
    ax2[0].set_ylabel("Phase [deg]", fontsize = 12)
    ax2[0].set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
    ax2[0].set_title("Prepared |g>", fontsize = 12)

    ax2[1].scatter(flow, sequence_ephase_deg, s=1, alpha = 0.3, color="black")
    ax2[1].hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", linewidth = 2, label = "|0>")
    ax2[1].hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", linewidth = 2, label = "|1>")
    ax2[1].hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", linewidth = 2, label = "|2>")
    ax2[1].legend(ncols = 1, bbox_to_anchor = (1, 1))
    ax2[1].set_xlabel('Time flow [s]', fontsize = 12)
    ax2[1].set_ylabel("Phase [deg]", fontsize = 12)
    ax2[1].set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
    ax2[1].set_title("Prepared |e>", fontsize = 12)
    fig2.tight_layout()

    ax2[2].scatter(flow, sequence_fphase_deg, s=1, alpha = 0.3, color="black")
    ax2[2].hlines(cloud_phase_deg[0], np.min(flow), np.max(flow), color = "#4242BDFF", linestyle = "-", linewidth = 2, label = "|0>")
    ax2[2].hlines(cloud_phase_deg[1], np.min(flow), np.max(flow), color = "#CC5454FF", linestyle = "-", linewidth = 2, label = "|1>")
    ax2[2].hlines(cloud_phase_deg[2], np.min(flow), np.max(flow), color = "#4AB64AFF", linestyle = "-", linewidth = 2, label = "|2>")
    ax2[2].legend(ncols = 1, bbox_to_anchor = (1, 1))
    ax2[2].set_xlabel('Time flow [s]', fontsize = 12)
    ax2[2].set_ylabel("Phase [deg]", fontsize = 12)
    ax2[2].set_ylim((np.min(cloud_phase_deg) - 30), (np.max(cloud_phase_deg) + 30))
    ax2[2].set_title("Prepared |f>", fontsize = 12)
    fig2.tight_layout()

    return (fig1, fig2)

def Readout_shaping_plot(delay,results,Photon_convert,Ac_info,log_scale):
    f01=[]
    def Starkshift_n(n_,fa):
        return fa-(2*n_*+1)*['X_eff']
    for i in range(len(results)):
        f01.append(results[i].attrs['f01_fit'])
    n= (Ac_info['f01_bare']-np.array(f01) - Ac_info['X_eff'])/(2*Ac_info['X_eff'])
    if Photon_convert:
        ylabel= 'Resonator \n photons'
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
        ax.plot(delay, n, color = "k",marker='o', alpha = 0.5,lw=1)
        ax.set_xlabel(r"$t_{XY}\ [\mu$s]", fontsize = 15)
        ax.set_ylabel(ylabel, fontsize = 15)
        ax.set_yscale("log")
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(ncols = 1, figsize = (6, 4), dpi = 200)
        ax.plot(delay, n, color = "k",marker='o', alpha = 0.5,lw=1)
        ax.set_xlabel(r"$t_{XY}\ [\mu$s]", fontsize = 15)
        ax.set_ylabel(ylabel, fontsize = 15)
        fig.tight_layout()

    return fig



def Cryoscope_plot(data:dict,times_idx:float,P_rescale:bool,Dis:float,Z_waveform:any,n_exp:float):
    if P_rescale is not True:
        Nor_f=1/1000
        y1_label= r"$a<X>\ $[mV]"
        y2_label= r"$a<Y>\ $[mV]"
    elif P_rescale is True:
        Nor_f= Dis
        y1_label= r"$<X> $"
        y2_label= r"$<Y> $"
    else: raise KeyError ('P_rescale is not bool') 
    
    X,Y,Z_Du= data['Y_pi_2']['data'][times_idx]/Nor_f, data['X_pi_2']['data'][times_idx]/Nor_f,1e9*data['X_pi_2']['first_samples']
    fig, ax = plt.subplots(nrows=1,figsize =(4.8,4),dpi =200)
    cmap = plt.get_cmap('jet')
    if P_rescale:
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        X=-1*(2*X-1)
        Y=2*Y-1
    ax.set_xlabel(y1_label,size ='12')
    ax.set_ylabel(y2_label,size ='12')
    m=ax.scatter(X,Y,c=Z_Du, vmin=Z_Du[0], vmax=Z_Du[-1], s=20, cmap=cmap,edgecolors=None, label=r"$data$")
    cbar=fig.colorbar(m,ax=ax)
    xscale=[min(Z_Du),max(Z_Du)]
    cbar.set_ticks(xscale)
    #cbar.ax.set_title(x_label)
    ax.axes.set_aspect('equal')
    fig.tight_layout()
    
    fig1, ax = plt.subplots(nrows=2,figsize =(6,4),dpi =200)
    if P_rescale:
        ax[0].set_ylim(-1,1)
        ax[1].set_ylim(-1,1)
    ax[0].set_ylabel(y1_label,size ='12')
    ax[1].set_xlabel(r"$\tau\ $[ns]",size ='12')
    ax[1].set_ylabel(y2_label,size ='12')
    ax[0].plot(Z_Du,X,'-o', color="b", alpha=0.5, lw=1)
    ax[1].plot(Z_Du,Y,'-o', color="r", alpha=0.5, lw=1)
    fig1.tight_layout()

    phase = np.unwrap(np.angle(X+Y*1j))
    phase = phase - phase[-1]
    dt = np.mean(np.diff(data['X_pi_2']['first_samples']))
    detuning = np.diff(phase)/(2*np.pi*dt) 
    detuning_sm= savgol_filter(phase/2/np.pi, 13, 3, deriv=1, delta=dt)#butter_lowpass_filter(savgol_filter(phase/2/np.pi, 13, 3, deriv=1, delta=dt),450*1e6,1/dt,20)
    Normalize_detuning = detuning/np.average(detuning[-int(len(Z_Du) / 2) :])
    Normalize_detuning_sm = detuning_sm / np.average(detuning_sm[-int(len(Z_Du) / 2) :])
    fig2, ax = plt.subplots(nrows=1,figsize =(6,4),dpi =200)
    ax.set_ylabel(y1_label,size ='12')
    ax.set_xlabel(r"$\tau\ $[ns]",size ='12')
    ax.set_ylabel("detuning [MHz]",size ='12')
    ax.plot(Z_Du[1:],detuning*1e-6, color="b", alpha=0.8, lw=2)
    ax.plot(Z_Du,detuning_sm*1e-6, color="r", alpha=0.8, lw=2)
    fig2.tight_layout()

    fig3, ax = plt.subplots(nrows=1,figsize =(6,4),dpi =200)
    ax.set_ylabel(y1_label,size ='12')
    ax.set_xlabel(r"$\tau\ $[ns]",size ='12')
    ax.set_ylabel("Normalized response",size ='12')
    ax.plot(Z_Du[1:],Normalize_detuning, color="b", alpha=0.8, lw=2)
    ax.plot(Z_Du,Normalize_detuning_sm, color="r", alpha=0.8, lw=2)
    fig3.tight_layout()

    result,fig4 = Pulse_distortion_analysis(t=Z_Du*1e-9,         
                                            detuning_raw= Normalize_detuning,        
                                            detuning=Normalize_detuning_sm,        
                                            V_input=Z_waveform, 
                                            n_exp=n_exp, )

    return fig,fig1,fig2,fig3,fig4,result



def Cryoscope_correction_plot(data_without_correction:dict,data_with_correction:dict):
    t1= data_without_correction['t'][1:] #due to np.diff
    t2= data_with_correction['t'][1:]
    y1= data_without_correction['Normalized_detuning']
    y2= data_with_correction['Normalized_detuning']
    mask1= y1 >= 0.2
    mask2= y2 >= 0.2

    fig1, ax = plt.subplots(nrows=1,figsize =(6,4),dpi =200)
    ax.set_ylabel("Normalized\n response",size ='12')
    ax.set_xlabel(r"$\tau\ $[ns]",size ='12')
    ax.plot(t1[mask1]*1e9,y1[mask1],'-', color="b", alpha=0.5, lw=1)
    ax.plot(t2[mask2]*1e9,y2[mask2],'-', color="r", alpha=0.5, lw=1)
    fig1.tight_layout()


def XY_Z_timing_plot(Z_delay,data,threshold,P_rescale,Dis):
    if P_rescale is not True:
        Nor_f=1/1000
        y_label= 'Contrast'+' [mV]'
    elif P_rescale is True:
        Nor_f= Dis
        y_label= r"$P_{1}\ $"
    else: raise KeyError ('P_rescale is not bool') 
    
    fig, ax = plt.subplots(nrows=1,figsize =(6,4),dpi =200)
    if P_rescale:
        ax.set_ylim(0,1)
    ax.set_xlabel("Z delay time [ns]",size ='12')
    ax.set_ylabel(y_label,size ='12')
    x=Z_delay*1e9
    y= data/Nor_f
    mask = y < threshold
    avg_x = np.mean(x[mask])
    ax.axvline(x=avg_x,color='k',linestyle='dashed',label="Pulse aligned time"+'\n'+r"$=%.0f $"%(avg_x)+"ns", alpha=0.8,lw=1.5)
    ax.plot(x,y,'-o', color="b", alpha=0.5, lw=1)
    ax.legend(loc="lower right",fontsize = 12)
    fig.tight_layout()
    return fig, avg_x/1e9

# %%
