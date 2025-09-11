
#%%
# from Hardware_setting import*
# from SQ_RB_seq import *

import numpy as np
import xarray as xr
from scipy import special

from scipy.integrate import quad
from lmfit import Model,Parameter


import matplotlib.pyplot as plt
from qcat.parser.qm_reader import load_xarray_h5
from qcat.NCU.Fit_library import gauss_func, gauss2d_func_model, gauss2d_func, bigauss2d_func_model, bigauss2d_func
from qcat.utilities.data_processing import find_nearest, rot, IQ_data_dis
from qcat.common_calculator.convertor import PetoT
from qcat.common_calculator.analytical import Relax_cal


import numpy as np
def Single_shot_ref_fit_analysis(data:tuple):

    I, Q= np.array(data[0]), np.array(data[1])
    bins=101
    I_=np.linspace(I.min(),I.max(),bins)  
    Q_=np.linspace(Q.min(),Q.max(),bins)
    hist, xedges, yedges = np.histogram2d(I,Q, bins=(bins,bins), density=True)
    I_guess, Q_guess, sig_guess=np.mean(I),np.mean(Q),(np.std(I)+np.std(Q))/2
    c_I=Parameter(name='c_I', value= I_guess, min=I_guess-5*sig_guess, max=I_guess+5*sig_guess) 
    c_Q=Parameter(name='c_Q', value= Q_guess, min=Q_guess-5*sig_guess, max=Q_guess+5*sig_guess)
    sigma=Parameter(name='sigma', value= sig_guess, min=0.1*sig_guess, max=3*sig_guess)
    X,Y= np.meshgrid(I_,Q_)
    result= gauss2d_func_model.fit(hist.transpose(),I=X,Q=Y,c_I=c_I,c_Q=c_Q,sigma=sigma,A=Parameter(name='A',value=np.max(hist), min=0.01*np.max(hist)) )     
    c_I_fit=result.best_values['c_I']
    c_Q_fit=result.best_values['c_Q']
    sigma_fit=result.best_values['sigma']
    A_fit=result.best_values['A']
    fit_pack=[c_I_fit,c_Q_fit,sigma_fit]
    fitting= gauss2d_func(X,Y,c_I_fit,c_Q_fit,sigma_fit,A_fit)
    #print ('Total prob. =',np.sum(hist)*((max(xedges)-min(xedges))/bins*(max(yedges)-min(yedges))/bins))
    
    return dict(data=[I,Q],data_hist=hist,coords=[X,Y],fitting=fitting,fit_pack=fit_pack)

def Outlier(I,Q,Ig,Qg,Ie,Qe,sigma): # only use in rotated IQ signals on I axis
    I,Q= np.array(I),np.array(Q)
    Dis_g= IQ_data_dis(I,Q,Ig,Qg)
    Dis_e= IQ_data_dis(I,Q,Ie,Qe)
    Outlier_event_I=[]
    Outlier_event_Q=[]
    Inner_event_I=[]
    Inner_event_Q=[]
    for i in range(len(I)): 
        if (I[i]<Ig and Dis_g[i]>3*sigma) or (I[i]>Ie and Dis_e[i]>3*sigma) or np.abs(Q[i])>3*sigma:
            Outlier_event_I.append(I[i])
            Outlier_event_Q.append(Q[i])
        else:
            Inner_event_I.append(I[i])
            Inner_event_Q.append(Q[i])
    Outlier_event_I,Outlier_event_Q,Inner_event_I,Inner_event_Q= np.array(Outlier_event_I),np.array(Outlier_event_Q),np.array(Inner_event_I),np.array(Inner_event_Q)
    return dict(Outlier_event=[Outlier_event_I,Outlier_event_Q],Inner_event=[Inner_event_I,Inner_event_Q],Outlier_P=len(Outlier_event_I)/len(I))

def Qubit_state_single_shot_fit_analysis(data:dict, T1:float=None,tau:float=None,f01:float=None,fixed_sigma_on=False,fixed_sigma_value=None):
    Ig_data,Qg_data,Ie_data,Qe_data= np.array(data['g'][0]), np.array(data['g'][1]) ,np.array(data['e'][0]) , np.array(data['e'][1])
    if len(Ig_data)<=2000:
        bins=21
    elif 2000<len(Ig_data)<10000:
        bins=101
    elif 10000<=len(Ig_data)<=20000:
        bins=151
    else:
        bins=201
    g_predict= Single_shot_ref_fit_analysis(data['g'])['fit_pack']
    Ig_guess, Qg_guess, sig_guess= g_predict[0],g_predict[1],g_predict[2]
    I_mixdata, Q_mixdata= np.hstack([Ig_data,Ie_data]), np.hstack([Qg_data,Qe_data])
    I_=np.linspace(I_mixdata.min(),I_mixdata.max(),bins)  
    Q_=np.linspace(Q_mixdata.min(),Q_mixdata.max(),bins)
    X,Y= np.meshgrid(I_,Q_)
    hist_Ie, edges_Ie= np.histogram(Ie_data, bins=bins, density=True)
    hist_Qe, edges_Qe= np.histogram(Qe_data, bins=bins, density=True)
    hist, xedges, yedges = np.histogram2d(I_mixdata, Q_mixdata, bins=(bins,bins), density=True)
    Ie_guess_idx, Qe_guess_idx=find_nearest(hist_Ie,np.max(hist_Ie)),find_nearest(hist_Qe,np.max(hist_Qe))
    Ie_guess, Qe_guess= edges_Ie[Ie_guess_idx],edges_Qe[Qe_guess_idx]
    #Parameter ini-guess
    cg_I=Parameter(name='cg_I', value= Ig_guess, min=Ig_guess-1*sig_guess, max=Ig_guess+1*sig_guess) 
    cg_Q=Parameter(name='cg_Q', value= Qg_guess, min=Qg_guess-1*sig_guess, max=Qg_guess+1*sig_guess)
    ce_I=Parameter(name='ce_I', value= Ie_guess, min=Ie_guess-5*sig_guess, max=Ie_guess+5*sig_guess) 
    ce_Q=Parameter(name='ce_Q', value= Qe_guess, min=Qe_guess-5*sig_guess, max=Qe_guess+5*sig_guess)
    
    Ag=Parameter(name='Ag',value=np.max(hist), min=0.1*np.max(hist)) 
    Ae=Parameter(name='Ae',value=np.max(hist), min=0.1*np.max(hist))
    if fixed_sigma_on:
        sigma=Parameter(name='sigma', value= fixed_sigma_value,vary=False)
    else:
        sigma=Parameter(name='sigma', value= sig_guess, min=0.01*sig_guess, max=3*sig_guess)
    # mixed data fit
    result= bigauss2d_func_model.fit(hist.transpose(),I=X,Q=Y,cg_I=cg_I,cg_Q=cg_Q,sigma=sigma,Ag=Ag,ce_I=ce_I,ce_Q=ce_Q,Ae=Ae)
    cg_I_fit=result.best_values['cg_I']
    cg_Q_fit=result.best_values['cg_Q']
    ce_I_fit=result.best_values['ce_I']
    ce_Q_fit=result.best_values['ce_Q']
    sigma_fit=result.best_values['sigma']
    # displace + rotate
    angle= np.angle(ce_I_fit-cg_I_fit+(ce_Q_fit-cg_Q_fit)*1j)
    rot_e_center= rot(ce_I_fit-cg_I_fit,ce_Q_fit-cg_Q_fit,angle)
    rot_g_IQ= rot(Ig_data-cg_I_fit,Qg_data-cg_Q_fit,angle)
    rot_e_IQ= rot(Ie_data-cg_I_fit,Qe_data-cg_Q_fit,angle)
    Ig_data_new, Qg_data_new, Ie_data_new, Qe_data_new= rot_g_IQ[0],rot_g_IQ[1],rot_e_IQ[0],rot_e_IQ[1]
    # along single quadrature fit
    R= 20*sigma_fit #range_factor
    xmin, xmax = np.minimum(0,rot_e_center[0])-R,np.maximum(0,rot_e_center[0])+R
    ymin, ymax = np.minimum(0,rot_e_center[1])-R,np.maximum(0,rot_e_center[1])+R
    hist_r_g, xedges, yedges = np.histogram2d(Ig_data_new, Qg_data_new, bins=(bins,bins),range=[[xmin, xmax], [ymin, ymax]], density=True)
    hist_r_e, xedges, yedges = np.histogram2d(Ie_data_new, Qe_data_new, bins=(bins,bins),range=[[xmin, xmax], [ymin, ymax]], density=True)
    New_axe_g_hist= np.sum(hist_r_g.transpose(), axis=0)
    New_axe_e_hist= np.sum(hist_r_e.transpose(), axis=0)
    I_ro = np.linspace(xmin, xmax,bins)
    I_fit= np.linspace(xmin, xmax,bins*5)
    def Reduced_bimodal_func(x,Ag,Ae):
        return gauss_func(x,0,sigma_fit,Ag)+gauss_func(x,rot_e_center[0],sigma_fit,Ae)
    bimodal_func_model = Model(Reduced_bimodal_func)
    result_g= bimodal_func_model.fit(New_axe_g_hist,x=I_ro,Ag=Parameter(name='Ag',value=np.max(New_axe_g_hist), min=0.5*np.max(New_axe_g_hist)) ,Ae=Parameter(name='Ae',value=0.05*np.max(New_axe_g_hist), min=0.001*np.max(New_axe_g_hist)))
    result_e= bimodal_func_model.fit(New_axe_e_hist,x=I_ro,Ag=Parameter(name='Ag',value=0.5*np.max(New_axe_e_hist), min=0.001*np.max(New_axe_e_hist)),Ae=Parameter(name='Ae',value=np.max(New_axe_e_hist), min=0.5*np.max(New_axe_e_hist)))
    Agg_fit=result_g.best_values['Ag']
    Aeg_fit=result_g.best_values['Ae']
    Age_fit=result_e.best_values['Ag']
    Aee_fit=result_e.best_values['Ae']
    
    inter_point = rot_e_center[0]/2
    def G_gg(I):
        return Agg_fit*np.exp(-I**2/(2*sigma_fit**2))
    def G_eg(I):
        return Aeg_fit*np.exp(-(I-rot_e_center[0])**2/(2*sigma_fit**2))
    def G_ge(I):
        return Age_fit*np.exp(-I**2/(2*sigma_fit**2))
    def G_ee(I):
        return Aee_fit*np.exp(-(I-rot_e_center[0])**2/(2*sigma_fit**2))

    Ggg= quad(G_gg,rot_e_center[0]-R,rot_e_center[0]+R)
    Geg= quad(G_eg,rot_e_center[0]-R,rot_e_center[0]+R)
    Gge= quad(G_ge,rot_e_center[0]-R,rot_e_center[0]+R)
    Gee= quad(G_ee,rot_e_center[0]-R,rot_e_center[0]+R)
    overlap_gg= quad(G_gg,inter_point,rot_e_center[0]+R)
    transi_eg= quad(G_eg,inter_point,rot_e_center[0]+R)
    transi_ge= quad(G_ge,rot_e_center[0]-R,inter_point)
    overlap_ee= quad(G_ee,rot_e_center[0]-R,inter_point)
    overlap= (overlap_gg[0])/(Ggg[0])
    Peg=  (Geg[0])/(Geg[0]+Ggg[0])
    Pge= (Gge[0])/(Gee[0]+Gge[0])
    Thermal= (Geg[0])/(Geg[0]+Ggg[0])
    Relax= (Gge[0])/(Gee[0]+Gge[0])-Thermal




    if f01 is None or tau is None or T1 is None:
        Wa = None
        Pre_decay = None
        T = None
    else:
        Wa = 2 * np.pi * f01
        T = PetoT(Thermal, Wa)
        Relax_predict= Relax_cal(0,tau,T1)
        Pre_decay= Relax-Relax_predict

    D = rot_e_center[0]
    SNR = D / sigma_fit
    overlap_predict = (1 / 2) * (1 - special.erf(np.sqrt(SNR ** 2 / 8)))
    M = np.array([[1 - Peg, Peg], [Pge, 1 - Pge]])
    F_s = 1 - overlap
    F_g = 1 - Peg
    F_e = 1 - Pge
    F = 1 - (1 / 2) * (Peg + Pge)

    Outlier_g_info = Outlier(Ig_data_new, Qg_data_new, 0, 0, rot_e_center[0], rot_e_center[1], sigma_fit)
    Outlier_e_info = Outlier(Ie_data_new, Qe_data_new, 0, 0, rot_e_center[0], rot_e_center[1], sigma_fit)
    fit_pack = [rot_e_center, Agg_fit, Aeg_fit, Age_fit, Aee_fit, sigma_fit, New_axe_g_hist, New_axe_e_hist]
    error_pack = dict(Ig=cg_I_fit, Qg=cg_Q_fit, Ie=ce_I_fit, Qe=ce_Q_fit,
                     D=D, sigma=sigma_fit, SNR=SNR, overlap=overlap, overlap_predict=overlap_predict,
                     Pgg=1-Peg, Peg=Peg, Pge=Pge, Pee=1-Pge, Thermal=Thermal, eff_T_mK=T,
                     Relax=Relax, Relax_predict=Relax_predict, Pre_decay=Pre_decay,
                     F_s=F_s, F_g=F_g, F_e=F_e, F=F,
                     pre_g_outlier=Outlier_g_info['Outlier_P'], pre_e_outlier=Outlier_e_info['Outlier_P'])

    return dict(rot_IQdata=[rot_g_IQ, rot_e_IQ], I_ro=I_ro, I_fit=I_fit, fit_pack=fit_pack, error_pack=error_pack, M=M,
                Ig=cg_I_fit, Qg=cg_Q_fit, Ie=ce_I_fit, Qe=ce_Q_fit, Outlier_g_info=Outlier_g_info, Outlier_e_info=Outlier_e_info)

plt.show()
if __name__ == '__main__':
    import xarray as xr
    from qcat.analysis.state_distinguish.visualization import Qubit_state_single_shot_1Q
    # Load the dataset
    file_path = r"d:\github\ASQMDriver\data\MIST\2025-09-08\#68_07_iq_blobs_210029\ds_raw.h5"
    ds = load_xarray_h5(file_path)



    # Print coordinate names
    print("Coordinate names:", list(ds.coords))

    # --- Select one qubit and extract Ig, Ie, Qg, Qe ---
    # This assumes the dataset has a 'qubit' axis and variables for Ig, Ie, Qg, Qe
    # Adjust variable names and axis selection as needed for your dataset
    qubit_axis = None
    for name in ds.dims:
        if 'qubit' in name.lower():
            qubit_axis = name
            break

    if qubit_axis is not None:
        n_qubits = ds.dims[qubit_axis]
        def get_var(varname, idx):
            for name in ds.data_vars:
                if name.lower() == varname.lower():
                    arr = ds[name]
                    if qubit_axis in arr.dims:
                        arr = arr.isel({qubit_axis: idx})
                    return arr.values.flatten()
            return None
        for qubit_idx in range(n_qubits):
            Ig = get_var('Ig', qubit_idx)
            Ie = get_var('Ie', qubit_idx)
            Qg = get_var('Qg', qubit_idx)
            Qe = get_var('Qe', qubit_idx)
            raw_data = {'g': [Ig, Qg], 'e': [Ie, Qe]}
            if all(v is not None for v in [Ig, Ie, Qg, Qe]):
                print(f"Extracted Ig, Ie, Qg, Qe for qubit {qubit_idx}.")
                result = Qubit_state_single_shot_fit_analysis(raw_data, T1=100, tau=10, f01=3.5)
                Qubit_state_single_shot_1Q(raw_data, result, None, False, None, None)
            else:
                print(f"Could not find all required variables: Ig, Ie, Qg, Qe for qubit {qubit_idx}.")
    else:
        print("No qubit axis found. Available dimensions:", list(ds.dims))
    plt.show()