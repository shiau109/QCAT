from resonator_tools.circuit import notch_port
from .electronic_delay import *
import scipy.io
import pandas as pd
from scipy.optimize import curve_fit 
import numpy as np
from typing import Tuple

def fit_resonator_batch( freq:np.ndarray, zdata:np.ndarray, power:np.ndarray=None, delay=None, Qc=None, alpha=None, amp_norm=None )->Tuple[pd.DataFrame,list,list]:

    # Fit part
    fitParas = []
    fitCurves = []
    powerdep = False
    if type(power) != type(None):
        powerdep = True
        print("power dependnet zdata")
    if zdata.ndim == 1:
        zdata = np.array([zdata])

    fitParas = []
    fitCurves_norm = []
    zdatas_norm = []   

    for xi in range(zdata.shape[0]):
        print(f"fitting power {power[xi]}")
        zdata_single = zdata[xi]
        fit_results, zdata_norm, fit_curve_norm = fit_resonator(freq, zdata_single, input_power=power[xi], delay=delay, Qc=Qc, alpha=alpha, amp_norm=amp_norm)

        fitParas.append(fit_results)
        zdatas_norm.append(zdata_norm)
        fitCurves_norm.append(fit_curve_norm)

    df_fitParas = pd.DataFrame(fitParas)
    
    return df_fitParas, zdatas_norm, fitCurves_norm

def fit_resonator( freq, zdata, input_power=None, delay=None, Qc=None, alpha=None, amp_norm=None):

    FitResonator = notch_port()  
    delay_auto, amp_norm_auto, alpha_auto, fr_auto, Ql_auto, A2, frcal =\
            FitResonator.do_calibration(freq,zdata,fixed_delay=delay)
    if amp_norm == None:
        amp_norm = amp_norm_auto
    if delay == None:
        # print("auto fit electrical delay")
        delay = delay_auto
    if alpha == None:
        alpha = alpha_auto


    fr = fr_auto
    Ql = Ql_auto
    zdata_norm = FitResonator.do_normalization(freq,zdata,delay,amp_norm,alpha,A2,frcal)
    FitResonator.fitresults = FitResonator.circlefit(freq,zdata_norm,fr,Ql)
    fit_results = FitResonator.fitresults

    fit_results["A"] = amp_norm
    fit_results["alpha"] = alpha
    fit_results["delay"] = delay
    
    if input_power != None:
        fit_results["photons"] = FitResonator.get_photons_in_resonator(input_power)
    
    if Qc != None:
        fit_results["Qi_dia_corr_fqc"] = 1./(1./fit_results["Ql"]-1./Qc)

    # fit_curve = FitResonator._S21_notch(
    # freq,fr=fit_results["fr"],
    # Ql=fit_results["Ql"],
    # Qc=fit_results["absQc"],
    # phi=fit_results["phi0"],
    # a=amp_norm,alpha=alpha,delay=delay)
        
    fit_curve_norm = FitResonator._S21_notch(
    freq,fr=fit_results["fr"],
    Ql=fit_results["Ql"],
    Qc=fit_results["absQc"],
    phi=fit_results["phi0"]
    )
    # print(zdata_norm.shape, fit_curve_norm.shape)
    return fit_results, zdata_norm, fit_curve_norm

def get_fixed_paras( fitParas:pd.DataFrame ):
    # Refined fitting
    chi = fitParas["chi_square"].to_numpy()
    weights = 1/chi**2
    min_chi_idx = chi.argmin()
    # print(fitParas["alpha"].to_numpy())
    # print(np.unwrap( fitParas["alpha"].to_numpy(), period=np.pi))

    # delay_refined = np.average(fitParas["delay"].to_numpy(), weights=weights)
    # amp_refined = np.average(fitParas["A"].to_numpy(), weights=weights)
    # Qc_refined = np.average(fitParas["Qc_dia_corr"].to_numpy(), weights=weights)
    # alpha_refined = np.average(fitParas["alpha"].to_numpy(), weights=weights)

    # fixed_alpha = np.average(np.unwrap( fitParas["alpha"].to_numpy(), period=np.pi), weights=weights)

    delay_refined = fitParas["delay"].to_numpy()[min_chi_idx]  
    amp_refined = fitParas["A"].to_numpy()[min_chi_idx]  
    Qc_refined = fitParas["Qc_dia_corr"].to_numpy()[min_chi_idx]  
    alpha_refined = fitParas["alpha"].to_numpy()[min_chi_idx]  

    return delay_refined, amp_refined, Qc_refined, alpha_refined

def find_row( file_name, colname, value ):

    df = pd.read_csv( file_name )
    searchedArr = df[[colname]].values
    idx = (np.abs(searchedArr - value)).argmin()
    #print( searchedArr[idx] )
    return df.iloc[[idx]]

def tan_loss(x,a,c,nc):
    return (c+a/(1+(x/nc))**0.5)

def fit_tanloss( n, loss, loss_err ):
    upper_bound = [1,1,1e4]
    lower_bound = [0,0,0]
    
    min_loss = np.amin(loss)
    max_loss = np.amax(loss)

    p0=[max_loss-min_loss,min_loss,0.1]
    try:
        popt, pcov = curve_fit(tan_loss, n, loss,sigma=loss_err**2, p0=p0, bounds=(lower_bound,upper_bound))
        p_sigma = np.sqrt(np.diag(pcov))
    except:
        popt = [0,0,0]
        p_sigma = [0,0,0]
    results_dict = {
        "A_TLS": [popt[0]],
        "const": [popt[1]],
        "nc": [popt[2]],
        "A_TLS_err": [p_sigma[0]],
        "const_err": [p_sigma[1]],
        "nc_err": [p_sigma[2]],
    }
    results = pd.DataFrame(results_dict)

    return results
    #paras[cav]={"fr (GHz)":float(int(cav[1:])/10000),"TLS":popt[0],"Const.":popt[1],"Nc":popt[2]}
