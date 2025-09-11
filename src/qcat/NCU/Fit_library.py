#%%
# from Hardware_setting import*
# from SQ_RB_seq import *
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
import numpy as np
import xarray as xr
from matplotlib import ticker
from scipy import special,optimize
# from resonator_tools import circuit
from statistics import median
from scipy.integrate import quad,dblquad
from scipy.stats import norm
from scipy.signal import  butter,sosfiltfilt,find_peaks,welch,savgol_filter,lfilter
from scipy.optimize import fsolve,root_scalar
from lmfit import Model,Parameter,minimize, report_fit,Parameters
# from quantify_scheduler.enums import BinMode
# from quantify_scheduler.backends.graph_compilation import SerialCompiler
# from hmmlearn import hmm
import math


#%% IQ displacement

def IQ_data_dis(I_data:np.ndarray,Q_data:np.ndarray,ref_I:float,ref_Q:float):
    Dis= np.sqrt((I_data-ref_I)**2+(Q_data-ref_Q)**2)
    return Dis   

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


def D_two_states(IQ1,IQ2,P_rescale):
    if P_rescale==True:
        Dis= np.sqrt((IQ1[0]-IQ2[0])**2+(IQ1[1]-IQ2[1])**2)
        if Dis==0:
            raise ValueError('D is zero, rescale population will diverge')
        else:
            pass
        return Dis  
    else:
        pass
 
def get_line_eq(x0, y0, x1, y1):
    m= -1/((y0-y1)/(x0-x1))
    return (x0+x1)/2,(y0+y1)/2, m
def line_equ(x,slope,x_cross,y_cross):
    return y_cross+slope*(x-x_cross)

def align_to_4ns(x):
    step = 4e-9  
    return math.ceil(x / step) * step

#%% fit function


def fft_oscillation_guess(data: np.ndarray, t: np.ndarray):
    amp = np.fft.fft(data)[: len(data) // 2] #use positive frequency
    freq = np.fft.fftfreq(len(data), t[1] - t[0])[: len(amp)]
    amp[0] = 0  # Remove DC part 
    power = np.abs(amp)
    f_guess = abs(freq[power == max(power)][0])
    phase_guess = 2 * np.pi - (2 * np.pi * t[data == max(data)] * f_guess)[0]
    return f_guess, phase_guess


def fft_beat_oscillation_guess(data: np.ndarray, t: np.ndarray):
    amp = np.fft.fft(data)[: len(data) // 2] #use positive frequency
    freq = np.fft.fftfreq(len(data), t[1] - t[0])[: len(amp)]
    amp[0] = 0  # Remove DC part 
    power = np.abs(amp)
    f1_guess = abs(freq[power == max(power)][0])
    f2_guess = abs(freq[power == sorted(power)[-2]][0])
    phase1_guess = 2 * np.pi - (2 * np.pi * t[data == max(data)] * f1_guess)[0]
    phase2_guess = 2 * np.pi - (2 * np.pi * t[data == max(data)] * f2_guess)[0]
    return f1_guess, f2_guess,phase1_guess,phase2_guess

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def rot(I,Q,angle):
    sin=np.sin(angle)
    cos=np.cos(angle)
    return I*cos+Q*sin, -I*sin+Q*cos
def Line_func(x,m,A):
    return m*x+A
def Detuning_ng_func(Vg, A, C_g, phi, f01_bar):
    return A * np.cos(2 * np.pi * C_g * Vg - phi) + f01_bar
def Photon_field_decay_func(t,kappa,w,A,B,phi,t_): 
    i=complex(0,1)
    t1=(t-t_)
    return np.abs(A*np.exp((-kappa/2)*t1)*(np.cos(w*t1)+i*np.sin(w*t1+phi))+B)
def Power_photon_field_decay_func(t,kappa,A,B,t_): 
    t1=(t-t_)
    return np.abs(A*np.exp(-kappa*t1)+B)
def Rabi_func(x,A,f,offset):
    return A*np.cos(2*np.pi*f*x+np.pi)+offset
def T1_func(D,A,T1,offset):
    return A*np.exp(-D/T1)+offset
def Ramsey_func(D,A,T2,f,phase,offset):
    return A*np.exp(-D/T2)*np.cos(2*np.pi*f*D+phase)+offset
def Ramsey_two_decay_beating_func(D,A1,A2,T2_1,T2_2,f1,f2,phase1,phase2,offset):
    return A1*np.exp(-D/T2_1)*np.cos(2*np.pi*f1*D+phase1)+A2*np.exp(-D/T2_2)*np.cos(2*np.pi*f2*D+phase2)+offset
def Loren_func(x,x0,gamma,A,base):
    return (A/np.pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))+base
def Loren_noise_func(f,fc,A,base):
    return 4*A**2/fc/(1+(2*np.pi*f/fc)**2)+base
def gauss_func(x,c,sigma,A):
    return A*np.exp(-(x-c)**2/2/sigma**2)    
def gauss2d_func(I,Q,c_I,c_Q,sigma,A):
    return A*np.exp(-(I-c_I)**2/2/sigma**2)*np.exp(-(Q-c_Q)**2/2/sigma**2)
def bigauss2d_func(I,Q,cg_I,cg_Q,sigma,Ag,ce_I,ce_Q,Ae):
    return gauss2d_func(I,Q,cg_I,cg_Q,sigma,Ag)+gauss2d_func(I,Q,ce_I,ce_Q,sigma,Ae)
def Relax_cal(inte_i,tf,T1):
    tau= tf - inte_i
    Relax= 1+(T1/tau)*(np.exp(-(inte_i+tau)/T1)-np.exp(-inte_i/T1))
    return Relax
def S21_notch(f,fr,Ql,Qc,phi,a,alpha,delay):
    i=complex(0,1) 
    return a*np.exp(complex(0,alpha))*np.exp(-2*i*np.pi*f*delay)*(1-Ql/Qc*np.exp(i*phi)/(1.+2*i*Ql*(f-fr)/fr))	
def Transmon(phi_ex,m,Ejmax,Ec,phi_offset,d):
    phase=m*(phi_ex-phi_offset)
    k=np.sqrt(np.cos(phase)**2+d**2*np.sin(phase)**2) # if squid is asymmetric
    return np.sqrt(8*Ejmax*k*Ec)-Ec
def Cavity_flux(phi_ex,m,phi_offset,f_bare,c):
    k=np.cos(m*phi_ex+phi_offset) # if squid is symmetric
    return f_bare+c*k
def Ej_transmon(phi_ex,m,Ejmax,phi_offset,d):
    phase=m*(phi_ex-phi_offset)
    k=np.sqrt(np.cos(phase)**2+d**2*np.sin(phase)**2)
    return Ejmax*k
def Ramsey_f(f,central,m,n):
    return m*np.abs((f-central))+n
def SQ_RB(N,a,p,b):
    return a*(p**N)+b
def f_eff_bare_func(amp,a,b,f_bare):
    return a*1/np.sqrt(1+amp**2*b)+f_bare
def f_r_state_func(amp,a,b,f_bare):
    return a*1/np.sqrt(1+amp**2*b)+f_bare
def Rabi_f(f,central,m,n):
    return m*((f-central)**2+n**2)**(1/2)
        
Line_model = Model(Line_func)
Photon_field_model = Model(Photon_field_decay_func)
Power_photon_field_model= Model(Power_photon_field_decay_func)
Rabi_model = Model(Rabi_func)
Rabi_f_model = Model(Rabi_f)
T1_func_model = Model(T1_func)
Ramsey_func_model = Model(Ramsey_func)
Ramsey_two_decay_beating_func_model = Model(Ramsey_two_decay_beating_func)
Loren_func_model = Model(Loren_func)
Loren_noise_model = Model(Loren_noise_func) 
Cavity_flux_model = Model(Cavity_flux)
Ramsey_f_model = Model(Ramsey_f)
SQ_RB_model = Model(SQ_RB)
f_eff_bare_model = Model(f_eff_bare_func)
f_r_state_func_model = Model(f_r_state_func)
gauss2d_func_model = Model(gauss2d_func, independent_vars=['I', 'Q'])
bigauss2d_func_model = Model(bigauss2d_func, independent_vars=['I', 'Q'])


def T1_fit_analysis(data:np.ndarray,freeDu:np.ndarray,T1_guess:float=10*1e-6):
    offset_guess= data[-1]
    T1= Parameter(name='T1', value= T1_guess, min=1*1e-6, max=500*1e-6) 
    result = T1_func_model.fit(data,D=freeDu,A=np.max(data)-offset_guess,T1=T1,offset=offset_guess)
    A_fit= result.best_values['A']
    T1_fit= result.best_values['T1']
    offset_fit= result.best_values['offset']
    para_fit= np.linspace(freeDu.min(),freeDu.max(),50*len(data))
    fitting= T1_func(para_fit,A_fit,T1_fit,offset_fit)
    return xr.Dataset(data_vars=dict(data=(['freeDu'],data),fitting=(['para_fit'],fitting)),coords=dict(freeDu=(['freeDu'],freeDu),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="T1",T1_fit=T1_fit))

def T2_fit_analysis(data:np.ndarray,freeDu:np.ndarray,T2_guess:float=20*1e-6):
    f_guess,phase_guess= fft_oscillation_guess(data,freeDu)
    T2=Parameter(name='T2', value= T2_guess, min=0.05*T2_guess, max=10*T2_guess) 
    up_lim_f= 10*1e6
    f_guess_=Parameter(name='f', value=f_guess , min=0, max=up_lim_f)
    result = Ramsey_func_model.fit(data,D=freeDu,A=abs(max(data)-min(data))/2,T2=T2,f=f_guess_,phase=phase_guess, offset=np.mean(data))
    A_fit= result.best_values['A']
    f_fit= result.best_values['f']
    phase_fit= result.best_values['phase']
    T2_fit= result.best_values['T2']
    offset_fit= result.best_values['offset']
    para_fit= np.linspace(freeDu.min(),freeDu.max(),50*len(data))
    fitting= Ramsey_func(para_fit,A_fit,T2_fit,f_fit,phase_fit,offset_fit)
    return xr.Dataset(data_vars=dict(data=(['freeDu'],data),fitting=(['para_fit'],fitting)),coords=dict(freeDu=(['freeDu'],freeDu),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="T2",T2_fit=T2_fit,f=f_fit))

def Charge_parity_switch_Ramsey_fit_analysis(data:np.ndarray,freeDu:np.ndarray,T2_guess:float=20*1e-6):
    f1_guess, f2_guess,phase1_guess,phase2_guess= fft_beat_oscillation_guess(data,freeDu)
    T2_1=Parameter(name='T2_1', value= T2_guess, min=0.05*T2_guess, max=10*T2_guess) 
    T2_2=Parameter(name='T2_1', value= T2_guess, min=0.05*T2_guess, max=10*T2_guess) 
    up_lim_f= 5*1e6
    f1_guess_=Parameter(name='f1', value=f1_guess , min=f1_guess*0.1, max=up_lim_f)
    f2_guess_=Parameter(name='f2', value=f2_guess , min=f2_guess*0.1, max=up_lim_f)
    result = Ramsey_two_decay_beating_func_model.fit(data,D=freeDu,A1=abs(max(data)-min(data))/2,A2=abs(max(data)-min(data))/2,T2_1=T2_1,T2_2=T2_2,f1=f1_guess_,f2=f2_guess_,phase1=phase1_guess,phase2=phase2_guess, offset=np.mean(data))
    A1_fit= result.best_values['A1']
    f1_fit= result.best_values['f1']
    phase1_fit= result.best_values['phase1']
    A2_fit= result.best_values['A2']
    f2_fit= result.best_values['f2']
    phase2_fit= result.best_values['phase2']
    T2_1_fit= result.best_values['T2_1']
    T2_2_fit= result.best_values['T2_2']
    offset_fit= result.best_values['offset']
    if f1_fit<f2_fit:
        f1_fit_=f1_fit
        f2_fit_=f2_fit
        T2_1_fit_=T2_1_fit
        T2_2_fit_=T2_2_fit
    else:
        f1_fit_=f2_fit
        f2_fit_=f1_fit
        T2_1_fit_=T2_2_fit
        T2_2_fit_=T2_1_fit
    para_fit= np.linspace(freeDu.min(),freeDu.max(),50*len(data))
    fitting= Ramsey_two_decay_beating_func(para_fit,A1_fit,A2_fit,T2_1_fit,T2_2_fit,f1_fit,f2_fit,phase1_fit,phase2_fit,offset_fit)
    return xr.Dataset(data_vars=dict(data=(['freeDu'],data),fitting=(['para_fit'],fitting)),coords=dict(freeDu=(['freeDu'],freeDu),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="Ramsey_charge_parity",T2_1=T2_1_fit_,f1=f1_fit_,T2_2=T2_2_fit_,f2=f2_fit_))

def Qubit_spectrum_analysis(q:str,data_bank:list,fit_f01:list,Ec_given:float,fit_window_data_index:list,bias_to_Ej_Ec:list,fit_filter:bool,fit_error_threshold:float):
    bias, f01=[],[]
    bias_fit,f01_fit=[],[]
    start,end= fit_window_data_index[0],fit_window_data_index[1]
    for i in range(start,end+1): 
        bias_fit.append(data_bank[i].add_kwargs['original_Z_sweep_samples'])
        for j in range(len(data_bank[i].add_kwargs['original_Z_sweep_samples'])):
            f01_fit.append(fit_f01[i]['data_fit'][j].attrs['f01_fit'])
    for i in range(len(data_bank)):
        bias.append(data_bank[i].add_kwargs['original_Z_sweep_samples'])
        for j in range(len(bias[i])):
            f01.append(fit_f01[i]['data_fit'][j].attrs['f01_fit'])  
    def Qubit_spectrum(phi_ex,m,Ejmax,Ec,phi_offset,d):
        phase=m*(phi_ex-phi_offset)
        k=np.sqrt(np.cos(phase)**2+d**2*np.sin(phase)**2) # if squid is asymmetric
        return np.sqrt(8*Ejmax*k*Ec)-Ec
    model = Model(Qubit_spectrum)
    k_guess=1
    predict_Ejmax= (max(np.array(f01))+Ec_given)**2/8/Ec_given/k_guess
    Ejmax_guess=Parameter(name='Ejmax', value=predict_Ejmax , min=0.9*predict_Ejmax, max=1.1*predict_Ejmax) #guess is based on sweet spot condition
    Ec_guess=Parameter(name='Ec', value=Ec_given , min=0.9*Ec_given, max=1.1*Ec_given)
    f_guess, phase_guess=fft_oscillation_guess(data=np.array(f01_fit), t=np.hstack(bias_fit))
    m_guess=Parameter(name='m', value=2*np.pi*f_guess/2,min=0.5*2*np.pi*f_guess/2, max=1.5*2*np.pi*f_guess/2)
    phi_guess=Parameter(name='phi_offset', value=phase_guess)
    result = model.fit(np.array(f01_fit),phi_ex=np.hstack(bias_fit),m=m_guess, Ejmax=Ejmax_guess,Ec=Ec_guess,phi_offset=phi_guess,d=0.5)
    m_fit= result.best_values['m']
    Ejmax_fit= result.best_values['Ejmax']
    Ec_fit= result.best_values['Ec']
    phi_offset_fit= result.best_values['phi_offset']
    d_fit= result.best_values['d']
    fit_compare= Qubit_spectrum(np.hstack(bias),m_fit,Ejmax_fit,Ec_fit,phi_offset_fit,d_fit)
    print('Ec_fit=',Ec_fit)
    if fit_filter:
        f01_=[]
        bias_=[]
        for i in range(len(f01)):
            if np.abs(fit_compare[i]-f01[i])/fit_compare[i]<fit_error_threshold:
                f01_.append(f01[i])
                bias_.append(np.hstack(bias)[i])
        f01_=np.array(f01_) 
        bias_=np.array(bias_)       
    else:
         f01_=np.array(f01)
         bias_=np.hstack(bias)
         
    max_z_range= max(np.abs(np.hstack(bias)))
    para_fit= np.linspace(-(max_z_range+0.05),max_z_range+0.05,1000*len(f01))
    fitting= Qubit_spectrum(para_fit,m_fit,Ejmax_fit,Ec_fit,phi_offset_fit,d_fit)

    return xr.Dataset(data_vars=dict(data=(['Z'],f01_),fitting=(['para_fit'],fitting)),coords=dict(Z=(['Z'],bias_),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="Zgate_twotone",q=q,Ec=Ec_fit,Ejmax_fit=Ejmax_fit,phi_offset_fit=phi_offset_fit,m_fit=m_fit,bias_to_Ej_Ec=bias_to_Ej_Ec,d_fit=d_fit))

def QS_fit_analysis(data:np.ndarray,f:np.ndarray):
    data_smooth = savgol_filter(data, window_length=max(5, len(data)//50|1), polyorder=2)
    peaks, props = find_peaks(data_smooth, prominence=np.ptp(data_smooth)*0.05, width=3)
    k = np.argmax(props["prominences"])
    pk_idx = peaks[k]
    fmin = f.min()
    fmax = f.max()
    width_max = fmax-fmin
    delta_f = np.diff(f)  
    min_delta_f = delta_f[delta_f > 0].min()
    width_min = min_delta_f
    width_guess = np.sqrt(width_min*width_max) 
    A= np.pi * width_guess * (np.max(data)-np.mean(data))
    #f_guess=Parameter(name='f', value= f[np.argmax(data)], min=f[np.argmax(data)]-50*1e6, max=f[np.argmax(data)]+50*1e6) 
    f_guess=Parameter(name='f', value= f[pk_idx], min=f[pk_idx]-50*1e6, max=f[pk_idx]+50*1e6) 
    result = Loren_func_model.fit(data,x=f,x0= f_guess,gamma=width_guess,A=A,base=np.mean(data))
    f01_fit= result.best_values['x0']
    bw= result.best_values['gamma']
    A_fit= result.best_values['A']
    base_fit= result.best_values['base']
    para_fit= np.linspace(fmin,fmax,50*len(data))
    fitting= Loren_func(para_fit,f01_fit,bw,A_fit,base_fit)
    return xr.Dataset(data_vars=dict(data=(['f'],data),fitting=(['para_fit'],fitting)),coords=dict(f=(['f'],f),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="QS",f01_fit=f01_fit,bandwidth=bw))


def Rabi_fit_analysis(data:np.ndarray,samples:np.ndarray,Du:float):
    f_guess,phase_guess= fft_oscillation_guess(data,samples)
    result = Rabi_model.fit(data,x=samples,A=abs(max(data)-min(data))/2,f=f_guess, offset=np.mean(data))
    A_fit= result.best_values['A'] 
    f_fit= result.best_values['f']
    offset_fit= result.best_values['offset']
    pi_2= 1/(2*f_fit)
    para_fit= np.linspace(samples.min(),samples.max(),50*len(data))
    fitting= Rabi_func(para_fit,A_fit,f_fit,offset_fit)
    return xr.Dataset(data_vars=dict(data=(['samples'],data),fitting=(['para_fit'],fitting)),coords=dict(samples=(['samples'],samples),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="Rabi",pi_2=pi_2,pi_Du=Du,f_Rabi=f_fit))

def Swap_fit_analysis(data:np.ndarray,samples:np.ndarray,xlabel:str):
    f_guess,phase_guess= fft_oscillation_guess(data,samples)
    T2=Parameter(name='T2', value= max(samples), min=min(samples)) 
    up_lim_f= 2*f_guess
    f_guess_=Parameter(name='f', value=f_guess , min=0, max=up_lim_f)
    result = Ramsey_func_model.fit(data,D=samples,A=abs(max(data)-min(data))/2,T2=T2,f=f_guess_,phase=phase_guess, offset=np.mean(data))
    A_fit= result.best_values['A']
    f_fit= result.best_values['f']
    phase_fit= result.best_values['phase']
    T2_fit= result.best_values['T2']
    offset_fit= result.best_values['offset']
    para_fit= np.linspace(samples.min(),samples.max(),50*len(data))
    fitting= Ramsey_func(para_fit,A_fit,T2_fit,f_fit,phase_fit,offset_fit)
    return xr.Dataset(data_vars=dict(data=(['samples'],data),fitting=(['para_fit'],fitting)),coords=dict(samples=(['samples'],samples),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="Swap",T2_fit=T2_fit,f=f_fit,xlabel=xlabel))

def iSwap_fit_analysis(data:np.ndarray,freeDu:np.ndarray,T2_guess:float=20*1e-6):
    f_guess,phase_guess= fft_oscillation_guess(data,freeDu)
    T2=Parameter(name='T2', value= T2_guess, min=0.05*T2_guess, max=10*T2_guess) 
    up_lim_f= 10*1e6
    f_guess_=Parameter(name='f', value=f_guess , min=0, max=up_lim_f)
    result = Ramsey_func_model.fit(data,D=freeDu,A=abs(max(data)-min(data))/2,T2=T2,f=f_guess_,phase=phase_guess, offset=np.mean(data))
    A_fit= result.best_values['A']
    f_fit= result.best_values['f']
    phase_fit= result.best_values['phase']
    T2_fit= result.best_values['T2']
    offset_fit= result.best_values['offset']
    para_fit= np.linspace(freeDu.min(),freeDu.max(),50*len(data))
    fitting= Ramsey_func(para_fit,A_fit,T2_fit,f_fit,phase_fit,offset_fit)
    return xr.Dataset(data_vars=dict(data=(['freeDu'],data),fitting=(['para_fit'],fitting)),coords=dict(freeDu=(['freeDu'],freeDu),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="iSwap",T2_fit=T2_fit,f=f_fit))


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

def Thermal_population_single_shot_fit_analysis(data:list,IeQe_guess:list):

    I_mixdata, Q_mixdata= np.array(data[0]), np.array(data[1])
    bins=101
    g_predict= Single_shot_ref_fit_analysis(data)['fit_pack']
    Ig_guess, Qg_guess, sig_guess= g_predict[0],g_predict[1],g_predict[2]
    I_=np.linspace(I_mixdata.min(),I_mixdata.max(),bins)  
    Q_=np.linspace(Q_mixdata.min(),Q_mixdata.max(),bins)
    X,Y= np.meshgrid(I_,Q_)

    hist, xedges, yedges = np.histogram2d(I_mixdata, Q_mixdata, bins=(bins,bins), density=True)
    #Parameter ini-guess
    cg_I=Parameter(name='cg_I', value= Ig_guess, min=Ig_guess-2*sig_guess, max=Ig_guess+2*sig_guess) 
    cg_Q=Parameter(name='cg_Q', value= Qg_guess, min=Qg_guess-2*sig_guess, max=Qg_guess+2*sig_guess)
    ce_I=Parameter(name='ce_I', value= IeQe_guess[0], min=IeQe_guess[0]-2*sig_guess, max=IeQe_guess[0]+2*sig_guess) 
    ce_Q=Parameter(name='ce_Q', value= IeQe_guess[1], min=IeQe_guess[1]-2*sig_guess, max=IeQe_guess[1]+2*sig_guess)
    sigma=Parameter(name='sigma', value= sig_guess, min=0.1*sig_guess, max=3*sig_guess)
    Ag=Parameter(name='Ag',value=np.max(hist), min=0.5*np.max(hist)) 
    Ae=Parameter(name='Ae',value=0.01*np.max(hist), min=0.0001*np.max(hist)) 
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
    rot_g_IQ= rot(I_mixdata-cg_I_fit,Q_mixdata-cg_Q_fit,angle)
    Ig_data_new, Qg_data_new= rot_g_IQ[0],rot_g_IQ[1]
    # along single quadrature fit
    R= 20*sigma_fit #range_factor
    xmin, xmax = np.minimum(0,rot_e_center[0])-R,np.maximum(0,rot_e_center[0])+R
    ymin, ymax = np.minimum(0,rot_e_center[1])-R,np.maximum(0,rot_e_center[1])+R
    hist_r_g, xedges, yedges = np.histogram2d(Ig_data_new, Qg_data_new, bins=(bins,bins),range=[[xmin, xmax], [ymin, ymax]], density=True)
    New_axe_g_hist= np.sum(hist_r_g.transpose(), axis=0)
    I_ro = np.linspace(xmin, xmax,bins)
    I_fit= np.linspace(xmin, xmax,bins*5)
    def Reduced_bimodal_func(x,Ag,Ae):
        return gauss_func(x,0,sigma_fit,Ag)+gauss_func(x,rot_e_center[0],sigma_fit,Ae)
    bimodal_func_model = Model(Reduced_bimodal_func)
    result_g= bimodal_func_model.fit(New_axe_g_hist,x=I_ro,Ag=Parameter(name='Ag',value=np.max(New_axe_g_hist), min=0.5*np.max(New_axe_g_hist)) ,Ae=Parameter(name='Ae',value=0.01*np.max(New_axe_g_hist), min=0))
    Agg_fit=result_g.best_values['Ag']
    Aeg_fit=result_g.best_values['Ae']
    inter_point = rot_e_center[0]/2
    def G_gg(I):
        return Agg_fit*np.exp(-I**2/(2*sigma_fit**2))
    def G_eg(I):
        return Aeg_fit*np.exp(-(I-rot_e_center[0])**2/(2*sigma_fit**2))

    Ggg= quad(G_gg,rot_e_center[0]-R,rot_e_center[0]+R)
    Geg= quad(G_eg,rot_e_center[0]-R,rot_e_center[0]+R)

    overlap_gg= quad(G_gg,inter_point,rot_e_center[0]+R)
    #transi_eg= quad(G_eg,inter_point,100*rot_e_center[0])
    overlap= (overlap_gg[0])/(Ggg[0])
    Thermal= (Geg[0])/(Geg[0]+Ggg[0])
    D= rot_e_center[0]
    SNR= D/sigma_fit
    overlap_predict= (1/2)*(1-special.erf(np.sqrt(SNR**2/8)))
    fit_pack= [rot_e_center,Agg_fit,Aeg_fit,sigma_fit,New_axe_g_hist]
    error_pack= dict(D=D,sigma=sigma_fit,SNR=SNR,overlap=overlap,overlap_predict=overlap_predict,Thermal=Thermal)
    return dict(rot_IQdata=[rot_g_IQ],I_ro=I_ro,I_fit=I_fit,fit_pack=fit_pack,error_pack=error_pack,fitIeQe=[ce_I_fit,ce_Q_fit])

def Qubit_state_single_shot_fit_analysis(data:dict, T1:float,tau:float,f01:float,fixed_sigma_on=False,fixed_sigma_value=None):
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
    Relax_predict= Relax_cal(0,tau,T1)
    Pre_decay= Relax-Relax_predict
    hbar = 1.054571800*1e-34
    kB = 1.38e-23    
    Wa= 2*np.pi*f01
    
    def PetoT(Pe):
        Pe = np.clip(Pe,1e-10, 1-1e-10)  
        Pg = 1 - Pe
        try:
            T = (-hbar*Wa)/(kB*np.log(Pe/Pg)) * 1000  
        except Exception as e:
            T = np.nan
        return T

    T= PetoT(Thermal)
    D= rot_e_center[0]
    SNR= D/sigma_fit
    overlap_predict= (1/2)*(1-special.erf(np.sqrt(SNR**2/8)))
    M=np.array([[1-Peg,Peg],[Pge,1-Pge]])
    F_s= 1-overlap
    F_g= 1-Peg
    F_e= 1-Pge
    F=1-(1/2)*(Peg+Pge)
    
    Outlier_g_info= Outlier(Ig_data_new, Qg_data_new,0,0,rot_e_center[0],rot_e_center[1],sigma_fit)
    Outlier_e_info= Outlier(Ie_data_new, Qe_data_new,0,0,rot_e_center[0],rot_e_center[1],sigma_fit)
    fit_pack= [rot_e_center,Agg_fit,Aeg_fit,Age_fit,Aee_fit,sigma_fit,New_axe_g_hist,New_axe_e_hist]
    error_pack= dict(Ig=cg_I_fit,Qg=cg_Q_fit,Ie=ce_I_fit,Qe=ce_Q_fit,
                     D=D,sigma=sigma_fit,SNR=SNR,overlap=overlap,overlap_predict=overlap_predict,
                     Pgg=1-Peg,Peg=Peg,Pge=Pge,Pee=1-Pge,Thermal=Thermal,eff_T_mK=T,
                     Relax=Relax,Relax_predict=Relax_predict,Pre_decay=Pre_decay,
                     F_s=F_s,F_g=F_g,F_e=F_e,F=F,
                     pre_g_outlier=Outlier_g_info['Outlier_P'],pre_e_outlier=Outlier_e_info['Outlier_P'])

    return dict(rot_IQdata=[rot_g_IQ,rot_e_IQ],I_ro=I_ro,I_fit=I_fit,fit_pack=fit_pack,error_pack=error_pack,M=M,Ig=cg_I_fit,Qg=cg_Q_fit,Ie=ce_I_fit,Qe=ce_Q_fit,Outlier_g_info=Outlier_g_info,Outlier_e_info=Outlier_e_info)


def Get_threshold(gdata:tuple,edata:tuple,readout_Q:str):
    if readout_Q=='Q1':
        I_index,Q_index= 0,1
    elif readout_Q=='Q2':
        I_index,Q_index= 2,3
    else: raise KeyError ('readout_Q is incorrect for getting threshold')
        
    Ig_data,Qg_data,Ie_data,Qe_data= np.array(gdata[I_index]), np.array(gdata[Q_index]) ,np.array(edata[I_index]) , np.array(edata[Q_index])
    if len(Ig_data)<10000:
        bins=51
    elif 10000<len(Ig_data)<20000:
        bins=101
    else:
        bins=201
    g_predict= Single_shot_ref_fit_analysis(tuple([Ig_data,Qg_data]))['fit_pack']
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
    sigma=Parameter(name='sigma', value= sig_guess, min=0.01*sig_guess, max=3*sig_guess)
    Ag=Parameter(name='Ag',value=np.max(hist), min=0.1*np.max(hist)) 
    Ae=Parameter(name='Ae',value=np.max(hist), min=0.1*np.max(hist))  
    # mixed data fit
    result= bigauss2d_func_model.fit(hist.transpose(),I=X,Q=Y,cg_I=cg_I,cg_Q=cg_Q,sigma=sigma,Ag=Ag,ce_I=ce_I,ce_Q=ce_Q,Ae=Ae)
    cg_I_fit=result.best_values['cg_I']
    cg_Q_fit=result.best_values['cg_Q']
    ce_I_fit=result.best_values['ce_I']
    ce_Q_fit=result.best_values['ce_Q']
    sigma_fit=result.best_values['sigma']
    return dict(Ig=cg_I_fit,Qg=cg_Q_fit,Ie=ce_I_fit,Qe=ce_Q_fit,sig=sigma_fit)

def Fit_assignment(data:tuple,thres_info:dict,readout_Q:str,f01:list,T1:list,tau:list):
    if readout_Q=='Q1':
        I_index,Q_index= 0,1
        f01=f01[0]
        T1= T1[0]
        tau=tau[0]
    elif readout_Q=='Q2':
        I_index,Q_index= 2,3
        f01=f01[1]
        T1= T1[1]
        tau=tau[1]
    else: raise KeyError ('readout_Q is incorrect for getting threshold')
    
    I_data, Q_data= data[I_index],data[Q_index]
    cg_I_fit,cg_Q_fit,ce_I_fit,ce_Q_fit,sigma_fit= thres_info['Ig'],thres_info['Qg'],thres_info['Ie'],thres_info['Qe'],thres_info['sig']
    bins=101
    # displace + rotate
    angle= np.angle(ce_I_fit-cg_I_fit+(ce_Q_fit-cg_Q_fit)*1j)
    rot_e_center= rot(ce_I_fit-cg_I_fit,ce_Q_fit-cg_Q_fit,angle)
    rot_IQ= rot(I_data-cg_I_fit,Q_data-cg_Q_fit,angle)
    I_data_new, Q_data_new= rot_IQ[0],rot_IQ[1]
    # along single quadrature fit
    R= 10*sigma_fit #range_factor
    xmin, xmax = np.minimum(0,rot_e_center[0])-R,np.maximum(0,rot_e_center[0])+R
    ymin, ymax = np.minimum(0,rot_e_center[1])-R,np.maximum(0,rot_e_center[1])+R
    hist_r, xedges, yedges = np.histogram2d(I_data_new, Q_data_new, bins=(bins,bins),range=[[xmin, xmax], [ymin, ymax]], density=True)
    New_axe_hist= np.sum(hist_r.transpose(), axis=0)
    I_ro = np.linspace(xmin, xmax,bins)
    I_fit= np.linspace(xmin, xmax,bins*5)
    def Reduced_bimodal_func(x,Ag,Ae):
        return gauss_func(x,0,sigma_fit,Ag)+gauss_func(x,rot_e_center[0],sigma_fit,Ae)
    bimodal_func_model = Model(Reduced_bimodal_func)
    result_g= bimodal_func_model.fit(New_axe_hist,x=I_ro,Ag=Parameter(name='Ag',value=np.max(New_axe_hist), min=0) ,Ae=Parameter(name='Ae',value=np.max(New_axe_hist), min=0))
   
    Ag_fit=result_g.best_values['Ag']
    Ae_fit=result_g.best_values['Ae']
    
    inter_point = rot_e_center[0]/2
    def G_g(I):
        return Ag_fit*np.exp(-I**2/(2*sigma_fit**2))
    def G_e(I):
        return Ae_fit*np.exp(-(I-rot_e_center[0])**2/(2*sigma_fit**2))
    Gg= quad(G_g,-100*rot_e_center[0],100*rot_e_center[0])
    Ge= quad(G_e,-100*rot_e_center[0],100*rot_e_center[0])
    overlap_g= quad(G_g,inter_point,100*rot_e_center[0])
    overlap_e= quad(G_e,-100*rot_e_center[0],inter_point)
    overlap= ((overlap_g[0])/(Gg[0])+(overlap_e[0])/(Ge[0]))/2
    Pe_cal= (Ge[0])/(Gg[0]+Ge[0])
    Pg_cal= (Gg[0])/(Gg[0]+Ge[0])
    Pe=Pe_cal/(Pe_cal+Pg_cal)
    Pg=Pg_cal/(Pe_cal+Pg_cal)
    Thermal= (Ge[0])/(Gg[0]+Ge[0])
    Relax= (Gg[0])/(Gg[0]+Ge[0])# -Thermal is minused at Multiplex_assignment_matrix_analysis
    Relax_predict= Relax_cal(0,tau,T1)
    Pre_decay= Relax-Relax_predict # is revised at Multiplex_assignment_matrix_analysis
    hbar = 1.054571800*1e-34
    kB = 1.38e-23    
    Wa= 2*np.pi*f01
    
    def PetoT(Pe):
        Pg= 1-Pe
        T= (-hbar*Wa)/(kB*np.log(Pe/Pg))*1000
        return T 
    T= PetoT(Thermal)
    D= rot_e_center[0]
    SNR= D/sigma_fit
    overlap_predict= (1/2)*(1-special.erf(np.sqrt(SNR**2/8)))
    fit_pack= [rot_e_center,Ag_fit,Ae_fit,sigma_fit,New_axe_hist]
    return dict(rot_IQdata=[rot_IQ],I_ro=I_ro,I_fit=I_fit,fit_pack=fit_pack,Pg=Pg,Pe=Pe,overlap=overlap,overlap_predict=overlap_predict,Thermal=Thermal,eff_T_mK=T,Relax=Relax,Relax_predict=Relax_predict,Pre_decay=Pre_decay,D=D,sigma=sigma_fit,SNR=SNR)

def SQ_threshold_acquisition_statistic(data:tuple,SQ_thres:dict):
    cg_I_fit,cg_Q_fit,ce_I_fit,ce_Q_fit= SQ_thres['Ig'],SQ_thres['Qg'],SQ_thres['Ie'],SQ_thres['Qe']
    x_cross,y_cross, slope = get_line_eq(cg_I_fit,cg_Q_fit,ce_I_fit,ce_Q_fit)

    I_data, Q_data= data[0],data[1]
    Cl,Ch=0,0 
    for i in range(len(I_data)):
        if Q_data[i]-line_equ(I_data[i],slope,x_cross,y_cross) <= 0:
            Cl+=1 
        elif Q_data[i]-line_equ(I_data[i],slope,x_cross,y_cross) > 0:
            Ch+=1
        
    if cg_Q_fit-line_equ(cg_I_fit,slope,x_cross,y_cross) > 0: 
        Cg,Ce= Ch,Cl
    elif cg_Q_fit-line_equ(cg_I_fit,slope,x_cross,y_cross) < 0: 
        Cg,Ce= Cl,Ch

    total_C= Cg+Ce

    return dict(Pg=Cg/total_C,Pe=Ce/total_C)

def Multi_threshold_acquisition_statistic(data:any,Q1_thres:dict,Q2_thres:dict):
    cg_I1_fit,cg_Q1_fit,ce_I1_fit,ce_Q1_fit= Q1_thres['Ig'],Q1_thres['Qg'],Q1_thres['Ie'],Q1_thres['Qe']
    cg_I2_fit,cg_Q2_fit,ce_I2_fit,ce_Q2_fit= Q2_thres['Ig'],Q2_thres['Qg'],Q2_thres['Ie'],Q2_thres['Qe']
    x1_cross,y1_cross, slope1 = get_line_eq(cg_I1_fit,cg_Q1_fit,ce_I1_fit,ce_Q1_fit)
    x2_cross,y2_cross, slope2 = get_line_eq(cg_I2_fit,cg_Q2_fit,ce_I2_fit,ce_Q2_fit)
    
    I1_data, Q1_data,I2_data, Q2_data= data[0],data[1],data[2],data[3]
    Cll,Chh,Clh,Chl=0,0,0,0 
    for i in range(len(I1_data)):
        if Q1_data[i]-line_equ(I1_data[i],slope1,x1_cross,y1_cross) <= 0 and Q2_data[i]-line_equ(I2_data[i],slope2,x2_cross,y2_cross) <= 0:
            Cll+=1
        elif Q1_data[i]-line_equ(I1_data[i],slope1,x1_cross,y1_cross) > 0 and Q2_data[i]-line_equ(I2_data[i],slope2,x2_cross,y2_cross) <= 0:
            Chl+=1
        elif Q1_data[i]-line_equ(I1_data[i],slope1,x1_cross,y1_cross) <= 0 and Q2_data[i]-line_equ(I2_data[i],slope2,x2_cross,y2_cross) > 0:
            Clh+=1   
        elif Q1_data[i]-line_equ(I1_data[i],slope1,x1_cross,y1_cross) > 0 and Q2_data[i]-line_equ(I2_data[i],slope2,x2_cross,y2_cross) > 0:
            Chh+=1
        
    if cg_Q1_fit-line_equ(cg_I1_fit,slope1,x1_cross,y1_cross) > 0 and cg_Q2_fit-line_equ(cg_I2_fit,slope2,x2_cross,y2_cross) > 0: 
        Cgg,Cge,Ceg,Cee= Chh,Chl,Clh,Cll
    elif cg_Q1_fit-line_equ(cg_I1_fit,slope1,x1_cross,y1_cross) < 0 and cg_Q2_fit-line_equ(cg_I2_fit,slope2,x2_cross,y2_cross) > 0: 
        Cgg,Cge,Ceg,Cee= Clh,Cll,Chh,Chl
    elif cg_Q1_fit-line_equ(cg_I1_fit,slope1,x1_cross,y1_cross) > 0 and cg_Q2_fit-line_equ(cg_I2_fit,slope2,x2_cross,y2_cross) < 0:
        Cgg,Cge,Ceg,Cee= Chl,Chh,Cll,Clh
    elif cg_Q1_fit-line_equ(cg_I1_fit,slope1,x1_cross,y1_cross) < 0 and cg_Q2_fit-line_equ(cg_I2_fit,slope2,x2_cross,y2_cross) < 0:    
        Cgg,Cge,Ceg,Cee= Cll,Clh,Chl,Chh
    total_C= Cgg+Cge+Ceg+Cee
    
    return dict(Pgg=Cgg/total_C,Pge=Cge/total_C,Peg=Ceg/total_C,Pee=Cee/total_C)
  
def Multiplex_assignment_matrix_analysis(data:dict, T1:list,f01:list, tau:list):
    
    Q1= Get_threshold(gdata=data[str(['g', 'g'])],edata=data[str(['e', 'g'])],readout_Q='Q1')
    Q2= Get_threshold(gdata=data[str(['g', 'g'])],edata=data[str(['g', 'e'])],readout_Q='Q2')
    Q1_pre_00= Fit_assignment(data=data[str(['g', 'g'])],thres_info=Q1,readout_Q='Q1',f01=f01,T1=T1,tau=tau)
    Q2_pre_00= Fit_assignment(data=data[str(['g', 'g'])],thres_info=Q2,readout_Q='Q2',f01=f01,T1=T1,tau=tau)
    Q1_pre_p0= Fit_assignment(data=data[str(['e', 'g'])],thres_info=Q1,readout_Q='Q1',f01=f01,T1=T1,tau=tau)
    Q2_pre_p0= Fit_assignment(data=data[str(['e', 'g'])],thres_info=Q2,readout_Q='Q2',f01=f01,T1=T1,tau=tau)
    Q1_pre_0p= Fit_assignment(data=data[str(['g', 'e'])],thres_info=Q1,readout_Q='Q1',f01=f01,T1=T1,tau=tau)
    Q2_pre_0p= Fit_assignment(data=data[str(['g', 'e'])],thres_info=Q2,readout_Q='Q2',f01=f01,T1=T1,tau=tau)
    Q1_pre_pp= Fit_assignment(data=data[str(['e', 'e'])],thres_info=Q1,readout_Q='Q1',f01=f01,T1=T1,tau=tau)
    Q2_pre_pp= Fit_assignment(data=data[str(['e', 'e'])],thres_info=Q2,readout_Q='Q2',f01=f01,T1=T1,tau=tau)
    
    # _ _ => Q1 Q2
    # Assign. 00 01 10 11
    #Pre. 00=>                
    #Pre. 01=>                
    #Pre. 10=>   
    #Pre. 11=>
    Pre_00= Multi_threshold_acquisition_statistic(data[str(['g', 'g'])],Q1,Q2)
    Pre_01= Multi_threshold_acquisition_statistic(data[str(['g', 'e'])],Q1,Q2)
    Pre_10= Multi_threshold_acquisition_statistic(data[str(['e', 'g'])],Q1,Q2)
    Pre_11= Multi_threshold_acquisition_statistic(data[str(['e', 'e'])],Q1,Q2)
    
    P_0000= Pre_00['Pgg']
    P_0001= Pre_00['Pge']
    P_0010= Pre_00['Peg']
    P_0011= Pre_00['Pee']
    P_0100= Pre_01['Pgg']
    P_0101= Pre_01['Pge']
    P_0110= Pre_01['Peg']
    P_0111= Pre_01['Pee']
    P_1000= Pre_10['Pgg']
    P_1001= Pre_10['Pge']
    P_1010= Pre_10['Peg']
    P_1011= Pre_10['Pee']
    P_1100= Pre_11['Pgg']
    P_1101= Pre_11['Pge']
    P_1110= Pre_11['Peg']
    P_1111= Pre_11['Pee']
    
    
    assign_matrix=np.array([[P_0000,P_0001,P_0010,P_0011],          
                            [P_0100,P_0101,P_0110,P_0111],           
                            [P_1000,P_1001,P_1010,P_1011],         
                            [P_1100,P_1101,P_1110,P_1111]])    
    
    fit_results= dict(Q1_pre_00=Q1_pre_00,Q2_pre_00=Q2_pre_00,
                      Q1_pre_0p=Q1_pre_0p,Q2_pre_0p=Q2_pre_0p,
                      Q1_pre_p0=Q1_pre_p0,Q2_pre_p0=Q2_pre_p0,
                      Q1_pre_pp=Q1_pre_pp,Q2_pre_pp=Q2_pre_pp)
    error_pack_Q1=dict(D=Q1_pre_00['D'],sigma=Q1_pre_00['sigma'],SNR=Q1_pre_00['SNR'],overlap=Q1_pre_00['overlap'],overlap_predict=Q1_pre_00['overlap_predict'],Thermal=Q1_pre_00['Thermal'],eff_T_mK=Q1_pre_00['eff_T_mK'],Relax=Q1_pre_p0['Relax']-Q1_pre_00['Thermal'],Relax_predict=Q1_pre_p0['Relax_predict'],Pre_decay=Q1_pre_p0['Pre_decay']-Q1_pre_00['Thermal'])
    error_pack_Q2=dict(D=Q2_pre_00['D'],sigma=Q2_pre_00['sigma'],SNR=Q2_pre_00['SNR'],overlap=Q2_pre_00['overlap'],overlap_predict=Q2_pre_00['overlap_predict'],Thermal=Q2_pre_00['Thermal'],eff_T_mK=Q2_pre_00['eff_T_mK'],Relax=Q2_pre_0p['Relax']-Q2_pre_00['Thermal'],Relax_predict=Q2_pre_0p['Relax_predict'],Pre_decay=Q2_pre_0p['Pre_decay']-Q2_pre_00['Thermal'])

    return dict(fit_results=fit_results,M=assign_matrix,Threshold_info=dict(Q1=Q1,Q2=Q2),error_pack_Q1=error_pack_Q1,error_pack_Q2=error_pack_Q2)

def QND_fidelity_analysis(data:dict):
    Q1= Get_threshold(gdata=data['g'],edata=data['e'],readout_Q='Q1')
    Pre_0= Multi_threshold_acquisition_statistic(data['g'],Q1,Q1)
    Pre_1= Multi_threshold_acquisition_statistic(data['e'],Q1,Q1)
    P_0_00=Pre_0['Pgg']
    P_0_01=Pre_0['Pge']
    P_0_10=Pre_0['Peg']
    P_0_11=Pre_0['Pee']
    P_1_00=Pre_1['Pgg']
    P_1_01=Pre_1['Pge']
    P_1_10=Pre_1['Peg']
    P_1_11=Pre_1['Pee']
    
    QND_matrix_pre_0=np.array([[P_0_00,P_0_10],          
                                  [P_0_01,P_0_11]])
    
    QND_matrix_pre_1=np.array([[P_1_00,P_1_10],          
                                  [P_1_01,P_1_11]])
    
    
    return dict(M1=QND_matrix_pre_0,M2=QND_matrix_pre_1)

def Ramsey_spec_F_fit(Ramsey_f_data:list,fxy:np.ndarray,fit_window_data_index:list):
    central_guess= Parameter(name='central', value=fxy[Ramsey_f_data.index(min(Ramsey_f_data))] , min=min(fxy), max=max(fxy))
    New_ramsey_F,New_fxy=[],[]
    start1,end1= fit_window_data_index[0][0],fit_window_data_index[0][1]
    start2,end2= fit_window_data_index[1][0],fit_window_data_index[1][1]
    for i in range(start1,end1+1):
        New_ramsey_F.append(Ramsey_f_data[i])
        New_fxy.append(fxy[i])        
    for i in range(start2,end2+1):
        New_ramsey_F.append(Ramsey_f_data[i])
        New_fxy.append(fxy[i])  
    result= Ramsey_f_model.fit(np.array(New_ramsey_F),f=np.array(New_fxy),central=central_guess,m=1,n=0)
    central=result.best_values['central']
    m=result.best_values['m']
    n=result.best_values['n']
    fitting= Ramsey_f(fxy,central,m,n)
    return dict(data=np.array(Ramsey_f_data),fitting=fitting,fxy=fxy,central=central,m=m,n=n)

def Rabi_spec_F_fit(Rabi_f:list,fxy:np.ndarray):
    central_guess= Parameter(name='central', value=fxy[Rabi_f.index(min(Rabi_f))] , min=min(fxy), max=max(fxy))
    result= Rabi_f_model.fit(np.array(Rabi_f),f=fxy,central=central_guess,m=1,n=min(Rabi_f))
    central=result.best_values['central']
    m=result.best_values['m']
    n=result.best_values['n']
    fitting=result.best_fit
    return dict(data=np.array(Rabi_f),fitting=fitting,fxy=fxy,central=central,m=m,n=n)

def Notch_type_resonator_fit(f:np.ndarray,amp:np.ndarray,phase:np.ndarray,target_Ql,electric_delay:float): 
    # f:Hz #fcrop: the fitting frequency range

    #Normalize
    dip_idx=list(amp).index(min(amp))
    if dip_idx>len(amp)/2:
        amp=amp/np.mean(amp[0:10])
    else:
        amp=amp/np.mean(amp[-10:])
    #fit    
    port = circuit.notch_port()    
    dtype='linmagphasedeg' # dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
    port.f_data = f 
    port.z_data_raw = port._ConvToCompl(amp,phase,dtype=dtype)
    port.autofit(electric_delay=electric_delay,fcrop=None,Ql_guess=target_Ql,fr_guess=f[dip_idx]) 
    #print("Fit results:", port.fitresults) 
    #port.plotall()
    fit= port.fitresults
    i=complex(0,1) 
    Qi= 1/(1/fit['Ql']-(np.real(1/(fit['absQc']*np.exp(-i*fit['phi0'])))))
    
    para= port.do_calibration(f,port._ConvToCompl(amp,phase,dtype=dtype))
    a= para[1]
    alpha= para[2]
    delay= para[0]
    print('Fit_electrical delay=',port._delay)
    para_fit= np.linspace(min(f),max(f),50*len(amp))
    S21= S21_notch(para_fit,fit['fr'],fit['Ql'],fit['absQc'],fit['phi0'],a,alpha,delay)
    return xr.Dataset(data_vars=dict(S21=(['para_fit'],S21),amp=(['f'],amp),pha=(['f'],phase),amp_fitting=(['para_fit'],np.abs(S21)),pha_fitting=(['para_fit'],(360/(2*np.pi))*(np.angle(S21)))),coords=dict(f=(['f'],f),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="RS",fr=fit['fr'],Ql=fit['Ql'],Qc=fit['absQc'],Qi=Qi,phi0=fit['phi0'],a=a,alpha=alpha,delay=delay))
     
def Readout_F_opt_Fit(result:list,f_samples:np.ndarray,target_Ql,electric_delay):

    Ig,Qg,Ie,Qe=[],[],[],[]
    for i in range(len(result)):
        Ig.append(result[i]['Ig'])
        Qg.append(result[i]['Qg'])
        Ie.append(result[i]['Ie'])
        Qe.append(result[i]['Qe'])
    i=complex(0,1)
    Ig,Qg,Ie,Qe= np.array(Ig),np.array(Qg),np.array(Ie),np.array(Qe)
    Amp_g, phase_g = np.sqrt(Ig**2+Qg**2), np.angle(Ig+i*Qg)/np.pi*180#np.arctan2(Qg,Ig)/np.pi*180
    Amp_e, phase_e = np.sqrt(Ie**2+Qe**2), np.angle(Ie+i*Qe)/np.pi*180#np.arctan2(Qe,Ie)/np.pi*180
    g_fit=Notch_type_resonator_fit(f=f_samples,amp=Amp_g,phase=phase_g,target_Ql=target_Ql,electric_delay=electric_delay) 
    e_fit=Notch_type_resonator_fit(f=f_samples,amp=Amp_e,phase=phase_e,target_Ql=target_Ql,electric_delay=electric_delay)
    fr_g= g_fit.attrs['fr']
    fr_e= e_fit.attrs['fr']
    fr_bare= (fr_g+fr_e)/2
    return dict(g_fit=g_fit,e_fit=e_fit),fr_bare


def SQ_RB_analysis(data:np.ndarray,Set:np.ndarray,N:np.ndarray):
    data=np.array(data).reshape(len(N),len(Set)).transpose()
    avg_seq_F= np.mean(data,axis=0)
    p=Parameter(name='p', value= 0.95, min=0, max=1) 
    result= SQ_RB_model.fit(avg_seq_F,N=N,a=avg_seq_F[0]-avg_seq_F[-1],p=p,b=avg_seq_F[-1])
    a_fit=result.best_values['a']   
    p_fit=result.best_values['p']
    b_fit=result.best_values['b'] 
    para_fit= np.linspace(min(N),max(N),len(N)*20)
    fitting= SQ_RB(para_fit,a_fit,p_fit,b_fit)
    return xr.Dataset(data_vars=dict(data=(['N'],avg_seq_F),fitting=(['para_fit'],fitting)),coords=dict(N=(['N'],N),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="SQ_RB",a_fit=a_fit,p_fit=p_fit,b_fit=b_fit,raw=data))

def SQ_PB_analysis(data:dict,Set:np.ndarray,N:np.ndarray,Dis:float):
    data_I=np.array(data['I']).reshape(len(N),len(Set)).transpose()/Dis
    data_X=np.array(data['X_pi_2']).reshape(len(N),len(Set)).transpose()/Dis
    data_Y=np.array(data['Y_pi_2']).reshape(len(N),len(Set)).transpose()/Dis
    sigmax=-1*(2*data_Y-1)
    sigmay= 2*data_X-1
    sigmaz= 2*data_I-1
    P_nor= sigmax**2+sigmay**2+sigmaz**2
    avg_seq_F= np.mean(P_nor,axis=0)
    p=Parameter(name='p', value= 0.98, min=0, max=1) 
    result= SQ_RB_model.fit(avg_seq_F,N=N,a=avg_seq_F[0]-avg_seq_F[-1],p=p,b=avg_seq_F[-1])
    a_fit=result.best_values['a']   
    p_fit=result.best_values['p']
    b_fit=result.best_values['b'] 
    para_fit= np.linspace(min(N),max(N),len(N)*20)
    fitting= SQ_RB(para_fit,a_fit,p_fit,b_fit)
    return xr.Dataset(data_vars=dict(data=(['N'],avg_seq_F),fitting=(['para_fit'],fitting)),coords=dict(N=(['N'],N),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="SQ_RB",a_fit=a_fit,p_fit=p_fit,b_fit=b_fit,raw=P_nor))


def Ac_Stark_shift_analysis(voltage:np.ndarray,Analysis_result:dict,X_eff:float,fit_window_data_index:list):
    f01=[]
    power_l= voltage**2/50 # unit: W
    
    for i in range(len(voltage)):
        f01.append(Analysis_result['data_fit'][i].attrs['f01_fit']) 
            
    def Starkshift_P(x,A,fa):
        return fa-(2*x*A*+1)*X_eff

    gmodel = Model(Starkshift_P)
    result = gmodel.fit(np.array(f01)[fit_window_data_index[0]:fit_window_data_index[1]],x=power_l[fit_window_data_index[0]:fit_window_data_index[1]],A=1e5,fa=f01[0])
    
    coeff=result.best_values['A']
    fa_fit=result.best_values['fa']
    n=coeff*power_l

    def Starkshift_n(n_,fa):
        return fa-(2*n_*+1)*X_eff
    para_fit= np.linspace(min(power_l),max(power_l),20*len(power_l))
    fitting= Starkshift_P(para_fit,coeff,fa_fit)

    return xr.Dataset(data_vars=dict(data=(['P'],np.array(f01)),fitting=(['para_fit'],fitting)),coords=dict(P=(['P'],power_l),para_fit=(['para_fit'],para_fit)),attrs=dict(exper="Ac-Stark",coeff=coeff,fa_0=fa_fit,X_eff=X_eff))


def T1_from_single_shot_analysis(data,Reset_time,R_integration):
    #data must be normalized as 1 and 0
    up_idx=[]
    down_idx=[]
    Stay_g=[]
    for i in range(len(data)-1):
        if data[i+1]-data[i]==1:
            up_idx.append(i+1)
        elif data[i+1]-data[i]==-1:
            down_idx.append(i)
        else:
            pass
    if data[0]==1:
        down_idx=down_idx[1:]  
    if data[-1]==1:
        up_idx=up_idx[:-1]    
    T1_us= (np.array(down_idx)-np.array(up_idx)+1)*(Reset_time+R_integration)*1e6
    for i in range(len(down_idx)-1):
        Stay_g.append((up_idx[i+1]-down_idx[i])*(Reset_time+R_integration)*1e6) 
    return T1_us, Stay_g

def Cross_correlation(Data1,Data2):
    # Calculate means
    x_mean = np.mean(Data1)
    y_mean = np.mean(Data2)
    
    # Calculate numerator
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(Data1, Data2))
    
    # Calculate denominators
    x_sq_diff = sum((a - x_mean) ** 2 for a in Data1)
    y_sq_diff = sum((b - y_mean) ** 2 for b in Data2)
    denominator = np.sqrt(x_sq_diff * y_sq_diff)
    correlation = numerator / denominator
    return correlation


def Four_states_single_shot_analysis_pre_ge(data:dict,IQf_guess:list,IQd_guess:list,f01:float):
    def FourGaussian(I, Q, Ig, Qg, Ie, Qe, If, Qf, Id, Qd, sigma, Ag, Ae, Af, Ad):
        funcGround  = Ag * np.exp(-(I - Ig)**2 / 2 / sigma**2) * np.exp(-(Q - Qg)**2 / 2 / sigma**2)
        funcExcited = Ae * np.exp(-(I - Ie)**2 / 2 / sigma**2) * np.exp(-(Q - Qe)**2 / 2 / sigma**2)
        funcSecond  = Af * np.exp(-(I - If)**2 / 2 / sigma**2) * np.exp(-(Q - Qf)**2 / 2 / sigma**2)
        funcThird   = Ad * np.exp(-(I - Id)**2 / 2 / sigma**2) * np.exp(-(Q - Qd)**2 / 2 / sigma**2)
        return (funcGround + funcExcited + funcSecond + funcThird)
    Ig_data,Qg_data = np.array(data['g'][0]), np.array(data['g'][1]) 
    Ie_data,Qe_data = np.array(data['e'][0]), np.array(data['e'][1])

    g_data = np.vstack((Ig_data, Qg_data))
    e_data = np.vstack((Ie_data, Qe_data))
    I_data = np.hstack((Ig_data, Ie_data))
    Q_data = np.hstack((Qg_data, Qe_data))

    if len(Ig_data)<10000:
        bins=51
    elif 10000<len(Ig_data)<20000:
        bins=101
    else:
        bins=201

    g_predict= Single_shot_ref_fit_analysis(data['g'])['fit_pack']
    Ig_guess, Qg_guess, sig_guess= g_predict[0],g_predict[1],g_predict[2]
    
    I_mixdata, Q_mixdata= np.hstack((Ig_data, Ie_data)), np.hstack((Qg_data, Qe_data))
    
    I_=np.linspace(I_mixdata.min(),I_mixdata.max(),bins)
    Q_=np.linspace(Q_mixdata.min(),Q_mixdata.max(),bins)
    X,Y= np.meshgrid(I_,Q_)

    hist_Ie, edges_Ie = np.histogram(Ie_data, bins = bins, density = True)
    hist_Qe, edges_Qe = np.histogram(Qe_data, bins = bins, density = True)
    Ie_guess_idx, Qe_guess_idx = np.argmax(hist_Ie),np.argmax(hist_Qe)
    Ie_guess, Qe_guess= edges_Ie[Ie_guess_idx],edges_Qe[Qe_guess_idx]

    If_guess, Qf_guess = IQf_guess[0],IQf_guess[1]
    Id_guess, Qd_guess = IQd_guess[0],IQd_guess[1]

    I_=np.linspace(I_data.min(),I_data.max(),bins)
    Q_=np.linspace(Q_data.min(),Q_data.max(),bins)
    X,Y= np.meshgrid(I_,Q_)

    hist, xedges, yedges = np.histogram2d(I_data, Q_data, bins = (bins,bins), density=True)

    Ag_guess = np.max(hist)
    Ae_guess = np.max(hist)
    Af_guess = np.max(hist) * 0.05
    Ad_guess = np.max(hist) * 0.05

    Ig    = Parameter(name = 'Ig', value = Ig_guess, min = Ig_guess - 1*sig_guess, max = Ig_guess + 1*sig_guess)
    Qg    = Parameter(name = 'Qg', value = Qg_guess, min = Qg_guess - 1*sig_guess, max = Qg_guess + 1*sig_guess)
    Ie    = Parameter(name = 'Ie', value = Ie_guess, min = Ie_guess - 1*sig_guess, max = Ie_guess + 1*sig_guess)
    Qe    = Parameter(name = 'Qe', value = Qe_guess, min = Qe_guess - 1*sig_guess, max = Qe_guess + 1*sig_guess)
    If    = Parameter(name = 'If', value = If_guess, min = If_guess - 1*sig_guess, max = If_guess + 1*sig_guess)
    Qf    = Parameter(name = 'Qf', value = Qf_guess, min = Qf_guess - 1*sig_guess, max = Qf_guess + 1*sig_guess)
    Id    = Parameter(name = 'Id', value = Id_guess, min = Id_guess - 1*sig_guess, max = Id_guess + 1*sig_guess)
    Qd    = Parameter(name = 'Qd', value = Qd_guess, min = Qd_guess - 1*sig_guess, max = Qd_guess + 1*sig_guess)
    sigma = Parameter(name = 'sigma', value = sig_guess, min = 0.01 * sig_guess, max=3 * sig_guess)
    Ag    = Parameter(name = 'Ag',value = Ag_guess, min = 0.1 * Ag_guess)
    Ae    = Parameter(name = 'Ae',value = Ae_guess, min = 0.1 * Ae_guess)
    Af    = Parameter(name = 'Ag',value = Af_guess, min=0, max = Ag_guess) 
    Ad    = Parameter(name = 'Ae',value = Ad_guess, min=0, max = Ag_guess)

    model = Model(FourGaussian, independent_vars=['I', 'Q'])
    result = model.fit(hist.transpose(), I=X, Q=Y, Ig=Ig, Qg=Qg, Ie=Ie, Qe=Qe, If=If, Qf=Qf, Id=Id, Qd=Qd, sigma=sigma, Ag=Ag, Ae=Ae, Af=Af, Ad=Ad)

    Ig_fit    = result.best_values['Ig']
    Qg_fit    = result.best_values['Qg']
    Ie_fit    = result.best_values['Ie']
    Qe_fit    = result.best_values['Qe']
    If_fit    = result.best_values['If']
    Qf_fit    = result.best_values['Qf']
    Id_fit    = result.best_values['Id']
    Qd_fit    = result.best_values['Qd']
    sigma_fit = result.best_values['sigma']
    Ag_fit    = result.best_values['Ag']
    Ae_fit    = result.best_values['Ae']
    Af_fit    = result.best_values['Af']
    Ad_fit    = result.best_values['Ad']

    Cg = np.array([Ig_fit, Qg_fit])
    Ce = np.array([Ie_fit, Qe_fit])
    Cf = np.array([If_fit, Qf_fit])
    Cd = np.array([Id_fit, Qd_fit])

    I_C =  np.array([Ig_fit, Ie_fit, If_fit, Id_fit])
    Q_C =  np.array([Qg_fit, Qe_fit, Qf_fit, Qd_fit])

    Dg_e = np.linalg.norm(Cg-Ce)
    Dg_f = np.linalg.norm(Cg-Cf)
    Df_d = np.linalg.norm(Cf-Cd)
    Dd_e = np.linalg.norm(Cd-Ce)
    D = np.array([Dg_e, Dg_f, Df_d, Dd_e])
    SNR = D/sigma_fit
    overlap = (1/2)*(1-special.erf(np.sqrt(SNR**2/8)))

    # Find A,B,C circumcentre
    def circumcenter(A, B, C):
        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
        Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
        return (Ux, Uy)
    
    U = circumcenter(Cg, Ce, Cf)
    r = np.array([np.linalg.norm(Cg - U), np.linalg.norm(Ce - U), np.linalg.norm(Cf- U), np.linalg.norm(Cd- U)])

    def CircleFunction(I, Q, Cent_x, Cent_y):
        return np.sqrt((I - Cent_x)**2 + (Q - Cent_y)**2)
    
    Cent_x = Parameter(name = "cent_x", value = U[0], min = U[0] - 1*sigma_fit, max = U[0] + 1*sigma_fit)
    Cent_y = Parameter(name = "cent_y", value = U[1], min = U[1] - 1*sigma_fit, max = U[1] + 1*sigma_fit)
    model_cir  = Model(CircleFunction, independent_vars=['I', 'Q'])
    result_cir = model_cir.fit(r, I = I_C, Q = Q_C, Cent_x = Cent_x, Cent_y = Cent_y)

    Cent_x_fit = result_cir.best_values['Cent_x']
    Cent_y_fit = result_cir.best_values['Cent_y']
    Cent = np.array([Cent_x_fit, Cent_y_fit])

    phase_g = np.angle((Cg-Cent)[0] + 1j*(Cg-Cent)[1])
    phase_e = np.angle((Ce-Cent)[0] + 1j*(Ce-Cent)[1])
    phase_f = np.angle((Cf-Cent)[0] + 1j*(Cf-Cent)[1])
    phase_d = np.angle((Cd-Cent)[0] + 1j*(Cd-Cent)[1])
    cloud_phase  = np.array([phase_g, phase_e, phase_f, phase_d])

    idx_phase = np.argsort(cloud_phase)
    sorted_phase = np.sort(cloud_phase)
    thresholds = np.array(range(len(sorted_phase)), dtype = float)

    for i in range(len(sorted_phase)):
        if i < (len(sorted_phase)-1):
            thres = (sorted_phase[i] + sorted_phase[i+1]) / 2
        else:
            thres = (sorted_phase[i] + sorted_phase[0] + 2*np.pi) / 2
        thres = (thres + np.pi) % (2*np.pi) - np.pi
        thresholds[idx_phase[i]] = thres
    
    def fit_prepared_state(I_data_in, Q_data_in):
        Ig   = Parameter(name = 'Ig', value = Ig_fit, min = Ig_fit - 1*sigma_fit, max = Ig_fit + 1*sigma_fit)
        Qg   = Parameter(name = 'Qg', value = Qg_fit, min = Qg_fit - 1*sigma_fit, max = Qg_fit + 1*sigma_fit)
        Ie   = Parameter(name = 'Ie', value = Ie_fit, min = Ie_fit - 1*sigma_fit, max = Ie_fit + 1*sigma_fit)
        Qe   = Parameter(name = 'Qe', value = Qe_fit, min = Qe_fit - 1*sigma_fit, max = Qe_fit + 1*sigma_fit)
        If   = Parameter(name = 'If', value = If_fit, min = If_fit - 1*sigma_fit, max = If_fit + 1*sigma_fit)
        Qf   = Parameter(name = 'Qf', value = Qf_fit, min = Qf_fit - 1*sigma_fit, max = Qf_fit + 1*sigma_fit)
        Id   = Parameter(name = 'Id', value = Id_fit, min = Id_fit - 1*sigma_fit, max = Id_fit + 1*sigma_fit)
        Qd   = Parameter(name = 'Qd', value = Qd_fit, min = Qd_fit - 1*sigma_fit, max = Qd_fit + 1*sigma_fit)
        sigma = Parameter(name = 'sigma', value = sigma_fit, min = 0.01 * sigma_fit, max=3 * sigma_fit)
        Ag   = Parameter(name = 'Ag',value = Ag_fit, min = 0.1 * Ag_fit)
        Ae   = Parameter(name = 'Ae',value = Ae_fit, min = 0.1 * Ae_fit)
        Af   = Parameter(name = 'Ag',value = Af_fit, min=0, max = Ag_fit)
        Ad   = Parameter(name = 'Ae',value = Ad_fit, min=0, max = Ag_fit)
        hist, xedges, yedges = np.histogram2d(I_data_in, Q_data_in, bins = (bins,bins), density=True)
        model = Model(FourGaussian, independent_vars=['I', 'Q'])
        result = model.fit(hist.transpose(), I=X, Q=Y, Ig=Ig, Qg=Qg, Ie=Ie, Qe=Qe, If=If, Qf=Qf, Id=Id, Qd=Qd, sigma=sigma, Ag=Ag, Ae=Ae, Af=Af, Ad=Ad)

        Ig_sec_fit    = result.best_values['Ig']
        Qg_sec_fit    = result.best_values['Qg']
        Ie_sec_fit    = result.best_values['Ie']
        Qe_sec_fit    = result.best_values['Qe']
        If_sec_fit    = result.best_values['If']
        Qf_sec_fit    = result.best_values['Qf']
        Id_sec_fit    = result.best_values['Id']
        Qd_sec_fit    = result.best_values['Qd']
        sigma_sec_fit = result.best_values['sigma']
        Ag_sec_fit    = result.best_values['Ag']
        Ae_sec_fit    = result.best_values['Ae']
        Af_sec_fit    = result.best_values['Af']
        Ad_sec_fit    = result.best_values['Ad']

        funcGround   = lambda y, x: Ag_sec_fit * np.exp(-(x - Ig_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qg_sec_fit)**2 / 2 / sigma_sec_fit**2)
        funcExcited  = lambda y, x: Ae_sec_fit * np.exp(-(x - Ie_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qe_sec_fit)**2 / 2 / sigma_sec_fit**2)
        funcSecond   = lambda y, x: Af_sec_fit * np.exp(-(x - If_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qf_sec_fit)**2 / 2 / sigma_sec_fit**2)
        funcThird    = lambda y, x: Ad_sec_fit * np.exp(-(x - Id_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qd_sec_fit)**2 / 2 / sigma_sec_fit**2)

        volume_g, gerr = dblquad(funcGround  , np.min(I_data_in) - 20 * sigma_sec_fit, np.max(I_data_in) + 20 * sigma_sec_fit, np.min(Q_data_in) - 20 * sigma_sec_fit, np.max(Q_data_in) + 20 * sigma_sec_fit)
        volume_e, eerr = dblquad(funcExcited , np.min(I_data_in) - 20 * sigma_sec_fit, np.max(I_data_in) + 20 * sigma_sec_fit, np.min(Q_data_in) - 20 * sigma_sec_fit, np.max(Q_data_in) + 20 * sigma_sec_fit)
        volume_f, ferr = dblquad(funcSecond  , np.min(I_data_in) - 20 * sigma_sec_fit, np.max(I_data_in) + 20 * sigma_sec_fit, np.min(Q_data_in) - 20 * sigma_sec_fit, np.max(Q_data_in) + 20 * sigma_sec_fit)
        volume_d, derr = dblquad(funcThird   , np.min(I_data_in) - 20 * sigma_sec_fit, np.max(I_data_in) + 20 * sigma_sec_fit, np.min(Q_data_in) - 20 * sigma_sec_fit, np.max(Q_data_in) + 20 * sigma_sec_fit)

        volume_total = volume_g + volume_e + volume_f + volume_d
        Pi_g = volume_g / volume_total
        Pi_e = volume_e / volume_total
        Pi_f = volume_f / volume_total
        Pi_d = volume_d / volume_total

        return ([Pi_g, Pi_e, Pi_f, Pi_d])
    
    ratio_g = fit_prepared_state(Ig_data, Qg_data)
    ratio_e = fit_prepared_state(Ie_data, Qe_data)
    hbar = 1.054571800*1e-34
    kB = 1.38e-23    
    Wa= 2*np.pi*f01
    Thermal= ratio_g[1]
    def PetoT(Pe):
        Pg= 1-Pe
        T= (-hbar*Wa)/(kB*np.log(Pe/Pg))*1000
        return T 
    T= PetoT(Thermal)
    M = np.array([ratio_g, ratio_e])
    F_s = 1-overlap
    F_g = ratio_g[0]
    F_e = ratio_e[1]
    F = (F_g + F_e) / 2

    fit_pack = [Cg, Ce, Cf, Cd, Ag_fit, Ae_fit, Af_fit, Ad_fit, sigma_fit, Cent]
    error_pack = dict(Ig=Ig_fit,Qg=Qg_fit,Ie=Ie_fit,Qe=Qe_fit,If=If_fit,Qf=Qf_fit,Id=Id_fit,Qd=Qd_fit,D=D,sigma=sigma_fit,SNR=SNR,overlap=overlap,Thermal=Thermal,T=T,F_s=F_s,F_g=F_g,F_e=F_e,F=F)
    return dict(IQdata=[g_data,e_data],I_fit=X,Q_fit=Y,fit_pack=fit_pack,error_pack=error_pack,M=M,Ig=Ig_fit,Qg=Qg_fit,Ie=Ie_fit,Qe=Qe_fit,If=If_fit,Qf=Qf_fit,Id=Id_fit,Qd=Qd_fit,Cent=Cent,thresholds=thresholds)

def Four_states_single_shot_analysis_pre_gef(data:dict,IQf_guess:list,IQd_guess:list,f01:float):
    def FourGaussian(I, Q, Ig, Qg, Ie, Qe, If, Qf, Id, Qd, sigma, Ag, Ae, Af, Ad):
        funcGround  = Ag * np.exp(-(I - Ig)**2 / 2 / sigma**2) * np.exp(-(Q - Qg)**2 / 2 / sigma**2)
        funcExcited = Ae * np.exp(-(I - Ie)**2 / 2 / sigma**2) * np.exp(-(Q - Qe)**2 / 2 / sigma**2)
        funcSecond  = Af * np.exp(-(I - If)**2 / 2 / sigma**2) * np.exp(-(Q - Qf)**2 / 2 / sigma**2)
        funcThird   = Ad * np.exp(-(I - Id)**2 / 2 / sigma**2) * np.exp(-(Q - Qd)**2 / 2 / sigma**2)
        return (funcGround + funcExcited + funcSecond + funcThird)
    Ig_data,Qg_data = np.array(data['g'][0]), np.array(data['g'][1]) 
    Ie_data,Qe_data = np.array(data['e'][0]), np.array(data['e'][1])
    If_data,Qf_data = np.array(data['f'][0]), np.array(data['f'][1])

    g_data = np.vstack((Ig_data, Qg_data))
    e_data = np.vstack((Ie_data, Qe_data))
    f_data = np.vstack((If_data, Qf_data))
    I_data = np.hstack((Ig_data, Ie_data, If_data))
    Q_data = np.hstack((Qg_data, Qe_data, Qf_data))

    if len(Ig_data)<10000:
        bins=51
    elif 10000<len(Ig_data)<20000:
        bins=101
    else:
        bins=201

    g_predict= Single_shot_ref_fit_analysis(data['g'])['fit_pack']
    Ig_guess, Qg_guess, sig_guess= g_predict[0],g_predict[1],g_predict[2]
    
    I_mixdata, Q_mixdata= np.hstack((Ig_data, Ie_data, If_data)), np.hstack((Qg_data, Qe_data, Qf_data))
    
    I_=np.linspace(I_mixdata.min(),I_mixdata.max(),bins)
    Q_=np.linspace(Q_mixdata.min(),Q_mixdata.max(),bins)
    X,Y= np.meshgrid(I_,Q_)

    hist_Ie, edges_Ie = np.histogram(Ie_data, bins = bins, density = True)
    hist_Qe, edges_Qe = np.histogram(Qe_data, bins = bins, density = True)
    Ie_guess_idx, Qe_guess_idx = np.argmax(hist_Ie),np.argmax(hist_Qe)
    Ie_guess, Qe_guess= edges_Ie[Ie_guess_idx],edges_Qe[Qe_guess_idx]

    If_guess, Qf_guess = IQf_guess[0],IQf_guess[1]
    Id_guess, Qd_guess = IQd_guess[0],IQd_guess[1]

    I_=np.linspace(I_data.min(),I_data.max(),bins)
    Q_=np.linspace(Q_data.min(),Q_data.max(),bins)
    X,Y= np.meshgrid(I_,Q_)

    hist, xedges, yedges = np.histogram2d(I_data, Q_data, bins = (bins,bins), density=True)

    Ag_guess = np.max(hist)
    Ae_guess = np.max(hist)
    Af_guess = np.max(hist) * 0.05
    Ad_guess = np.max(hist) * 0.05

    Ig    = Parameter(name = 'Ig', value = Ig_guess, min = Ig_guess - 1*sig_guess, max = Ig_guess + 1*sig_guess)
    Qg    = Parameter(name = 'Qg', value = Qg_guess, min = Qg_guess - 1*sig_guess, max = Qg_guess + 1*sig_guess)
    Ie    = Parameter(name = 'Ie', value = Ie_guess, min = Ie_guess - 1*sig_guess, max = Ie_guess + 1*sig_guess)
    Qe    = Parameter(name = 'Qe', value = Qe_guess, min = Qe_guess - 1*sig_guess, max = Qe_guess + 1*sig_guess)
    If    = Parameter(name = 'If', value = If_guess, min = If_guess - 1*sig_guess, max = If_guess + 1*sig_guess)
    Qf    = Parameter(name = 'Qf', value = Qf_guess, min = Qf_guess - 1*sig_guess, max = Qf_guess + 1*sig_guess)
    Id    = Parameter(name = 'Id', value = Id_guess, min = Id_guess - 1*sig_guess, max = Id_guess + 1*sig_guess)
    Qd    = Parameter(name = 'Qd', value = Qd_guess, min = Qd_guess - 1*sig_guess, max = Qd_guess + 1*sig_guess)
    sigma = Parameter(name = 'sigma', value = sig_guess, min = 0.01 * sig_guess, max=3 * sig_guess)
    Ag    = Parameter(name = 'Ag',value = Ag_guess, min = 0.1 * Ag_guess)
    Ae    = Parameter(name = 'Ae',value = Ae_guess, min = 0.1 * Ae_guess)
    Af    = Parameter(name = 'Ag',value = Af_guess, min=0, max = Ag_guess) 
    Ad    = Parameter(name = 'Ae',value = Ad_guess, min=0, max = Ag_guess)

    model = Model(FourGaussian, independent_vars=['I', 'Q'])
    result = model.fit(hist.transpose(), I=X, Q=Y, Ig=Ig, Qg=Qg, Ie=Ie, Qe=Qe, If=If, Qf=Qf, Id=Id, Qd=Qd, sigma=sigma, Ag=Ag, Ae=Ae, Af=Af, Ad=Ad)

    Ig_fit    = result.best_values['Ig']
    Qg_fit    = result.best_values['Qg']
    Ie_fit    = result.best_values['Ie']
    Qe_fit    = result.best_values['Qe']
    If_fit    = result.best_values['If']
    Qf_fit    = result.best_values['Qf']
    Id_fit    = result.best_values['Id']
    Qd_fit    = result.best_values['Qd']
    sigma_fit = result.best_values['sigma']
    Ag_fit    = result.best_values['Ag']
    Ae_fit    = result.best_values['Ae']
    Af_fit    = result.best_values['Af']
    Ad_fit    = result.best_values['Ad']

    Cg = np.array([Ig_fit, Qg_fit])
    Ce = np.array([Ie_fit, Qe_fit])
    Cf = np.array([If_fit, Qf_fit])
    Cd = np.array([Id_fit, Qd_fit])

    I_C =  np.array([Ig_fit, Ie_fit, If_fit, Id_fit])
    Q_C =  np.array([Qg_fit, Qe_fit, Qf_fit, Qd_fit])

    Dg_e = np.linalg.norm(Cg-Ce)
    Dg_f = np.linalg.norm(Cg-Cf)
    Df_d = np.linalg.norm(Cf-Cd)
    Dd_e = np.linalg.norm(Cd-Ce)
    D = np.array([Dg_e, Dg_f, Df_d, Dd_e])
    SNR = D/sigma_fit
    overlap = (1/2)*(1-special.erf(np.sqrt(SNR**2/8)))

    # Find A,B,C circumcentre
    def circumcenter(A, B, C):
        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
        Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
        return (Ux, Uy)
    
    U = circumcenter(Cg, Ce, Cf)
    r = np.array([np.linalg.norm(Cg - U), np.linalg.norm(Ce - U), np.linalg.norm(Cf- U), np.linalg.norm(Cd- U)])

    def CircleFunction(I, Q, Cent_x, Cent_y):
        return np.sqrt((I - Cent_x)**2 + (Q - Cent_y)**2)
    
    Cent_x = Parameter(name = "cent_x", value = U[0], min = U[0] - 1*sigma_fit, max = U[0] + 1*sigma_fit)
    Cent_y = Parameter(name = "cent_y", value = U[1], min = U[1] - 1*sigma_fit, max = U[1] + 1*sigma_fit)
    model_cir  = Model(CircleFunction, independent_vars=['I', 'Q'])
    result_cir = model_cir.fit(r, I = I_C, Q = Q_C, Cent_x = Cent_x, Cent_y = Cent_y)

    Cent_x_fit = result_cir.best_values['Cent_x']
    Cent_y_fit = result_cir.best_values['Cent_y']
    Cent = np.array([Cent_x_fit, Cent_y_fit])

    phase_g = np.angle((Cg-Cent)[0] + 1j*(Cg-Cent)[1])
    phase_e = np.angle((Ce-Cent)[0] + 1j*(Ce-Cent)[1])
    phase_f = np.angle((Cf-Cent)[0] + 1j*(Cf-Cent)[1])
    phase_d = np.angle((Cd-Cent)[0] + 1j*(Cd-Cent)[1])
    cloud_phase  = np.array([phase_g, phase_e, phase_f, phase_d])

    idx_phase = np.argsort(cloud_phase)
    sorted_phase = np.sort(cloud_phase)
    thresholds = np.array(range(len(sorted_phase)), dtype = float)

    for i in range(len(sorted_phase)):
        if i < (len(sorted_phase)-1):
            thres = (sorted_phase[i] + sorted_phase[i+1]) / 2
        else:
            thres = (sorted_phase[i] + sorted_phase[0] + 2*np.pi) / 2
        thres = (thres + np.pi) % (2*np.pi) - np.pi
        thresholds[idx_phase[i]] = thres
    
    def fit_prepared_state(I_data_in, Q_data_in):
        Ig   = Parameter(name = 'Ig', value = Ig_fit, min = Ig_fit - 1*sigma_fit, max = Ig_fit + 1*sigma_fit)
        Qg   = Parameter(name = 'Qg', value = Qg_fit, min = Qg_fit - 1*sigma_fit, max = Qg_fit + 1*sigma_fit)
        Ie   = Parameter(name = 'Ie', value = Ie_fit, min = Ie_fit - 1*sigma_fit, max = Ie_fit + 1*sigma_fit)
        Qe   = Parameter(name = 'Qe', value = Qe_fit, min = Qe_fit - 1*sigma_fit, max = Qe_fit + 1*sigma_fit)
        If   = Parameter(name = 'If', value = If_fit, min = If_fit - 1*sigma_fit, max = If_fit + 1*sigma_fit)
        Qf   = Parameter(name = 'Qf', value = Qf_fit, min = Qf_fit - 1*sigma_fit, max = Qf_fit + 1*sigma_fit)
        Id   = Parameter(name = 'Id', value = Id_fit, min = Id_fit - 1*sigma_fit, max = Id_fit + 1*sigma_fit)
        Qd   = Parameter(name = 'Qd', value = Qd_fit, min = Qd_fit - 1*sigma_fit, max = Qd_fit + 1*sigma_fit)
        sigma = Parameter(name = 'sigma', value = sigma_fit, min = 0.01 * sigma_fit, max=3 * sigma_fit)
        Ag   = Parameter(name = 'Ag',value = Ag_fit, min = 0.1 * Ag_fit)
        Ae   = Parameter(name = 'Ae',value = Ae_fit, min = 0.1 * Ae_fit)
        Af   = Parameter(name = 'Ag',value = Af_fit, min=0, max = Ag_fit)
        Ad   = Parameter(name = 'Ae',value = Ad_fit, min=0, max = Ag_fit)
        hist, xedges, yedges = np.histogram2d(I_data_in, Q_data_in, bins = (bins,bins), density=True)
        model = Model(FourGaussian, independent_vars=['I', 'Q'])
        result = model.fit(hist.transpose(), I=X, Q=Y, Ig=Ig, Qg=Qg, Ie=Ie, Qe=Qe, If=If, Qf=Qf, Id=Id, Qd=Qd, sigma=sigma, Ag=Ag, Ae=Ae, Af=Af, Ad=Ad)

        Ig_sec_fit    = result.best_values['Ig']
        Qg_sec_fit    = result.best_values['Qg']
        Ie_sec_fit    = result.best_values['Ie']
        Qe_sec_fit    = result.best_values['Qe']
        If_sec_fit    = result.best_values['If']
        Qf_sec_fit    = result.best_values['Qf']
        Id_sec_fit    = result.best_values['Id']
        Qd_sec_fit    = result.best_values['Qd']
        sigma_sec_fit = result.best_values['sigma']
        Ag_sec_fit    = result.best_values['Ag']
        Ae_sec_fit    = result.best_values['Ae']
        Af_sec_fit    = result.best_values['Af']
        Ad_sec_fit    = result.best_values['Ad']

        funcGround   = lambda y, x: Ag_sec_fit * np.exp(-(x - Ig_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qg_sec_fit)**2 / 2 / sigma_sec_fit**2)
        funcExcited  = lambda y, x: Ae_sec_fit * np.exp(-(x - Ie_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qe_sec_fit)**2 / 2 / sigma_sec_fit**2)
        funcSecond   = lambda y, x: Af_sec_fit * np.exp(-(x - If_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qf_sec_fit)**2 / 2 / sigma_sec_fit**2)
        funcThird    = lambda y, x: Ad_sec_fit * np.exp(-(x - Id_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qd_sec_fit)**2 / 2 / sigma_sec_fit**2)

        volume_g, gerr = dblquad(funcGround  , np.min(I_data_in) - 20 * sigma_sec_fit, np.max(I_data_in) + 20 * sigma_sec_fit, np.min(Q_data_in) - 20 * sigma_sec_fit, np.max(Q_data_in) + 20 * sigma_sec_fit)
        volume_e, eerr = dblquad(funcExcited , np.min(I_data_in) - 20 * sigma_sec_fit, np.max(I_data_in) + 20 * sigma_sec_fit, np.min(Q_data_in) - 20 * sigma_sec_fit, np.max(Q_data_in) + 20 * sigma_sec_fit)
        volume_f, ferr = dblquad(funcSecond  , np.min(I_data_in) - 20 * sigma_sec_fit, np.max(I_data_in) + 20 * sigma_sec_fit, np.min(Q_data_in) - 20 * sigma_sec_fit, np.max(Q_data_in) + 20 * sigma_sec_fit)
        volume_d, derr = dblquad(funcThird   , np.min(I_data_in) - 20 * sigma_sec_fit, np.max(I_data_in) + 20 * sigma_sec_fit, np.min(Q_data_in) - 20 * sigma_sec_fit, np.max(Q_data_in) + 20 * sigma_sec_fit)

        volume_total = volume_g + volume_e + volume_f + volume_d
        Pi_g = volume_g / volume_total
        Pi_e = volume_e / volume_total
        Pi_f = volume_f / volume_total
        Pi_d = volume_d / volume_total

        return ([Pi_g, Pi_e, Pi_f, Pi_d])
    
    ratio_g = fit_prepared_state(Ig_data, Qg_data)
    ratio_e = fit_prepared_state(Ie_data, Qe_data)
    ratio_f = fit_prepared_state(If_data, Qf_data)
    hbar = 1.054571800*1e-34
    kB = 1.38e-23    
    Wa= 2*np.pi*f01
    Thermal= ratio_g[1]
    def PetoT(Pe):
        Pg= 1-Pe
        T= (-hbar*Wa)/(kB*np.log(Pe/Pg))*1000
        return T 
    T= PetoT(Thermal)
    M = np.array([ratio_g, ratio_e, ratio_f])
    F_s = 1-overlap
    F_g = ratio_g[0]
    F_e = ratio_e[1]
    F_f = ratio_f[2]
    F = (F_g + F_e + F_f) / 3

    fit_pack = [Cg, Ce, Cf, Cd, Ag_fit, Ae_fit, Af_fit, Ad_fit, sigma_fit, Cent]
    error_pack = dict(Ig=Ig_fit,Qg=Qg_fit,Ie=Ie_fit,Qe=Qe_fit,If=If_fit,Qf=Qf_fit,Id=Id_fit,Qd=Qd_fit,D=D,sigma=sigma_fit,SNR=SNR,overlap=overlap,Thermal=Thermal,T=T,F_s=F_s,F_g=F_g,F_e=F_e,F=F)
    return dict(IQdata=[g_data,e_data,f_data],I_fit=X,Q_fit=Y,fit_pack=fit_pack,error_pack=error_pack,M=M,Ig=Ig_fit,Qg=Qg_fit,Ie=Ie_fit,Qe=Qe_fit,If=If_fit,Qf=Qf_fit,Id=Id_fit,Qd=Qd_fit,Cent=Cent,thresholds=thresholds)

def Four_states_threshold_acquisition_pre_ge(data:dict): # Input analysis_result
    Cent = data["Cent"]
    Cg        = data['fit_pack'][0]
    Ce        = data['fit_pack'][1]
    Cf        = data['fit_pack'][2]
    Cd        = data['fit_pack'][3]

    phase_g = np.angle((Cg-Cent)[0] + 1j*(Cg-Cent)[1])
    phase_e = np.angle((Ce-Cent)[0] + 1j*(Ce-Cent)[1])
    phase_f = np.angle((Cf-Cent)[0] + 1j*(Cf-Cent)[1])
    phase_d = np.angle((Cd-Cent)[0] + 1j*(Cd-Cent)[1])
    thresholds = data["thresholds"]
    
    g_data = data['IQdata'][0]
    e_data = data['IQdata'][1]
    Ig_data, Qg_data = g_data[0], g_data[1]
    Ie_data, Qe_data = e_data[0], e_data[1]

    Ig_data = Ig_data - Cent[0]
    Qg_data = Qg_data - Cent[1]
    Ie_data = Ie_data - Cent[0]
    Qe_data = Qe_data - Cent[1]

    g_phase = np.angle(Ig_data + 1j*Qg_data)
    e_phase = np.angle(Ie_data + 1j*Qe_data)

    def sortingByThreshold(labels, phase, ratio_list):
        n = len(labels)
        for i in range(n):
            th1 = thresholds[i]
            th2 = thresholds[(i + 1) % n]

            if th1 < th2:
                if phase < th1 or phase > th2:
                    ratio_list[i] += 1
            else:
                if th1 > phase > th2:
                    ratio_list[i] += 1
    
    labels = [0, 1, 2, 3]
    ratio_g = np.array([0, 0, 0, 0])
    ratio_e = np.array([0, 0, 0, 0])
    for phase in g_phase:
        sortingByThreshold(labels, phase, ratio_g)

    for phase in e_phase:
        sortingByThreshold(labels, phase, ratio_e)
        
    ratio_g = ratio_g/np.sum(ratio_g)
    ratio_e = ratio_e/np.sum(ratio_e)
    M = np.array([ratio_g, ratio_e])
    return M

def Four_states_threshold_acquisition_pre_gef(data:dict): # Input analysis_result
    Cent = data["Cent"]
    Cg        = data['fit_pack'][0]
    Ce        = data['fit_pack'][1]
    Cf        = data['fit_pack'][2]
    Cd        = data['fit_pack'][3]

    phase_g = np.angle((Cg-Cent)[0] + 1j*(Cg-Cent)[1])
    phase_e = np.angle((Ce-Cent)[0] + 1j*(Ce-Cent)[1])
    phase_f = np.angle((Cf-Cent)[0] + 1j*(Cf-Cent)[1])
    phase_d = np.angle((Cd-Cent)[0] + 1j*(Cd-Cent)[1])
    thresholds = data["thresholds"]
    
    g_data = data['IQdata'][0]
    e_data = data['IQdata'][1]
    f_data = data['IQdata'][2]
    Ig_data, Qg_data = g_data[0], g_data[1]
    Ie_data, Qe_data = e_data[0], e_data[1]
    If_data, Qf_data = f_data[0], f_data[1]

    Ig_data = Ig_data - Cent[0]
    Qg_data = Qg_data - Cent[1]
    Ie_data = Ie_data - Cent[0]
    Qe_data = Qe_data - Cent[1]
    If_data = If_data - Cent[0]
    Qf_data = Qf_data - Cent[1]

    g_phase = np.angle(Ig_data + 1j*Qg_data)
    e_phase = np.angle(Ie_data + 1j*Qe_data)
    f_phase = np.angle(If_data + 1j*Qf_data)

    def sortingByThreshold(labels, phase, ratio_list):
        n = len(labels)
        for i in range(n):
            th1 = thresholds[i]
            th2 = thresholds[(i + 1) % n]

            if th1 < th2:
                if phase < th1 or phase > th2:
                    ratio_list[i] += 1
            else:
                if th1 > phase > th2:
                    ratio_list[i] += 1
    
    labels = [0, 1, 2, 3]
    ratio_g = np.array([0, 0, 0, 0])
    ratio_e = np.array([0, 0, 0, 0])
    ratio_f = np.array([0, 0, 0, 0])
    for phase in g_phase:
        sortingByThreshold(labels, phase, ratio_g)

    for phase in e_phase:
        sortingByThreshold(labels, phase, ratio_e)
    
    for phase in f_phase:
        sortingByThreshold(labels, phase, ratio_f)
        
    ratio_g = ratio_g/np.sum(ratio_g)
    ratio_e = ratio_e/np.sum(ratio_e)
    ratio_f = ratio_f/np.sum(ratio_f)
    M = np.array([ratio_g, ratio_e, ratio_f])
    return M

def Phase_threshold_circle_fit(IQ_g ,IQ_e, IQ_f, IQ_d,analysis_cloud_number): # IQ_g should be np.array
    IQ_g ,IQ_e, IQ_f, IQ_d= np.array(IQ_g) ,np.array(IQ_e), np.array(IQ_f), np.array(IQ_d)
    if analysis_cloud_number==4:
        I_C = np.array([IQ_g[0] ,IQ_e[0], IQ_f[0], IQ_d[0]])
        Q_C = np.array([IQ_g[1] ,IQ_e[1], IQ_f[1], IQ_d[1]])

        def circumcenter(A, B, C):
            D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
            Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
            Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
            return (Ux, Uy)
        
        U = circumcenter(IQ_g, IQ_e, IQ_f)
        r = np.array([np.linalg.norm(IQ_g - U), np.linalg.norm(IQ_e - U), np.linalg.norm(IQ_f- U), np.linalg.norm(IQ_d- U)])

        def CircleFunction(I, Q, Cent_x, Cent_y):
            return np.sqrt((I - Cent_x)**2 + (Q - Cent_y)**2)

        Cent_x = Parameter(name = "cent_x", value = U[0], min = U[0] - 0.25*r[0], max = U[0] + 0.25*r[0])
        Cent_y = Parameter(name = "cent_y", value = U[1], min = U[1] - 0.25*r[0], max = U[1] + 0.25*r[0])
        model_cir  = Model(CircleFunction, independent_vars=['I', 'Q'])
        result_cir = model_cir.fit(r, I = I_C, Q = Q_C, Cent_x = Cent_x, Cent_y = Cent_y)

        Cent_x_fit = result_cir.best_values['Cent_x']
        Cent_y_fit = result_cir.best_values['Cent_y']
        Cent = np.array([Cent_x_fit, Cent_y_fit])

        Cg_phase = np.angle((IQ_g-Cent)[0] + 1j*(IQ_g-Cent)[1])
        Ce_phase = np.angle((IQ_e-Cent)[0] + 1j*(IQ_e-Cent)[1])
        Cf_phase = np.angle((IQ_f-Cent)[0] + 1j*(IQ_f-Cent)[1])
        Cd_phase = np.angle((IQ_d-Cent)[0] + 1j*(IQ_d-Cent)[1])
        cloud_phase  = np.array([Cg_phase, Ce_phase, Cf_phase, Cd_phase])
        cloud_phase_deg = np.rad2deg(cloud_phase)
    return cloud_phase, [Cent_x_fit,Cent_y_fit]

def Phase_threshold_acquisition(data,cloud_phase,Center_fit,analysis_cloud_number): # IQ_g should be np.array
    if analysis_cloud_number==4:
        I_data, Q_data = data[0]-Center_fit[0], data[1]-Center_fit[1]
        phase_data = np.angle(I_data + 1j*Q_data)
        abs_data = np.abs(I_data + 1j*Q_data)

        idx_phase = np.argsort(cloud_phase)
        sorted_phase = np.sort(cloud_phase)
        thresholds = np.array(range(len(sorted_phase)), dtype = float)

        for i in range(len(sorted_phase)):
            if i < (len(sorted_phase)-1):
                thres = (sorted_phase[i] + sorted_phase[i+1]) / 2
            else:
                thres = (sorted_phase[i] + sorted_phase[0] + 2*np.pi) / 2
            thres = (thres + np.pi) % (2*np.pi) - np.pi
            thresholds[idx_phase[i]] = thres

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
        ratio_g = np.array([0, 0, 0, 0]) # maybe somtimes this will be used.

        sorted_data = [[], [], [], []] # maybe somtimes this will be used.
        filtered_sequence_label = np.array([]) # maybe somtimes this will be used.
        sequence_label = np.array([])
        sequence_phase = np.array([])
        for i in range(len(phase_data)):
            label, pos = sortingByThreshold(labels, phase_data[i], abs_data[i], ratio_g)
            sorted_data[label].append(pos)
            sequence_phase = np.append(sequence_phase, pos[1])
            sequence_label = np.append(sequence_label, label)
            if label <= 1:
                filtered_sequence_label = np.append(filtered_sequence_label, label)
                
        sequence_phase_deg = np.rad2deg(sequence_phase)

    return filtered_sequence_label,sequence_label,sequence_phase_deg

def Parity_transformation(sequence):
    p=[]
    p_m=0
    p_p=0
    for i in range(0,len(sequence)-1):
        if sequence[i]==sequence[i+1]:
            p.append(1)
            p_p+=1
        elif sequence[i]!=sequence[i+1]:
            p.append(-1)
            p_m+=1
    # print('odd events=',p_m)
    # print('even events=',p_p)
    return np.array(p)


def Three_states_single_shot_fit_analysis(data:dict, T1:float,tau:float,f01:float): #Input processed data
    Ig_data,Qg_data = np.array(data['g'][0]), np.array(data['g'][1]) 
    Ie_data,Qe_data = np.array(data['e'][0]), np.array(data['e'][1])
    If_data,Qf_data = np.array(data['f'][0]), np.array(data['f'][1])

    g_data = np.vstack((Ig_data, Qg_data))
    e_data = np.vstack((Ie_data, Qe_data))
    f_data = np.vstack((If_data, Qf_data))

    if len(Ig_data)<10000:
        bins=51
    elif 10000<len(Ig_data)<20000:
        bins=101
    else:
        bins=201

    g_predict= Single_shot_ref_fit_analysis(data['g'])['fit_pack']
    Ig_guess, Qg_guess, sig_guess= g_predict[0],g_predict[1],g_predict[2]
    I_mixdata, Q_mixdata= np.hstack((Ig_data, Ie_data, If_data)), np.hstack((Qg_data, Qe_data, Qf_data))
    I_=np.linspace(I_mixdata.min(),I_mixdata.max(),bins)
    Q_=np.linspace(Q_mixdata.min(),Q_mixdata.max(),bins)
    X,Y= np.meshgrid(I_,Q_)

    #Finding the position of peaks of excited and second excited state
    hist_Ie, edges_Ie= np.histogram(Ie_data, bins=bins, density=True)
    hist_Qe, edges_Qe= np.histogram(Qe_data, bins=bins, density=True)
    hist_If, edges_If= np.histogram(If_data, bins=bins, density=True)
    hist_Qf, edges_Qf= np.histogram(Qf_data, bins=bins, density=True)
    Ie_guess_idx, Qe_guess_idx = np.argmax(hist_Ie),np.argmax(hist_Qe)
    Ie_guess, Qe_guess= edges_Ie[Ie_guess_idx],edges_Qe[Qe_guess_idx]
    If_guess_idx, Qf_guess_idx = np.argmax(hist_If),np.argmax(hist_Qf)
    If_guess, Qf_guess= edges_If[If_guess_idx],edges_Qf[Qf_guess_idx]

    hist, xedges, yedges = np.histogram2d(I_mixdata, Q_mixdata, bins=(bins,bins), density=True)
    
    #Parameter ini-guess
    Ig    = Parameter(name = 'Ig', value = Ig_guess, min = Ig_guess - 1*sig_guess, max = Ig_guess + 1*sig_guess)
    Qg    = Parameter(name = 'Qg', value = Qg_guess, min = Qg_guess - 1*sig_guess, max = Qg_guess + 1*sig_guess)
    Ie    = Parameter(name = 'Ie', value = Ie_guess, min = Ie_guess - 5*sig_guess, max = Ie_guess + 5*sig_guess)
    Qe    = Parameter(name = 'Qe', value = Qe_guess, min = Qe_guess - 5*sig_guess, max = Qe_guess + 5*sig_guess)
    If    = Parameter(name = 'If', value = If_guess, min = If_guess - 5*sig_guess, max = If_guess + 5*sig_guess)
    Qf    = Parameter(name = 'Qf', value = Qf_guess, min = Qf_guess - 5*sig_guess, max = Qf_guess + 5*sig_guess)
    Ag    = Parameter(name = 'Ag',value = np.max(hist), min = 0.1 * np.max(hist)) 
    Ae    = Parameter(name = 'Ae',value = np.max(hist), min = 0.1 * np.max(hist))
    Af    = Parameter(name = 'Af',value = np.max(hist), min = 0.1 * np.max(hist))
    sigma = Parameter(name = 'sigma', value = sig_guess, min = 0.01 * sig_guess, max = 3 * sig_guess)

    # mixed data fit
    def Three_gaussian_2d_func(I, Q, Ig, Qg, Ie, Qe, If, Qf, sigma, Ag, Ae, Af):
        funcGround  = Ag * np.exp(-(I - Ig)**2 / 2 / sig_guess**2) * np.exp(-(Q - Qg)**2 / 2 / sig_guess**2)
        funcExcited = Ae * np.exp(-(I - Ie)**2 / 2 / sig_guess**2) * np.exp(-(Q - Qe)**2 / 2 / sig_guess**2)
        funcSecond  = Af * np.exp(-(I - If)**2 / 2 / sig_guess**2) * np.exp(-(Q - Qf)**2 / 2 / sig_guess**2)
        return (funcGround + funcExcited + funcSecond)

    Three_gaussian_2D_Model = Model(Three_gaussian_2d_func, independent_vars=['I', 'Q'])
    result = Three_gaussian_2D_Model.fit(hist.transpose(), I = X, Q = Y, Ig = Ig, Qg = Qg, Ie = Ie, Qe = Qe, If = If, Qf = Qf, sigma = sigma, Ag = Ag, Ae = Ae, Af = Af)

    Ig_fit    = result.best_values['Ig']
    Qg_fit    = result.best_values['Qg']
    Ie_fit    = result.best_values['Ie']
    Qe_fit    = result.best_values['Qe']
    If_fit    = result.best_values['If']
    Qf_fit    = result.best_values['Qf']
    Ag_fit    = result.best_values['Ag']
    Ae_fit    = result.best_values['Ae']
    Af_fit    = result.best_values['Af']
    sigma_fit = result.best_values['sigma']

    Cg = np.array([Ig_fit, Qg_fit])
    Ce = np.array([Ie_fit, Qe_fit])
    Cf = np.array([If_fit, Qf_fit])

    # Dis and SNR
    Dge = np.linalg.norm(Cg-Ce)
    Def = np.linalg.norm(Ce-Cf)
    Dfg = np.linalg.norm(Cf-Cg)
    D = np.array([Dge, Def, Dfg])
    SNR = D/sigma_fit
    overlap = (1/2)*(1-special.erf(np.sqrt(SNR**2/8)))
    def circumcenter(A, B, C):
        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
        Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
        return (Ux, Uy)
    
    Cent = circumcenter(Cg, Ce, Cf)
    phase_g = np.angle((Cg-Cent)[0] + 1j*(Cg-Cent)[1])
    phase_e = np.angle((Ce-Cent)[0] + 1j*(Ce-Cent)[1])
    phase_f = np.angle((Cf-Cent)[0] + 1j*(Cf-Cent)[1])
    cloud_phase  = np.array([phase_g, phase_e, phase_f])

    idx_phase = np.argsort(cloud_phase)
    sorted_phase = np.sort(cloud_phase)
    thresholds = np.array(range(len(sorted_phase)), dtype = float)

    for i in range(len(sorted_phase)):
        if i < (len(sorted_phase)-1):
            thres = (sorted_phase[i] + sorted_phase[i+1]) / 2
        else:
            thres = (sorted_phase[i] + sorted_phase[0] + 2*np.pi) / 2
        thres = (thres + np.pi) % (2*np.pi) - np.pi
        thresholds[idx_phase[i]] = thres

    def fitPrepared_state(I_data, Q_data):
        hist, xedges, yedges = np.histogram2d(I_data, Q_data, bins = bins, density = True)

        Ig    = Parameter(name = 'Ig', value = Ig_fit, min = Ig_fit - 1*sigma_fit, max = Ig_fit + 1*sigma_fit)
        Qg    = Parameter(name = 'Qg', value = Qg_fit, min = Qg_fit - 1*sigma_fit, max = Qg_fit + 1*sigma_fit)
        Ie    = Parameter(name = 'Ie', value = Ie_fit, min = Ie_fit - 1*sigma_fit, max = Ie_fit + 1*sigma_fit)
        Qe    = Parameter(name = 'Qe', value = Qe_fit, min = Qe_fit - 1*sigma_fit, max = Qe_fit + 1*sigma_fit)
        If    = Parameter(name = 'If', value = If_fit, min = If_fit - 1*sigma_fit, max = If_fit + 1*sigma_fit)
        Qf    = Parameter(name = 'Qf', value = Qf_fit, min = Qf_fit - 1*sigma_fit, max = Qf_fit + 1*sigma_fit)
        Ag    = Parameter(name = 'Ag', value = np.max(hist), min = 0.05 * np.max(hist), max = 2 * np.max(hist)) 
        Ae    = Parameter(name = 'Ae', value = np.max(hist), min = 0.05 * np.max(hist), max = 2 * np.max(hist))
        Af    = Parameter(name = 'Af', value = np.max(hist), min = 0.05 * np.max(hist), max = 2 * np.max(hist))
        sigma = Parameter(name = 'sigma', value = sigma_fit, min = 0.01 * sigma_fit, max = 3 * sigma_fit)
        
        model = Model(Three_gaussian_2d_func, independent_vars=['I', 'Q'])
        result = model.fit(hist.transpose(), I = X, Q = Y, Ig = Ig, Qg = Qg, Ie = Ie, Qe = Qe, If = If, Qf = Qf, sigma = sigma, Ag = Ag, Ae = Ae, Af = Af)

        Ig_sec_fit = result.best_values['Ig']
        Qg_sec_fit = result.best_values['Qg']
        Ie_sec_fit = result.best_values['Ie']
        Qe_sec_fit = result.best_values['Qe']
        If_sec_fit = result.best_values['If']
        Qf_sec_fit = result.best_values['Qf']
        Ag_sec_fit = result.best_values['Ag']
        Ae_sec_fit = result.best_values['Ae']
        Af_sec_fit = result.best_values['Af']
        sigma_sec_fit = result.best_values['sigma']

        funcGround  = lambda y, x: Ag_sec_fit * np.exp(-(x - Ig_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qg_sec_fit)**2 / 2 / sigma_sec_fit**2)
        funcExcited = lambda y, x: Ae_sec_fit * np.exp(-(x - Ie_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qe_sec_fit)**2 / 2 / sigma_sec_fit**2)
        funcSecond  = lambda y, x: Af_sec_fit * np.exp(-(x - If_sec_fit)**2 / 2 / sigma_sec_fit**2) * np.exp(-(y - Qf_sec_fit)**2 / 2 / sigma_sec_fit**2)

        volume_g = dblquad(funcGround , np.min(I_data) - 20 * sigma_sec_fit, np.max(I_data) + 20 * sigma_sec_fit, np.min(Q_data) - 20 * sigma_sec_fit, np.max(Q_data) + 20 * sigma_sec_fit)
        volume_e = dblquad(funcExcited, np.min(I_data) - 20 * sigma_sec_fit, np.max(I_data) + 20 * sigma_sec_fit, np.min(Q_data) - 20 * sigma_sec_fit, np.max(Q_data) + 20 * sigma_sec_fit)
        volume_f = dblquad(funcSecond , np.min(I_data) - 20 * sigma_sec_fit, np.max(I_data) + 20 * sigma_sec_fit, np.min(Q_data) - 20 * sigma_sec_fit, np.max(Q_data) + 20 * sigma_sec_fit)
        volume_total = volume_g[0] + volume_e[0] + volume_f[0]
        Pig = volume_g[0] / volume_total
        Pie = volume_e[0] / volume_total
        Pif = volume_f[0] / volume_total
        return ([Pig, Pie, Pif])

    ratio_g = fitPrepared_state(Ig_data, Qg_data)
    ratio_e = fitPrepared_state(Ie_data, Qe_data)
    ratio_f = fitPrepared_state(If_data, Qf_data)
    M = np.array([ratio_g, ratio_e, ratio_f])
    F_s = 1-overlap
    F_g = ratio_g[0]
    F_e = ratio_e[1]
    F_f = ratio_f[2]
    F = (F_g + F_e + F_f)/3

    fit_pack= [Cg,Ce,Cf,Ag_fit,Ae_fit,Af_fit,sigma_fit]
    error_pack= dict(Ig=Ig_fit,Qg=Qg_fit,Ie=Ie_fit,Qe=Qe_fit,If=If_fit,Qf=Qf_fit,D=D,sigma=sigma_fit,SNR=SNR,overlap=overlap,F_s=F_s,F_g=F_g,F_e=F_e,F_f=F_f,F=F)
    return dict(IQdata=[g_data,e_data,f_data],I_fit=X, Q_fit=Y,fit_pack=fit_pack,error_pack=error_pack,M=M,Ig=Ig_fit,Qg=Qg_fit,Ie=Ie_fit,Qe=Qe_fit,If=If_fit,Qf=Qf_fit,thresholds=thresholds,Cent=Cent)

def Three_state_threshold_acquisition(data:dict): # Input analysis_result
    Ig_data, Qg_data = data['IQdata'][0][0], data['IQdata'][0][1]
    Ie_data, Qe_data = data['IQdata'][1][0], data['IQdata'][1][1]
    If_data, Qf_data = data['IQdata'][2][0], data['IQdata'][2][1]

    Cent = data["Cent"]
    thresholds = data["thresholds"]

    Ig_data = Ig_data - Cent[0]
    Qg_data = Qg_data - Cent[1]
    Ie_data = Ie_data - Cent[0]
    Qe_data = Qe_data - Cent[1]
    If_data = If_data - Cent[0]
    Qf_data = Qf_data - Cent[1]

    g_phase = np.angle(Ig_data + 1j*Qg_data)
    e_phase = np.angle(Ie_data + 1j*Qe_data)
    f_phase = np.angle(If_data + 1j*Qf_data)

    def sortingByThreshold(labels, phase, ratio_list):
        n = len(labels)
        for i in range(n):
            th1 = thresholds[i]
            th2 = thresholds[(i + 1) % n]

            if th1 < th2:
                if phase < th1 or phase > th2:
                    ratio_list[i] += 1
            else:
                if th1 > phase > th2:
                    ratio_list[i] += 1
    
    labels = [0, 1, 2]
    ratio_g = np.array([0, 0, 0])
    ratio_e = np.array([0, 0, 0])
    ratio_f = np.array([0, 0, 0])
    for phase in g_phase:
        sortingByThreshold(labels, phase, ratio_g)

    for phase in e_phase:
        sortingByThreshold(labels, phase, ratio_e)
    
    for phase in f_phase:
        sortingByThreshold(labels, phase, ratio_f)

    ratio_g = ratio_g/np.sum(ratio_g)
    ratio_e = ratio_e/np.sum(ratio_e)
    ratio_f = ratio_f/np.sum(ratio_f)
    M = np.array([ratio_g, ratio_e])

    M_count = np.array((ratio_g, ratio_e, ratio_f))
    
    return M_count
    
def SQ_threshold_acquisition(data:tuple,IQ_g:list,IQ_e:list):
    cg_I,cg_Q,ce_I,ce_Q= IQ_g[0],IQ_g[1],IQ_e[0],IQ_e[1]
    if cg_I!=ce_I or cg_Q!=ce_Q:
        I_data, Q_data= data[0],data[1]
        angle= np.angle(ce_I-cg_I+(ce_Q-cg_Q)*1j)
        ce_I_rot= rot(ce_I-cg_I,ce_Q-cg_Q,angle)[0]
        I_rot, Q_rot= rot(I_data-cg_I,Q_data-cg_Q,angle)
        thres= ce_I_rot/2
        label=[]
        for i in range(len(I_rot)):
            if I_rot[i]>thres:
                label.append(1)
            else:
                label.append(0)
    else:
        raise ValueError('SQ_threshold_acquisition needs the correct threshold information')
    return dict(ce_I_rot=ce_I_rot,I_rot=I_rot,label=label)


def data_filter(data,sigma_factor=3):
    filter_data=[]
    for i in data:
        if np.abs(i-np.median(data))< sigma_factor*np.std(data):
            filter_data.append(i)
        else:
            pass
    return np.array(filter_data)

def data_flow_filter(flow,data,sigma_factor=3):
    filter_flow,filter_data=[],[]
    for i in range(len(data)):
        if np.abs(data[i]-np.median(data))< sigma_factor*np.std(data):
            filter_flow.append(flow[i])
            filter_data.append(data[i])
        else:
            pass
    return np.array(filter_flow), np.array(filter_data)


def charge_spec_fit(x, f1, f2,f01,f_guess):
    """
    Default: f1<f2 (two Ramsey fitting frequencies)

    Fits scattered (x, f1, f2) data to two cosine functions 
    using Hough Transform for initialization and non-linear least squares fitting.

    Returns:
    - popt1, errors1: Fitted parameters and errors for first function
    - popt2, errors2: Fitted parameters and errors for second function
    """
    import scipy.optimize as opt
    f1,f2= np.array(f1)*1e-6,np.array(f2)*1e-6
    x_data = np.concatenate([x, x])
    y_data = np.concatenate([f1, f2])


    # Initialize parameters using Hough Transform
    A_lower_bound = 0.5 * abs(np.max(f1) - np.min(f1))
    A_upper_bound = 1.5 * abs(np.max(f1) - np.min(f1))
    #f_guess,phi_guess= fft_oscillation_guess(f1,x)

    C_g_lower_bound = 0.9 * f_guess
    C_g_upper_bound = 1.1 * f_guess
    f01_bar_est = np.mean(y_data)
    
    # Hough Transform-like accumulator space
    step=20
    A_vals = np.linspace(A_lower_bound, A_upper_bound, step)
    phi_vals = np.linspace(-np.pi, np.pi, step)
    C_g_vals = np.linspace(C_g_lower_bound, C_g_upper_bound, step)
    accumulator = np.zeros((len(A_vals), len(phi_vals), len(C_g_vals)))

    # Voting in (A, phi, C_g) space
    for i in range(len(x_data)):
        x_i = x_data[i]  
        y_i = y_data[i]  
        for ai, A_ in enumerate(A_vals):
            for pi, phi_ in enumerate(phi_vals):
                for ci, C_g_ in enumerate(C_g_vals):
                    predicted_y = Detuning_ng_func(x_i,A_,C_g_,phi_,f01_bar_est) 
                    if np.abs(predicted_y - y_i)/y_i < 0.008:
                        accumulator[ai, pi, ci] += 1

    # Find best (A, phi, C_g) triplet
    A1_idx, phi1_idx, Cg1_idx = np.unravel_index(
        np.argmax(accumulator), accumulator.shape
    )
    A1, phi1, C_g1 = A_vals[A1_idx], phi_vals[phi1_idx], C_g_vals[Cg1_idx]

    # Remove region around the first max
    accumulator[
        max(A1_idx - 5, 0) : min(A1_idx + 5, len(A_vals)),
        max(phi1_idx - 5, 0) : min(phi1_idx + 5, len(phi_vals)),
        max(Cg1_idx - 3, 0) : min(Cg1_idx + 3, len(C_g_vals)),
    ] = 0

    # Find second best (A, phi, C_g) triplet
    A2_idx, phi2_idx, Cg2_idx = np.unravel_index(
        np.argmax(accumulator), accumulator.shape
    )
    A2, phi2, C_g2 = A_vals[A2_idx], phi_vals[phi2_idx], C_g_vals[Cg2_idx]

    # Fit data using curve fitting
    y_fit1 = Detuning_ng_func(x_data, A1, C_g1, phi1, f01_bar_est)
    y_fit2 = Detuning_ng_func(x_data, A2, C_g2, phi2, f01_bar_est)

    final_labels = np.argmin(
        np.vstack([np.abs(y_data - y_fit1), np.abs(y_data - y_fit2)]), axis=0
    )
    
    popt1, pcov1 = opt.curve_fit(
        Detuning_ng_func,
        x_data[final_labels == 0],
        y_data[final_labels == 0],
        p0=[A1, C_g1, phi1, f01_bar_est],
        bounds=([A_lower_bound,C_g_lower_bound,-np.pi,f01_bar_est*0.8],[A_upper_bound,C_g_upper_bound, np.pi,f01_bar_est*1.2]))
    
    delta_f01= popt1[0]
    fitting=dict(Vg=x,Vg_fit=np.linspace(min(x),max(x),501), A_fit=popt1[0],C_g_fit=popt1[1],phi_fit=popt1[2],f01_bar_fit=popt1[3])
    Vg_0= popt1[2]/2/np.pi/popt1[1] 

    return dict(fitting=fitting,delta_f01=delta_f01,f01_bar=popt1[3]+f01/1e6,Vg_0=Vg_0,x_data=x_data,y_data=y_data,final_labels=final_labels)



def Flux_crosstalk_linear_fit(V_target,V_meas,fit_window_data_index):
    start,end= fit_window_data_index[0],fit_window_data_index[1]
    V_meas_w,V_target_w = V_meas[start:end],V_target[start:end]
    m_guess_value=(V_meas_w[-1]-V_meas_w[0])/(V_target_w[-1]-V_target_w[0])
    m_guess=Parameter(name='m', value=m_guess_value)
    A_guess=Parameter(name='A', value= V_meas_w[0]-m_guess_value*V_meas_w[0]) 
    result= Line_model.fit(V_meas_w,x=V_target_w,m=m_guess,A=A_guess)
    m_fit=result.best_values['m']   
    A_fit=result.best_values['A']
    para_fit= np.linspace(min(V_target),max(V_target),len(V_target)*20)
    fitting= Line_func(para_fit,m_fit,A_fit)

    return dict(para_fit=para_fit,fitting=fitting,m_fit=m_fit)


def exponential_decay(t, *params):
    """Sum of exponentials for step response fit."""
    n = len(params) // 2
    y = np.zeros_like(t)
    for i in range(n):
        A = params[2 * i]
        tau = params[2 * i + 1]
        y += A * (1 - np.exp(-t / tau))
    return y


def inverse_filter_calc(exponential, Ts=1e-9, reg=1e-1, max_gain=None):
    a_list = [np.exp(-Ts / tau) for _, tau in exponential]
    num = np.array([1.0])
    for alpha in a_list:
        num = np.convolve(num, [1.0, -alpha])
    num = num / (np.sum(num) + reg)
    if max_gain is not None and np.max(np.abs(num)) > max_gain:
        scale = max_gain / np.max(np.abs(num))
        num *= scale

    den = np.array([1.0])
    return num, den

def exponential_correction(A, tau, Ts=1e-9):
    tau_s = tau * Ts
    k1 = Ts + 2 * tau_s * (A + 1)
    k2 = Ts - 2 * tau_s * (A + 1)
    c1 = Ts + 2 * tau_s
    c2 = Ts - 2 * tau_s
    feedback_tap = k2 / k1
    feedforward_taps = np.array([c1, c2]) / k1
    return feedforward_taps, feedback_tap

def filter_calc(exponential):  # exponential list [(A,tau),...]
    b = np.zeros((2, len(exponential)))
    feedback_taps = np.zeros(len(exponential))
    for i, (A, tau) in enumerate(exponential):
        b[:, i], feedback_taps[i] = exponential_correction(A, tau)
    feedforward_taps = b[:, 0]
    for i in range(len(exponential) - 1):
        feedforward_taps = np.convolve(feedforward_taps, b[:, i + 1])
    if np.abs(max(np.abs(feedforward_taps))) >= 2:
        feedforward_taps = 2 * feedforward_taps / max(np.abs(feedforward_taps))
    return feedforward_taps, feedback_taps

def Pulse_distortion_analysis(t, detuning_raw, detuning, V_input,
                               n_exp=1,
                               Lh=None,
                               plot=True):
    t = np.asarray(t)
    detuning = np.asarray(detuning)
    V_input = np.asarray(V_input)

    if not (len(t) == len(detuning) == len(V_input)):
        raise ValueError("t, detuning, V_input must have same length.")

    t_ns = t * 1e9
    t_s = t  
    Ts = np.mean(np.diff(t_s))
    N = len(t)

    # Mask filter the t >= 0
    mask = t >= 0
    tm = t_s[mask]
    tm_ns = t_ns[mask]
    y = detuning[mask]
    V = V_input[mask]

    # Normalize step response
    baseline = np.mean(detuning[t < 0]) if np.any(t < 0) else np.mean(detuning)
    y = y - baseline
    tail_n = max(1, N // 10)
    norm_factor = np.mean(y[-tail_n:])
    if np.isclose(norm_factor, 0):
        norm_factor = np.mean(np.abs(y[-tail_n:])) + 1e-12
    step_resp = y / norm_factor

    # Fit step response with multi-exponentials
    if n_exp < 1:
        n_exp = 1
    T_total = tm[-1] - tm[0] if len(tm) > 1 else Ts
    tau_min = max(Ts, 1e-12)           
    tau_max = max(10*T_total, 1e-9)    

    p0 = []
    lower = []
    upper = []
    for i in range(n_exp):
        A_guess = (1.0 / n_exp) * (1 if i % 2 == 0 else 0.8)
        tau_guess = tau_min * ( (tau_max/tau_min) ** ( (i+1)/(n_exp+1) ) )
        p0 += [A_guess, tau_guess]
        lower += [-20.0, tau_min]
        upper += [ 20.0, tau_max]
    try:
        popt, _ = optimize.curve_fit(lambda x, *params: exponential_decay(x, *params),
                                     tm, step_resp, p0=p0, bounds=(lower, upper), maxfev=50000)
    except:
        popt, _ = optimize.curve_fit(lambda x, *params: exponential_decay(x, *params),
                                     tm, step_resp, p0=p0, maxfev=50000)

    fitted_step = exponential_decay(tm, *popt)
    fitted_step /= fitted_step[-1]  # normalize to 1 at steady-state
    impulse_response = np.diff(fitted_step, prepend=fitted_step[0])


    if Lh is None:
        Lh = N
    Lh = min(Lh, N)
    h_est_trunc = impulse_response[:Lh].copy()

    step_input = np.ones_like(t)[mask]
    without_filter = exponential_decay(tm, *popt)  

    exp_list = [(popt[i], popt[i+1]) for i in range(0, len(popt), 2)]
    fir_taps, iir_taps = inverse_filter_calc(exp_list, Ts)
    fir_taps_o, iir_taps_o = filter_calc(exp_list)
    
    predistorted_dac = lfilter(fir_taps, iir_taps, step_input)
    # max_amp = 1.5*V_input[int(len(V_input)/2)]  
    # predistorted_dac = np.clip(predistorted_dac, None, max_amp)

    with_filter = lfilter(fir_taps_o, fir_taps_o, step_input)
    with_filter /= with_filter[-10:].mean()  
    no_filter = step_resp

    if plot:
        fig, ax = plt.subplots(nrows=4, figsize=(6, 8), dpi=100)
        # Step response fit
        ax[0].plot(tm_ns, step_resp, 'o', label='Measured step', lw=3)
        ax[0].plot(tm_ns, fitted_step, '--r', label=f'Fit (n_exp={n_exp})', lw=3)
        ax[0].legend(fontsize=12)
        ax[0].set_xlim(-10, max(tm_ns))
        ax[0].set_ylim(0, 1.2)
        ax[0].set_ylabel('Normalized detuning', size=12)
        ax[0].set_title('Predistortion Verification', size=12)
        ax[0].tick_params(axis='both', which='major', labelsize=12)

        # Impulse response
        ax[1].plot(tm_ns, h_est_trunc, 'b', label='Impulse response')
        ax[1].legend(fontsize=12)
        ax[1].set_ylabel('Impulse\n response', size=12)
        ax[1].tick_params(axis='both', which='major', labelsize=12)

        # Response comparison
        ax[2].plot(tm_ns, no_filter, label='Original response', alpha=0.8, lw=4)
        ax[2].plot(tm_ns, with_filter, label='Predistorted response', lw=3)
        ax[2].plot(tm_ns, step_input, '--k', label='Ideal', lw=1.5)
        ax[2].set_ylabel('Detuning\n (seen by QPU)', size=12)
        ax[2].legend(fontsize=12)
        ax[2].tick_params(axis='both', which='major', labelsize=12)

        # DAC waveform
        ax[3].plot(tm_ns, predistorted_dac*V_input[int(len(V_input)/2)]/predistorted_dac[int(len(V_input)/2)], label='Predistorted DAC', lw=3)
        ax[3].set_xlabel('Time [ns]', size=12)
        ax[3].set_ylabel('Amplitude\n (from DAC)', size=12)
        ax[3].legend(fontsize=12)
        ax[3].tick_params(axis='both', which='major', labelsize=12)

        fig.tight_layout()
    else:
        fig=None


    return {"t": t,
            "Normalized_detuning":detuning_raw,
            "impulse_response": impulse_response,
            "h_est_trunc": h_est_trunc,
            "fitted_step": fitted_step,
            "fit_para":popt,
            "exp_list": exp_list,
            "fir_taps": fir_taps,
            "iir_taps": iir_taps,
            "predistorted_dac": predistorted_dac,
            "with_filter": with_filter},fig



def apply_predistortion(t_input, V_input, result, plot=True):
    step_input = V_input
    Ts = np.mean(np.diff(t_input))
    popt= result['fit_para']
    without_filter = exponential_decay(t_input, *popt)  
    exp_list = [(popt[i], popt[i+1]) for i in range(0, len(popt), 2)]
    fir_taps, iir_taps = inverse_filter_calc(exp_list, Ts)

    predistorted_dac = lfilter(fir_taps, iir_taps, step_input)
    predistorted_dac_nor= predistorted_dac*V_input[int(len(V_input)/2)]/predistorted_dac[int(len(V_input)/2)]

    if plot:
        fig, ax = plt.subplots(nrows=1, figsize=(6,4), dpi=200)
        ax.plot(t_input*1e9, V_input, 'k', label='Ideal pulse', lw=3)
        ax.plot(t_input*1e9,predistorted_dac_nor , 'r--', label='Predistorted pulse', lw=3)
        ax.legend(fontsize=12)
        ax.set_xlabel('Time [ns]',size=12)
        ax.set_ylabel('Amplitude\n(from DAC)', size=12)
        ax.set_title('Predistortion Verification', size=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        fig.tight_layout()

    return predistorted_dac_nor



def Random_I_X_circuit_base(N:int,Set:int):
    base= []
    for i in range(Set):
        seq = np.random.randint(0, 2, size=N)
        base.append(seq)
    
    return base

def Outlier_sim(SNR):
    output_array = []
    if (type(SNR) == int) or (type(SNR) == float):
        c1 = 0.5 * (1-np.exp(-9/2))
        c2 = (norm.cdf(SNR)-0.5) * special.erf(3/np.sqrt(2))
        func_right_cricle = lambda x, SNR: np.exp(-(SNR+3*x)**2/2) * special.erf(3/np.sqrt(2) * np.sqrt(1-x**2))
        c3, c3_error = quad(func_right_cricle, 0, 1, args = (SNR,))
        return 1 - (c1 + c2 + 3/np.sqrt(2*np.pi) * c3)
    else:
        SNR = list(SNR)
        for i in range(len(SNR)):
            delta = SNR[i]
            c1 = 0.5 * (1-np.exp(-9/2))
            c2 = (norm.cdf(delta)-0.5) * special.erf(3/np.sqrt(2))
            func_right_cricle = lambda x, delta: np.exp(-(delta+3*x)**2/2) * special.erf(3/np.sqrt(2) * np.sqrt(1-x**2))
            c3, c3_error = quad(func_right_cricle, 0, 1, args = (delta,))
            output_array.append((1 - (c1 + c2 + 3/np.sqrt(2*np.pi) * c3)))
        return np.array(output_array)

