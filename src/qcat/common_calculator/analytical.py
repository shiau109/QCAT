import numpy as np

def Relax_cal(inte_i,tf,T1):
    tau= tf - inte_i
    Relax= 1+(T1/tau)*(np.exp(-(inte_i+tau)/T1)-np.exp(-inte_i/T1))
    return Relax

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

def resonator_freq_response(para,P_in):
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
    f_g= para['f_eff_bare'] + X*n_g/np.pi
    f_e= para['f_eff_bare'] - X*n_e/np.pi
    return f_g,f_e