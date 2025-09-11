import numpy as np
hbar = 1.0545718e-34
kB = 1.38e-23    

def PetoT(Pe, Wa):
    Pe = np.clip(Pe,1e-10, 1-1e-10)  
    Pg = 1 - Pe
    try:
        T = (-hbar*Wa)/(kB*np.log(Pe/Pg)) * 1000  
    except Exception as e:
        T = np.nan
    return T

def PtoV(P):
    return np.sqrt(P*1e-3*50)

def VtoN(V, coeff):
    return V**2/50*coeff

def NtoV(N, coeff):
    return np.sqrt(N/coeff*50)