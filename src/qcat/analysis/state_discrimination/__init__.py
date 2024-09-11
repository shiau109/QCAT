import numpy as np

def p01_to_Teff( p01, frequency ):
    """
    Parameters:\n
    frequency unit in Hz
    """
    n = p01/(1-2*p01)
    HDB = (1.0546/1.3806) *1e-11 # 1.0546e-34 / 1.3806e-23
    effective_T = frequency*2*np.pi*HDB/np.log(1+1/n)

    return effective_T