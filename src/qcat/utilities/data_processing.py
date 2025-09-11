import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def rot(I,Q,angle):
    sin=np.sin(angle)
    cos=np.cos(angle)
    return I*cos+Q*sin, -I*sin+Q*cos

def IQ_data_dis(I_data:np.ndarray,Q_data:np.ndarray,ref_I:float,ref_Q:float):
    Dis= np.sqrt((I_data-ref_I)**2+(Q_data-ref_Q)**2)
    return Dis   
