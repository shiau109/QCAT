import numpy as np
from lmfit import Model
from lmfit.model import ModelResult


def exp_gaussian_cos(t, T1, amp, t_phi, detune, offset):
    return amp * (np.exp(-t/(2*T1)) * np.exp(-t**2/t_phi**2) * np.cos(2 * np.pi * t * detune)) +offset

def _qubit_ramsey_model():
    # Create a model from the damped_oscillation function
    model = Model(exp_gaussian_cos)

    # Create a parameters object
    params = model.make_params( amp=0.02, t_phi=10, detune=1, offset=0)

    return model, params

def qubit_ramsey_fitting( time, T1, data )->ModelResult:

    model, params = _qubit_ramsey_model()
    params['amp'].set(guess_amp( data ), min=0, max=data[0]-max(data))
    params['t_phi'].set(guess_t_phi(time,T1,data), min=0, max=time[-1])
    params['detune'].set(guess_detune( time, data ), min=0)
    params['offset'].set(guess_offset( data ), min=data[-1]-1.0, max=data[-1]+1.0)
    
    result = model.fit(data, params, t=time, T1=T1)
    return result

def guess_amp( data ):
    amp = (data[0]-data[-1])
    return amp 

def guess_t_phi(time, T1, data ): # Unfinish
    return T1

def guess_detune(time, data):
    max_val_idx = np.argmax(data)
    min_val_idx = np.argmin(data)
    max_val_time = time[max_val_idx]
    min_val_time = time[min_val_idx]
    return 1 / np.abs(2*(max_val_time-min_val_time)) # Mhz

def guess_offset( data ):
    return data[-1]