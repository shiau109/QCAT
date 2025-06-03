import numpy as np
from lmfit import Model
from lmfit.model import ModelResult


def exp_cos(n, amp, tau, period, offset):
    return amp * np.exp(-n/tau) * np.cos(2 * np.pi * n/period)+offset
def gate_amp_model():
    model = Model(exp_cos)
    
    # Create a parameters object
    params = model.make_params(amp=0.02, tau=500, period = 10, offset=0)
    
    return model, params

def gate_amp_fitting( seq, data )->ModelResult:
    
    model, params = gate_amp_model()
    params['amp'].set(guess_amp( data ))
    params['offset'].set(guess_offset( data ))
    params['tau'].set(guess_tau(seq,data), min=0, max=seq[-1]) 
    params['period'].set(guess_period(data))
    
    result = model.fit(data, params, n=seq)
    return result

def guess_tau( seq, data ):
    # Calculate absolute differences between array elements and target value
    ooe = guess_amp(data)/np.e +guess_offset(data)
    absolute_diff = np.abs(data - ooe)
    
    # Find index of the minimum absolute difference
    nearest_index = np.argmin(absolute_diff)
    return seq[nearest_index] 

def guess_amp( data ):
    amp = (data[0]-data[-1])
    return amp 

def guess_offset( data ):
    return data[-1]

def guess_period(data):
    max_val_idx = np.argmax(data)
    min_val_idx = np.argmin(data)
    return np.abs(2*(max_val_idx-min_val_idx))
    