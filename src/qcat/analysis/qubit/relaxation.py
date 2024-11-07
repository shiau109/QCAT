import numpy as np
from lmfit import Model, Parameters
from lmfit.models import ExponentialModel
from lmfit.model import ModelResult

import pandas as pd
import typing
def exp_decay(t, amp, tau, offset):
    return amp * np.exp(-t/tau) +offset



def _qubit_relaxation_model():
    # Create a model from the damped_oscillation function
    model = Model(exp_decay)

    # Create a parameters object
    params = model.make_params(amp=0.02, tau=500, offset=0)

    # params['amp'].set(min=0.0, max=1.0) 
    # params['tau'].set(min=0, max=1e6) 

    # params['offset'].set(min=-1.0, max=1.0)
    return model, params

def qubit_relaxation_fitting( time, data )->ModelResult:

    model, params = _qubit_relaxation_model()
    # max_val = np.max(data)
    # min_val = np.min(data)
    params['amp'].set(data[0]-data[-1])#, vary=False)
    params['offset'].set(data[-1])#averaging(data,6))
    params['tau'].set(guess_tau(time,data), min=0, max=time[-1]) 
    
    result = model.fit(data, params, t=time)
    return result

def averaging(data, n_avg):
    value = 0
    for i in range(n_avg):
        value += data[-i]
        if n_avg != 0:
            return i/n_avg
        else: return 1

def qubit_relaxation_statistic( time, data:np.ndarray ):
    """
    Parameters:\n
    time: numpy array\n
    data: numpy array with shape (M,N)\n

    """
    all_result = []
    for i in range(data.shape[0]):
        
        fit_data = data[i]
        all_result.append( qubit_relaxation_fitting(time,fit_data).params.valuesdict() )

    df_result = pd.DataFrame(all_result)
    return df_result

def guess_tau( time, data ):
    # Calculate absolute differences between array elements and target value
    ooe = guess_amp(data)/np.e +guess_offset(data)
    absolute_diff = np.abs(data - ooe)
    
    # Find index of the minimum absolute difference
    nearest_index = np.argmin(absolute_diff)
    return time[nearest_index] 

def guess_amp( data ):
    amp = (data[0]-data[-1])
    return amp 

def guess_offset( data ):
    return data[-1]