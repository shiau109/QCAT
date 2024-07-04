import numpy as np
from lmfit import Model
import pandas as pd

def damped_oscillation(t, amp, tau, freq, phi, offset):
    return amp * np.exp(-t/tau) * np.cos(freq *2 *np.pi *t + phi) +offset



def _resonator_decay_model():
    # Create a model from the damped_oscillation function
    model = Model(damped_oscillation)

    # Create a parameters object
    params = model.make_params(amp=0.02, tau=500, freq=0.1*np.pi, phi=0, offset=0)

    params['amp'].set(min=0.0, max=0.1) 
    params['tau'].set(min=10, max=1e3) 
    params['freq'].set(min=0.02, max=0.5) # Frequency between 0 and 10 radians per second
    params['phi'].set(min=-np.pi*2, max=0)  # Phase between -π and π
    params['offset'].set(vary=False)
    return model, params

def resonator_decay_fitting( time_resolution, data, damp_freq=None ):
    T = time_resolution   # Sampling period (s)
    L = data.shape[-1]   # Length of the signal
    t = np.linspace(0, L-1, L) * T  # Time vector
    model, params = _resonator_decay_model()
    if damp_freq != None:
        params['freq'].set( damp_freq)#, vary=False)
    params['amp'].set(np.max(data)*1.2)#, vary=False)
    max_time = t[np.argmax(data)]
    max_period = max_time*damp_freq
    decimal_part = max_period - int(max_period)
    params['phi'].set(-decimal_part*np.pi*2)#, vary=False)

    result = model.fit(data, params, t=t)
    return result

def resonator_freqResponse_decay( data:np.ndarray, time_resolution:float, damp_freq:float=None ):
    """
    Parameters:\n
    data: numpy array with shape (M,N)\n
    damp_freq: unit in GHz \n
    time_resolution: unit in ns\n

    """
    freq_num = data.shape[0]
    Fs = 1/time_resolution 
    T = time_resolution   # Sampling period (s)
    L = data.shape[1]   # Length of the signal
    t = np.linspace(0, L-1, L) * T  # Time vector

    if damp_freq == None:
        fft_signal = data[int(freq_num/2)]
        fft_signal = fft_signal-np.mean(fft_signal)
        damp_freq = freq_guess(Fs, fft_signal)
        print(f"Initial guess {damp_freq:.2f} GHz")

    all_result = []
    for i in range(data.shape[0]):
        
        fit_data = data[i]
        all_result.append( resonator_decay_fitting(time_resolution,fit_data,damp_freq).params.valuesdict() )

    df_result = pd.DataFrame(all_result)
    return df_result

def freq_guess( Fs, data ):
    L = data.shape[-1]
    fft_values = np.fft.fft(data)
    P2 = np.abs(fft_values/L)
    P1 = P2[:L//2+1]
    P1[1:-1] = 2*P1[1:-1]
    frequencies = Fs * np.arange(0, (L/2)+1) / L
    index_of_peak = np.argmax(P1)
    dominant_frequency = frequencies[index_of_peak]
    return dominant_frequency 
