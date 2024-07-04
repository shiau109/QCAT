import numpy as np



def fft_chavron( time, data:np.ndarray ):

    dt = time[1]-time[0]
    time_len = time.shape[-1]

    shifted_data = data-np.mean(data, axis=1, keepdims=True)
    # Generate the frequency array for the FFT results
    frequencies = np.fft.fftfreq(time_len, d=dt)[:time_len//2]
    # Optionally, sort frequencies and corresponding FFT result
    # sorted_freq = np.fft.fftshift(frequencies)[:time_len//2+1]  # Shift zero freq to center    
    fft_cols = np.fft.fft(shifted_data, axis=1)[:,:time_len//2] 
    magnitude_spectrum = np.abs(fft_cols)  # Taking the magnitude of the FFT results for the first channel


    return magnitude_spectrum, frequencies

def get_main_tone( freq, magnitude ):
    # List of indices
    max_indices = np.argmax(magnitude, axis=1)
    # Retrieving values
    main_freq = freq[max_indices]
    main_mag = magnitude[np.arange(magnitude.shape[0]),max_indices]
    return main_freq, main_mag