
from abc import ABC, abstractmethod


from numpy import fft, ndarray, pi


# fft function made by Wei-En
def fft_oscillation_guess(data:ndarray, t:ndarray):
    amp = fft.fft(data)[: len(data) // 2] #use positive frequency
    freq = fft.fftfreq(len(data), t[1] - t[0])[: len(amp)]
    amp[0] = 0  # Remove DC part 
    power = abs(amp)
    f_guess = abs(freq[power == max(power)][0])
    phase_guess = 2 * pi - (2 * pi * t[data == max(data)] * f_guess)[0]
    return f_guess, phase_guess

class FunctionFitting( ABC ):
    def __init__():
        pass

    @abstractmethod
    def model_function( **arg ):
        pass

    @abstractmethod
    def guess():
        pass

    @abstractmethod
    def fit():
        pass
    
    def fitting_curve( self, x ):
        return self.model(x)


