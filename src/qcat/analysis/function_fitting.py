
from abc import ABC, abstractmethod

from lmfit import Model, Parameter
from lmfit.model import ModelResult
from numpy import fft, ndarray, pi, mean, linspace, array
from xarray import Dataset

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


class CosineFitting( FunctionFitting ):
    def __init__( self ):
        super().__init__()
        self.cosine_model = Model(self.model_function)
        self.x:ndarray = []
        self.y:ndarray = []
        self.__amp:float = None
        self.__f:float = None
        self.__phi:float = None
        self.__offset:float = None
        self.__fit_results:ModelResult = None

    @property
    def fit_results( self )->ModelResult:
        return self.__fit_results
    @property
    def pit_paras( self )->tuple:
        return (self.__amp, self.__f, self.__phi, self.__offset)
    

    def model( self, x:ndarray|float ):
        return self.model_function(x, *self.pit_paras())

    def model_function(self,x, A,f,phi,offset):
        from numpy import cos
        return A*cos(2*pi*f*x+phi)+offset
    
    def guess(self, data:ndarray, x:ndarray):
        self.x = x
        self.y = data

        f_guess, self.phi_guess= fft_oscillation_guess(data,x)
        self.f_guess= Parameter(name='f', value=f_guess , min=0, max=f_guess*10)
        self.A_guess = abs(max(data)-min(data))/2
        self.offset_guess = mean(data)

    def fit(self, data:ndarray=None, x:ndarray=None):
        """ if `data` or `x` was given, use them to fitting. Otherwise, fitting by the data and x from `self.guess()` """
        if data is not None:
            if data.reshape(-1).shape[0] != 0:
                self.y = data
        if x is not None:
            if x.reshape(-1).shape[0] != 0:
                self.x = x
        
        self.__fit_results = self.cosine_model.fit(self.y,x=self.x, A=self.A_guess,f=self.f_guess,phi=self.phi_guess,offset=self.offset_guess)
        self.__amp = self.fit_results.best_values['A']
        self.__f = self.fit_results.best_values['f']
        self.__phi = self.fit_results.best_values['phase']
        self.__offset = self.fit_results.best_values['offset']

    def get_fit_pack( self )->Dataset:
        extended_x = linspace(self.x.min(),self.x.max(),50*array(self.x).shape[0])
        fitting_curve = self.fitting_curve(extended_x)
        fit_pack_ds = Dataset(data_vars=dict(data=(['freeDu'],self.y),fitting=(['para_fit'],fitting_curve)),
                       coords=dict(freeDu=(['freeDu'],self.x),para_fit=(['para_fit'],extended_x)),
                       attrs=dict(exper="xyfcali",f=self.__f,phase=self.__phi,coefs=list(self.pit_paras())))

        return fit_pack_ds