from xarray import DataArray
from lmfit import Model,Parameter
from lmfit.model import ModelResult
from numpy import ndarray, fft, linspace
from numpy import cos, abs, exp, max, min, mean, argmax
from numpy import pi

from .function_fitting import FunctionFitting


class FitDampedOscillation(FunctionFitting):
    """
    Input dataarray is 
    
    """
    def __init__(self, data:DataArray=None):

        self._data_parser(data)
        self.model = Model(self.model_function)
        self.params = None
    def _data_parser( self, data:DataArray ):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")

        self.y = data.values
        self.x = data.coords["x"].values

    def model_function( self, x, a, tau, f, phi, c ):
        return a*exp(-x/tau)*cos(2*pi*f*x+phi)+c

    def guess( self ):
    # fft function made by Wei-En

        y = self.y
        t = self.x
        
        dt = t[1] - t[0]
        max_val = max(y)
        min_val = min(y)
        # frequency guess
        amp = fft.fft(y)[: len(y) // 2] #use positive frequency
        freq = fft.fftfreq(len(y), dt)[: len(amp)]
        amp[0] = 0  # Remove DC part 
        power = abs(amp)
        f_guess = abs(freq[argmax(power)])
        f_guess_dict = dict(value=f_guess, min=-1/dt/2, max=1/dt/2 )

        # phi_guess = 2 * pi - (2 * pi * t[y == max(y)] * f_guess)[0]
        phi_guess_dict = dict(value=0, min=-pi, max=pi )

        #c guess
        c_guess_dict = dict(value=mean(y), min=min_val, max=max_val)


        #amp guess
        a_guess = (max_val-min_val)/2
        a_guess_dict = dict(value=a_guess, min=0, max=a_guess*2)


        #tau guess
        tau_guess_dict = dict(value=t[-1]/2, min=0, max=t[-1]*2)
        self.params = self.model.make_params( 
                    a=a_guess_dict,  
                    tau=tau_guess_dict,
                    f=f_guess_dict,
                    phi=phi_guess_dict,
                    c=c_guess_dict)

        return self.params
    def fit(self, data:DataArray=None)-> ModelResult:
        
        
        if data is not None:
            self._data_parser(data)


        if self.params is None:
            self.guess()

        result = self.model.fit( self.y, self.params, x=self.x )
        self.result = result
        return result

