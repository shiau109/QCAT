from xarray import DataArray
from lmfit import Model,Parameter
from lmfit.model import ModelResult
from numpy import ndarray, fft, linspace
from numpy import cos, abs, exp, max, min, mean, argmax
from numpy import pi

from .function_fitting import FunctionFitting


class FitBasePowerLaw(FunctionFitting):
    """
    Input dataarray is 
    
    """
    def __init__(self, data:DataArray=None):

        self._data_parser(data)

        self.model = Model(self.model_function)
        
    def _data_parser( self, data:DataArray ):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")
        self.y = data.values
        self.x = data.coords["x"].values

    def model_function( self, x, a, base, c ):
        return a * (base**x) + c
    
    def guess( self ):
        x = self.x
        y = self.y

        max_y = max(y)
        min_y = min(y)
        y_range = (max_y-min_y)

        #base guess
        base_guess_dict = dict(value=0.9, min=0, max=1)
    
        #c guess
        c_guess_dict = dict(value=mean(y), min=min_y-y_range*2, max=max_y)

        #amp guess
        a_guess = y_range
        a_guess_dict = dict(value=a_guess, min=-a_guess*2, max=a_guess*2)


        params = self.model.make_params( 
                    a=a_guess_dict,  
                    base=base_guess_dict,
                    c=c_guess_dict)
        self.params = params

        return params
    def fit(self, data:DataArray=None)-> ModelResult:
        
        
        if data is not None:
            self._data_parser(data)


        
        self.guess()

        result = self.model.fit( self.y, self.params, x=self.x )
        self.result = result
        return result

