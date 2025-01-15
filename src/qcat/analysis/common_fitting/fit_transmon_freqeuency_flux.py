from qcat.analysis.function_fitting import FunctionFitting
from xarray import DataArray
from lmfit import Model,Parameter
from lmfit.model import ModelResult
import numpy as np
from qcat.analysis.function_fitting import FunctionFitting


class FitTransmonFrequencyFlux(FunctionFitting):
    """
    Input datarray
    value unit in GHz
    coord name is 'x'

    
    """
    def __init__(self, data:DataArray=None):
        self.Ec_design = 0.2
        self._data_parser(data)
        self.params = None

    def _data_parser( self, inputdata:DataArray ):
        if not isinstance(inputdata, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")

        self.freq = self.values
        self.x = inputdata.coords["x"] # us


    def model_function( self, x, Ec, offset, period, Ej_sum, d):

        quan_flux = self.iv2quantFlux( x, offset, period  )
        Ej_eff = self.effective_Ej( quan_flux, Ej_sum, d)
        return self.tramsmon_frequency(Ej_eff,Ec)
    def tramsmon_frequency( self, Ej, Ec):
        return np.sqrt(8*Ec*Ej)-Ec

    def effective_Ej( self, quan_flux, Ej_sum, d ):
        return Ej_sum*np.sqrt(np.cos( np.pi*quan_flux )**2+(d*np.sin( np.pi*quan_flux ))**2)

    def iv2quantFlux( self, iv, offset, period ):
        return (iv-offset)/period
    
    def guess( self ):

        y = self.freq
        x = self.x

        max_val = max(y)

        min_x = min(x)
        max_x = max(x)
        x_range = max_x -min_x

        max_index = np.argmax(y)
        # Get the corresponding x-value
        max_y_x_pos = x[max_index]

        # Ec guess
        Ec_guess = self.Ec_design
        Ec_guess_dict = dict(value=Ec_guess, min=0.01, max=1 )

        #offset guess
        offset_guess_dict = dict(value=max_y_x_pos, min=min_x -x_range*2, max=max_x +x_range*2)

        #period guess
        period_guess_dict = dict(value=x_range, min=0, max=x_range*5)
        
        #Ej_sum guess
        Ej_guess = (max_val+Ec_guess)**2/8./Ec_guess
        Ej_sum_guess_dict = dict(value=Ej_guess, min=0, max=x_range*5)

        #d guess
        d_guess_dict = dict(value=0, min=0, max=1)

        params = self.model.make_params( 
                    Ec=Ec_guess_dict,  
                    offset=offset_guess_dict,
                    period=period_guess_dict,
                    Ej_sum=Ej_sum_guess_dict,
                    d=d_guess_dict)
        self.params = params

        return params
    def fit(self, data:DataArray=None)-> ModelResult:
        
        
        if data is not None:
            self._data_parser(data)


        self.model = Model(self.model_function)
        if self.params is None: self.guess()

        result = self.model.fit( self.freq, self.params, x=self.x )
        self.result = result
        return result
    
if __name__ == '__main__':

    import xarray as xr
    data = xr.DataArray(
        data= np.array(6.0, 4.73),  # Data (NumPy array or similar)
        dims=["x"],     # Dimension names
        coords={                    # Coordinates (mapping of dimension names to arrays)
            "x": [0.12,0.21],
        },
        name="example_data",        # Optional: Name of the DataArray
        attrs={                     # Optional: Additional metadata
            "description": "Random data for testing",
            "units": "unknown"
        }
    )

    my_fit = FitTransmonFrequencyFlux(data)
    paras = my_fit.guess()
    paras.add("Ec",value=0.15,vary=False)
    paras.add("offset",value=-0.02,vary=False)
    paras.add("d",value=0,vary=False)