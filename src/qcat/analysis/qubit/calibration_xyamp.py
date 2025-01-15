from qcat.analysis.base import QCATAna
from qcat.analysis.common_fitting.fit_cosine import FitCosine
from qcat.visualization.qubit.plot_calibration_xy import PainterXYAmp
from xarray import DataArray
import numpy as np

class CalibrationXYAmp( QCATAna ):
    """
    Input dataarray  

    """
    def __init__( self, data:DataArray ):

        self._import_data(data)

    def _import_data( self, data ):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")
        
        check_coords = ["amplitude"]
        for coord_name in check_coords:
            if coord_name not in data.coords:
                raise ValueError(f"No coord name called {coord_name} in the input array")
                
        self.data = data



    def _start_analysis( self ):


        plot_data = []
        fit_results = []
        correction_values = []
        sequence_names = ["x180","x90"]
        for sequence in sequence_names:
            fit_data = self.data.sel(sequence=sequence).rename({"amplitude": "x"})
            plot_data.append(fit_data)
            fit_cos = FitCosine( fit_data )
            # guess_para.add("f",value=0.5)
            # guess_para.add("phi",value=0)
            # fit_cos.params = guess_para
            fit_cos.fit()
            fit_results.append(fit_cos.result)

            x = fit_data["x"].values
            fitted_params = fit_cos.result.params
            # Wrapper function for minimization
            def wrapped_func(x):
                return fit_cos.model_function(x, a=fitted_params["a"].value,f=fitted_params["f"].value, phi=fitted_params["phi"].value, c=fitted_params["c"].value)
            # Perform minimization
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(wrapped_func, bounds=(np.min(x), np.max(x)), method="bounded")

            # Get the x-value of the minimum
            correction_value = result.x
            correction_values.append(correction_value)


        # Plot
        painter = PainterXYAmp( self.data, fit_results, correction_values )
        self.fig = painter.plot()


    def _export_result( self, *args, **kwargs ):
        pass

    # def run( self, save_path:str = None ):
        
    #     self.raw_data = self._import_data()

    #     if self.raw_data is not None:
    #         self.result = self._start_analysis()

    #         if save_path is not None:
    #             self.save_path = save_path
    #             self._export_result()

    #         return self.result

    #     else:
    #         print("Import data failed.")

    

    
