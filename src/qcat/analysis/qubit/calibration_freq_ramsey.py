from qcat.analysis.base import QCATAna
from qcat.analysis.common_fitting.fit_damped_oscillation import FitDampedOscillation
from xarray import DataArray
from qcat.visualization.qubit.plot_ramsey import PainterT2Ramsey
import numpy as np

class CalibrationRamsey( QCATAna ):
    """
    Input dataarray  

    """
    def __init__( self, data:DataArray ):

        self._import_data(data)

    def _import_data( self, data ):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")
        
        check_coords = ["time"]
        for coord_name in check_coords:
            if coord_name not in data.coords:
                raise ValueError(f"No coord name called {coord_name} in the input array")
        
        self.data = data
        self.data = self.data.assign_coords(time=self.data.coords["time"].values/1000.)


    def _start_analysis( self ):
        fit_ramsey = FitDampedOscillation( self.data.rename({"time": "x"}) )
        guess_para = fit_ramsey.guess()
        init_phi = np.pi/2
        guess_para.add("phi",min=init_phi*0.9, max=init_phi*1.1)
        fit_ramsey.fit()


        # Plot
        painter = PainterT2Ramsey( self.data, fit_ramsey.result )
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

    

    
