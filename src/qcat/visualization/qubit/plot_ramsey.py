

from qcat.visualization.painter import Painter

import numpy as np
import matplotlib.pyplot as plt
from xarray import DataArray
class PainterT2Ramsey( Painter ):
    
    def __init__(self, plot_data, fit_result=None):


        self._data_parser(plot_data)
        self.name = plot_data.name

        self.fit_result = fit_result

    def _data_parser( self, data:DataArray ):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")

        self.plot_data = data

        return self.plot_data
    
    def plot( self ):
        time = self.plot_data.coords["time"].values
        y = self.plot_data.values
        title = self.name
        fig, ax = plt.subplots(1)

        ax.set_title(f"{title} T2 Ramsey I data")
        ax.set_xlabel("Wait time (us)")
        ax.set_ylabel(f"voltage (mV)")
        ax.plot( time, y,"o", label="data",markersize=1)
        if self.fit_result is not None:
            ax.plot( time, self.fit_result.best_fit, label="fit")
            f = self.fit_result.params['f'].value
            phi = self.fit_result.params['phi'].value
            ax.text(0.1, 0.9, f"Driving Detuned: {f:.3f}", fontsize=10, transform=ax.transAxes)  
            ax.text(0.1, 0.8, f"Phase check: {phi:.3f} (should be pi/2)", fontsize=10, transform=ax.transAxes)  


        plt.tight_layout()
        
        return fig