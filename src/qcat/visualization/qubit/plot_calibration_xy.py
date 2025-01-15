

from qcat.visualization.painter import Painter

import numpy as np
import matplotlib.pyplot as plt
from xarray import DataArray

from matplotlib.cm import get_cmap
class PainterXYAmp( Painter ):
    """
    DataArray with coords name 
    """
    def __init__(self, plot_data:DataArray, fit_result=None, correction_value=None):

        self.name = plot_data.name

        self._data_parser(plot_data)
        self.fit_result = fit_result
        self.correction_value = correction_value
    def _data_parser( self, data:list[DataArray] ):

        for one_trace_data in data:
            if not isinstance(one_trace_data, DataArray):
                raise ValueError(f"Input data must be an list of xarray.DataArray. current type {type(one_trace_data)}")
            check_coords = ["amplitude"]
            for coord_name in check_coords:
                if coord_name not in one_trace_data.coords:
                    raise ValueError(f"No coord name called {coord_name} in the input array")
            
        self.plot_data = data

        return self.plot_data
    
    def plot( self ):


        fig, ax = plt.subplots(1)

        ax.set_title(f"{self.name} xy calibration")
        ax.set_xlabel("amplitude ratio")
        ax.set_ylabel(f"signal")

        # Generate a list of 10 colors
        color_list = ['red', 'blue', 'green', 'orange', 'purple']
        for i, plot_data in enumerate(self.plot_data):
            x = plot_data.coords["amplitude"]
            y = plot_data.values
            sequence = plot_data.coords["sequence"].values
            ax.scatter( x, y, s=1, color=color_list[i], label=sequence)
            if self.fit_result[i] is not None:
                ax.plot( x, self.fit_result[i].best_fit, color=color_list[i])
            if self.correction_value[i] is not None:

                ax.text(0.5, 0.8-i*0.05, f"{sequence}: {self.correction_value[i]:.3f}", fontsize=10, transform=ax.transAxes)
                ax.axvline(self.correction_value[i], color=color_list[i], linestyle='--', linewidth=1)     
        ax.legend()
        plt.tight_layout()
        
        return fig