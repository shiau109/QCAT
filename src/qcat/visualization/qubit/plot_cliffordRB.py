

from qcat.visualization.painter import Painter

import numpy as np
import matplotlib.pyplot as plt
from xarray import Dataset

class PainterCliffordRB( Painter ):
    """
    DataArray with coords name 
    """
    def __init__(self, plot_data:Dataset, fit_result=None, fidelity=None, label=None):

        self.label = label

        self._data_parser(plot_data)
        self.fit_result = fit_result
        self.fidelity = fidelity

    def _data_parser( self, data:Dataset ):

        if not isinstance(data, Dataset):
            raise ValueError(f"Input data must be an xarray.DataArray.\n current type:{type(data)}")
        check_coords = ["gate_num"]
        for coord_name in check_coords:
            if coord_name not in data.coords:
                raise ValueError(f"No coord name called {coord_name} in the Dataset")
        
        check_vars = ["p0","std"]
        for var_name in check_vars:
            if var_name not in list(data.data_vars):
                raise ValueError(f"No var name called {var_name} in the Dataset")
                    
        self.plot_data = data

        return self.plot_data
    
    def plot( self ):
        
        x = self.plot_data.coords["gate_num"].values
        y = self.plot_data["p0"].values
        yerr = self.plot_data["std"].values
        if self.label is not None: title = self.label
        else: title = ""

        fig, ax = plt.subplots(1)

        ax.errorbar(x, y, yerr=yerr, marker=".")
        ax.set_title(f"{title} Single qubit RB")
        # ax.set_xlabel("Number of Clifford gates")
        # ax.set_ylabel("Sequence Fidelity")
        if self.fit_result is not None:
            ax.plot( x, self.fit_result.best_fit,"o", label="data",markersize=1,linestyle="--", linewidth=2)
            # ax.set_xscale('log')
            p = self.fit_result.params["base"].value
            r_c = self.fidelity["Clifford_gate_fidelity"]
            r_g = self.fidelity["native_gate_fidelity"]
            ax.text(0.04, 
                    0.96, 
                    f"Base: p = {np.format_float_scientific(p, precision=2)}\n"#A+-{stdevs[1]:.2}\n"
                    f"Error rate: 1-p = {np.format_float_scientific(1-p, precision=2)}\n"#+-{stdevs[1]:.2}\n"
                    f"Clifford set infidelity: r_c = {np.format_float_scientific(r_c, precision=2)}\n"#+-{r_c_std:.2}\n"
                    f"Gate infidelity: r_g = {np.format_float_scientific(r_g, precision=2)}",#+-{r_g_std:.2}", 
                    fontsize=9, 
                    color="black",
                    ha='left', 
                    va='top',
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.5))
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel("Population $P_0$", fontsize=20)
        ax.set_xlabel("Number of Clifford gates", fontsize=20)
        plt.tight_layout()
        
        return fig