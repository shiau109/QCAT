from qcat.analysis.base import QCATAna
from qcat.analysis.function_fitting.fit_powerlaw_base import FitBasePowerLaw
from qcat.visualization.qubit.plot_cliffordRB import PainterCliffordRB
from xarray import Dataset
import numpy as np

class Clifford1QRB( QCATAna ):
    """
    Input dataarray  

    """
    def __init__( self, data:Dataset, label=None ):

        self._import_data(data)
        self.label = label
    def _import_data( self, data ):

        if not isinstance(data, Dataset):
            raise ValueError(f"Input data must be an xarray.Dataset.\n current type:{type(data)}")
        check_coords = ["gate_num"]
        for coord_name in check_coords:
            if coord_name not in data.coords:
                raise ValueError(f"No coord name called {coord_name} in the Dataset")
        
        check_vars = ["p0","std"]
        for var_name in check_vars:
            if var_name not in list(data.data_vars):
                raise ValueError(f"No var name called {var_name} in the Dataset")
                    
        self.data = data

    def _start_analysis( self, plot=True ):


        fit_data = self.data.rename({"gate_num": "x"})
        # print(fit_data["p0"])
        fit_base = FitBasePowerLaw( fit_data["p0"] )

        fit_base.fit()

        x = fit_data["x"].values
        one_minus_p = 1 - fit_base.result.params["base"].value
        r_c = one_minus_p * (1 - 1 / 2**1)
        r_g = r_c / 1.875  # 1.875 is the average number of gates in clifford operation
        r_c_std = fit_base.result.params['base'].stderr * (1 - 1 / 2**1)
        r_g_std = r_c_std / 1.875
        # Get the x-value of the minimum
        self.fidelity = {
            "native_gate_infidelity": r_g,
            "native_gate_infidelity_err": r_g_std,
            "Clifford_gate_infidelity": r_c,
            "fitting_base":fit_base.result.params["base"].value,
            "fitting_base_stderr": fit_base.result.params['base'].stderr
        }

        # Plot
        if plot:
            painter = PainterCliffordRB( self.data, fit_base.result, self.fidelity, self.label )
            self.fig = painter.plot()


    def _export_result( self, *args, **kwargs ):
        pass

if __name__ == "__main__":

    import xarray as xr
    import matplotlib.pyplot as plt
    dataset = xr.open_dataset(r"d:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\20250112_130414_1QRB_1_-0.0\1QRB.nc")
    # Plot

    sq_data = dataset["q1_ro"]
    # change format (temp)
    data = xr.Dataset(
        {
            "p0": (("gate_num",), -sq_data.sel(mixer="val").values),
            "std": (("gate_num",), sq_data.sel(mixer="err").values),
        },
        coords={"gate_num": sq_data.coords["x"].values },
    )
    label = sq_data.name
    # Analysis
    my_ana = Clifford1QRB( data, label )
    my_ana._start_analysis()


    plt.show()
