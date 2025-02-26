from qcat.analysis.base import QCATAna
from qcat.analysis.function_fitting.fit_damped_oscillation import FitDampedOscillation
from xarray import DataArray
from qcat.visualization.qubit.plot_ramsey import PainterT2Ramsey
import numpy as np
import matplotlib.pyplot as plt
class ZZInteraction( QCATAna ):
    """
    Input dataarray  

    """
    def __init__( self, data:DataArray ):

        self._import_data(data)

    def _import_data( self, data ):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")
        
        check_coords = ["coupler","time"]
        for coord_name in check_coords:
            if coord_name not in data.coords:
                raise ValueError(f"No coord name called {coord_name} in the input array")
        
        self.data = data
        self.data = self.data.assign_coords(time=self.data.coords["time"].values/1000.)


    def _start_analysis( self ):
        x = self.data.coords["coupler"].values
        y = self.data.coords["time"].values
        x = np.sqrt(8*0.2*27*abs(np.cos((x+0.115)/0.7*np.pi)))-0.2
        zz_interaction = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            fit_data = self.data.isel(coupler=i)
            fit_ramsey = FitDampedOscillation( fit_data.rename({"time": "x"}) )
            guess_para = fit_ramsey.guess()
            init_phi = np.pi/2
            guess_para.add("phi",min=init_phi*0.9, max=init_phi*1.1)
            fit_ramsey.fit()
            zz_interaction[i]=fit_ramsey.result.params["f"].value




        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

        c = ax1.pcolormesh(x, y, self.data.values.T, cmap='bwr', shading='auto')
        # fig.("Conditional")
        ax1.set_ylabel('Evolutionary Time [ns]')
        # fig.colorbar(c, ax=ax1, label='Intensity')  # 添加 colorbar
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_ylabel("Time (us)", fontsize=20)

        ax2.plot(x, np.abs(zz_interaction), label="Crosstalk", color='blue')
        # ax2.set_title("Crosstalk X Flux")
        ax2.set_xlabel("Flux [V]")
        ax2.set_ylabel("Crosstalk [MHz]")    
        ax2.set_yscale("log")    
        # ax3.plot(flux, compute_fc(flux, 7.323, 4.663579, 0.225, 0.132), label="fc", color='blue')
        # ax3.set_title("fc(z) X Flux")
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_xlabel("Coupler Frequency (GHz)", fontsize=20)
        ax2.set_ylabel("ZZ interaction (MHz)", fontsize=20)
        ax2.set_ylim(0.01, 3)
        ax2.set_xlim(5.1, 6.5)
        ax2.set_ylim(0.01, 3)
        fig.tight_layout()
        # plt.tight_layout(rect=[0, 0, 1, 0.96])

    def _export_result( self, *args, **kwargs ):
        pass

if __name__ == "__main__":
    import xarray as xr
    import numpy as np

    dataset = xr.open_dataset(r"d:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\20250112_122247_find_ZZfree_q1_q2\find_ZZfree_q1_q2.nc")
    sq_data = dataset["q1_ro"]
    print(sq_data)

    sq_data = sq_data.sel(mixer="I").rename({"flux": "coupler"})
    # change format (temp)
    # data = xr.DataArray(
    #     data=np.random.rand(3, 4),  # The core data (a NumPy array or similar)
    #     dims=["time", "space"],     # Names of dimensions
    #     coords={                    # Coordinates for the dimensions
    #         "time": ["2023-01-01", "2023-01-02", "2023-01-03"],
    #         "space": ["x", "y", "z", "w"],
    #     },
    #     name="example_data",        # Optional: Name of the DataArray
    #     attrs={                     # Optional: Additional metadata
    #         "description": "Random data",
    #         "units": "arbitrary",
    #     }
    # )
    label = sq_data.name
    # Analysis
    my_ana = ZZInteraction( sq_data )
    my_ana._start_analysis()

    # Plot
    plt.show()