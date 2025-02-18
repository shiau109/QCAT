import numpy as np 
import qutip as qt
import matplotlib.pyplot as plt
from qutip import Bloch
import random
from scipy.optimize import curve_fit
from application.test.TPS_RB_simulation import RandomizedBenchmarking
from qcat.common_calculator import chi_qubit_coupler
import xarray as xr
from qcat.analysis.qubit.clifford_1QRB import Clifford1QRB

if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # Main Simulation
    myRB = RandomizedBenchmarking()

    # Define a range of sequence lengths.
    length_power = np.arange(10)
    sequence_lengths = 2**length_power  # e.g., [1, 2, 4, 8, ...]
    myRB.n_trials = 100  # Number of random sequences per sequence length.
    myRB.p_depol = 0.00
    rb_fidelities = myRB.simulate_rb( sequence_lengths )




    # change format (temp)
    data = xr.Dataset(
        {
            "p0": (("gate_num",), rb_fidelities),
        },
        coords={"gate_num": sequence_lengths },
    )
    label = "test"
        
    my_ana = Clifford1QRB( data, label )
    my_ana._start_analysis( plot=False )

