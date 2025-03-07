import numpy as np 
import qutip as qt
import matplotlib.pyplot as plt
from qutip import Bloch
import random
from scipy.optimize import curve_fit
from TPS_RB_simulation import RandomizedBenchmarking,RandomizedBenchmarkingZZ
from qcat.common_calculator.chi_qubit_coupler import chi_qc
import xarray as xr
from qcat.analysis.qubit.clifford_1QRB import Clifford1QRB

if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # Main Simulation
    myRB = RandomizedBenchmarkingZZ()
    myRB.zz_interaction = 0.001
    # myRB = RandomizedBenchmarking()

    # Define a range of sequence lengths.
    length_power = np.arange(11)
    sequence_lengths = 2**length_power  # e.g., [1, 2, 4, 8, ...]
    myRB.n_trials = 100  # Number of random sequences per sequence length.
    myRB.p_depol = 0.013
    myRB.gate_gen.rot_err = 0.00

    detuning = np.linspace( -0.005, 0.005, 11 )


    sim_points = detuning.shape[-1]
    r_g_list = []
    chi_error_list = []
    for i in range(sim_points):
        myRB.gate_gen.detuning = detuning[i]
        print(f"{i+1}/{sim_points} chi_error_list{myRB.gate_gen.detuning}")

        rb_fidelities, std = myRB.simulate_rb( sequence_lengths )
        # change format (temp)
        data = xr.Dataset(
            {
                "p0": (("gate_num",), rb_fidelities),
                "std":  (("gate_num",), std),
            },
            coords={"gate_num": sequence_lengths },
        )
        label = "test"
            
        my_ana = Clifford1QRB( data, label )
        my_ana._start_analysis( plot=False )
        print(f"native_gate_fidelity:{my_ana.fidelity['native_gate_fidelity']}")
        r_g_list.append( my_ana.fidelity["native_gate_fidelity"] )
        chi_error_list.append( myRB.gate_gen.detuning )
    y = np.array(r_g_list)
    fig, ax = plt.subplots(1)
    ax.plot( detuning, y*100,"o")

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Coupler Frequency (GHz)", fontsize=20)
    ax.set_ylabel("Native Gate Error Rate (%)", fontsize=20)

    # ax.set_ylim(0.2, 1.0)

    # ax.set_xlim(5.1, 6.5)
    output_dataarray = xr.Dataset(
        data_vars = dict(
            r_g = (["frequency"],np.array(r_g_list)),
            chi_error_list = (["frequency"],np.array(chi_error_list)),
        ),
        coords=dict(
            frequency = detuning,
        ),
    )
    print(output_dataarray)
    output_dataarray.to_netcdf(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\simulation\RB_detuned_zz.nc")
    plt.show()