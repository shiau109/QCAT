import numpy as np 
import qutip as qt
import matplotlib.pyplot as plt
from qutip import Bloch
import random
from scipy.optimize import curve_fit
from TPS_RB_simulation import RandomizedBenchmarking
from qcat.common_calculator.chi_qubit_coupler import chi_qc
import xarray as xr
from qcat.analysis.qubit.clifford_1QRB import Clifford1QRB

if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # Main Simulation
    myRB = RandomizedBenchmarking()

    # Define a range of sequence lengths.
    length_power = np.arange(11)
    sequence_lengths = 2**length_power  # e.g., [1, 2, 4, 8, ...]
    myRB.n_trials = 100  # Number of random sequences per sequence length.
    myRB.p_depol = 0.013
    myRB.gate_gen.rot_err = 0.00

    qubit_frequency = 4.727
    alpha_q1 = -0.2
    alpha_qc = -0.15

    detuning = np.linspace( 0.5, 1.5, 100 )
    coupling = 0.065*np.sqrt( (qubit_frequency+detuning)/qubit_frequency )
    fig, ax = plt.subplots(1)
    ax.plot( qubit_frequency+detuning, chi_qc(coupling, detuning, alpha_q1, alpha_qc),"o")    
    opt_chi = chi_qc(0.070*np.sqrt( (qubit_frequency+1)/qubit_frequency ), 1, alpha_q1, alpha_qc)

    sim_points = 10
    detuning = np.linspace( 0.5, 1.5, 10 )
    coupling = 0.070*np.sqrt( (qubit_frequency+detuning)/qubit_frequency )
    r_g_list = []
    for i in range(sim_points):
        myRB.gate_gen.detuning = chi_qc(coupling[i], detuning[i], alpha_q1, alpha_qc) -opt_chi
        rb_fidelities, std = myRB.simulate_rb( sequence_lengths )
        print(f"{i} detuning {myRB.gate_gen.detuning}")
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
        print(my_ana.fidelity["native_gate_fidelity"])
        r_g_list.append( my_ana.fidelity["native_gate_fidelity"] )

    y = np.array(r_g_list)
    fig, ax = plt.subplots(1)
    ax.plot( qubit_frequency+detuning, y*100,"o")

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Coupler Frequency (GHz)", fontsize=20)
    ax.set_ylabel("Native Gate Error Rate (%)", fontsize=20)
    # ax.set_yscale("log")
    # ax.set_ylim(0.002, 0.01)

    ax.set_ylim(0.2, 1.0)

    ax.set_xlim(5.1, 6.5)
    plt.show()