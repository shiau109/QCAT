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

    myRB = RandomizedBenchmarking()

    # Define a range of sequence lengths.
    length_power = np.arange(11)
    sequence_lengths = 2**length_power  # e.g., [1, 2, 4, 8, ...]
    myRB.n_trials = 100  # Number of random sequences per sequence length.
    myRB.p_depol = 0.013
    myRB.gate_gen.rot_err = 0.00

    qubit_frequency = 4.727
    opt_coupler = 6.65
    alpha_q1 = -0.2
    alpha_qc = -0.165

    x = np.array([-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])


    coupler_freq = np.sqrt(8 * 0.165 * 42 * abs(np.cos((x + 0.115) / 0.627 * np.pi))) - 0.165
    print(  np.sqrt(8 * 0.165 * 42 * abs(np.cos((0.115) / 0.627 * np.pi))) - 0.165)
    # coupler_freq = np.linspace( 5.5, 7.5, 7 )
    coupling = 0.080

    detuning = coupler_freq -qubit_frequency
    opt_detuning = opt_coupler-qubit_frequency

    corr_coupling = coupling*np.sqrt( (qubit_frequency+detuning)/qubit_frequency )


    fig, ax = plt.subplots(1)
    ax.plot( qubit_frequency+detuning, chi_qc(coupling, detuning, alpha_q1, alpha_qc),"o")    
    opt_chi = chi_qc(coupling*np.sqrt( (qubit_frequency+opt_detuning)/qubit_frequency ), opt_detuning, alpha_q1, alpha_qc)
    print(f"opt_detuning:{opt_detuning} chi:{opt_chi}")
    sim_points = detuning.shape[-1]
    r_g_list = []
    chi_error_list = []
    for i in range(sim_points):
        myRB.gate_gen.detuning = chi_qc(corr_coupling[i], detuning[i], alpha_q1, alpha_qc) -opt_chi
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
    ax.plot( qubit_frequency+detuning, y*100,"o")

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Coupler Frequency (GHz)", fontsize=20)
    ax.set_ylabel("Native Gate Error Rate (%)", fontsize=20)

    # ax.set_ylim(0.2, 1.0)

    # ax.set_xlim(5.1, 6.5)
    output_dataarray = xr.Dataset(
        data_vars = dict(
            r_g = (["flux"],np.array(r_g_list)),
            chi_error_list = (["flux"],np.array(chi_error_list)),
            frequency = (["flux"],qubit_frequency+detuning)
        ),
        coords=dict(
            flux = x,
        ),
        attrs=dict(
            opt_chi = opt_chi,
        ),
    )
    print(output_dataarray)
    output_dataarray.to_netcdf(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\simulation\RB_dispersive.nc")
    plt.show()