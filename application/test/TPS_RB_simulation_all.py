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
from qcat.common_calculator.qcq_zz_interaction import ZZ_interaction

if __name__ == '__main__':
    x = np.array([-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])
    x = np.array([-0.08,-0.04,0.0,0.04,0.08]) #Fast test
    
    qubit_frequency = 4.727
    opt_coupler = 6.65
    design_coupler = 6.3
    alpha_q1 = -0.2
    alpha_qc = -0.16

    # Main Simulation
    myRB = RandomizedBenchmarkingZZ()
    # myRB.zz_interaction = 0.001
    # myRB = RandomizedBenchmarking()

    # Define a range of sequence lengths.
    length_power = np.arange(11)
    sequence_lengths = 2**length_power  # e.g., [1, 2, 4, 8, ...]
    myRB.n_trials = 100  # Number of random sequences per sequence length.
    myRB.p_depol = 0.013
    myRB.gate_gen.rot_err = 0.0

    crosstalk_freq = -0.15*(x-0.12)**2
    opt_crosstalk_freq =  -0.15*(0-0.12)**2
    # crosstalk_freq = 0.025*x
    opt_crosstalk_freq = 0
    coupler_freq = np.sqrt(8 * 0.16 *43.73 * abs(np.cos((x +0.015 +0.10) /0.612 * np.pi))) - 0.16



    opt_coupler =  np.sqrt(8 * 0.16 *43.73 * abs(np.cos((0 +0.015 +0.10) /0.612 * np.pi))) - 0.16

    # coupler_freq = np.linspace( 5.5, 7.5, 7 )

    detuning = qubit_frequency -coupler_freq
    opt_detuning = qubit_frequency -opt_coupler

    # chi = chi_qc(coupling, detuning, alpha_q1, alpha_qc)


    calc = ZZ_interaction()
    calc.w2 = coupler_freq
    coupling = 0.085 *0.85
    corr_coupling = coupling*np.sqrt( (coupler_freq)/design_coupler )
    calc.g23 = corr_coupling
    calc.g12 = corr_coupling
    calc.g13 = 0.0038 *1.071 *0.85


    zz2_nanjing, zz3_nanjing, zz4_nanjing = calc.nanjing_formula()
    total_zz = zz2_nanjing +zz3_nanjing +zz4_nanjing
    chi = chi_qc(corr_coupling, detuning, alpha_q1, alpha_qc)
    opt_chi = chi_qc(coupling*np.sqrt( (opt_coupler)/design_coupler ), opt_detuning, alpha_q1, alpha_qc)
    print(f"opt_detuning:{opt_detuning} chi:{opt_chi}")

    sim_points = x.shape[-1]
    r_g_list = []
    rb_shift_list = []
    for i in range(sim_points):
        myRB.gate_gen.detuning = chi[i] -opt_chi+ crosstalk_freq[i]-opt_crosstalk_freq
        
        # Compute Nanjing's formulas and Tsinghua's formulas

        # myRB.zz_interaction = total_zz[i]
        myRB.zz_interaction = 0

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
        print(f"native_gate_fidelity:{my_ana.fidelity['native_gate_infidelity']}")
        r_g_list.append( my_ana.fidelity["native_gate_infidelity"] )
        rb_shift_list.append( myRB.gate_gen.detuning )
    y = np.array(r_g_list)
    fig, ax = plt.subplots(1)
    ax.plot( coupler_freq, y*100,"o")

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Coupler Frequency (GHz)", fontsize=20)
    ax.set_ylabel("Native Gate Error Rate (%)", fontsize=20)

    # ax.set_ylim(0.2, 1.0)

    # ax.set_xlim(5.1, 6.5)
    output_dataarray = xr.Dataset(
        data_vars = dict(
            r_g = (["flux"], np.array(r_g_list)),
            chi = (["flux"], chi),
            coupler_freq = (["flux"], coupler_freq),
            corr_coupling = (["flux"], corr_coupling),
            rb_shift_list =  (["flux"], np.array(rb_shift_list)),
            crosstalk_freq = (["flux"], crosstalk_freq),
            zz_interaction = (["flux"], total_zz),
        ),
        coords=dict(
            flux = x,
        ),
        attrs=dict(
            opt_chi = opt_chi,
            opt_coupler = opt_coupler,
            opt_crosstalk_freq = opt_crosstalk_freq
        ),
    )
    print(output_dataarray)
    output_dataarray.to_netcdf(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\simulation\RB_detuned_wozz_fast.nc")
    plt.show()