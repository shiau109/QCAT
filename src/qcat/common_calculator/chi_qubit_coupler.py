import numpy as np
import matplotlib.pyplot as plt

def chi_qc( coupling, detuning, anharmonicity_q, anharmonicity_c ):
    chi = 2*coupling**2*(1/(detuning+anharmonicity_c)-1/(detuning-anharmonicity_q))

    return chi

if __name__ == '__main__':
    f_q = 4.7
    E_c_q = -0.2
    E_c_c = -0.165
    f_c = np.linspace( 5.5, 7.5, 100 )
    detuning = f_c-f_q
    coupling = 0.085*np.sqrt( (f_q+detuning)/f_q )
    # plt.plot( detuning, coupling )

    plt.plot( f_c, chi_qc(coupling, detuning, E_c_q, E_c_c), label="Float")
    plt.plot( f_c, chi_qc(coupling[0], detuning, E_c_q, E_c_c), label="Fix")
    plt.plot( f_c, chi_qc(coupling[-1], detuning, E_c_q, E_c_c), label="Fix2")

    plt.legend()
    plt.show()