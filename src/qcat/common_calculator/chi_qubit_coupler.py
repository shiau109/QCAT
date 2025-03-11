import numpy as np
import matplotlib.pyplot as plt

def chi_qc( coupling, detuning, anharmonicity_q, anharmonicity_c ):
    chi = 2*coupling**2*( ac_term(detuning,anharmonicity_c) -aq_term(detuning,anharmonicity_q))

    return chi
def ac_term(detuning,anharmonicity_c):
    return 1/(detuning-anharmonicity_c)

def aq_term(detuning,anharmonicity_q):
    return 1/(detuning+anharmonicity_q)

if __name__ == '__main__':
    f_q = 4.7
    E_c_q = -0.2
    E_c_c = -0.16
    f_c = np.linspace( 5., 7.5, 100 )
    detuning = f_q -f_c
    # coupling = 0.085 *0.95
    coupling = 0.085 *0.87
    coupling = coupling*np.sqrt( (f_c)/6.3 )
    # plt.plot( detuning, coupling )
    plt.plot( f_c, chi_qc(coupling, detuning, E_c_q, E_c_c), label="Float")
    plt.plot( f_c, 2*coupling**2 * ac_term( detuning, E_c_c), label="c")
    plt.plot( f_c, 2*coupling**2 * aq_term( detuning, E_c_q), label="q")

    # plt.plot( f_c, chi_qc(coupling[0], detuning, E_c_q, E_c_c), label="Fix")
    # plt.plot( f_c, chi_qc(coupling[-1], detuning, E_c_q, E_c_c), label="Fix2")

    plt.legend()
    plt.show()