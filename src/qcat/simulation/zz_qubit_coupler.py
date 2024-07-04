import numpy as np
import matplotlib.pyplot as plt

def chi_qc( coupling, detuning, anharmonicity_q, anharmonicity_c ):
    chi = 2*coupling**2*(1/(detuning+anharmonicity_c)-1/(detuning-anharmonicity_q))

    return chi

detuning = np.linspace( 500, 2000, 100 )
coupling = 65*np.sqrt( (4500+detuning)/4500 )
# plt.plot( detuning, coupling )

plt.plot( detuning, chi_qc(coupling, detuning, -200, -150), label="Float")
plt.plot( detuning, chi_qc(coupling[0], detuning, -200, -150), label="Fix")
plt.plot( detuning, chi_qc(coupling[-1], detuning, -200, -150), label="Fix2")

plt.legend()
plt.show()