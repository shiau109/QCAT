import matplotlib.pyplot as plt
import numpy as np
# h 6.626e-34
# Kb 1.38e-23
def radiation_energy( temperature ):
    return 5.67e-8 * temperature **4


def gamma_qp( qp_ratio ):
    return np.sqrt(2)/(7e3*95e-15)*qp_ratio*(43.5/4.9)**1.5
    # return np.sqrt(2)/(8.5e3*95e-15)*qp_ratio*(43.5/5.06)**1.5

def pop_qp( qp_ratio ):
    return 2.17*qp_ratio*(43.5/4.9)**3.65





const_t1 = 18/1e6
const_gamma = 1/const_t1
temperature = np.linspace( 4, 100, 100 )
# qp_ratio = np.logspace( -7.0, -5.0, 200)
qp_ratio = 1.5e-6* radiation_energy( temperature )

p01 = pop_qp(qp_ratio)
t1 = 1e6* 1/ (gamma_qp( qp_ratio ) )
# t1 = 1e6* 1/ (gamma_qp( qp_ratio ) )

fig, ax = plt.subplots(3, sharex=True)


ax[0].plot( temperature, gamma_qp( qp_ratio )/1e6 )
ax[1].plot( temperature, p01 )
ax[2].plot( temperature, qp_ratio )
# ax[0].plot( qp_ratio, t1 )
# ax[1].plot( qp_ratio, p01 )

# ax[1].set_ylim( 0, 20 )

# ax[0].set_xscale('log')
ax[2].set_yscale('log')
# ax.set_yscale('log')

plt.show()