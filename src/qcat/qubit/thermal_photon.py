import matplotlib.pyplot as plt
import numpy as np
# h 6.626e-34
# Kb 1.38e-23


def freq_theral_energy_ratio( f, t):
    """
    f unit is GHz
    t unit in K
    """
    return f/t *6.626/1.38 *1e-2

def fc_ratio( f ):
    """
    f unit is GHz
    """
    return f/3 *10
def blackbody_photon( f, t ):
    return 2*fc_ratio(f)**2/( np.exp(freq_theral_energy_ratio(f, t))-1 )


freq = np.linspace( 0.1, 20000, 1000 )


fig, ax = plt.subplots()
temp_list = [4,20,40,60]
total_n = []
for t in temp_list:
    n_sd = blackbody_photon( freq, t )
    ax.plot( freq, n_sd, label=f"{t}" )
    total_n.append( np.sum(n_sd) )
fig1, ax1 = plt.subplots()
ax1.plot(temp_list, total_n, label="n integ")
ax1.plot(temp_list, np.array(temp_list)**4*5e2, label="T**4")
# ax.plot( freq, 1/(np.exp(freq_theral_energy_ratio(freq, 4))-1) )
ax1.legend()
# ax[1].set_ylim( 0, 20 )

# ax[0].set_xscale('log')
# ax.set_yscale('log')
# ax.set_yscale('log')

plt.show()
