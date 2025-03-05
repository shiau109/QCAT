import numpy as np
import matplotlib.pyplot as plt

def chi_function(delta_qc, g, chi):
    term = (delta_qc / 2) - np.sqrt((delta_qc ** 2) / 4 + g ** 2)
    numerator = -g ** 2 + term ** 2
    denominator = g ** 2 + term ** 2
    return chi * (1 - (numerator / denominator))

# Define parameters
g = 80.0  # Coupling strength (modify as needed)
chi = 1.0  # Scale factor (modify as needed)
delta_qc_values = np.linspace(-1000, 1000, 1000)  # Range of Delta_qc
chi_values = chi_function(delta_qc_values, g, chi)

# Plot the function
plt.figure(figsize=(8, 5))
plt.plot(delta_qc_values, chi_values, label="$\chi(\Delta_{qc})$", color='b')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel("$\Delta_{qc}$")
plt.ylabel("$\chi$")
plt.title("Plot of $\chi$ as a function of $\Delta_{qc}$")
plt.legend()
plt.grid()
plt.show()
