# plotting.py
import matplotlib.pyplot as plt
import numpy as np

# 繪圖函數
def plot_fit(photons_sorted, delta_tot, fitted_delta, fr_value):
    plt.scatter(photons_sorted, delta_tot, label='Data')
    plt.plot(photons_sorted, fitted_delta, color='red', label='Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N_{photons}')
    plt.ylabel('Δ_tot')
    plt.title(f'Fitting Result for fr={fr_value}')
    plt.legend()
    plt.show()
