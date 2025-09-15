

import matplotlib.pyplot as plt
import xarray as xr

def plot_results(rawdata:xr.Dataset, analysis_result:dict=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    ax.plot(rawdata.coords['idle_time'].values, rawdata, '-', label='Raw Data')
    if analysis_result is not None:
        # Show best_fit curve if available
        if 'best_fit' in analysis_result:
            ax.plot(rawdata.coords['idle_time'].values, analysis_result['best_fit'], '-', label='Fit')
        ax.legend()
        # Add textbox with fit parameters
        params = analysis_result
        k1 = params.get('kappa_1', float('nan'))
        k2 = params.get('kappa_2', float('nan'))
        tau1 = 1/k1 if k1 != 0 else float('nan')
        tau2 = 1/k2 if k2 != 0 else float('nan')
        f1 = params.get('f_1', float('nan'))
        f2 = params.get('f_2', float('nan'))
        textstr = (
            f"κ₁ = {k1:.4g} (τ₁={tau1:.4g})\n"
            f"a₁ = {params.get('a_1', float('nan')):.4g}\n"
            f"f₁ = {f1:.4g}\n"
            f"ϕ₁ = {params.get('phi_1', float('nan')):.4g}\n"
            f"κ₂ = {k2:.4g} (τ₂={tau2:.4g})\n"
            f"a₂ = {params.get('a_2', float('nan')):.4g}\n"
            f"f₂ = {f2:.4g}\n"
            f"ϕ₂ = {params.get('phi_2', float('nan')):.4g}\n"
            f"f+/-df = {(f1+f2)/2:.4g} +/- {abs(f1-f2)/2:.4g}\n"
        )
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('idle time', fontsize=20)
    ax.set_ylabel('state', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    fig.tight_layout()

    return fig

def plot_fft(freq, amp):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(freq, amp, label='FFT (positive freq)')
    ax.set_xlabel('Frequency', fontsize=20)
    ax.set_ylabel('Amplitude', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.legend()
    fig.tight_layout()
    return fig