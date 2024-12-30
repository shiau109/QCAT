import matplotlib.pyplot as plt
import numpy as np
def plot_qubit_relaxation( time, signal, ax, fit_result=None ):
    """
    x shape (M,) 1D array
    y shape (N,M)
    N is 1(I only) or 2(both IQ)
    """
    # c = ax.pcolormesh(dfs, amp_log_ratio, np.abs(s21), cmap='RdBu')# , vmin=z_min, vmax=z_max)
    # ax.set_title('pcolormesh')
    # fig.show()
    # Plot
        
    ax.plot( time, signal,"o", label="data",markersize=1)
    ax.set_ylabel(f"voltage (mV)")
    ax.set_xlabel("Wait time (us)")
    if fit_result is not None:
        ax.plot( time, fit_result.best_fit, label="fit")

    return ax

def plot_time_dep_qubit_T1_relaxation_2Dmap( time, evo_time, signal, ax, fit_result=None ):
    """
    x shape (M,) 1D array
    y shape (N,M)
    N is 1(I only) or 2(both IQ)
    """
    # c = ax.pcolormesh(dfs, amp_log_ratio, np.abs(s21), cmap='RdBu')# , vmin=z_min, vmax=z_max)
    # ax.set_title('pcolormesh')
    # fig.show()
    # Plot
        
    ax.set_title('Time dependent T1')
    ax.pcolormesh( time, evo_time, signal.transpose(), cmap='RdBu')# , vmin=z_min, vmax=z_max)
    if fit_result is not None:
        _plot_T1_trace(time, fit_result, ax)

    return ax

def plot_and_save_T1_spectrum(dataset, time, flux, acc_T1):
    for ro_name, data in dataset.data_vars.items():
        fig, ax = plt.subplots()
        ax.set_title('pcolormesh')
        ax.set_xlabel("Flux")
        ax.set_ylabel("T1 (us)")
        pcm = ax.pcolormesh(flux+dataset.attrs["z_offset"],time, data.values[0].T, cmap='RdBu')# , vmin=z_min, vmax=z_max)
        ax.plot(flux+dataset.attrs["z_offset"], acc_T1)
        ax.axvline(x=dataset.attrs["z_offset"], color='black', linestyle='--', label='idle')
        ax.text(0.04, 
                0.96, 
                f"Mean T1 = {np.mean(acc_T1):.4}+-{np.std(acc_T1):.2}",
                fontsize=9, 
                color="black",
                ha='left', 
                va='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.5))
        ax.legend()
        
        plt.colorbar(pcm, label='Value')


    plt.show()
    
def plot_and_save_T1_spectrum_rep(dataset, rep, flux, acc_T1):
    for ro_name, data in dataset.data_vars.items():
        fig, ax = plt.subplots()
        ax.set_title('pcolormesh')
        ax.set_xlabel("Flux")
        ax.set_ylabel("rep time")
        pcm = ax.pcolormesh(flux+dataset.attrs["z_offset"],rep, acc_T1, cmap='RdBu')# , vmin=z_min, vmax=z_max)
        ax.axvline(x=dataset.attrs["z_offset"], color='black', linestyle='--', label='ref IF')
        ax.legend()
        
        plt.colorbar(pcm, label='Value')


plt.show()


def plot_time_dep_qubit_T2_relaxation_2Dmap( time, evo_time, signal, ax, fit_result=None ):
    """
    x shape (M,) 1D array
    y shape (N,M)
    N is 1(I only) or 2(both IQ)
    """
        
    ax.set_title('Time dependent T2')
    ax.pcolormesh( time, evo_time, signal.transpose(), cmap='RdBu')
    if fit_result is not None:
        _plot_T2_trace(time, fit_result, ax)

    return ax

def plot_qubit_T1_relaxation_hist( T1_array, ax=None ):
    """
    x shape (M,) 1D array
    y shape (N,M)
    N is 1(I only) or 2(both IQ)
    """

    mean_t1 = np.mean(T1_array)
    bin_width = mean_t1 *0.05
    start_value = np.mean(T1_array)*0.5
    end_value = np.mean(T1_array)*1.5
    custom_bins = [start_value + i * bin_width for i in range(int((end_value - start_value) / bin_width) + 1)]
    ax.hist(T1_array, custom_bins, density=False, alpha=0.7, label='Histogram')# color='blue', 
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    # p = gaussian(x, mu, sigma)
    # ax.plot(x, p, 'k', linewidth=2, label=f'Fit result: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}')
    # ax.legend()
    # print(f'Mean: {mu:.2f}')
    # print(f'Standard Deviation: {sigma:.2f}')


    return ax

def plot_qubit_T2_relaxation_hist( T2_array, ax=None ):
    """
    x shape (M,) 1D array
    y shape (N,M)
    N is 1(I only) or 2(both IQ)
    """

    mean_t2 = np.mean(T2_array)
    bin_width = mean_t2 *0.05
    start_value = np.mean(T2_array)*0.5
    end_value = np.mean(T2_array)*1.5
    custom_bins = [start_value + i * bin_width for i in range(int((end_value - start_value) / bin_width) + 1)]
    ax.hist(T2_array, custom_bins, density=False, alpha=0.7, label='Histogram')# color='blue', 
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)


def _plot_T1_trace( x, T1_array, ax):
    ax.plot(x, T1_array)

def _plot_T2_trace( x, T2_array, ax):
    ax.plot(x, T2_array)