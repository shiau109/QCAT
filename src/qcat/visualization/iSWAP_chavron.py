
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from qcat.coupling.iswap import fft_chavron, get_main_tone

def rotate_data(data, iq_rotate=0):
    idata = data[0]
    qdata = data[1]
    zdata = (idata +1j*qdata)*np.exp(1j*iq_rotate)
    return zdata

def plot_iSWAP_analysis_result( data, tuning ,time, output=None  ):
    """
    
    """
    

    fig, ax = plt.subplots(2)
    ax[0].set_ylim( 0, 250)
    _plot_iSWAP_chavron(data.transpose(), tuning, time, ax[0])

    fft_mag, freq = fft_chavron( time, data )

    main_freq, main_amp = get_main_tone( freq, fft_mag )
    
    max_freq_idx = np.argmax(main_amp)

    _plot_chavron_fft(fft_mag.transpose(), tuning, freq, main_freq, ax[1])

    if output != None:
        fig.savefig(output)
    
    return main_freq[max_freq_idx]

def _plot_iSWAP_chavron( data, tuning, time, ax:plt.Axes ):
    """
    data shape ( 2, N, M )

    """
    # abs_freq = freq_LO+freq_IF+amp
    
    ax.pcolormesh( tuning, time, data, cmap='RdBu')# , vmin=z_min, vmax=z_max)
    # ax[0].axvline(x=freq_LO+freq_IF, color='b', linestyle='--', label='ref IF')
    # ax[0].axvline(x=freq_LO, color='r', linestyle='--', label='LO')
    # ax[0].legend()

def _plot_chavron_fft( data, tuning, freq, main_freq, ax:plt.Axes ):
    ax.plot( tuning, main_freq, 'o', ms=2 )
    ax.pcolormesh( tuning, freq, data, cmap='gray')  # Use logarithmic scaling for better visibility

