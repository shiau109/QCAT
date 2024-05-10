

import matplotlib.pyplot as plt
from qcat.zline_crosstalk.ramsey_2dfft import analysis_crosstalk_value
import numpy as np

def plot_analysis( z1, z2, data:np.ndarray ):
    """
    z1 is crosstalk voltage (other)\n
    z2 is compensation voltage (self)
    """
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(10, 5)

    crosstalk, freq_axes, mag = analysis_crosstalk_value( z1, z2, data )
    _plot_rawdata( z1, z2, data.transpose(), crosstalk, ax[0] )

    _plot_2Dfft( freq_axes[0], freq_axes[1], mag.transpose(), ax[1] )


def _plot_rawdata( x, y, z, slope, ax=None ):
    """
    x is crosstalk voltage (other)\n
    y is compensation voltage (self)
    """
    ax.pcolormesh(x, y, z, cmap='gray')
    ax.plot([x[0],x[-1]],[ x[0]*slope, x[-1]*slope ],color="r",linewidth=5)
    ax.set_title('Original Image')
    ax.set_xlabel(f"Crosstalk Delta Voltage (mV)")
    ax.set_ylabel(f"Compensation Delta Voltage (mV)")

def _plot_2Dfft( x, y, z, ax=None  ):
    """
    x is crosstalk voltage (other)\n
    y is compensation voltage (self)
    """
    # plt.pcolormesh(f_z_crosstalk, f_z_target, np.log1p(magnitude_spectrum), cmap='gray')  # Use log scale for better visualization
    ax.pcolormesh( x, y, z, cmap='gray')  # Use log scale for better visualization
    # ax.plot([-f_z_crosstalk_pos/1000,f_z_crosstalk_pos/1000],[-f_z_target_pos/1000,f_z_target_pos/1000],"o",color="r",markersize=5)
    # ax.plot([-f_z_crosstalk_pos/1000,f_z_crosstalk_pos/1000],[-f_z_target_pos/1000,f_z_target_pos/1000],color="r",linewidth=1)
    ax.set_xlabel(f"crosstalk wavenumber (1/mV)")
    ax.set_ylabel(f"compensation wavenumber (1/mV)")
    ax.set_title('2D Fourier Transform (Magnitude Spectrum)')
