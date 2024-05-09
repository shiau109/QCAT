

import matplotlib.pyplot as plt
from qcat.zline_crosstalk.ramsey_2dfft import analysis_crosstalk_value


def plot_analysis( z1, z2, data ):
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(10, 5)

    crosstalk, freq_axes, mag = analysis_crosstalk_value( z1, z2, data )
    _plot_rawdata( z1, z2, data.transpose(), crosstalk, ax[0] )

    _plot_2Dfft( freq_axes[0], freq_axes[1], mag.transpose(), ax[1] )


def _plot_rawdata( x, y, z, slope, ax=None ):
    ax.pcolormesh(x, y, z, cmap='gray')
    ax.plot([x[0],x[-1]],[ x[0]*slope, x[-1]*slope ],color="r",linewidth=5)
    ax.set_title('Original Image')
    ax.set_xlabel(f"Target Delta Voltage (mV)")
    ax.set_ylabel(f"Crosstalk Delta Voltage (mV)")

def _plot_2Dfft( x, y, z, ax=None  ):

    # plt.pcolormesh(f_z_crosstalk, f_z_target, np.log1p(magnitude_spectrum), cmap='gray')  # Use log scale for better visualization
    ax.pcolormesh( x, y, z, cmap='gray')  # Use log scale for better visualization
    # ax.plot([-f_z_crosstalk_pos/1000,f_z_crosstalk_pos/1000],[-f_z_target_pos/1000,f_z_target_pos/1000],"o",color="r",markersize=5)
    # ax.plot([-f_z_crosstalk_pos/1000,f_z_crosstalk_pos/1000],[-f_z_target_pos/1000,f_z_target_pos/1000],color="r",linewidth=1)
    ax.set_xlabel(f"Q1 wavenumber (1/mV)")
    ax.set_ylabel(f"Q2 wavenumber (1/mV)")
    ax.set_title('2D Fourier Transform (Magnitude Spectrum)')
