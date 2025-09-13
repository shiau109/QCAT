
import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
import numpy as np
import xarray as xr

    
def plot_results( rawdata:xr.Dataset, analysis_result:xr.Dataset=None):

    output_figs = {}
    fig_raw, ax_raw = plt.subplots(figsize=(8, 6), dpi=100)
    rawdata = rawdata.transpose('time','flux')
    rawdata.plot(ax=ax_raw)

    if analysis_result is not None:
        fig_zzOnly, ax_zzOnly = plt.subplots(figsize=(8, 6), dpi=100)
        zz_line = analysis_result["f"].plot(ax = ax_zzOnly, color='blue', label='ZZ strength')
        t2_line = (1/analysis_result["tau"]).plot( ax=ax_zzOnly, color='red', label='1/T2' )
        ax_zzOnly.set_xlabel('Flux', fontsize=20)
        ax_zzOnly.set_ylabel("ZZ (MHz)", fontsize=20)
        ax_zzOnly.xaxis.set_tick_params(labelsize=16)
        ax_zzOnly.locator_params(axis='x', nbins=7)
        ax_zzOnly.yaxis.set_tick_params(labelsize=16)
        # Add legend (testbox)
        ax_zzOnly.legend(fontsize=14, loc='best', frameon=True)

        output_figs["zz_value"] = fig_zzOnly
        zz_strength_line = (1/analysis_result["f"]).plot( ax=ax_raw, color='blue', label='ZZ period' )  # Overlay line
        inv_t2_line = analysis_result["tau"].plot( ax=ax_raw, color='red', label='T2' )
        # Add legend (testbox)
        ax_raw.legend(fontsize=14, loc='best', frameon=True)

    ax_raw.set_xlabel('Flux', fontsize=20)
    ax_raw.set_ylabel("Free evolution time (us)", fontsize=20)
    ax_raw.xaxis.set_tick_params(labelsize=16)
    ax_raw.locator_params(axis='x', nbins=7)
    ax_raw.yaxis.set_tick_params(labelsize=16)
    output_figs["raw_data"] = fig_raw


    # fig.tight_layout()
    return output_figs