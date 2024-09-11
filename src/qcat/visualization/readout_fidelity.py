
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qcat.analysis.state_discrimination.readout_fidelity import GMMROFidelity, G1DROFidelity
from qcat.analysis.state_discrimination import p01_to_Teff
def plot_readout_fidelity( dataset:xr.DataArray, gmm_ROfidelity:GMMROFidelity, g1d_ROfidelity:G1DROFidelity, frequency=None, output=None, plot=True, detail_output:bool=False):


    """
    Parameters:\n
    data:
        3 dim with shape (2*2*N)\n
        shape[0] is I and Q\n
        shape[1] is prepare state\n
        shape[2] is N times single shot\n
    
    """
    # dataset = xr.DataArray(data, coords= [("mixer",["I","Q"]), ("prepared_state",[0,1]), ("shot_idx",data.shape[2])] )

    plt.rcParams['figure.figsize'] = [16.0, 8.0]
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_iq_training = plt.subplot(gs[0, 0])
    ax_iq_training.set_title(f"Training data", fontsize=20  )
    ax_iq_training.tick_params(axis='both', labelsize=15)
    # ax_iq_training.tick_params(axis='y', labelsize=15)

    # ax_iq_training.yticks(fontsize=20)
    ax_iq_0 = plt.subplot(gs[0,1])
    ax_iq_0.set_title("Prepare |0>", fontsize=20  )
    ax_iq_0.tick_params(axis='both', labelsize=15)
    ax_hist_0 = plt.subplot(gs[1,1])
    ax_hist_0.tick_params(axis='both', labelsize=15)

    ax_iq_1 = plt.subplot(gs[0,2])
    ax_iq_1.set_title("Prepare |1>", fontsize=20  )
    ax_iq_1.tick_params(axis='both', labelsize=15)
    ax_hist_1 = plt.subplot(gs[1,2])
    ax_hist_1.tick_params(axis='both', labelsize=15)

    # Training scatter plot
    training_data = dataset.stack( new_index=('prepared_state','index')).values
    state = np.concatenate((gmm_ROfidelity.state_data_array[0], gmm_ROfidelity.state_data_array[1]))

    _plot_iq_shots( training_data[0], training_data[1], state, ax_iq_training )
    make_ellipses( gmm_ROfidelity.discriminator.cluster_model, ax_iq_training )

   
    ax_p_iq = [ax_iq_0, ax_iq_1] 
    ax_hist = [ax_hist_0, ax_hist_1] 

    state_probability = gmm_ROfidelity.state_probability
    # prepare i state plot
    for i in [0,1]:
        # scatter plot
        p_data = dataset.sel(prepared_state=i).values
        _plot_iq_shots(p_data[0],p_data[1],gmm_ROfidelity.state_data_array[i],ax_p_iq[i],i,gmm_ROfidelity.state_probability[i])
        
        # Histogram plot
        g1d_prob, bin_center, hist, p_result = g1d_ROfidelity.g1d_dist[i]
        _plot_1Ddistribution( bin_center, hist, p_result, i, g1d_ROfidelity._get_gaussian_area(p_result), ax_hist[i])
 
        
    dis = g1d_ROfidelity.discriminator.signal
    sigma = np.mean(g1d_ROfidelity.discriminator.noise)
    # print(sigma)
    snr = g1d_ROfidelity.discriminator.snr
    # Text infor
    fig.text(0.05,0.35,f"Readout Fidelity={1-(state_probability[0][0]+state_probability[1][1])/2:.3f}", fontsize = 20)
    fig.text(0.05,0.3,f"IQ distance/STD={dis:.2f}/{sigma:.2f}", fontsize = 20)
    fig.text(0.05,0.25,f"Voltage SNR={snr:.2f}", fontsize = 20)
    fig.text(0.05,0.20,f"Power SNR={np.log10(snr)*20:.2f} dB", fontsize = 20)
    
    if frequency != None:

        p01 = g1d_ROfidelity.g1d_dist[0][0][1]
        
        effective_T = p01_to_Teff(p01, frequency)
        fig.text(0.05,0.15,f"Effective temperature (mK)={effective_T*1000:.1f}", fontsize = 20)
    else:
        p01 = state_probability[0][1]
        effective_T = 0
    if output != None :
        full_path = f"{output}.png"
        print(f"Saving plot at {full_path}")
        fig.savefig(f"{full_path}")
        if plot:
            plt.show()
        else:
             plt.close()
    else:
        if plot:
            plt.show()
        else:
            plt.close()
    if detail_output:
        return fig, p01, effective_T*1000, np.log10(snr)*20
    else:
        return fig

import matplotlib as mpl
colors = ["blue", "red"]

def make_ellipses(gmm, ax:plt.Axes):
    for n, color in enumerate(colors):
        match gmm.covariance_type:
            case "full":
                covariances = gmm.covariances_[n][:2, :2]
            case "tied":
                covariances = gmm.covariances_[:2, :2]
            case "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            case "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            case _:
                covariances = gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        # v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        v = 3.0* np.sqrt(v)
        # print(f"ellipses{v/3}")
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color, fill=False
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")



def _plot_iq_shots( idata, qdata, label, ax:plt.Axes, prepare_state:int=None, probability=None ):
    """
    Parameters:
    idata is 1d array
    qdata is 1d array
    
    """
    cmap = mcolors.ListedColormap(["blue", "red"])
    data_pts = idata.shape[-1]
    ax.scatter( idata, qdata, marker='o', c=label, cmap=cmap, s=1000/data_pts, alpha=0.5)
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel('I Voltage Signal', fontsize=20)
    ax.set_ylabel('Q Voltage Signal', fontsize=20)
    if prepare_state is not None:
        ax.text(0.07,0.9,f"P({prepare_state}|0)={probability[0]:.3f}", fontsize = 20, transform=ax.transAxes)
        ax.text(0.07,0.8,f"P({prepare_state}|1)={probability[1]:.3f}", fontsize = 20, transform=ax.transAxes)
    
from lmfit.models import GaussianModel
from lmfit.model import ModelResult
def _plot_1Ddistribution( bin_center, hist, results, prepare_state:int, probability, ax:plt.Axes):

    sigma = np.array([results.params["g0_sigma"],results.params["g1_sigma"]])
    _plot_histogram(bin_center, hist, ax)

    # print(results.fit_report())
    
    _plot_gaussian_fit_curve( bin_center, results, ax)
    ax.set_yscale('log')
    ax.set_ylim(1e-3,np.max(hist)*1.5)
    ax.text(0.07,0.9,f"P({prepare_state}|0)={probability[0]:.3f}", fontsize = 20, transform=ax.transAxes)
    ax.text(0.07,0.8,f"P({prepare_state}|1)={probability[1]:.3f}", fontsize = 20, transform=ax.transAxes)
    
    ax.set_xlabel('Projected Voltage Signal', fontsize=18)

def _plot_histogram(bin_center, data, ax:plt.Axes):
    """
    data type
    1 dim with shape (N)
    shape[0] is N times single shot
    
    """
    cmap = mcolors.ListedColormap(["blue", "red"])
    width = bin_center[1] -bin_center[0]
    ax.bar(bin_center, data, width=width)



def _plot_gaussian_fit_curve(xdata, result:ModelResult, ax:plt.Axes):
    # ax.plot(xdata, result.init_fit, '-', label='init fit') 
    ax.plot(xdata, result.best_fit, '--', color="black", label='best fit', linewidth=2)

    # popt, pcov = curve_fit(double_gaussian, xdata, ydata, p0=guess)
    # (c1, mu1, sigma1, c2, mu2, sigma2) = popt
    gm = GaussianModel()
    pars = gm.make_params()
    pars['center'].set(value=result.params["g0_center"].value)
    pars['amplitude'].set(value=result.params["g0_amplitude"].value)
    pars['sigma'].set(value=result.params["g0_sigma"].value)
    gm.eval(pars, x=xdata)
    ax.plot(xdata, gm.eval(pars, x=xdata), '--', color="b",label='state 0', linewidth=2)
    
    pars['center'].set(value=result.params["g1_center"].value)
    pars['amplitude'].set(value=result.params["g1_amplitude"].value)
    pars['sigma'].set(value=result.params["g1_sigma"].value) 

    gm.eval(pars, x=xdata)    
    ax.plot(xdata, gm.eval(pars, x=xdata), '--', color="r",label='state 1', linewidth=2)


