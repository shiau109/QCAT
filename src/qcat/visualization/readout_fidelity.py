
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qcat.state_discrimination.discriminator import get_proj_distance, train_GMModel, train_1DGaussianModel, p01_to_Teff, get_probability, get_sigma
def plot_readout_fidelity( data, frequency=None, output=None, plot=True ):


    """
    Parameters:\n
    data:
        3 dim with shape (2*2*N)\n
        shape[0] is I and Q\n
        shape[1] is prepare state\n
        shape[2] is N times single shot\n
    
    """
    trained_GMModel = train_GMModel(data)
    dataset = xr.Dataset(
        {"single_shot":(["mixer","state","shot_idx"],data)},
        coords={ "mixer":np.array(["I","Q"]), "state": np.array([0,1]), "shot_idx":np.arange(data.shape[2]) }
    )
    
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

    # scatter plot
    training_data = trained_GMModel.training_data.transpose()
    _plot_iq_shots( training_data[0], training_data[1], trained_GMModel.get_prediction(trained_GMModel.training_data), ax_iq_training )
    make_ellipses( trained_GMModel.model, ax_iq_training )

    # scatter prepare 0 plot
    p0_data = dataset["single_shot"].sel(state=0).values
    _plot_iq_shots(p0_data[0],p0_data[1],trained_GMModel.get_prediction(p0_data.transpose()),ax_iq_0)
    prepare_0_dist = trained_GMModel.get_state_population(p0_data.transpose())
    prepare_0_dist = prepare_0_dist/np.sum(prepare_0_dist)

    ax_iq_0.text(0.07,0.9,f"P(0|0)={prepare_0_dist[0]:.3f}", fontsize = 20, transform=ax_iq_0.transAxes)
    ax_iq_0.text(0.07,0.8,f"P(0|1)={prepare_0_dist[1]:.3f}", fontsize = 20, transform=ax_iq_0.transAxes)

    # scatter prepare 1 plot
    p1_data = dataset["single_shot"].sel(state=1).values
    _plot_iq_shots(p1_data[0],p1_data[1],trained_GMModel.get_prediction(p1_data.transpose()),ax_iq_1)
    prepare_1_dist = trained_GMModel.get_state_population(p1_data.transpose())
    prepare_1_dist = prepare_1_dist/np.sum(prepare_1_dist)
    ax_iq_1.text(0.07,0.9,f"P(1|0)={prepare_1_dist[0]:.3f}", fontsize = 20, transform=ax_iq_1.transAxes)
    ax_iq_1.text(0.07,0.8,f"P(1|1)={prepare_1_dist[1]:.3f}", fontsize = 20, transform=ax_iq_1.transAxes)

    # Histogram plot

    centers, sigmas = trained_GMModel.output_1D_paras()

    train_data_proj = get_proj_distance(centers.transpose(), data) 
    pos = get_proj_distance(centers.transpose(), centers.transpose()) 
    dis = np.abs(pos[1]-pos[0])
    sigma = np.max( sigmas )

    
    # # Histogram plot
    trained_1DGModel = train_1DGaussianModel( train_data_proj, (pos, sigmas)  )
    trained_1DGModel._results.fit_report()

    bin_center, hist_0, p0_result = trained_1DGModel.fit_distribution(train_data_proj[0])
    bin_center, hist_1, p1_result = trained_1DGModel.fit_distribution(train_data_proj[1])
 
    # print(p0_result.fit_report(),p1_result.fit_report())
    plot_1Ddistribution( bin_center, hist_0, p0_result, 0, ax_hist_0)
    plot_1Ddistribution( bin_center, hist_1, p1_result, 1, ax_hist_1)

    # Text infor
    snr = dis/sigma
    fig.text(0.05,0.35,f"Readout Fidelity={1-(prepare_0_dist[1]+prepare_1_dist[0])/2:.3f}", fontsize = 20)
    fig.text(0.05,0.3,f"IQ distance/STD={dis:.2f}/{sigma:.2f}", fontsize = 20)
    fig.text(0.05,0.25,f"Voltage SNR={snr:.2f}", fontsize = 20)
    fig.text(0.05,0.20,f"Power SNR={np.log10(snr)*20:.2f} dB", fontsize = 20)
    
    if frequency != None:

        probability = get_probability(p0_result)
        p01 = probability[1]
        
        effective_T = p01_to_Teff(p01, frequency)
        fig.text(0.05,0.15,f"Effective temperature (mK)={effective_T*1000:.1f}", fontsize = 20)
    else:
        effective_T = 0
    if output != None :
        full_path = f"{output}.png"
        print(f"Saving plot at {full_path}")
        fig.savefig(f"{full_path}")
        if plot:
            plt.show()
    else:
        if plot:
            plt.show()
        else:
            plt.close()
    return fig, effective_T*1000, np.log10(snr)*20

import matplotlib as mpl
colors = ["blue", "red"]

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        # match gmm.covariance_type:
        #     case "full":
        #         covariances = gmm.covariances_[n][:2, :2]
        #     case "tied":
        #         covariances = gmm.covariances_[:2, :2]
        #     case "diag":
        #         covariances = np.diag(gmm.covariances_[n][:2])
        #     case "spherical":
        #         covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        #     case _:
        #         covariances = gmm.covariances_[n][:2, :2]
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        else:
                covariances = gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color, fill=False
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")



def _plot_iq_shots( idata, qdata, label, ax ):
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

from lmfit.models import GaussianModel
from lmfit.model import ModelResult
def plot_1Ddistribution( bin_center, hist, results, prepare_state:int, ax):

    sigma = np.array([results.params["g0_sigma"],results.params["g1_sigma"]])
    _plot_histogram(bin_center, hist, ax)

    # print(results.fit_report())
    
    _plot_gaussian_fit_curve( bin_center, results, ax)
    ax.set_yscale('log')
    ax.set_ylim(1e-3,np.max(hist)*1.5)
    peak_value = np.array([results.params["g0_amplitude"].value, results.params["g1_amplitude"].value])
    area = peak_value * sigma*np.sqrt(2*np.pi)
    probability = area/np.sum(area)
    ax.text(0.07,0.9,f"P({prepare_state}|0)={probability[0]:.3f}", fontsize = 20, transform=ax.transAxes)
    ax.text(0.07,0.8,f"P({prepare_state}|1)={probability[1]:.3f}", fontsize = 20, transform=ax.transAxes)
    
    ax.set_xlabel('Projected Voltage Signal', fontsize=18)

def _plot_histogram(bin_center, data, ax):
    """
    data type
    1 dim with shape (N)
    shape[0] is N times single shot
    
    """
    cmap = mcolors.ListedColormap(["blue", "red"])
    width = bin_center[1] -bin_center[0]
    ax.bar(bin_center, data, width=width)



def _plot_gaussian_fit_curve(xdata, result:ModelResult, ax):
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


