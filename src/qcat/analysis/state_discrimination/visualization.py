import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from qcat.NCU.Visualized_library import plot_textbox_small, add_headers
from qcat.NCU.Fit_library import gauss_func, SQ_threshold_acquisition_statistic



def plot_outliers(data, outlier_mask, analysis_result=None):
    """
    Plot scatter plots of I vs Q for each prepared_state, showing only the outlier points as defined by outlier_mask.
    Args:
        data (xr.Dataset): Dataset with variables 'I', 'Q', coords 'shot_idx', 'prepared_state'.
        outlier_mask (dict): Dictionary mapping prepared_state index to boolean mask array (same length as shot_idx for that state).
    Returns:
        fig: matplotlib Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)

    for i in range(2):
        mask = outlier_mask[i]

        # Extract I and Q for both prepared states
        I_vals = data['I'].sel(prepared_state=i).values[mask]
        Q_vals = data['Q'].sel(prepared_state=i).values[mask]
        axes[i].scatter(I_vals, Q_vals, s=10, alpha=0.8, color='orange', marker='o', edgecolor='none', label='Outlier')

        # Optionally plot means as black dots
        if analysis_result is not None:

            trained_paras = analysis_result.get('trained_paras', None)
            if trained_paras is not None and 'means' in trained_paras:
                means = trained_paras['means']
                plot_gmm_means_on_axes(axes[i], means)
            if trained_paras is not None and 'covariances' in trained_paras:
                plot_gmm_circles_on_axis(axes[i], trained_paras)

            y_offset = 0.98
            if 'outlier_probability' in analysis_result:
                outlier_prob = analysis_result['outlier_probability']
                text_msg = f"Outlier prob.: {outlier_prob[i]:.3e}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                        
    fig.tight_layout()
    return fig, axes

def Single_shot_Rawdata_plot(data:dict):
    Ig_data,Qg_data,Ie_data,Qe_data= 1000*np.array(data['g'][0]), 1000*np.array(data['g'][1]) ,1000*np.array(data['e'][0]) , 1000*np.array(data['e'][1])
    I,Q= np.hstack([Ig_data,Ie_data]), np.hstack([Qg_data,Qe_data])
    
    fig, axes = plt.subplots(ncols =2,figsize =(6,3),dpi =200)
    axes[0].scatter(Ig_data,Qg_data, color="blue", alpha=0.5, s=1)   
    axes[1].scatter(Ie_data,Qe_data, color="red", alpha=0.5, s=1)      
    axes[0].set_xlabel(r"$I\ $[mV]",size ='15')
    axes[0].set_ylabel(r"$Q\ $[mV]",size ='15')
    axes[1].set_xlabel(r"$I\ $[mV]",size ='15')
    axes[1].set_ylabel(r"$Q\ $[mV]",size ='15')
    axes[0].set_title('Prepare |g>')
    axes[1].set_title('Prepare |e>')
    axes[0].set_xlim(np.mean(I)-5*np.std(I),np.mean(I)+5*np.std(I))
    axes[0].set_ylim(np.mean(Q)-5*np.std(I),np.mean(Q)+5*np.std(I))
    axes[1].set_xlim(np.mean(I)-5*np.std(I),np.mean(I)+5*np.std(I))
    axes[1].set_ylim(np.mean(Q)-5*np.std(I),np.mean(Q)+5*np.std(I))
    axes[0].axes.set_aspect('equal')
    axes[1].axes.set_aspect('equal')
    fig.tight_layout()
    return fig

def Qubit_state_single_shot_plot(results:dict):
    Pgg,Pee,OE= results['error_pack']['Pgg'],results['error_pack']['Pee'],results['error_pack']['overlap']
    Peg,Pge=1-Pgg,1-Pee
    ce_I,ce_Q,sig=1000*results['fit_pack'][0][0],1000*results['fit_pack'][0][1],1000*results['fit_pack'][5]
    Inte_g_data,Inte_e_data= results['fit_pack'][6],results['fit_pack'][7]
    Ig,Qg= results['rot_IQdata'][0][0],results['rot_IQdata'][0][1]
    Ie,Qe= results['rot_IQdata'][1][0],results['rot_IQdata'][1][1]
    I,Q= 1000*np.hstack([Ig,Ie]), 1000*np.hstack([Qg,Qe])
    I_ro,I_fit= 1000*results['I_ro'],1000*results['I_fit']
    Mgg= gauss_func(I_fit,0,sig,results['fit_pack'][1])
    Meg= gauss_func(I_fit,ce_I,sig,results['fit_pack'][2])
    Mge= gauss_func(I_fit,0,sig,results['fit_pack'][3])
    Mee= gauss_func(I_fit,ce_I,sig,results['fit_pack'][4])
    
    fig, axes = plt.subplots(ncols =2,figsize =(6,3),dpi =200)
    fig1, ax1 = plt.subplots(nrows=1,ncols =2,figsize =(7,3.5),dpi =200)

    Outlier_event_g,Inner_event_g= results['Outlier_g_info']['Outlier_event'],results['Outlier_g_info']['Inner_event']
    Outlier_event_e,Inner_event_e= results['Outlier_e_info']['Outlier_event'],results['Outlier_e_info']['Inner_event']
    Outlier_P_g, Outlier_P_e= results['Outlier_g_info']['Outlier_P'],results['Outlier_e_info']['Outlier_P']
    axes[0].scatter(1000*Inner_event_g[0], 1000*Inner_event_g[1], color="blue", alpha=0.5, s=0.5)
    axes[1].scatter(1000*Inner_event_e[0], 1000*Inner_event_e[1], color="red", alpha=0.5, s=0.5)
    axes[0].scatter(1000*Outlier_event_g[0], 1000*Outlier_event_g[1], color="grey", alpha=0.5, s=0.5)
    axes[1].scatter(1000*Outlier_event_e[0], 1000*Outlier_event_e[1], color="grey", alpha=0.5, s=0.5)
    text_msg1=''
    text_msg1 += r"$\rm{Outlier}= %.1f $"%(Outlier_P_g*100)+'%'
    plot_textbox_small(axes[0],text_msg1,x=0.47,y=0.93,fontsize=10)
    text_msg2=''
    text_msg2 += r"$\rm{Outlier}= %.1f $"%(Outlier_P_e*100)+'%'
    plot_textbox_small(axes[1],text_msg2,x=0.47,y=0.93,fontsize=10)

    Inte_data=[Inte_g_data,Inte_e_data]
    Mg=[Mgg,Mge]
    Me=[Meg,Mee]
    Pg=[Pgg,Pge]
    Pe=[Peg,Pee]

    ax1[0].plot(I_ro, Inte_data[0],'o',color='b',alpha=0.3,ms=3)
    ax1[1].plot(I_ro, Inte_data[1],'o',color='r',alpha=0.3,ms=3)
  
    for j in range(2):
        ax1[j].plot(I_fit, Mg[j],'--b',alpha=0.8,lw=1)
        ax1[j].plot(I_fit, Me[j],'--r',alpha=0.8,lw=1)
        ax1[j].set_ylim(10**(int(np.log10(np.max(Inte_data[0])))-3),10**(int(np.log10(np.max(Inte_data[0])))+1))
        ax1[j].set_xlim(ce_I/2-14*sig,ce_I/2+14*sig)
        ax1[j].axvline(x=ce_I/2,color='grey',linestyle='dashed',alpha=0.5,lw=1)
        ax1[j].set_yscale('log')
        text_msg=''
        text_msg += r"$P_{g}= %.1f $"%(Pg[j]*100)+'%'+'\n'
        text_msg += r"$P_{e}= %.1f $"%(Pe[j]*100)+'%'+'\n'
        text_msg += r"$\varepsilon_{o}=%.1f $"%(OE*100)+'%'
        plot_textbox_small(ax1[j],text_msg,fontsize=10)
        ax1[j].set_xlabel(r"$I^{'}\ $[mV]",size ='15')

    axes[0].scatter(0,0,c='k',s=15)
    axes[0].scatter(ce_I,ce_Q,c='k',s=15)
    axes[1].scatter(0,0,c='k',s=15)
    axes[1].scatter(ce_I,ce_Q,c='k',s=15)
    axes[0].add_patch(Ellipse(xy=[0,0],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[0].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[1].add_patch(Ellipse(xy=[0,0],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[1].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*2,height=sig*2,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[0].add_patch(Ellipse(xy=[0,0],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[0].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[1].add_patch(Ellipse(xy=[0,0],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[1].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*4,height=sig*4,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[0].add_patch(Ellipse(xy=[0,0],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[0].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[1].add_patch(Ellipse(xy=[0,0],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))
    axes[1].add_patch(Ellipse(xy=[ce_I,ce_Q],width=sig*6,height=sig*6,fill=False, alpha=0.8, facecolor= None, edgecolor="k", linewidth=0.8, linestyle='--',angle=0))

    axes[0].set_xlabel(r"$I\ $[mV]",size ='15')
    axes[0].set_ylabel(r"$Q\ $[mV]",size ='15')
    axes[1].set_xlabel(r"$I\ $[mV]",size ='15')
    axes[1].set_ylabel(r"$Q\ $[mV]",size ='15')
    axes[0].set_xlim(np.minimum(min(I),min(Q))-sig*2,np.maximum(max(I),max(Q))+sig*2)
    axes[0].set_ylim(np.minimum(min(I),min(Q))-sig*2,np.maximum(max(I),max(Q))+sig*2)
    axes[1].set_xlim(np.minimum(min(I),min(Q))-sig*2,np.maximum(max(I),max(Q))+sig*2)
    axes[1].set_ylim(np.minimum(min(I),min(Q))-sig*2,np.maximum(max(I),max(Q))+sig*2)
    axes[0].axes.set_aspect('equal')
    axes[1].axes.set_aspect('equal')
    axes[0].set_title('Prepare |g>')
    axes[1].set_title('Prepare |e>')
    fig.tight_layout()

    font_kwargs = dict(fontweight="bold", fontsize='12',color='k',alpha=0.9)
    row_headers = [""]
    col_headers = ['Prepare |g>','Prepare |e>']
    add_headers(fig1, col_headers=col_headers, row_headers=row_headers, **font_kwargs)
    ax1[0].set_ylabel(r'$PDF$',size ='15')
    fig1.tight_layout()
    return fig, fig1

def Single_shot_2D_hist_plot(data):
    I,Q= data[0],data[1]
    bins=101
    I_=np.linspace(I.min(),I.max(),bins)  
    Q_=np.linspace(Q.min(),Q.max(),bins)
    hist, xedges, yedges = np.histogram2d(I,Q, bins=(bins,bins), density=True)
    X,Y= np.meshgrid(I_,Q_)
    fig, axes = plt.subplots(nrows =1,figsize =(5,4),dpi =200)
    cmap = plt.get_cmap('jet')
    vmax= np.max(hist)
    pcm = axes.pcolormesh(1000*X,1000*Y, hist.transpose(), vmin=0,vmax=vmax, cmap=cmap,shading='auto')
    cbar =fig.colorbar(pcm, axes=axes, extend='both', orientation='vertical')
    cbar.axes.tick_params(labelsize=10)
    axes.set_xlabel(r"$I\ $[mV]",size ='15')
    axes.set_ylabel(r"$Q\ $[mV]",size ='15')
    cbar.set_label(r'$PDF$',size ='10')
    axes.set_title('Single shot data histogram')
    axes.axes.set_aspect('equal')
    fig.tight_layout()
    
    return fig 

def Qubit_state_single_shot_1Q(Processed_data,Analysis_result,anal_info,Save,Save_graph_path,id):
    # fig1= Single_shot_Rawdata_plot(Processed_data)
    fig2,fig3= Qubit_state_single_shot_plot(Analysis_result)
    # Direct_g=SQ_threshold_acquisition_statistic(data=Processed_data['g'],SQ_thres=dict(Ig=Analysis_result['Ig'],Qg=Analysis_result['Qg'],Ie=Analysis_result['Ie'],Qe=Analysis_result['Qe']))
    # Direct_e=SQ_threshold_acquisition_statistic(data=Processed_data['e'],SQ_thres=dict(Ig=Analysis_result['Ig'],Qg=Analysis_result['Qg'],Ie=Analysis_result['Ie'],Qe=Analysis_result['Qe']))
    # M_direct=np.array([[Direct_g['Pg'],Direct_g['Pe']],[Direct_e['Pg'],Direct_e['Pe']]])
    # fig4= Plot_assignment_matrix_SQ(Analysis_result['M'],Q=Raw_data_collect['g'].meas_q,title='Single shot fitting')
    # fig5= Plot_assignment_matrix_SQ(M_direct,Q=Raw_data_collect['g'].meas_q,title='Direct counting')
    # fig6= Single_shot_2D_hist_plot(Processed_data['g']) 
    # fig7= Single_shot_2D_hist_plot(Processed_data['e']) 
    # show_args(Analysis_result['error_pack'],title='Readout analysis (error budget)')
    return {'scatter': fig2, 'hist':fig3 }

def plot_prepared_state_scatter(data, analysis_result=None):
    """
    Plot two scatter plots of I vs Q for prepared_state=0 and prepared_state=1, sharing the same axis limits.
    Optionally plot GMM means as black dots if analysis_result is provided.
    Args:
        data (xr.Dataset): Dataset with variables 'I', 'Q', coords 'shot_idx', 'prepared_state'.
        analysis_result (dict, optional): Dictionary with GMM parameters (expects 'means').
    Returns:
        fig: matplotlib Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    I_list = []
    Q_list = []
    for i in range(2):
        # Extract I and Q for both prepared states
        I_list.append(data['I'].sel(prepared_state=i).values)
        Q_list.append(data['Q'].sel(prepared_state=i).values)



    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    for i in range(2):
        # Default color is blue, but if state_label is provided, use it for coloring
        if analysis_result is not None and 'state_label' in analysis_result:
            labels = np.array(analysis_result['state_label'][i])
            # Use a colormap for 2 classes
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(labels / (labels.max() if labels.max() > 0 else 1))
            axes[i].scatter(I_list[i], Q_list[i], s=6, alpha=0.7, c=colors, marker='o', edgecolor='none')
        else:
            axes[i].scatter(I_list[i], Q_list[i], s=1, alpha=0.5, color='blue')


        # Optionally plot GMM means as black dots
        if analysis_result is not None:

            trained_paras = analysis_result.get('trained_paras', None)
            if trained_paras is not None and 'means' in trained_paras:
                means = trained_paras['means']
                plot_gmm_means_on_axes(axes[i], means)
            if trained_paras is not None and 'covariances' in trained_paras:
                plot_gmm_circles_on_axis(axes[i], trained_paras)

            y_offset = 0.98
            if 'direct_counts' in analysis_result:
                direct_counts = analysis_result['direct_counts']
                text_msg = f"Direct counts:\n{direct_counts[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                y_offset -= 0.18  # Move next box down
            if 'gaussian_norms' in analysis_result:
                gaussian_norms = analysis_result['gaussian_norms']
                text_msg = f"Gaussian norms:\n{gaussian_norms[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

    fig.tight_layout()
    return fig, axes

def plot_2d_histogram(hist_dataset, analysis_result=None):
    """
    Plot 2D histogram (density) for each prepared_state from hist_dataset.
    If analysis_result is provided, overlay GMM means and covariances.
    Args:
        hist_dataset (xr.Dataset): Dataset with variable 'density' and coords 'prepared_state', 'x', 'y'.
        analysis_result (dict, optional): GMM fit results to overlay.
        axes: matplotlib Axes or None.
        cmap (str): Colormap for the plot (default 'viridis').
    Returns:
        fig: matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    n_states = hist_dataset.sizes['prepared_state']
    fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 5), dpi=150)
    if n_states == 1:
        axes = [axes]
    
    x = hist_dataset['x'].values
    y = hist_dataset['y'].values
    for i, state in enumerate(hist_dataset.coords['prepared_state'].values):
        density = hist_dataset['density'].sel(prepared_state=state).values.T  # shape (len(y), len(x)), so transpose for imshow
        pcm = axes[i].imshow(
            density,
            origin='lower',
            aspect='auto',
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap='viridis',
            norm=LogNorm() if np.any(density > 0) else None
        )
        axes[i].set_title(f"prepared_state={state}")
        axes[i].set_xlabel('I')
        axes[i].set_ylabel('Q')
        fig.colorbar(pcm, ax=axes[i], label='Density (log scale)')

        # Optionally plot GMM means as black dots
        if analysis_result is not None:

            trained_paras = analysis_result.get('trained_paras', None)
            if trained_paras is not None and 'means' in trained_paras:
                means = trained_paras['means']
                plot_gmm_means_on_axes(axes[i], means)
            if trained_paras is not None and 'covariances' in trained_paras:
                plot_gmm_circles_on_axis(axes[i], trained_paras)

            y_offset = 0.98
            if 'direct_counts' in analysis_result:
                direct_counts = analysis_result['direct_counts']
                text_msg = f"Direct counts:\n{direct_counts[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                y_offset -= 0.18  # Move next box down
            if 'gaussian_norms' in analysis_result:
                gaussian_norms = analysis_result['gaussian_norms']
                text_msg = f"Gaussian norms:\n{gaussian_norms[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

    plt.tight_layout()
    return fig, axes
    


def plot_gmm_means_on_axes(axes, means):
    """
    Plot GMM means as black dots on each axis in axes.
    Args:
        axes: list of matplotlib Axes
        means: array-like, shape (2, 2), GMM means for two components
    """
    axes.scatter(means[0][0], means[0][1], c='k', s=40, marker='o')
    axes.scatter(means[1][0], means[1][1], c='k', s=40, marker='o')

def plot_gmm_circles_on_axis(axes, analysis_result, n_std=[1,2,3], **circle_kwargs):
    """
    Plot GMM means as centers and covariances as radii (dashed circles) on a given axis.
    Args:
        axes: matplotlib Axes
        analysis_result: dict, expects 'means' (N,2) and 'covariances' (N,) from GMM
        n_std: list or float, number(s) of standard deviations for the radius
        circle_kwargs: additional kwargs for Ellipse
    """
    from matplotlib.patches import Ellipse
    means = analysis_result['means']
    covariances = analysis_result['covariances']
    # Accept n_std as a list or a single float
    if isinstance(n_std, (int, float)):
        n_std_list = [n_std]
    else:
        n_std_list = list(n_std)
    for i in range(means.shape[0]):
        center = means[i]
        for n in n_std_list:
            radius = np.sqrt(covariances[i]) * n
            circle = Ellipse(xy=center, width=2*radius, height=2*radius, angle=0,
                             edgecolor='k', facecolor='none', linestyle='--', linewidth=1.5, **circle_kwargs)
            axes.add_patch(circle)

def axis_formatter(axes, lim_I, lim_Q, i):

    from matplotlib.ticker import ScalarFormatter
    axes.set_xlim(lim_I)
    axes.set_ylim(lim_Q)
    axes.set_aspect('equal')
    # Use ScalarFormatter for scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    axes.yaxis.set_major_formatter(formatter)
    # Force scientific notation if needed
    axes.ticklabel_format(style='sci', axis='both', scilimits=(-2,2))
    # Get offset text (exponential part)
    axes.xaxis.offsetText.set_visible(True)
    axes.yaxis.offsetText.set_visible(True)
    # Set axis labels with exponent if present
    xlabel = r"$I$"
    ylabel = r"$Q$"
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(f'prepared_state={i}')
    
def compute_shared_axis_limits(data, n_std=5):
    """
    Compute shared axis limits for I and Q from a dataset with 'prepared_state' axis.
    Args:
        data: xarray Dataset with variables 'I', 'Q', and 'prepared_state' coordinate
        n_std: number of standard deviations for the axis limits (default 5)
    Returns:
        lim_I: tuple (min, max) for I axis
        lim_Q: tuple (min, max) for Q axis
    """
    I_list = []
    Q_list = []
    for i in range(2):
        I_list.append(data['I'].sel(prepared_state=i).values)
        Q_list.append(data['Q'].sel(prepared_state=i).values)
    all_I = np.concatenate(I_list)
    all_Q = np.concatenate(Q_list)
    I_mean, Q_mean = np.mean(all_I), np.mean(all_Q)
    I_std, Q_std = np.std(all_I), np.std(all_Q)
    lim_I = (I_mean - n_std*I_std, I_mean + n_std*I_std)
    lim_Q = (Q_mean - n_std*Q_std, Q_mean + n_std*Q_std)
    return lim_I, lim_Q