
from matplotlib import pyplot as plt
import pandas as pd
from typing import List, Tuple
from matplotlib.gridspec import GridSpec
import numpy as np
from qcat.analysis.resonator.photon_dep.analysis_method import *
import copy
default_plot_style = {
    "marker_style":"o",
    "marker_size":1,
    "marker_color":None,
    "legend_label":None,
    "xlabel":None,
    "ylabel":None,
    "xscale":"linear",
    "yscale":"linear"
}
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

def plot_cavityS21_fitting(freq:np.ndarray, raw:np.ndarray, fit:np.ndarray, dependency:np.ndarray, title=None, output_fd=None):
    
    fig = plt.figure(facecolor='white',figsize=(20,9))
    fig.tight_layout()
    faxmin = freq.min()
    faxmax = freq.max()
    faxmid = (faxmin+faxmax)/2
    faxdf = faxmax-faxmin
    t = [ faxmid-faxdf/2.5, faxmid, faxmid+faxdf/2.5]
     
    gs = GridSpec(2, 2)
    
    ax_amp = plt.subplot(gs[0,0])
    # ax_amp.set_xlabel("Frequency (GHz)")
    ax_amp.set_ylabel("Ampltude")
    # ax_amp.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax_amp.locator_params(tight=True)
    ax_amp.set_xticks(t)

    ax_pha = plt.subplot(gs[1,0])
    ax_pha.set_xlabel("Frequency (GHz)")
    ax_pha.set_ylabel("Phase (rad)")
    ax_pha.locator_params(tight=True)
    ax_pha.set_xticks(t)

    ax_iq = plt.subplot(gs[0:,1])
    ax_iq.set_xlabel("In-phase")
    ax_iq.set_ylabel("Quadrature")
    ax_iq.set_title(title)
    ax_iq.locator_params(tight=True)

    ax_iq.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax_iq.yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.subplots_adjust(left=0.15,
                        bottom=0.15, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.25, 
                        hspace=0.25)
    plot_style = {}

    color = plt.cm.rainbow(np.linspace(0, 1, dependency.shape[-1]))

    for rawdata, fitcurve, dep, c in zip(raw,fit,dependency, color):
        plot_style["marker_color"] = c
        plot_style["legend_label"] = None

        # plot amplitude subplot
        plot_rawdata = pd.DataFrame()
        plot_rawdata["x"] = freq
        plot_rawdata["y"] = np.abs(rawdata)
        plot_style["marker_style"] = "o"
        
        ax_amp = plot_basic( plot_rawdata, plot_style, ax_amp )

        plot_fitcurve = pd.DataFrame()
        plot_fitcurve["x"] = freq
        plot_fitcurve["y"] = np.abs(fitcurve)
        plot_style["marker_style"] = "-"
        ax_amp = plot_basic( plot_fitcurve, plot_style, ax_amp )


        # plot phase subplot
        plot_rawdata = pd.DataFrame()
        plot_rawdata["x"] = freq
        plot_rawdata["y"] = np.unwrap(np.angle(rawdata))
        plot_style["marker_style"] = "o"
        ax_pha = plot_basic( plot_rawdata, plot_style, ax_pha )

        plot_fitcurve = pd.DataFrame()
        plot_fitcurve["x"] = freq
        plot_fitcurve["y"] = np.unwrap(np.angle(fitcurve))
        plot_style["marker_style"] = "-"
        ax_pha = plot_basic( plot_fitcurve, plot_style, ax_pha )

        # plot IQ subplot
        plot_rawdata = pd.DataFrame()
        plot_rawdata["x"] = rawdata.real
        plot_rawdata["y"] = rawdata.imag
        plot_style["marker_style"] = "o"
        ax_iq = plot_basic( plot_rawdata, plot_style, ax_iq )

        plot_fitcurve = pd.DataFrame()
        plot_fitcurve["x"] = fitcurve.real
        plot_fitcurve["y"] = fitcurve.imag
        plot_style["marker_style"] = "-"
        plot_style["legend_label"] =f"{dep}"
        ax_iq = plot_basic( plot_fitcurve, plot_style, ax_iq )
    # ax_amp.legend()
    # ax_pha.legend()
    ax_iq.legend()

    if output_fd != None :
        full_path = f"{output_fd}/{title}_fitcurve.png"
        print(f"Saving plot at {full_path}")
        plt.savefig(f"{full_path}")
        plt.close()
    else:
        plt.show()

def plot_resonatorFitting( labels, plot_data, title, color=None, output_fd=None  ):
    
    plotObj = _plotFrame_resonatorFitting()
    fig = plotObj[0]
    for i, l in enumerate(labels):
        if type(color)==type(None):
            c = None
        else:
            c = color[i]

        freq = plot_data[i][0]
        rawdata = plot_data[i][1]
        fitcurve = plot_data[i][2]
        _plot_add_resonatorFitting(freq, rawdata, fitcurve, l, color=c, plotObj=plotObj)
        fig.legend()
    if output_fd != None :
        full_path = f"{output_fd}/{title}.png"
        print(f"Saving plot at {full_path}")
        fig.savefig(f"{full_path}")
    else:
        plt.show()

def _plot_add_resonatorFitting(freq:np.ndarray, rawdata:np.ndarray, fitcurve:np.ndarray, legend_label, color=None, plotObj=None):
    ax_amp = plotObj[1]
    ax_pha = plotObj[2]
    ax_iq = plotObj[3]
    # plot amplitude subplot
    _plot_abs( freq, rawdata, fitcurve, None, ax_amp, color )
    _plot_angle( freq, rawdata, fitcurve, None, ax_pha, color )
    _plot_iq( rawdata, fitcurve, legend_label, ax_iq, color )

def _plot_abs(freq:np.ndarray, data:np.ndarray, fitcurve:np.ndarray, legend_label, axObj:plt.Axes, color=None):
    data = np.abs(data)
    fitcurve = np.abs(fitcurve)

    axObj.plot( freq, data, 'o', ms=1, c=color)# , label=legend_label )
    axObj.plot( freq, fitcurve, '-', linewidth=1, c=color)# , label=legend_label )

def _plot_angle(freq:np.ndarray, data:np.ndarray, fitcurve:np.ndarray, legend_label, axObj:plt.Axes, color=None):
    # data = np.angle(data)
    # fitcurve = np.angle(data)
    axObj.plot( freq, np.arctan2(data.imag,data.real), 'o', ms=1, c=color)# , label=legend_label )
    axObj.plot( freq, np.arctan2(fitcurve.imag,fitcurve.real), '-', linewidth=1, c=color)# , label=legend_label )

def _plot_iq( data:np.ndarray, fitcurve:np.ndarray, legend_label, axObj:plt.Axes, color=None):
    axObj.plot( data.real, data.imag, 'o', ms=1, c=color)# , label=legend_label )
    axObj.plot( fitcurve.real, fitcurve.imag, '-', linewidth=1, c=color, label=legend_label )

def _plotFrame_resonatorFitting():

    fig = plt.figure(facecolor='white',figsize=(20,9))
    fig.tight_layout()
     
    gs = GridSpec(2, 2)
    
    ax_amp = plt.subplot(gs[0,0])
    # ax_amp.set_xlabel("Frequency (GHz)")
    ax_amp.set_ylabel("Ampltude")
    # ax_amp.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax_amp.locator_params(tight=True)

    ax_pha = plt.subplot(gs[1,0])
    ax_pha.set_xlabel("Frequency (GHz)")
    ax_pha.set_ylabel("Phase (rad)")
    ax_pha.locator_params(tight=True)

    ax_iq = plt.subplot(gs[0:,1])
    ax_iq.set_xlabel("In-phase")
    ax_iq.set_ylabel("Quadrature")
    ax_iq.locator_params(tight=True)

    ax_iq.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax_iq.yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.subplots_adjust(left=0.15,
                        bottom=0.15, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.25, 
                        hspace=0.25)
    return fig, ax_amp, ax_pha, ax_iq

def plot_powerdeploss_fitting( powerloss, tanloss_result, title=None, output_fd=None ):

    fig = plt.figure(facecolor='white')
    axObj = fig.add_subplot()
    axObj.set_title(title)
    axObj.set_xlabel("n")
    axObj.set_ylabel("loss")

    n = powerloss["photons"].to_numpy()
    loss = powerloss["loss"].to_numpy()
    loss_err = powerloss["loss_err"].to_numpy()

    
    # Raw data 
    # Prepare plot data format
    powdep_loss = pd.DataFrame()
    powdep_loss["x"] = n
    powdep_loss["y"] = loss
    powdep_loss["yerr"] = loss_err
    # Prepare plot style format
    plot_style = {}
    plot_style["marker_style"] = "o"
    plot_basic( powdep_loss, plot_style, axObj)
    # Fitting curve
    # Prepare plot data format
    powdep_loss_fit = pd.DataFrame()
    fit_n = 10**(np.linspace(-2,6,100))
    powdep_loss_fit["x"] = fit_n
    powdep_loss_fit["y"] = tan_loss(fit_n, tanloss_result["A_TLS"].values, tanloss_result["const"].values, tanloss_result["nc"].values)
    # Prepare plot style format
    plot_style = {}
    plot_style["marker_style"] = "-"
    plot_basic( powdep_loss_fit, plot_style, axObj )

    axObj.set_xlabel("Photon Number")
    axObj.set_ylabel("Loss")
    axObj.set_xscale("log")
    axObj.set_yscale("log")
    if output_fd != None :
        plt.savefig(f"{output_fd}/{title}_fitcurve.png")
        plt.close()
    else:
        plt.show()

def plot_power_dependent_Q( powerQ_result, cav_label=None, output_fd=None ):
    
    plot_style = {}
    plot_style["marker_size"] = 5

    fig = plt.figure(facecolor='white')
    plt.subplots_adjust(left=0.15,
                        bottom=0.15, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.25, 
                        hspace=0.25)
    axObj = fig.add_subplot()

    xy_cols = [("photons","Qi_dia_corr","Qi_dia_corr_err"),
        ("photons","Qc_dia_corr","absQc_err"),
        ("photons","Ql","Ql_err")]
    plot_list = assemble_plot_df( powerQ_result, xy_cols )
    labels = ["Internal Q","Coupling Q", "Loaded Q"]
    for i, p in enumerate(plot_list):
        plot_style["legend_label"] = labels[i]
        plot_basic(p, plot_style=plot_style, axObj=axObj)

    axObj.set_xlabel("Photons")
    axObj.set_ylabel("Quality factor")
    axObj.set_xscale("log")
    axObj.set_yscale("log")
    axObj.set_title(cav_label)
    axObj.legend()
    if output_fd != None :
        plt.savefig(f"{output_fd}/{cav_label}_Q.png")
        plt.close()
    else:
        plt.show()

def assemble_plot_df( data:pd.DataFrame, xy_cols:List )->List[pd.DataFrame]:
    """
    Transform a dataframe to the format for plotting
    """

    plot_dfs = []
    for xy_col in xy_cols:
        
        x_col = xy_col[0]
        y_col = xy_col[1]

        plot_df = pd.DataFrame()
        x = data[x_col].to_numpy()
        plot_df["x"] = x

        y = data[y_col].to_numpy()
        plot_df["y"] = y
        
        if len(xy_col)==3:
            yerr_col = xy_col[2]
            if yerr_col != None:
                yerr = data[yerr_col].to_numpy()
                plot_df["yerr"] = yerr

        plot_dfs.append(plot_df)

    return plot_dfs



# Basic ploting
def plot_basic( data:pd.DataFrame, plot_style:dict=default_plot_style, axObj=None, output=None):
    """
    output : str output path and name
    default_plot_style = {
        "marker_style":"o",
        "marker_size":1,
        "marker_color":None,
        "legend_label":None,
        "xlabel":None,
        "ylabel":None,
        "xscale":"linear",
        "yscale":"linear"
    }
    """
    temp_ps = copy.deepcopy(default_plot_style)
    temp_ps.update(plot_style)
    plot_style = temp_ps
    mk_style = plot_style["marker_style"]
    mk_size = plot_style["marker_size"]
    mk_color = plot_style["marker_color"]
    leg_label = plot_style["legend_label"]
    show_plot = False
    if axObj == None :
        print("No axObj")
        show_plot = True
        fig = plt.figure(figsize=(16,9),facecolor='white')
        axObj = fig.add_subplot()
        # axObj.set_title(plot_style["title"])
        axObj.set_xlabel(plot_style["xlabel"])
        axObj.set_ylabel(plot_style["ylabel"])
        axObj.set_xscale(plot_style["xscale"])
        axObj.set_yscale(plot_style["yscale"])
        

    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    if "yerr" in data:
        yerr = data["yerr"].to_numpy()
    else:
        yerr = None
    axObj.errorbar( x, y, yerr=yerr, fmt=mk_style, ms=mk_size, c=mk_color, label=leg_label )
    if show_plot:
        print("Plot!!")
        plt.show()
    if output != None:
        plt.savefig(f"{output}.png")
        plt.close()


    return axObj

# Older style
def plot_multiCav_powerQ(dfs:List[pd.DataFrame], sample_name, assignment:pd.DataFrame, output=None ):
    
    plt.figure(facecolor='white', figsize=(8,5)) #figsize=(16,9),
    for df in dfs:
        print(f"ploting {df.Name}")
        if df.Name in assignment['measurement_label'].to_numpy():
            asi = assignment.loc[assignment['measurement_label'] == df.Name]
            a_marker_style = asi["marker_style"].values[0]
            a_color = asi["color"].values[0]
            a_clw = asi["center_linewidth"].values[0]
            x_axis = df["photons"].to_numpy()
            Qi = df["Qi_dia_corr"].to_numpy()
            Qi_err = df["Qi_dia_corr_err"].to_numpy()
            plt.errorbar(x_axis, Qi, yerr=Qi_err, ms = 5, fmt=a_marker_style, c=a_color, label=f"{df.Name}-{a_clw}um")
    plt.xlabel("Photon Number n",fontsize=16)
    #plt.xticks(fontsize=13)
    plt.ylabel("Qi",fontsize=16)
    #plt.yticks(fontsize=13)
    plt.legend()
    plt.title( f"{sample_name}",fontsize=16)
    plt.yscale('log')
    plt.xscale('log')
    #plt.show()
    if output != None :
        plt.savefig(f"{output}/{sample_name}_allQi.png")
    plt.close()

def plot_df(df:pd.DataFrame, xycols, axObj=None, log_scale=(False,False), title=None, output=None):
    """
    xycols : a array of 4 value tuple: ( xcol, ycol, yerrcol, label )
        xcol, ycol, yerrcol are names of column
        label is shown text in legend

    title : a 3 value tuple: ( Graph, x axis, y axis )

    log_scale : a list of 2 value tuple: ( xcol, ycol )
        xcol, ycol are bool, True for log scale

    output : str output path and name

    """
    if axObj == None:
        fig = plt.figure(figsize=(16,9),facecolor='white')
        axObj = fig.add_subplot()
    if type(xycols) != list:
        xycols = [xycols]
    for xcol, ycol, yerrcol, label in xycols:
        x = df[xcol].to_numpy()
        y = df[ycol].to_numpy()
        if yerrcol != None:
            yerr = df[yerrcol].to_numpy()
            axObj.errorbar( x, y, yerr=yerr, fmt="o", label=label )
        else:
            axObj.plot( x, y, "o", label=label )
    # plt.title( f"{df.Name}",fontsize=16)

    if title != None:
        axObj.set_title(title[0],fontsize=16)
        axObj.set_xlabel(title[1],fontsize=16)
        axObj.set_ylabel(title[2],fontsize=16)

    #plt.yticks(fontsize=13)
    axObj.legend()
    if log_scale[0] : axObj.set_xscale('log')
    if log_scale[1] : axObj.set_yscale('log')
    #plt.show()
    if output != None:
        plt.savefig(f"{output}.png")
        plt.close()
    return axObj

def plot_multiRes_powerQ_free( import_folder, assignment:pd.DataFrame, output=None ):
    
    xy_cols = [("photons","Qi_dia_corr","Qi_dia_corr_err"),
        ("photons","Qc_dia_corr","absQc_err"),
        ("photons","Ql","Ql_err")]
    # plot_list = assemble_plot_df(power_dep,xy_cols)
    y_labels = ["Internal Q", "Coupling Q", "Loaded Q"]
    for i, xy_col in enumerate(xy_cols):
        fig, ax = _plot_multiRes_powerQ_flex(xy_col, assignment, import_folder, "free_result")
        ax.set_ylabel(y_labels[i])
        fig.legend()
        fig.suptitle(f'All Resonator (Free)')

        #plt.show()
        if output != None :
            plt.savefig(f"{output}/all_result_{xy_cols[i][1]}_free.png")    
   
def plot_multiRes_powerQ_refined( import_folder, assignment:pd.DataFrame, output=None, xy_axis=None ):
    
    xy_cols = [("photons","Qi_dia_corr_fqc","Qi_dia_corr_err"),
        ("photons","Qc_dia_corr_fixed","absQc_err"),
        ("photons","Ql","Ql_err")]
    # plot_list = assemble_plot_df(power_dep,xy_cols)
    y_labels = ["Internal Q", "Coupling Q", "Loaded Q"]
    for i, xy_col in enumerate(xy_cols):
        fig, ax = _plot_multiRes_powerQ_flex(xy_col, assignment, import_folder, "refined_result", xy_axis)
        ax.set_ylabel(y_labels[i])
        fig.legend(fontsize=16,loc='upper right', bbox_to_anchor=(1, 0.9))
        fig.suptitle(f'All Resonator (Refined)')

        #plt.show()
        if output != None :
            plt.savefig(f"{output}/all_result_{xy_cols[i][1]}_refined.png")

def _plot_multiRes_powerQ_flex( xy_col, assignment:pd.DataFrame, import_folder, file_name, xy_axis=None ):

    fig, ax = _plotFrame_multiRes_powerQ()
    if not (xy_axis is None):
        ax.set_xlim(xy_axis[0][0],xy_axis[0][1])
        ax.set_ylim(xy_axis[1][0],xy_axis[1][1])
    ax.tick_params(axis='both', labelsize=20)
    
    for label in assignment['measurement_label'].to_numpy():
        print(f"ploting {label}")
        file_path = f"{import_folder}/{label}/{file_name}.csv"
        power_dep = pd.read_csv( file_path )
        df_plot = assemble_plot_df(power_dep,[xy_col])[0]
        asi = assignment.loc[assignment['measurement_label'] == label]
        _add_plot_multiRes_powerQ(df_plot, asi, ax)

    # fig.legend()
    # fig.suptitle(f'All Resonator (Free)')

    return fig, ax 
def _add_plot_multiRes_powerQ( data:pd.DataFrame, asi:pd.DataFrame, ax:plt.Axes):


    a_marker_style = asi["marker_style"].values[0]
    a_color = asi["color"].values[0]
    if "design_parameter" in asi.columns:
        para = f" {asi['design_parameter'].values[0]}"
    else:
        lw = asi["center_linewidth"].values[0]
        para = f"-{lw}um"
        
    label = asi['measurement_label'].values[0]

    ax.errorbar(data["x"].values, data["y"].values, yerr=data["yerr"].values, ms = 5, fmt=a_marker_style, c=a_color, label=f"{label}{para}")

def _plotFrame_multiRes_powerQ( )->Tuple[plt.Figure,plt.Axes]:

    fig = plt.figure(facecolor='white',figsize=(10,8))

    ax = fig.subplots()
    # ax_amp.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.set_xscale('log')
    ax.set_yscale('log')

    
    fig.subplots_adjust(left=0.15,
                        bottom=0.15, 
                        right=0.8, 
                        top=0.9, 
                        wspace=0.25, 
                        hspace=0.25)
    return fig, ax

def plot_singleRes_powerQ_refined( powerQ_result, cav_label=None, output_fd=None ):

    fig, ax = _plotFrame_singleRes_power()
    ax.set_ylabel("Quality factor")

    xy_cols = [("photons","Qi_dia_corr_fqc","Qi_dia_corr_err"),
        ("photons","Qc_dia_corr_fixed","absQc_err"),
        ("photons","Ql","Ql_err")]
    plot_list = assemble_plot_df( powerQ_result, xy_cols )
    labels = ["Internal Q","Coupling Q", "Loaded Q"]
    for i, p in enumerate(plot_list):
        ax.errorbar( p["x"], p["y"], yerr=p["yerr"], fmt="o", ms=5, label=f"{labels[i]}" )

    ax.set_title(f"{cav_label}_refined")
    fig.legend()
    if output_fd != None :
        plt.savefig(f"{output_fd}/{cav_label}_powerQ_refined.png")
        plt.close()
    else:
        plt.show()

def plot_singleRes_powerloss_refined( powerQ_result, cav_label=None, output_fd=None ):

    fig, ax = _plotFrame_singleRes_power()
    ax.set_ylabel("Loss")

    xy_cols = [("photons","Qi_dia_corr_fqc","Qi_dia_corr_err"),
        ("photons","Qc_dia_corr_fixed","absQc_err"),
        ("photons","Ql","Ql_err")]
    plot_list = assemble_plot_df( powerQ_result, xy_cols )
    labels = ["Internal Loss","Coupling Loss", "Loaded Loss"]
    for i, p in enumerate(plot_list):
        ax.errorbar( p["x"], 1/p["y"],yerr=1/p["y"]*p["yerr"]/p["y"], fmt="o", ms=5, label=f"{labels[i]}" )
    ax.set_title(f"{cav_label}_refined")
    fig.legend()
    if output_fd != None :
        plt.savefig(f"{output_fd}/{cav_label}_powerloss_refined.png")
        plt.close()
    else:
        plt.show()

def plot_singleRes_powerQ_free( powerQ_result, cav_label=None, output_fd=None ):

    fig, ax = _plotFrame_singleRes_power()
    ax.set_ylabel("Quality factor")

    xy_cols = [("photons","Qi_dia_corr","Qi_dia_corr_err"),
        ("photons","Qc_dia_corr","absQc_err"),
        ("photons","Ql","Ql_err")]
    plot_list = assemble_plot_df( powerQ_result, xy_cols )
    labels = ["Internal Q","Coupling Q", "Loaded Q"]
    for i, p in enumerate(plot_list):
        ax.errorbar( p["x"], p["y"], yerr=p["yerr"], fmt="o", ms=5, label=f"{labels[i]}" )
    ax.set_title(f"{cav_label}_free")
    fig.legend()
    if output_fd != None :
        plt.savefig(f"{output_fd}/{cav_label}_powerQ_free.png")
        plt.close()
    else:
        plt.show()

def plot_singleRes_powerloss_free( powerQ_result, cav_label=None, output_fd=None ):

    fig, ax = _plotFrame_singleRes_power()
    ax.set_ylabel("Loss")

    xy_cols = [("photons","Qi_dia_corr","Qi_dia_corr_err"),
        ("photons","Qc_dia_corr","absQc_err"),
        ("photons","Ql","Ql_err")]
    plot_list = assemble_plot_df( powerQ_result, xy_cols )
    labels = ["Internal Loss","Coupling Loss", "Loaded Loss"]
    for i, p in enumerate(plot_list):
        ax.errorbar( p["x"], 1/p["y"], yerr=1/p["y"]*p["yerr"]/p["y"], fmt="o", ms=5, label=f"{labels[i]}" )
    ax.set_title(f"{cav_label}_free")
    fig.legend()
    if output_fd != None :
        plt.savefig(f"{output_fd}/{cav_label}_powerloss_free.png")
        plt.close()
    else:
        plt.show()

def _plotFrame_singleRes_power( )->Tuple[plt.Figure,plt.Axes]:
    fig = plt.figure(facecolor='white')
    plt.subplots_adjust(left=0.15,
                        bottom=0.15, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.25, 
                        hspace=0.25)
    
    axObj = fig.add_subplot()
    axObj.set_xlabel("Photons")
    axObj.set_xscale("log")
    axObj.set_yscale("log")
    
    axObj.legend()
    # fig.suptitle( title )
    return fig, axObj