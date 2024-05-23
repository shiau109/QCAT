

from qcat.utility.io_translator import mat_to_numpy, to_dataset
import numpy as np

from .analysis_method import *
from qcat.visualization.photon_dep_loss import *
# from typing import List
from xarray import Dataset
from qcat.utility.file_structure import save_power_dep


class ResonatorData():
    def __init__( self, freq:np.ndarray, zdata:np.ndarray, power:float=None ):
        self.__freq = freq
        self.__rawdata = zdata
        self.power = power

    @property
    def rawdata( self )->np.ndarray:
        """
        Give a complex number 1d numpy array
        """
        return self.__rawdata
    @rawdata.setter
    def rawdata( self, value:np.ndarray ):
        self.__rawdata = value

    @property
    def freq( self )->np.ndarray:
        """
        Give a real number 1d numpy array
        """
        return self.__freq
    @freq.setter
    def freq( self, value:np.ndarray ):
        self.__freq = value

    def fit( self, delay=None, Qc_real=None, alpha=None, amp_norm=None):

        FitResonator = notch_port()  
        freq = self.freq
        zdata = self.rawdata

        delay_auto, amp_norm_auto, alpha_auto, fr_auto, Ql_auto, A2, frcal =\
                FitResonator.do_calibration(freq,zdata,fixed_delay=delay)
        if amp_norm == None:
            amp_norm = amp_norm_auto
        if delay == None:
            # print("auto fit electrical delay")
            delay = delay_auto
        if alpha == None:
            alpha = alpha_auto


        fr = fr_auto
        Ql = Ql_auto
        zdata_norm = FitResonator.do_normalization(freq,zdata,delay,amp_norm,alpha,A2,frcal)
        FitResonator.fitresults = FitResonator.circlefit(freq,zdata_norm,fr,Ql)
        fit_results = FitResonator.fitresults

        fit_results["A"] = amp_norm
        fit_results["alpha"] = alpha
        fit_results["delay"] = delay

        input_power = self.power
        if input_power != None:
            fit_results["photons"] = FitResonator.get_photons_in_resonator(input_power)
        
        if Qc_real != None:
            fit_results["Qi_dia_corr_fqc"] = 1./(1./fit_results["Ql"]-1./Qc_real)
            fit_results["Qc_dia_corr_fixed"] = Qc_real
            
        fit_curve_norm = FitResonator._S21_notch(
        freq,fr=fit_results["fr"],
        Ql=fit_results["Ql"],
        Qc=fit_results["absQc"],
        phi=fit_results["phi0"]
        )

        self.zdata_norm = zdata_norm
        self.fit_curve_norm = fit_curve_norm
        # print(zdata_norm.shape, fit_curve_norm.shape)
        return fit_results, zdata_norm, fit_curve_norm
    
    

class PhotonDepResonator():
    def __init__( self, name:str ):
        self.name = name
        self._resonator_data = []
        self._result = []
    
    @property
    def resonator_data( self )->List[ResonatorData]:
        return self._resonator_data
    
    def import_mat( self, file_name, attenuation ):
        power, freq, s21 = mat_to_numpy( file_name )
        zdata_2d = s21.transpose()

        mk_power = power-attenuation
        for i, p in enumerate(mk_power):
            r_data = ResonatorData( freq*1e9, zdata_2d[i], p )
            self.resonator_data.append( r_data )

    def import_array( self, power, freq, s21 ):

        mk_power = power
        for i, p in enumerate(mk_power):
            r_data = ResonatorData( freq*1e9, s21[i], p )
            self.resonator_data.append( r_data )


    def free_analysis( self, output_fd ):

        print(f"{self.name} start free analysis")
        
        alldata_results = []
        alldata_plot = []
        alldata_power = []
        for r_data in self.resonator_data:

            df_fitParas, zdatas_norm, fitCurves_norm = r_data.fit()
            alldata_results.append(df_fitParas)
            alldata_power.append(r_data.power)
            alldata_plot.append((r_data.freq,zdatas_norm,fitCurves_norm))

        df_powerQ_results = pd.DataFrame(alldata_results)
        df_powerQ_results.Name = self.name

        ## Save result
        colors = plt.cm.rainbow(np.linspace(0, 1, len(alldata_power)))
        plot_resonatorFitting( alldata_power, alldata_plot, f"{self.name}_free", colors, output_fd=output_fd)
        save_power_dep(df_powerQ_results, f"{output_fd}/free_result.csv")

        return df_powerQ_results
    
    def refined_analysis( self, output_fd ):

        df_fitResult_free = self.free_analysis(output_fd)
        delay_refined, amp_refined, Qc_refined, alpha_refined = get_fixed_paras( df_fitResult_free )

        print(f"delay: {delay_refined:.3e},\namp_norm: {amp_refined:.3e},\nQc: {Qc_refined:.3e},\nalpha: {alpha_refined:.3e}")
        
        alldata_results = []
        alldata_power = []
        alldata_plot = []
        for r_data in self.resonator_data:
            df_fitParas, zdatas_norm, fitCurves_norm = r_data.fit(delay=delay_refined,Qc_real=Qc_refined)
            alldata_results.append(df_fitParas)
            alldata_power.append(r_data.power)
            alldata_plot.append((r_data.freq,zdatas_norm,fitCurves_norm))
        df_fitResult_fixed = pd.DataFrame(alldata_results)
        df_fitResult_fixed.Name = self.name
        ## Save result
        colors = plt.cm.rainbow(np.linspace(0, 1, len(alldata_power)))
        plot_resonatorFitting( alldata_power, alldata_plot, f"{self.name}_refined", colors, output_fd=output_fd)
        save_power_dep(df_fitResult_fixed, f"{output_fd}/refined_result.csv")

        return df_fitResult_fixed
