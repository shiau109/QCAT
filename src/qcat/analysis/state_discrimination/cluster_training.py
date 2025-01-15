import numpy as np
from numpy.typing import NDArray

from sklearn.mixture import GaussianMixture
import xarray as xr

from qcat.analysis.base import QCATAna

class GMMClusterTrainer( QCATAna ):

    def __init__( self ):
        super().__init__()
        self.__model = GaussianMixture(n_components=2, random_state=0)

    @property
    def model( self )->GaussianMixture:
        return self.__model
    
    def _import_data( self, data:NDArray ):
        """        
        Used to check input data \n
        Parameters:\n 
        \t data: numpy array with shape (n,2). \n
        axis 0 (n) is shot number \n
        axis 1 (2) is IQ channel data \n

        """
        if data.ndim == 2 and data.shape[-1]==2:
            self.raw_data = data
            return self.raw_data
        else:
            print("training data should be 2D array with shape (n,2)")

    def _start_analysis( self ):
        try:
            self.__model.fit( self.raw_data )
        except:
            print(f"GaussianMixture fit fail")
        self.__model.weights_ = [0.5,0.5]

        self.result = self.__model
        return self.result
    
    def _export_result( self ):

        return self.result
    

    def _model_paras( self ):
        """
        four para in dict
        means
        weights
        covariances
        precisions_cholesk
        """
        output_dict = {
            "means":self.__model.means_,
            "weights":self.__model.weights_,
            "covariances":self.__model.covariances_,
            "precisions_cholesky":self.__model.precisions_cholesky_,
        }

        return output_dict

    def _rebuild_model( self, paras:dict ):

        self.__model.means_ = paras["means"]
        self.__model.weights_ = paras["weights"]
        self.__model.covariances_ = paras["covariances"]
        self.__model.precisions_cholesky_ = paras["precisions_cholesky"]

class GMMLabelAssign():
    def __init__( self, cluster_model:GaussianMixture ):
        self.cluster_model = cluster_model
        

    def _import_data( self, data:xr.DataArray ):
        """        
        Input ground state data\n
        Parameters:\n 
        \t data: numpy array with shape (2,2). \n
        axis 0 (2) is prepare state \n
        axis 1 (2) is IQ channel data \n

        """
        if "mixer" not in list(data.dims):
            print("No axis is called 'mixer' in coords")
            return None
        if "prepared_state" not in list(data.dims):
            print("No axis is called 'prepared_state' in coords")
            return None     
        data = data.transpose("prepared_state","mixer")
        self.raw_data = data

    def _start_analysis( self ):
        prepare_state = self.raw_data.coords["prepared_state"].values
        state_iq_point = self.raw_data.values
        predict_state = self.cluster_model.predict(state_iq_point)
        mapping_arr = predict_state[prepare_state]
        # self.state_map = state_map
        self.mapping_arr = mapping_arr
        self.result = mapping_arr
        return self.result
    
    def _export_result( self ):

        return self.result

class GMMLabelMap():
    def __init__( self, label_assign:GMMLabelAssign ):
        self.label_assign = label_assign
        

    def _import_data( self, data:xr.DataArray ):
        """        
        Input ground state data\n
        Parameters:\n 
        \t data: numpy array with shape (2,2). \n
        axis 0 (2) is prepare state \n
        axis 1 (2) is IQ channel data \n

        """
        if "prepared_state" not in list(data.dims):
            print("No axis is called 'prepared_state' in coords")
            data = None     
        self.raw_data = data

    def _start_analysis( self ):

        mapping_arr = np.array(self.label_assign.mapping_arr)
        label_data = self.raw_data.values

        # print(self.label_assign.state_map,self.label_assign.label_map)
        # flat_array = label_data.flatten()
        state_data = mapping_arr[label_data]#.reshape(label_data.shape)
        self.result = state_data
        return self.result

    def _export_result( self )->np.ndarray:

        return self.result


from lmfit.models import GaussianModel

class G1DClusterTrainer( QCATAna ):

    def __init__( self ):
        super().__init__()
        gm0 = GaussianModel(prefix="g0_", name="g0")
        gm1 = GaussianModel(prefix="g1_", name="g1")
        self.__model = gm0 + gm1

        self.mu = None
        self.sigma = None
        self.guess_vary = False

    @property
    def model( self )->GaussianMixture:
        return self.__model
    
    def _import_data( self, data:NDArray ):
        """
        input numpy array with shape (2,N)
        shape[0]: state
        shape[1]: N element is point number
        """
        self.training_data = data
        # print("guess",guess)

    def _start_analysis( self ):

        if self.mu is None:
            self.mu = np.mean(self.training_data, axis=1)
        if self.sigma is None:
            self.sigma = np.std( self.training_data, axis=1 )

        mu = self.mu
        sigma = self.sigma
        sigma_mean = np.mean( sigma )
        guess_vary = self.guess_vary

        dis = np.abs(mu[1]-mu[0])
        est_peak_h = 1/sigma_mean

        # print("est_peak_h",est_peak_h)
        
        bin_center = np.linspace(-(dis+2.5*sigma_mean), dis+2.5*sigma_mean,50)
        width = bin_center[1] -bin_center[0]
        bins = np.append(bin_center,bin_center[-1]+width) -width/2

        hist, bin_edges = np.histogram(self.training_data.flatten(), bins, density=True)

        self.__model.set_param_hint('g0_center',value=mu[0], vary=guess_vary)
        self.__model.set_param_hint('g1_center',value=mu[1], vary=guess_vary)
        self.__model.set_param_hint('g0_amplitude',value=est_peak_h, min=0, max=est_peak_h*2, vary=True)
        self.__model.set_param_hint('g1_amplitude',value=est_peak_h, min=0, max=est_peak_h*2, vary=True)
        self.__model.set_param_hint('g0_sigma',value=sigma[0], vary=guess_vary)
        self.__model.set_param_hint('g1_sigma',value=sigma[1], vary=guess_vary)

        params = self.__model.make_params()
        result = self.__model.fit(hist,params,x=bin_center)
        self.result = self.__model
        return self.result
    
    def _export_result( self ):

        return self.result

def get_sigma( covariances ):
    v, w = np.linalg.eigh(covariances)
    v = np.sqrt(v/2)
    return  np.sqrt(v[0]**2+v[1]**2)
    
def output_1D_paras( self ):
    sigma_0 = get_sigma(self.__model.covariances_[0])
    sigma_1 = get_sigma(self.__model.covariances_[1])
    sigmas_1d = np.array([sigma_0,sigma_1]) 
    centers = self.__model.means_ 
    # centers_1d = get_proj_distance(centers.transpose(), centers.transpose())    
    return centers, sigmas_1d