
import numpy as np
from sklearn.mixture import GaussianMixture

class Discriminator():

    def __init__( self ):
        self.__model = GaussianMixture(n_components=2, random_state=0)

    @property
    def model( self )->GaussianMixture:
        return self.__model

    def import_training_data( self, data ):
        """
        input numpy array with shape (n,2)
        n is point number
        """
        self.training_data = data
        self.__model.fit(data)
        self.__model.weights_ = [0.5,0.5]

        # return self
    def relabel_model( self, ground_data ):
        """
        input numpy array with shape (2,n)
        """

        gp = np.array([np.mean(ground_data, axis=1)])
        # print( gp )
        # print( gp.shape )

        # print(self.__model.predict( gp ))
        if self.__model.predict( gp ) == 1:
            self.__model.means_ = np.flip(self.__model.means_,0)
            self.__model.weights_ = np.flip(self.__model.weights_,0)
            self.__model.covariances_ = np.flip(self.__model.covariances_,0)
            self.__model.precisions_cholesky_ = np.flip(self.__model.precisions_cholesky_,0)


    def output_paras( self ):
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

    def rebuild_model( self, paras:dict ):

        self.__model.means_ = paras["means"]
        self.__model.weights_ = paras["weights"]
        self.__model.covariances_ = paras["covariances"]
        self.__model.precisions_cholesky_ = paras["precisions_cholesky"]


    def get_prediction( self, data ):
        """
        input numpy array with shape (n,2)
        n is point number
        """
        self.__predict_label = self.model.predict( data )
        return self.__predict_label

    def get_state_population( self, data ):
        self.get_prediction(data)
        s_pop = np.bincount(self.__predict_label)
        if s_pop.shape[-1] == 1:
            s_pop =  np.append(s_pop, [0])
        return s_pop
    
    def output_1D_paras( self ):
        sigma_0 = get_sigma(self.__model.covariances_[0])
        sigma_1 = get_sigma(self.__model.covariances_[1])
        sigmas = np.array([sigma_0,sigma_1]) 
        centers = self.__model.means_    
        return centers, sigmas
    
def get_sigma( covariances ):
    v, w = np.linalg.eigh(covariances)
    v = np.sqrt(v/2)
    return  np.sqrt(v[0]**2+v[1]**2)

def get_proj_distance( proj_pts:np.ndarray, iq_data ):
    """
    proj_pts with shape (2,2)\n
    shape[0] is IQ\n
    shape[1] is state\n
    iq_data with shape (2,N,M...)\n
    shape[0] is IQ\n
    shape[1] is point idx\n
    Return
    with shape (N,M...)
    """
    # Matrix method
    # p0 = proj_pts[0]
    # p1 = proj_pts[1]
    # ref_point = (p0+p1)/2
    # shifted_iq = iq_data.transpose()-ref_point
    # v_01 = p1 -p0 

    # v_01_dis = np.sqrt( v_01[0]**2 +v_01[1]**2 )

    # shifted_iq = shifted_iq.transpose()
    # v_01 = np.array([v_01])
    # projectedDistance = v_01@shifted_iq/v_01_dis
    # return projectedDistance[0]

    z_pos = proj_pts[0]+1j*proj_pts[1]
    z_dir = z_pos[1]-z_pos[0]


    z_data = iq_data[0]+1j*iq_data[1]
    projectedDistance = z_data*np.exp( -1j*np.angle(z_dir) )

    return projectedDistance.real

def train_GMModel( data ):
    """
    data type
    3 dim with shape (2*2*N)
    shape[0] is I and Q
    shape[1] is prepare state
    shape[2] is N times single shot
    
    """
    new_shape = (data.shape[0], -1)  # -1 lets numpy calculate the necessary size
    training_data = data.reshape(new_shape)
    my_model = Discriminator()
    my_model.import_training_data(training_data.transpose())

    my_model.relabel_model(np.array([data[0][0],data[1][0]]))
    return my_model

from lmfit.models import GaussianModel
from lmfit.model import ModelResult
class Discriminator1D():

    def __init__( self ):
        gm0 = GaussianModel(prefix="g0_", name="g0")
        gm1 = GaussianModel(prefix="g1_", name="g1")
        self.__model = gm0 + gm1

    @property
    def model( self )->GaussianMixture:
        return self.__model

    def import_training_data( self, data, guess=None, guess_vary=False ):
        """
        input numpy array with shape (2,N)
        shape[0]: state
        shape[1]: N element is point number
        """
        
        self.training_data = data
        print("guess",guess)

        if not isinstance(type(guess),type(None)):
            mu, sigma = guess
            print("mu, sigma", mu, sigma)
        else:
            mu = np.mean(data, axis=1)
            sigma = np.std( data, axis=1 )

        sigma_mean = np.mean( sigma )

        dis = np.abs(mu[1]-mu[0])
        est_peak_h = 1/sigma_mean
        print("est_peak_h",est_peak_h)
        bin_center = np.linspace(-(dis+2.5*sigma_mean), dis+2.5*sigma_mean,50)

        width = bin_center[1] -bin_center[0]
        bins = np.append(bin_center,bin_center[-1]+width) -width/2

        hist, bin_edges = np.histogram(data.flatten(), bins, density=True)

        self.__model.set_param_hint('g0_center',value=mu[0], vary=guess_vary)
        self.__model.set_param_hint('g1_center',value=mu[1], vary=guess_vary)
        self.__model.set_param_hint('g0_amplitude',value=est_peak_h, min=0, max=est_peak_h*2, vary=True)
        self.__model.set_param_hint('g1_amplitude',value=est_peak_h, min=0, max=est_peak_h*2, vary=True)
        self.__model.set_param_hint('g0_sigma',value=sigma[0], vary=guess_vary)
        self.__model.set_param_hint('g1_sigma',value=sigma[1], vary=guess_vary)

        params = self.__model.make_params()
        self._results = self.__model.fit(hist,params,x=bin_center)

        return self._results

    def fit_distribution( self, data, bin_center=None ):
        """
        input numpy array with shape (n,)
        n is point number
        """

        if type(bin_center) == type(None):
            sigma = np.array([self._results.params["g0_sigma"],self._results.params["g1_sigma"]])
            pos = np.array([self._results.params["g0_center"],self._results.params["g1_center"]])
            bin_center = np.linspace(pos[0]-5.*sigma[0], pos[1]+5.*sigma[1],100)

        width = bin_center[1] -bin_center[0]
        bins = np.append(bin_center,bin_center[-1]+width) -width/2
        hist, _ = np.histogram(data, bins, density=True)
        params = self.__model.make_params()
        result = self.__model.fit(hist,params,x=bin_center)
        return bin_center, hist, result
    
def get_probability( result ):
    sigma = np.array([ result.params["g0_sigma"], result.params["g1_sigma"]])
    peak_value = np.array([ result.params["g0_amplitude"], result.params["g1_amplitude"]])
    area = peak_value * sigma*np.sqrt(2*np.pi)
    probability = area/np.sum(area) 
    return probability

def train_1DGaussianModel( training_data, guess=None )->Discriminator1D:
    """
    data type
    3 dim with shape (2*N)
    shape[0] is prepare state
    shape[1] is N times single shot
    
    """
    my_model = Discriminator1D()
    # combined_training_data = training_data.reshape((2*training_data.shape[-1]))
    my_model.import_training_data( training_data, guess=guess, guess_vary=False)
    return my_model

def p01_to_Teff( p01, frequency ):
    """
    Parameters:\n
    frequency unit in Hz
    """
    n = p01/(1-2*p01)
    HDB = (1.0546/1.3806) *1e-11 # 1.0546e-34 / 1.3806e-23
    effective_T = frequency*2*np.pi*HDB/np.log(1+1/n)

    return effective_T