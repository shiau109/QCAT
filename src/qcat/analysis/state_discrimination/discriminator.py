
import numpy as np
from sklearn.mixture import GaussianMixture
from qcat.analysis.state_discrimination.cluster_training import GMMLabelMap
from qcat.analysis.base import QCATAna
import xarray as xr


class Discriminator( QCATAna ):

    def __init__( self, cluster_model:GaussianMixture, label_map:GMMLabelMap ):
        self.cluster_model = cluster_model
        self.label_map = label_map

    def _import_data( self, *args, **kwargs ):
        """ Used to check input data """
        pass


    def _start_analysis( self, *args, **kwargs ):
        """ Used to start analysis which might be time costly """
        
        self.label_map = GMMLabelMap( self.label_assign )
        self.label_map._import_data(label_darray)
        self.label_map._start_analysis()
        state_darray = self.label_map._export_result()

        self.state_population = np.apply_along_axis(np.bincount, axis=-1, arr=state_darray, minlength=2)
        self.state_probability = self.state_population/ state_darray.shape[-1]

    def _export_result( self, *args, **kwargs ):
        """ Export result with a format from analysis"""
        pass
    
    

    
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

from lmfit.models import GaussianModel, Model
from lmfit.model import ModelResult

class G1DDiscriminator():

    def __init__( self, label_map:GMMLabelMap ):
        self.label_map = label_map
        gm0 = GaussianModel(prefix="g0_", name="g0")
        gm1 = GaussianModel(prefix="g1_", name="g1")
        self.__cluster_model = gm0 + gm1
        self.threshold = None

    @property
    def cluster_model( self )->Model:
        return self.__cluster_model 
       
    @cluster_model.setter
    def cluster_model( self, val ):

        if isinstance( val, Model):
            self.__cluster_model = val

        # elif isinstance( val, dict ):
        #     try:
        #         self.__model = GaussianMixture(n_components=2, random_state=0)
        #         self._rebuild_model(val)
        #     except:
        #         print("Can't rebuild model")
        else:
            print("Not correct type, please use GaussianModel object")

    def _import_data( self, data:xr.DataArray ):
        """        
        Used to check input data \n
        Parameters:\n 
        \t data: Dataset with var name "data", coords ["index"]. \n
        Return:\n 

        """
        
        for check_axis in ["index"]:
            if check_axis in list(data.dims):
                self.raw_data = data
                return self.raw_data
            else:
                print(f"No {check_axis} axis in dataset")
        

    def _start_analysis( self ):
        """ Used to start analysis which might be time costly """

        # training_darray = self.raw_data.stack( new_index=('index', 'prepared_state'))

        params = self.cluster_model.param_hints
        # print(params,params["g0_center"]["value"],params["g1_center"]["value"])
        self.threshold = (params["g0_center"]["value"]+params["g1_center"]["value"])/2
        self.signal = np.abs(params["g0_center"]["value"]-params["g1_center"]["value"])
        self.noise = np.array([ params["g0_sigma"]["value"], params["g1_sigma"]["value"]])
        self.snr = self.signal/np.mean(self.noise)
        
        # print(self.threshold)

        label_data = ( self.raw_data.values > self.threshold ).astype(int)

        # Create the DataArray with named dimensions
        label_darray = xr.DataArray( label_data, coords=self.raw_data.coords )
        # print(label_darray)

        self.label_map._import_data(label_darray)
        self.label_map._start_analysis()
        state_darray = self.label_map._export_result()

        # self.state_population = np.apply_along_axis(np.bincount, axis=-1, arr=state_darray, minlength=2)
        # self.state_probability = self.state_population/ state_darray.shape[-1]
        self.result = state_darray
        return state_darray

    def _export_result( self ):
        """ Export result with a format from analysis"""
        return self.result



from qcat.analysis.base import QCATAna
from qcat.analysis.state_discrimination.cluster_training import GMMLabelMap

class GMMDiscriminator(QCATAna):

    def __init__( self, cluster_model:GaussianMixture, label_map:GMMLabelMap ):
        super().__init__()
        self.__cluster_model = cluster_model
        self.label_map = label_map
    
    @property
    def cluster_model( self )->GaussianMixture:
        return self.__cluster_model 
       
    @cluster_model.setter
    def cluster_model( self, val ):

        if isinstance( val, GaussianMixture):
            self.__cluster_model = val

        # elif isinstance( val, dict ):
        #     try:
        #         self.__model = GaussianMixture(n_components=2, random_state=0)
        #         self._rebuild_model(val)
        #     except:
        #         print("Can't rebuild model")
        else:
            print("Not correct type, please use GaussianMixture object")

    def _import_data( self, data:xr.DataArray ):
        """        
        Used to check input data \n
        Parameters:\n 
        \t data: Dataset with var name "data", coords ["mixer", "index"]. \n
        Return:\n 

        """
        if "mixer" in list(data.dims):
            self.raw_data = data
            return self.raw_data
        else:
            print("No mixer axis in dataset")
        

    def _start_analysis( self ):
        """ Used to start analysis which might be time costly """

        training_darray = self.raw_data.stack( new_index=('index', 'prepared_state'))
        training_darray = training_darray.transpose( "new_index", "mixer" )

        self.__predict_label = self.cluster_model.predict( training_darray.values )

        # Create the DataArray with named dimensions
        label_darray = xr.DataArray( self.__predict_label, dims=["new_index"] )
        # print(label_darray)
        label_darray = label_darray.assign_coords( new_index=training_darray.coords["new_index"])
        label_darray = label_darray.unstack("new_index")
        label_darray = label_darray.transpose( "prepared_state", "index" )
        # print(label_darray)

        self.label_map._import_data(label_darray)
        self.label_map._start_analysis()
        state_darray = self.label_map._export_result()
        self.state_population = np.apply_along_axis(np.bincount, axis=-1, arr=state_darray, minlength=2)
        self.state_probability = self.state_population/ state_darray.shape[-1]
        self.result = state_darray
        return state_darray

    def _export_result( self ):
        """ Export result with a format from analysis"""
        return self.result

    def _export_1D_paras( self ):


        mapping_arr = self.label_map.label_assign.mapping_arr
        sigma = [None]*2
        centers_2d = [None]*2
        for label_i in [0,1]:
            state_i = mapping_arr[label_i]
            sigma[state_i] = get_sigma(self.__cluster_model.covariances_[label_i])
            centers_2d[state_i] = self.__cluster_model.means_[label_i]

        sigmas_1d = np.array(sigma)
        centers_2d = np.array(centers_2d)
        centers_1d = get_proj_distance(centers_2d.transpose(), centers_2d.transpose())    
        return centers_2d, centers_1d, sigmas_1d

    def _export_G1DDiscriminator( self )->G1DDiscriminator:
        g1d_discriminator = G1DDiscriminator( self.label_map )
        centers_2d, centers_1d, sigmas_1d = self._export_1D_paras()
        g1d_discriminator.cluster_model.set_param_hint('g0_center',value=centers_1d[0], vary=False)
        g1d_discriminator.cluster_model.set_param_hint('g1_center',value=centers_1d[1], vary=False)
        g1d_discriminator.cluster_model.set_param_hint('g0_sigma',value=sigmas_1d[0], vary=False)
        g1d_discriminator.cluster_model.set_param_hint('g1_sigma',value=sigmas_1d[1], vary=False)
        return g1d_discriminator
            


    


