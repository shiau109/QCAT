import numpy as np
from sklearn.mixture import GaussianMixture
import xarray as xr

from qcat.analysis.state_discrimination.cluster_training import GMMClusterTrainer, G1DClusterTrainer, GMMLabelAssign, GMMLabelMap
from qcat.analysis.state_discrimination.discriminator import get_proj_distance, GMMDiscriminator, G1DDiscriminator
from qcat.analysis.base import QCATAna

class GMMROFidelity( QCATAna ):

    def __init__( self ):
        super().__init__()
        
        self.discriminator = None
    

    @property
    def centers ( self )->np.ndarray:
        return self.discriminator.cluster_model.means_
    
    def _import_data( self, data:xr.DataArray ):
        """        
        Used to check input data \n
        Parameters:\n 
        \t data: DataArray object with coords ["mixer", "prepared_state", "index"]. \n
        Return:\n 

        """
        data_trans = data.transpose("prepared_state", "index", "mixer")
        self.raw_data = data_trans
        return self.raw_data
        

    def _start_analysis( self ):
        """ Used to start analysis which might be time costly """

        
        training_DataArray = self.raw_data.stack( new_index=('index', 'prepared_state'))
        training_DataArray = training_DataArray.transpose( "new_index", "mixer" )
        training_data = training_DataArray.values
 
        self.cluster_trainer = GMMClusterTrainer()
        self.cluster_trainer._import_data( training_data )
        self.cluster_trainer._start_analysis()

        trained_model = self.cluster_trainer._export_result()
        
        label_assign = self._create_state_label_mapping( trained_model )
        self.label_map = GMMLabelMap( label_assign )

        self.discriminator = GMMDiscriminator( trained_model, self.label_map )
        self.discriminator._import_data( self.raw_data )
        self.discriminator._start_analysis()
    
        self.state_data_array = self.discriminator._export_result()

        self.state_population = np.apply_along_axis(np.bincount, axis=-1, arr=self.state_data_array, minlength=2)
        self.state_probability = self.state_population/ self.state_data_array.shape[-1]

    def _export_result( self ):
        """ Export result with a format from analysis"""
        pass


        # return self
    def _create_state_label_mapping( self, trained_model )->GMMLabelAssign:

        ave_iq = self.raw_data.mean(dim="index")
        label_assign = GMMLabelAssign( trained_model )
        label_assign._import_data(ave_iq)
        label_assign._start_analysis()
        return label_assign
    
    def export_G1DROFidelity( self ):
        data = self.raw_data.transpose("mixer","prepared_state",  "index").values
        centers_2d, centers1d, sigmas = self.discriminator._export_1D_paras()
        train_data_proj = get_proj_distance(centers_2d.transpose(), data)
        dataset_proj = xr.DataArray(train_data_proj, coords= [ ("prepared_state",[0,1]), ("index",np.arange(data.shape[2]))] )

        g1d_fidelity = G1DROFidelity()
        # Set parameter for G1DDiscriminator which from GMMROFidelity result
        g1d_fidelity.discriminator = self.discriminator._export_G1DDiscriminator()        
        g1d_fidelity._import_data(dataset_proj)
        g1d_fidelity._start_analysis()
        return g1d_fidelity

class G1DROFidelity( QCATAna ):

    def __init__( self ):
        super().__init__()
        
        # self.label_map = None
        self.discriminator = None
    

    def _import_data( self, data:xr.DataArray ):
        """        
        Used to check input data \n
        Parameters:\n 
        \t data: DataArray object with coords [ "prepared_state", "index"]. \n
        Return:\n 

        """
        data_trans = data.transpose("prepared_state", "index")
        self.raw_data = data_trans
        return self.raw_data
        

    def _start_analysis( self ):
        """ Used to start analysis which might be time costly """

        


        if self.discriminator is None:
            training_DataArray = self.raw_data.stack( new_index=('index', 'prepared_state'))
            training_DataArray = training_DataArray.transpose( "new_index", "mixer" )
            # print(training_DataArray)
            training_data = training_DataArray.values
    
            self.cluster_trainer = G1DClusterTrainer()
            self.cluster_trainer._import_data( training_data )
            self.cluster_trainer._start_analysis()

            trained_model = self.cluster_trainer._export_result()

            label_assign = self._create_state_label_mapping( trained_model )
            label_map = GMMLabelMap( label_assign )
            self.discriminator = G1DDiscriminator( trained_model, label_map )
            
        self.discriminator._import_data( self.raw_data )
        self.discriminator._start_analysis()

        self.g1d_dist = []
        for i in range(np.array(self.raw_data.coords['prepared_state']).shape[0]):
            self.g1d_dist.append( self._fit_distribution(self.raw_data[i]) )
               

        self.state_data_array = self.discriminator._export_result()

        self.state_population = np.apply_along_axis(np.bincount, axis=-1, arr=self.state_data_array, minlength=2)
        self.state_probability = self.state_population/ self.state_data_array.shape[-1]

    def _export_result( self ):
        """ Export result with a format from analysis"""
        pass

        # return self
    def _create_state_label_mapping( self, trained_model )->GMMLabelAssign:

        ave_iq = self.raw_data.mean(dim="index")

        label_assign = GMMLabelAssign( trained_model )
        label_assign._import_data(ave_iq)
        label_assign._start_analysis()
        return label_assign
    
    def _fit_distribution( self, data, bin_center=None ):
        """
        input numpy array with shape (n,)
        n is point number
        """

        if type(bin_center) == type(None):
            params = self.discriminator.cluster_model.param_hints
            sigma = np.array([params["g0_sigma"]["value"],params["g1_sigma"]["value"]])
            pos = np.array( [ params["g0_center"]["value"], params["g1_center"]["value"] ])
            bin_center = np.linspace(pos[0]-5.*sigma[0], pos[1]+5.*sigma[1],100)

        width = bin_center[1] -bin_center[0]
        bins = np.append(bin_center,bin_center[-1]+width) -width/2
        # print("bins", bins[0], bins[-1])
        hist, _ = np.histogram(data, bins, density=True)
        params = self.discriminator.cluster_model.make_params()
        # print("hist", hist)

        result = self.discriminator.cluster_model.fit(hist,params,x=bin_center)
        probability = self._get_gaussian_area(result)
        
        return probability, bin_center, hist, result


    def _get_gaussian_area( self, result ):

        sigma = np.array([ result.params["g0_sigma"], result.params["g1_sigma"]])
        peak_value = np.array([ result.params["g0_amplitude"], result.params["g1_amplitude"]])
        area = peak_value * sigma*np.sqrt(2*np.pi)
        probability = area/np.sum(area) 
        return probability


    

if __name__ == '__main__':
    snr = 2.

    g = (0, 0)
    e = (1, 1)
    std_dev = 1/snr
    num_samples = 1000

    # Generate random numbers from the normal distribution
    Ig = np.random.normal(g[0], std_dev, num_samples)    
    Qg = np.random.normal(g[1], std_dev, num_samples) 
    Ie = np.random.normal(e[0], std_dev, num_samples)    
    Qe = np.random.normal(e[1], std_dev, num_samples) 
    # ( ["state","mixer","index"], 
    data =  np.array([[Ig, Qg], [Ie, Qe]])
    coords= [ ("compressed", [0,1]), ("mixer",["I","Q"]), ("index",np.arange( num_samples ))]

    dataarray = xr.DataArray( data=data, coords=coords )

    fidelity_qcat = GMMROFidelity()
    fidelity_qcat._import_data( dataarray )
    fidelity_qcat._start_analysis()
    print(fidelity_qcat.state_data_array)
    print(fidelity_qcat.state_population)
    print(fidelity_qcat.state_probability)

    import matplotlib.pyplot as plt
    plt.plot( dataarray.sel(mixer="I", compressed=0).values, dataarray.sel(mixer="Q", compressed=0).values, "o" )
    plt.plot( dataarray.sel(mixer="I", compressed=1).values, dataarray.sel(mixer="Q", compressed=1).values, "o" )
    # plt.plot( Ie, Qe, 'o' )
    plt.show()