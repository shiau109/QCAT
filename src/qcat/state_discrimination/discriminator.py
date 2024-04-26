
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
        return np.bincount(self.__predict_label)

def train_model( data ):
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
    my_model.relabel_model(data[0])
    return my_model

from lmfit.models import GaussianModel
from lmfit.model import ModelResult
class Discriminator1D():

    def __init__( self ):
        gm0 = GaussianModel(prefix="g0_", name="g0")
        gm1 = GaussianModel(prefix="g1_", name="g1")
        model = gm0 + gm1
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

    def get_label( self ):
        """
        input numpy array with shape (n,2)
        n is point number
        """

        return self.__predict_label

    def get_state_population( self ):
        return np.bincount(self.__predict_label)