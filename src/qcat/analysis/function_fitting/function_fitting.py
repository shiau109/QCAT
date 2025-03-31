
from abc import ABC, abstractmethod



class FunctionFitting( ABC ):
    def __init__():
        pass

    @abstractmethod
    def model_function( **arg ):
        pass

    @abstractmethod
    def guess():
        pass

    @abstractmethod
    def fit():
        pass
    
    def fitting_curve( self, x ):
        return self.model(x)