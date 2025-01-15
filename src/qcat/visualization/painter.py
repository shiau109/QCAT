import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from xarray import Dataset

class Painter(ABC):

    def __init__( self ):
        self.output_fig = []
        self.mode = "ave"

    
    @abstractmethod
    def plot( self, show:bool=True ):
        pass
        return self.output_fig
