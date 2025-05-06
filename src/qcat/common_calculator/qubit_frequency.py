
import numpy as np
from abc import ABC, abstractmethod
class Qubit(ABC):
    def __init__():
        pass

    @abstractmethod
    def frequency():
        pass


class Transmon(Qubit):
    def __init__( self, Ej, Ec ):
        self.Ej = Ej
        self.Ec = Ec
    
    def frequency( self ):
        return np.sqrt( 8*self.Ej*self.Ec ) -self.Ec

class FluxTunablTransmon(Transmon):
    def __init__( self, Ej_sum, Ec, junction_asymmetry_parameter=0 ):
        self.Ej_sum = Ej_sum
        self.Ec = Ec
        self.junction_asymmetry_parameter = junction_asymmetry_parameter

    def effective_Ej( self, flux_quantum_ratio ):
        cos_part = np.cos(np.pi*flux_quantum_ratio)
        sin_part = self.junction_asymmetry_parameter *np.sin(np.pi*flux_quantum_ratio)
        return self.Ej_sum *np.sqrt( cos_part**2 +sin_part**2 )

    def frequency( self, flux_quantum_ratio ):
        self.Ej = self.effective_Ej( flux_quantum_ratio )
        return super().frequency()
    


