
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
        sin_part = self.junction_asymmetry_parameter *np.sin(flux_quantum_ratio)
        return self.Ej_sum *np.sqrt( cos_part**2 +sin_part**2 )

    def frequency( self, flux_quantum_ratio ):
        self.Ej = self.effective_Ej( flux_quantum_ratio )
        return super().frequency()
    


import matplotlib.pyplot as plt
from qcat.simulation.chi_qubit_coupler import chi_qc
# Step 1: Define the function f(x, y)
def population(x, y, crosstalk):
    qubit = FluxTunablTransmon( 12000, 200 )
    coupler = FluxTunablTransmon( 40000, 150 )
    flux = x +crosstalk*y
    fq_idle = qubit.frequency( 0 )
    fc_idle = coupler.frequency( 0 )
    fq = qubit.frequency( flux *1 )
    fc = coupler.frequency( y *1 +crosstalk*x )
    detuning = fq-fc
    chi_ref = chi_qc( 70, fq_idle-fc_idle, qubit.Ec, coupler.Ec )
    chi = chi_qc( 70, detuning, qubit.Ec, coupler.Ec )
    print(fq_idle, fc_idle, fq_idle-fc_idle, chi_ref)
    # population = chi-chi_ref # fq-fq_idle

    population = np.cos( 2*np.pi*( fq-fq_idle )*1 ) # 
    return +population

# Step 2: Create a grid of x and y values
crosstalk = -0.05

x = np.linspace(-0.01, 0.01, 500)/0.8
y = x/crosstalk
X, Y = np.meshgrid(x, y)
# Step 3: Compute the function values over the grid
Z = population(X, Y, crosstalk)

# Step 4: Plot the 2D color map
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='Function value')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('2D Color Map of f(x, y)')
plt.show()
