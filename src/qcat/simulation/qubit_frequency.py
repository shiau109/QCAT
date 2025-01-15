
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
    


import matplotlib.pyplot as plt
from qcat.common_calculator.chi_qubit_coupler import chi_qc
# Step 1: Define the function f(x, y)
def population(x, y, crosstalk):
    qubit = FluxTunablTransmon( 18000, 200 )
    coupler = FluxTunablTransmon( 44000, 150 )
    q_flux = (x +crosstalk*y )/0.6
    c_flux = (y *1 )/0.75
    fq_idle = qubit.frequency( -0.017 )
    fc_idle = coupler.frequency( -0.017 )
    fq = qubit.frequency( q_flux -0.017)
    fc = coupler.frequency( c_flux -0.017 )
    detuning = fq-fc
    chi_ref = chi_qc( 70, fq_idle-fc_idle, qubit.Ec, coupler.Ec )
    chi = chi_qc( 70, detuning, qubit.Ec, coupler.Ec )
    print(fq_idle, fc_idle, fq_idle-fc_idle, chi_ref)
    # population = chi-chi_ref # fq-fq_idle

    population = (1+np.cos( 2*np.pi*( fq-fq_idle +chi-chi_ref  )*1 ))/2 # 
    return population

# Step 2: Create a grid of x and y values
crosstalk = 0.044

x = np.linspace(-0.01, 0.01, 500)
y = np.linspace(-0.01/0.05, 0.01/0.05, 501)
X, Y = np.meshgrid(x, y)
# Step 3: Compute the function values over the grid
Z = population(X, Y, crosstalk)

# Step 4: Plot the 2D color map
plt.figure(figsize=(8, 6))
plt.contourf(Y*1000, X*1000, Z, cmap='viridis')
plt.colorbar()#label='Function value')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.title('2D Color Map of f(x, y)')
plt.show()
