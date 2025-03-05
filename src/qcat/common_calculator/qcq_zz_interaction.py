import numpy as np
import matplotlib.pyplot as plt

class ZZ_interaction:
    # def __init__(self, 
    #              w1=4.727, 
    #              w3=4.910, 
    #              a1=-0.205, a2=-0.165, a3=-0.205, 
    #              g12=0.086, g23=0.086, g13=0.0045,
    #              w2_range=(5.5, 7.25), 
    #              w2_points=100):
    def __init__(self, 
                 w1=3.63, 
                 w3=3.757, 
                 a1=-0.205, a2=-0.165, a3=-0.205, 
                 g12=0.076, g23=0.076, g13=0.004,
                 w2_range=(4, 6), 
                 w2_points=100):
        """
        Initialize the calculator with default parameter values and compute w2.
        """
        self.w1 = w1
        self.w3 = w3
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

        self.w2 = np.linspace(w2_range[0], w2_range[1], w2_points)
        self.g12 = g12
        self.g23 = g23
        self.g13 = g13
        self._compute_differences()
        
    def _compute_differences(self):
        """
        Precompute the differences used in the formulas.
        """
        self.d12 = self.w1 - self.w2      # array
        self.d13 = self.w1 - self.w3      # scalar
        self.d32 = self.w3 - self.w2      # array
        
    def nanjing_formula(self):
        """
        Compute Nanjing’s formula:
            zz2 = 2*g13**2*(1/(d13-a3)-1/(d13+a1))
            zz3 = 2*g13*g12*g23*(2/(d13-a3)/d12 - 2/(d13+a1)/d32 + 2/(d12*d32))
            zz4 = 2*g12**2*g23**2*(1/d12**2/(d13-a3) - 1/d32**2/(d13+a1) 
                 + 1/(d12+d32-a2)*(1/d12+1/d32)**2)
        Returns a tuple: (zz2, zz3, zz4)
        """
        zz2 = 2 * self.g13**2 * (1.0/(self.d13 - self.a3) - 1.0/(self.d13 + self.a1))
        zz3 = 2 * self.g13 * self.g12 * self.g23 * (
            2.0/(self.d13 - self.a3)/self.d12 - 2.0/(self.d13 + self.a1)/self.d32 
            + 2.0/(self.d12 * self.d32)
        )
        zz4 = 2 * self.g12**2 * self.g23**2 * (
            1.0/self.d12**2/(self.d13 - self.a3) - 1.0/self.d32**2/(self.d13 + self.a1) +
            1.0/(self.d12 + self.d32 - self.a2) * (1.0/self.d12 + 1.0/self.d32)**2
        )
        return zz2, zz3, zz4

    def tsinghua_formula(self):
        """
        Compute Tsinghua’s formula:
            zz2 = 2*g13**2*(1/(d13-a3)-1/(d13+a1))
            zz3 = 2*g13*g12*g23*((2/(d13-a3)-1/d13)/d12 + (2/(-d13-a1)+1/d13)/d32)
            zz4 = 2*g12**2*g23**2*(1/d12+1/d32)**2/(d12+d32-a2)
                  + (g12**2*g23**2/d12**2)*(2/(d13-a3)-1/d13-1/d32)
                  + (g12**2*g23**2/d32**2)*(2/(-d13-a1)+1/d13-1/d12)
        Returns a tuple: (zz2, zz3, zz4)
        """
        zz2 = 2 * self.g13**2 * (1.0/(self.d13 - self.a3) - 1.0/(self.d13 + self.a1))
        zz3 = 2 * self.g13 * self.g12 * self.g23 * (
            (2.0/(self.d13 - self.a3) - 1.0/self.d13)/self.d12 +
            (2.0/(-self.d13 - self.a1) + 1.0/self.d13)/self.d32
        )
        zz4 = (
            2 * self.g12**2 * self.g23**2 * (1.0/self.d12 + 1.0/self.d32)**2/(self.d12 + self.d32 - self.a2)
            + (self.g12**2 * self.g23**2/self.d12**2) * (2.0/(self.d13 - self.a3) - 1.0/self.d13 - 1.0/self.d32)
            + (self.g12**2 * self.g23**2/self.d32**2) * (2.0/(-self.d13 - self.a1) + 1.0/self.d13 - 1.0/self.d12)
        )
        return zz2, zz3, zz4

# Example usage:
if __name__ == '__main__':
    # Instantiate the calculator with default values
    calc = ZZ_interaction()
    
    # Compute Nanjing's formulas
    zz2_nanjing, zz3_nanjing, zz4_nanjing = calc.nanjing_formula()
    
    # Compute Tsinghua's formulas
    zz2_tsinghua, zz3_tsinghua, zz4_tsinghua = calc.tsinghua_formula()
    
    # Optionally, you can plot one of the outputs
    plt.figure(figsize=(8, 5))
    plt.plot(calc.w2, zz2_nanjing+zz3_nanjing+zz4_nanjing, label="Nanjing zz2")
    plt.plot(calc.w2, zz2_tsinghua+zz3_tsinghua+zz4_tsinghua, label="Tsinghua zz2", linestyle='--')
    plt.xlabel("w2")
    plt.ylabel("zz2 values")
    plt.legend()
    plt.title("Comparison of zz2 computed from two formulas")
    plt.show()
