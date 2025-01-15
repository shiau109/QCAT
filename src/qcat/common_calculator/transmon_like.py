import math
import numpy as np
f_q = 2650

# ejec_ratio = 50
# E_c = f_q/ (np.sqrt(8*ejec_ratio)-1)
# E_j = E_c *ejec_ratio


E_c = 210
E_j = (f_q+E_c)**2/ (8*E_c)
ejec_ratio = E_j/E_c
print(E_c)
print(ejec_ratio)

def odd_even_diff( m, E_c, E_j ):
    oe = (-1)**m
    const = np.sqrt(2/np.pi)
    ejec_ratio = E_j/E_c
    return oe *E_c *const *2**(4*m+5)/math.factorial(m) *(ejec_ratio/2)**(m/2.+3/4.) *np.exp(-np.sqrt(8*ejec_ratio))

e_0 = odd_even_diff(0,E_c,E_j)
e_1 = odd_even_diff(1,E_c,E_j)
e_2 = odd_even_diff(2,E_c,E_j)
print( f"0: {e_0}, 1:{e_1}, 2:{e_2}")
print(f"1-0:{e_1-e_0},{np.log10(np.abs(e_1-e_0)/f_q)}")
print(f"2-1:{e_2-e_1},{np.log10(np.abs(e_2-e_1)/f_q)}")
