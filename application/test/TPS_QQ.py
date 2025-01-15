import matplotlib.pyplot as plt
#import numpy as np
from numpy import *
from qutip import *

dim = 2  # level number for each qubit or coupler = 0~(dim-1)
dim_tot = dim**2
j = arange(dim)

w1_max = 4.5
w2_max = 4.5



w1_idle = 4.727  # [GHz]
w2_idle = 4.901 # [GHz]
 
EC1 = 0.21  # [GHz]
EC2 = 0.21  # [GHz]

g12_max = 0.08

#w1_list的第一個頻率必須遠離w2_idle
w1_list = linspace(w1_idle-1, w1_idle+1, 300)




# Create operators
I = qeye(dim)
sm1 = destroy(dim) 
sm2 = destroy(dim)


sm1_ = tensor(sm1, I)
sm2_ = tensor(I, sm2)

idx = 0
evals_mat = zeros((len(w1_list), dim_tot))  
for w1 in w1_list:
    wj_1 = (w1 + EC1) * (j + 0.5) - EC1 * (6 * j**2 + 6 * j + 3) / 12
    wj_1 = wj_1 - wj_1[0]

    w2 = w2_idle  
    wj_2 = (w2 + EC2) * (j + 0.5) - EC2 * (6 * j**2 + 6 * j + 3) / 12
    wj_2 = wj_2 - wj_2[0]


    g12 = g12_max * sqrt(w1 * w2) / sqrt(w1_max * w2_max)
    # Qubit operator
    H1 = qdiags(wj_1, 0) 
    H2 = qdiags(wj_2, 0)


    H1_ = tensor(H1, I)
    H2_ = tensor(I, H2)
    H_12_ = g12 * (sm1_ + sm1_.dag()) * (sm2_ + sm2_.dag())


    H = H1_ + H2_ + H_12_
    evals, ekets = H.eigenstates()
    evals_mat[idx, :] = real(evals)
    
    if idx==0:
        first_w_ekets = ekets
                  
    idx += 1
    
    
########  Identify the eigenstates that are 000-, 001-,100-, 101-like #########
g = basis(dim,0)
e = basis(dim,1)

s00 = tensor(g,g) 
s01 = tensor(g,e) 
s10 = tensor(e,g) 
s11 = tensor(e,e) 

overlap_00 = []
overlap_01 = []
overlap_10 = []
overlap_11 = []

for ii in range(dim_tot):
    overlap_00.append(abs(s00.dag()*first_w_ekets[ii])) 
    overlap_01.append(abs(s01.dag()*first_w_ekets[ii]))
    overlap_10.append(abs(s10.dag()*first_w_ekets[ii]))
    overlap_11.append(abs(s11.dag()*first_w_ekets[ii]))
         
idx_00_ = overlap_00.index(max(overlap_00))
idx_01_ = overlap_01.index(max(overlap_01))
idx_10_ = overlap_10.index(max(overlap_10))
idx_11_ = overlap_11.index(max(overlap_11))


#################   Calculate the zz interaction strength ##################
E00_ = evals_mat[:, idx_00_]
E01_ = evals_mat[:, idx_01_]
E10_ = evals_mat[:, idx_10_]
E11_ = evals_mat[:, idx_11_]
zz_strength = (E11_-E01_)-(E10_-E00_)


###############   Plot zz interaction vs wc (or filling factor of the coupler)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(w1_list, abs(zz_strength)*1e3, "g")
ax.set_xlabel(r'$\omega_{1}/2\pi (GHz)$', fontsize=24)
ax.set_ylabel(r'$|\zeta/2\pi| (MHz)$', fontsize=24)
ax.set_yscale('log')
plt.show()