import numpy as np
import matplotlib.pyplot as plt


w1 = 4.727
w2 = np.linspace(5., 6.5, 200)
w3 = 4.910
 
a1 = -0.205
a2 = -0.150
a3 = -0.205
 
g12 = 0.070#0.0813;
g23 = 0.070#0.0813;
g13 = 0.005#0.0038; #
 
d12 = w1-w2
d13 = w1-w3
d32 = w3-w2
# ########  Nanjing¡¦s formula 
zz2_13 = 2*g13**2*(1./(d13-a3)-1./(d13+a1))
zz3_13 = 2*g13*g12*g23*(2./(d13-a3)/d12 - 2./(d13+a1)/d32 + 2./d12/d32)
zz4_13 = 2*g12**2*g23**2*(1./d12**2./(d13-a3) - 1./d32**2./(d13+a1) + 1./(d12+d32-a2)*(1./d12+1./d32)**2)
 
# ####### Tsinghua's formula
zz2_13_ = 2*g13**2*(1./(d13-a3)-1./(d13+a1))
zz3_13_ = 2*g13*g12*g23*((2./(d13-a3)-1./d13)/d12+ (2./(-d13-a1)+1/d13)/d32)
zz4_13_ = (2*g12**2*g23**2*(1./d12+1./d32)**2./(d12+d32-a2)
            +(g12**2*g23**2./d12**2)*(2./(d13-a3)-1./d13-1./d32)
            +(g12**2*g23**2./d32**2)*(2./(-d13-a1)+1./d13-1./d12))


fig, ax = plt.subplots(1)

ax.plot(w2,(zz2_13)*np.ones(w2.shape[0])*1e9/1e6,'r', label="$g_{direct}$")
# ax.plot(w2,(zz3_13)*1e9/1e6,'b')
ax.plot(w2,(zz3_13+zz4_13)*1e9/1e6,'b', label="$g_{indirect}$")

# ax.plot(w2,(zz3_13_)*1e9/1e6,'--b')
# ax.plot(w2,(zz4_13)*1e9/1e6,'g')
# ax.plot(w2,(zz4_13_)*1e9/1e6,'--g')
ax.plot(w2,(zz2_13+zz3_13+zz4_13)*1e9/1e6,'k', label="Total")
# ax.plot(w2,(zz2_13+zz3_13_+zz4_13_)*1e9/1e6,'--k')
ax.hlines(0, 5.1, 6.5, color="gray", linestyle="--", lw=2)

ax.set_xlim(5.1, 6.5)
ax.set_ylim(-3, 3)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Coupler Frequency (GHz)", fontsize=20)
ax.set_ylabel("ZZ interaction (MHz)", fontsize=20)
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1)


ax.plot(w2,np.abs(zz2_13+zz3_13+zz4_13)*1e9/1e6,'k')
ax.set_yscale("log")
ax.set_xlim(5.1, 6.5)
ax.set_ylim(0.01, 3)
# ax.set_xticks( [5,5.5,6,6.5],fontsize=14)
# ax.set_yticks( [0.01,0.1,1],fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Coupler Frequency (GHz)", fontsize=20)
ax.set_ylabel("ZZ interaction (MHz)", fontsize=20)
plt.tight_layout()
plt.show()