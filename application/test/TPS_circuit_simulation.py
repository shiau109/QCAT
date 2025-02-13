import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip import Bloch

# Define Pauli matrices (global)
X = qt.sigmax()
Y = qt.sigmay()
Z = qt.sigmaz()


# -----------------------------------------------------------------------------
def effective_hamiltonian(phi, Delta, Omega):
    """
    Build the effective Hamiltonian in the rotating frame.
    """
    return (Delta / 2) * Z + (Omega / 2) * (np.cos(phi) * X + np.sin(phi) * Y)

def apply_depolarization(rho, p):
    """
    Apply the depolarizing channel to a density matrix:
      rho -> (1-p)*rho + p*I/2
    """
    return (1 - p) * rho + p * qt.qeye(2) / 2

def circuit_application(ops:list, init_state, p_depol=0):
    """
    Repeatedly apply the operator 'op' to init_state.
    
    Parameters:
      ops       : the operators (Qobj) to be applied
      init_state: initial state (ket or density matrix)
      p_depol  : depolarization probability to apply after each repetition
      
    Returns:
      trajectory: 3xN array of Bloch coordinates after each repetition
      final_state: the final density matrix after all repetitions
    """
    # Convert to density matrix if necessary.
    if init_state.type == 'ket':
        state = qt.ket2dm(init_state)
    else:
        state = init_state
    trajectory = []
    
    for op in ops:
        state = op * state * op.dag()  # unitary evolution on density matrix
        if p_depol > 0:
            state = apply_depolarization(state, p_depol)
        x = np.real(qt.expect(X, state))
        y = np.real(qt.expect(Y, state))
        z = np.real(qt.expect(Z, state))
        trajectory.append([x, y, z])
    
    return np.array(trajectory).T, state
# -----------------------------------------------------------------------------
# Parameters
phi_err  = 0       # Phase error for the second operation
Delta       = 0.001     # Detuning between drive and qubit frequency
rot_err     = 0.00
t_pulse     = 40        # Pulse duration (arbitrary units)
Omega       = (np.pi / t_pulse) * (1 + rot_err)  # Drive amplitude
p_depol     = 0.01    # Depolarization probability (0 means no depolarization)

# -----------------------------------------------------------------------------
# Native gate
gate_I = (-1j * effective_hamiltonian(0, Delta, 0) *t_pulse).expm()

gate_X = (-1j * effective_hamiltonian(0, Delta, Omega) *t_pulse).expm()
gate_sX = (-1j * effective_hamiltonian(0, Delta, Omega/2) *t_pulse).expm()
gate_isX = (-1j * effective_hamiltonian(np.pi, Delta, Omega/2) *t_pulse).expm()

gate_Y = (-1j * effective_hamiltonian(np.pi/2, Delta, Omega) *t_pulse).expm()
gate_sY = (-1j * effective_hamiltonian(np.pi/2, Delta, Omega/2) *t_pulse).expm()
gate_isY = (-1j * effective_hamiltonian(-np.pi/2, Delta, Omega/2) *t_pulse).expm()



# -----------------------------------------------------------------------------
# 3. Plot Bloch points for the 7 native gates applied to the initial state.
# We'll compute the Bloch vector for each gate when applied to |0>.
native_gates = {
    "I":   gate_I,
    "X":   gate_X,
    "sX":  gate_sX,
    "isX": gate_isX,
    "Y":   gate_Y,
    "sY":  gate_sY,
    "isY": gate_isY
}

# Create a new Bloch sphere for native gate points.
b_native = Bloch()
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # List of distinct colors for each gate
initial_state = qt.basis(2, 0)

# Add each gate's output as an individual point.
for i, (label, op) in enumerate(native_gates.items()):
    # Apply the gate to the initial state.
    state = op * initial_state
    # Compute the Bloch vector coordinates.
    x = np.real(qt.expect(X, state))
    y = np.real(qt.expect(Y, state))
    z = np.real(qt.expect(Z, state))
    # Add the point as a separate dataset so that we can assign a color.
    b_native.point_color = [colors[i]]
    b_native.add_points([[x], [y], [z]], 's')
    # Set the color for the most recently added dataset.
    # Optionally, you can store labels or print them.
    print(f"Gate {label}: Bloch vector = ({x:.3f}, {y:.3f}, {z:.3f})")

b_native.make_sphere()
plt.title("Bloch Points for 7 Native Gates")
plt.show()


clifford_gate = [ gate_I, gate_X, gate_Y, gate_X*gate_Y, 
    gate_sY*gate_sX, gate_isY*gate_sX, gate_sY*gate_isX, gate_isY*gate_isX,
    gate_sX*gate_sY, gate_isX*gate_sY, gate_sX*gate_isY, gate_isX*gate_isY, 
    gate_sX, gate_isX, gate_sY, gate_isY, 

    gate_sX*gate_sY*gate_isX, gate_sX*gate_isY*gate_isX, 

    gate_sY*gate_X, gate_isY*gate_X, gate_sX*gate_Y, gate_isX*gate_Y,

    gate_sX*gate_sY*gate_sX, gate_isX*gate_isY*gate_isX, 

]
operations = [gate_I]*100
traj_repeated, final_state = circuit_application(operations, initial_state, p_depol)

# -----------------------------------------------------------------------------
# Plotting the trajectories on the Bloch sphere
b = Bloch()
b.add_points(traj_repeated, 's')   # Trajectory for repeated composite operations
b.make_sphere()
plt.show()