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
    # return (Delta / 2) * Z + (Omega / 2) * (np.cos(phi) * X + np.sin(phi) * Y)
    return (Delta / 2) * Z + (Omega / 2) * (np.cos(phi) * X + np.sin(phi) * Y)

def apply_depolarization(rho, p):
    """
    Apply the depolarizing channel to a density matrix:
      rho -> (1-p)*rho + p*I/2
    """
    return (1 - p) * rho + p * qt.qeye(2) / 2

def compute_bloch_trajectory(H, t_pulse, n_steps, init_state, p_depol=0):
    """
    Compute the Bloch trajectory for a pulse with Hamiltonian H.
    
    Parameters:
      H        : Hamiltonian (Qobj)
      t_pulse  : pulse duration
      n_steps  : number of time steps in the trajectory
      init_state: initial state (ket or density matrix)
      p_depol  : depolarization probability to be applied after the unitary evolution
      
    Returns:
      trajectory: 3xN array of Bloch coordinates ([<X>, <Y>, <Z>])
      final_U   : the unitary corresponding to the full pulse (t = t_pulse)
      final_rho : the final density matrix after the pulse and depolarization
    """
    # If the initial state is a ket, convert it to a density matrix.
    if init_state.type == 'ket':
        rho0 = qt.ket2dm(init_state)
    else:
        rho0 = init_state

    t_values = np.linspace(0, t_pulse, n_steps)
    trajectory = []
    
    for t in t_values:
        U = (-1j * H * t).expm()
        rho = U * rho0 * U.dag()
        if p_depol > 0:
            rho = apply_depolarization(rho, p_depol)
        # Compute expectation values for the Bloch vector.
        x = np.real(qt.expect(X, rho))
        y = np.real(qt.expect(Y, rho))
        z = np.real(qt.expect(Z, rho))
        trajectory.append([x, y, z])
    
    # Final full pulse evolution
    U_final = (-1j * H * t_pulse).expm()
    final_rho = U_final * rho0 * U_final.dag()
    if p_depol > 0:
        final_rho = apply_depolarization(final_rho, p_depol)
    
    return np.array(trajectory).T, U_final, final_rho

def repeated_application(op, reps, init_state, p_depol=0):
    """
    Repeatedly apply the operator 'op' to init_state.
    
    Parameters:
      op       : the operator (Qobj) to be applied
      reps     : number of repetitions
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
    
    for i in range(reps):
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
phi_initial = 0.0       # Phase error for the first operation (radians)
phi_second  = 0       # Phase error for the second operation
Delta       = 0.001     # Detuning between drive and qubit frequency
rot_err     = 0.01
t_pulse     = 40        # Pulse duration (arbitrary units)
Omega       = (np.pi / t_pulse) * (1 + rot_err)  # Drive amplitude
p_depol     = 0.01    # Depolarization probability (0 means no depolarization)

# -----------------------------------------------------------------------------
# Build Hamiltonians for each operation
H_eff_initial = effective_hamiltonian(phi_initial, Delta, Omega)
H_eff_second  = effective_hamiltonian(phi_second, Delta, Omega)

n_steps = 100  # Number of time steps for trajectory calculation

# -----------------------------------------------------------------------------
# 1. First operation: apply pulse to the |0> state.
initial_state = qt.basis(2, 0)
traj1, U1, state_after_first = compute_bloch_trajectory(H_eff_initial, t_pulse, n_steps, initial_state, p_depol)

# 2. Second operation: apply pulse to the state after the first pulse.
traj2, U2, state_after_second = compute_bloch_trajectory(H_eff_second, t_pulse, n_steps, state_after_first, p_depol)

# Composite operation: U2 * U1
composite_op = U2 * U1

# 3. Repeatedly apply the composite operation.
reps = 10
traj_repeated, final_state = repeated_application(composite_op, reps, state_after_second, p_depol)

# -----------------------------------------------------------------------------
# Plotting the trajectories on the Bloch sphere
b = Bloch()
b.add_points(traj1, 's',"red")         # Trajectory for the first pulse
b.add_points(traj2, 's',"blue")         # Trajectory for the second pulse
b.add_points(traj_repeated, 's')   # Trajectory for repeated composite operations
b.make_sphere()
plt.show()

# Print out the unitary operators for reference.
print("First operation U:\n", U1)
print("Second operation U:\n", U2)
