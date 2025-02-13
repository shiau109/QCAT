import numpy as np 
import qutip as qt
import matplotlib.pyplot as plt
from qutip import Bloch
import random
from scipy.optimize import curve_fit

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

class SQGateGenerator:
    def __init__( self ):
        self.phi_err   = 0         # Phase error for the second operation
        self.Delta     = 0.002     # Detuning between drive and qubit frequency
        self.rot_err   = 0.00
        self.t_pulse   = 40        # Pulse duration (arbitrary units)
        self.Omega     = (np.pi / self.t_pulse) * (1 + self.rot_err)  # Drive amplitude
        self.p_depol   = 0.0      # Depolarization probability (0 means no depolarization)

    def get_native_gate(self):
        # -----------------------------------------------------------------------------
        # Parameters
        Delta     = self.Delta     # Detuning between drive and qubit frequency
        t_pulse   = self.t_pulse      # Pulse duration (arbitrary units)
        Omega     = self.Omega # Drive amplitude
        p_depol   = self.p_depol      # Depolarization probability (0 means no depolarization)

        # -----------------------------------------------------------------------------
        # Native gate definitions
        gate_I   = (-1j * effective_hamiltonian(0, Delta, 0) * t_pulse).expm()

        gate_X   = (-1j * effective_hamiltonian(0, Delta, Omega) * t_pulse).expm()
        gate_sX  = (-1j * effective_hamiltonian(0, Delta, Omega/2) * t_pulse).expm()
        gate_isX = (-1j * effective_hamiltonian(np.pi, Delta, Omega/2) * t_pulse).expm()

        gate_Y   = (-1j * effective_hamiltonian(np.pi/2, Delta, Omega) * t_pulse).expm()
        gate_sY  = (-1j * effective_hamiltonian(np.pi/2, Delta, Omega/2) * t_pulse).expm()
        gate_isY = (-1j * effective_hamiltonian(-np.pi/2, Delta, Omega/2) * t_pulse).expm()


        # -----------------------------------------------------------------------------
        # 3. Plot Bloch points for the 7 native gates applied to the initial state.
        # We'll compute the Bloch vector for each gate when applied to |0>.
        self.native_gates = {
            "I":   gate_I,
            "X":   gate_X,
            "sX":  gate_sX,
            "isX": gate_isX,
            "Y":   gate_Y,
            "sY":  gate_sY,
            "isY": gate_isY
        }
        return self.native_gates
    def check_position( self ):
        # Create a new Bloch sphere for native gate points.
        b_native = Bloch()
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # List of distinct colors for each gate
        initial_state = qt.basis(2, 0)

        # Add each gate's output as an individual point.
        for i, (label, op) in enumerate(self.native_gates.items()):
            # Apply the gate to the initial state.
            state = op * initial_state
            # Compute the Bloch vector coordinates.
            x = np.real(qt.expect(X, state))
            y = np.real(qt.expect(Y, state))
            z = np.real(qt.expect(Z, state))
            # Here we add each point with the specified color.
            b_native.add_points([[x], [y], [z]], 's', colors=[colors[i]])
            print(f"Gate {label}: Bloch vector = ({x:.3f}, {y:.3f}, {z:.3f})")

        b_native.make_sphere()
        plt.title("Bloch Points for 7 Native Gates")
        plt.show()

    def get_clifford_gate( self ):
        # Ensure native gates are generated.
        self.get_native_gate()
        # Retrieve all native gates.
        gate_I   = self.native_gates["I"]
        gate_X   = self.native_gates["X"]
        gate_Y   = self.native_gates["Y"]
        gate_sX  = self.native_gates["sX"]
        gate_isX = self.native_gates["isX"]
        gate_sY  = self.native_gates["sY"]
        gate_isY = self.native_gates["isY"]

        # Pre-built Clifford gates (each is a composition of native gates).
        clifford_gate = [ 
            gate_I, 
            gate_X, 
            gate_Y, 
            gate_X * gate_Y, 
            gate_sY * gate_sX, 
            gate_isY * gate_sX, 
            gate_sY * gate_isX, 
            gate_isY * gate_isX,
            gate_sX * gate_sY, 
            gate_isX * gate_sY, 
            gate_sX * gate_isY, 
            gate_isX * gate_isY, 
            gate_sX, 
            gate_isX, 
            gate_sY, 
            gate_isY, 
            gate_sX * gate_sY * gate_isX, 
            gate_sX * gate_isY * gate_isX, 
            gate_sY * gate_X, 
            gate_isY * gate_X, 
            gate_sX * gate_Y, 
            gate_isX * gate_Y,
            gate_sX * gate_sY * gate_sX, 
            gate_isX * gate_isY * gate_isX, 
        ]
        return clifford_gate
# -----------------------------------------------------------------------------
# Randomized Benchmarking Simulation
#
# For each sequence length, we generate many random sequences composed of
# Clifford gates from the list above. For each sequence, we compute the ideal
# inverse (from the noiseless composition), then apply the sequence with noise
# (using depolarization after each gate) followed by the inverse. The fidelity
# of the final state with the initial state is measured and averaged.

def random_clifford_sequence(seq_length, clifford_list):
    """
    Generate a random sequence (list) of Clifford gates of the given length.
    """
    return [random.choice(clifford_list) for _ in range(seq_length)]

def simulate_rb(sequence_length, clifford_list, init_state, n_trials=30, p_depol=0):
    """
    Simulate randomized benchmarking for a given sequence length.
    
    Parameters:
      sequence_length: number of Clifford gates in the sequence.
      clifford_list  : list of Clifford gate operators.
      init_state     : initial state (ket).
      n_trials       : number of random sequences to average over.
      p_depol        : depolarization probability applied after each gate.
      
    Returns:
      Average fidelity over n_trials.
    """
    fidelities = []
    for trial in range(n_trials):
        # Generate a random sequence of Clifford gates.
        seq = random_clifford_sequence(sequence_length, clifford_list)
        
        # Compute the ideal overall gate (noiseless) for determining the inverse.
        overall_ideal = qt.qeye(2)
        for op in seq:
            overall_ideal = op * overall_ideal
        inv_gate = overall_ideal.dag()  # Ideal inverse.
        
        # Now apply the sequence to the initial state with noise.
        if init_state.type == 'ket':
            state = qt.ket2dm(init_state)
        else:
            state = init_state
        for op in seq:
            state = op * state * op.dag()
            # Apply depolarization noise after each gate if desired.
            if p_depol > 0:
                state = apply_depolarization(state, p_depol)
                
        # Apply the ideal inverse gate.
        state = inv_gate * state * inv_gate.dag()
        
        # Compute the fidelity with the initial state.
        proj_init = init_state * init_state.dag()
        fid = np.real(qt.expect(proj_init, state))
        fidelities.append(fid)
    return np.mean(fidelities)

initial_state = qt.basis(2, 0)
clifford_gate = SQGateGenerator().get_clifford_gate()
# Define a range of sequence lengths.
length_power = np.arange(10)
sequence_lengths = 2**length_power#np.arange(1, 50, 5)
rb_fidelities = []
n_trials = 100  # Number of random sequences per sequence length.

for m in sequence_lengths:
    avg_fid = simulate_rb(m, clifford_gate, initial_state, n_trials=n_trials, p_depol=p_depol)
    rb_fidelities.append(avg_fid)
    print(f"Sequence length {m}: Average Fidelity = {avg_fid:.4f}")

# Plot the RB decay curve.
plt.figure()
plt.scatter(sequence_lengths, rb_fidelities, label="Simulated Data")
plt.xlabel("Sequence Length")
plt.ylabel("Fidelity")
plt.title("Randomized Benchmarking Simulation")
plt.legend()

# Fit an exponential decay: F(m) = A * p^m + B
def exp_decay(m, A, p, B):
    return A * p**m + B

params, cov = curve_fit(exp_decay, sequence_lengths, rb_fidelities, p0=(0.5, 0.99, 0.5))
m_fit = np.linspace(sequence_lengths[0], sequence_lengths[-1], 100)
plt.plot(m_fit, exp_decay(m_fit, *params), 'r--', label="Fitted Curve")
plt.ylim(0,1)

plt.legend()
plt.show()

print("Fitted parameters: A =", params[0], "p =", params[1], "B =", params[2])
