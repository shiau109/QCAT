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
def effective_hamiltonian(phi, detuning, Omega):
    """
    Build the effective Hamiltonian in the rotating frame.
    """
    return (detuning / 2) * Z + (Omega / 2) * (np.cos(phi) * X + np.sin(phi) * Y)

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
class SQGateGenerator:
    def __init__( self ):
        self.phi_err   = 0         # Phase error for the second operation
        self.detuning  = 0.00     # Detuning between drive and qubit frequency (noisy)
        self.rot_err   = 0.00
        self.t_pulse   = 40        # Pulse duration (arbitrary units)
        self.Omega     = (np.pi / self.t_pulse) * (1 + self.rot_err)  # Drive amplitude
        self.p_depol   = 0.0       # Depolarization probability (0 means no depolarization)

    def get_native_gate(self):
        """
        Get the noisy native gates (with Δ ≠ 0, etc.).
        """
        detuning   = self.detuning      
        t_pulse = self.t_pulse      
        Omega   = self.Omega 
        # Native gate definitions (noisy version)
        gate_I   = (-1j * effective_hamiltonian(0, detuning, 0) * t_pulse).expm()
        gate_X   = (-1j * effective_hamiltonian(0, detuning, Omega) * t_pulse).expm()
        gate_sX  = (-1j * effective_hamiltonian(0, detuning, Omega/2) * t_pulse).expm()
        gate_isX = (-1j * effective_hamiltonian(np.pi, detuning, Omega/2) * t_pulse).expm()
        gate_Y   = (-1j * effective_hamiltonian(np.pi/2, detuning, Omega) * t_pulse).expm()
        gate_sY  = (-1j * effective_hamiltonian(np.pi/2, detuning, Omega/2) * t_pulse).expm()
        gate_isY = (-1j * effective_hamiltonian(-np.pi/2, detuning, Omega/2) * t_pulse).expm()
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


    def get_clifford_gate(self):
        """
        Build the noisy Clifford gates (each is a composition of native gates).
        """
        self.get_native_gate()  # generate noisy native gates
        gate_I   = self.native_gates["I"]
        gate_X   = self.native_gates["X"]
        gate_Y   = self.native_gates["Y"]
        gate_sX  = self.native_gates["sX"]
        gate_isX = self.native_gates["isX"]
        gate_sY  = self.native_gates["sY"]
        gate_isY = self.native_gates["isY"]

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





class RandomizedBenchmarking():
    def __init__( self ):

        self.gate_gen = SQGateGenerator()

        self.ideal_clifford_gate = SQGateGenerator().get_clifford_gate()
        self.initial_state = qt.basis(2, 0)

        self.n_trials = 100  # Number of random sequences per sequence length.
        self.p_depol = 0.00

    # -----------------------------------------------------------------------------
    # Randomized Benchmarking Simulation (using ideal inversion)
    #
    # In this version we generate a random sequence of indices, then form the ideal
    # sequence (for computing the inverse) and the corresponding noisy sequence (for evolution)
    def random_clifford_indices( self, seq_length, n_cliffords):
        """
        Generate a random sequence (list) of indices for Clifford gates.
        """
        return [random.randrange(n_cliffords) for _ in range(seq_length)]

    def simulate_rb_single_length(self, sequence_length ):
        """
        Simulate randomized benchmarking for a given sequence length using ideal inversion.
        
        Parameters:
        sequence_length: number of Clifford gates in the sequence.
        ideal_clifford_list: list of ideal Clifford gate operators.
        noisy_clifford_list: list of corresponding noisy Clifford gate operators.
        init_state     : initial state (ket).
        n_trials       : number of random sequences to average over.
        p_depol        : depolarization probability applied after each gate.
        
        Returns:
        Average fidelity over n_trials.
        """
        n_trials = self.n_trials
        p_depol = self.p_depol
        init_state = self.initial_state
        fidelities = []
        ideal_clifford_list = self.ideal_clifford_gate
        noisy_clifford_list = self.noisy_clifford_gate

        n_cliffords = len(ideal_clifford_list)
        for trial in range(n_trials):
            # Generate a random sequence of indices.
            indices = self.random_clifford_indices(sequence_length, n_cliffords)
            # print(indices)
            # Compute the overall ideal gate from the ideal Clifford list.
            overall_ideal = qt.qeye(2)
            for idx in indices:
                overall_ideal = ideal_clifford_list[idx] * overall_ideal
            inv_gate = overall_ideal.dag()  # Ideal inversion gate.
            
            # Now apply the noisy sequence to the initial state.
            if init_state.type == 'ket':
                state = qt.ket2dm(init_state)
            else:
                state = init_state
            for idx in indices:
                state = noisy_clifford_list[idx] * state * noisy_clifford_list[idx].dag()
                if p_depol > 0:
                    state = apply_depolarization(state, p_depol)
                    
            # Apply the ideal inverse gate.
            state = inv_gate * state * inv_gate.dag()
            
            # Compute the fidelity with the initial state.
            proj_init = init_state * init_state.dag()
            fid = np.real(qt.expect(proj_init, state))
            fidelities.append(fid)
        return np.mean(fidelities), np.std(fidelities)
    
    def simulate_rb( self, sequence_lengths ):
        self.noisy_clifford_gate = self.gate_gen.get_clifford_gate()
        rb_fidelities = []
        rb_std = []
        for m in sequence_lengths:
            avg_fid, std = self.simulate_rb_single_length(m)
            rb_fidelities.append(avg_fid)
            rb_std.append(std)
            # print(f"Sequence length {m}: Average Fidelity = {avg_fid:.4f}")
        return rb_fidelities, rb_std

if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # Main Simulation
    myRB = RandomizedBenchmarking()

    # Define a range of sequence lengths.
    length_power = np.arange(10)
    sequence_lengths = 2**length_power  # e.g., [1, 2, 4, 8, ...]
    myRB.n_trials = 100  # Number of random sequences per sequence length.
    myRB.p_depol = 0.00
    myRB.gate_gen.detuning = 0.005
    myRB.gate_gen.rot_err = 0.01

    rb_fidelities, _ = myRB.simulate_rb( sequence_lengths )



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
