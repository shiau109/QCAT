"""
Hankel SVD Analysis Tool for Signal Processing and Pole Extraction

This module provides a comprehensive class for analyzing time series signals using
Hankel matrix decomposition, SVD, and ESPRIT/HSVD pole extraction methods.

Author: Created for T1 parametric drive analysis
Date: January 2026
"""

import numpy as np
from scipy.linalg import svd, hankel, lstsq, eigvals
from typing import List, Dict, Tuple, Optional, Union
import warnings

class HankelSVDAnalyzer:
    """
    A comprehensive class for Hankel matrix SVD analysis and pole extraction.
    
    This class provides methods for:
    1. SVD decomposition of Hankel matrices
    2. Signal reconstruction from SVD modes
    3. Complex pole extraction using ESPRIT/HSVD method
    4. Batch analysis with varying number of components
    
    Attributes:
        signal (np.ndarray): Original input signal
        dt (float): Time step between signal samples (in seconds)
        hankel_matrix (np.ndarray): Constructed Hankel matrix
        U (np.ndarray): Left singular vectors from SVD
        s (np.ndarray): Singular values from SVD
        Vt (np.ndarray): Right singular vectors (transposed) from SVD
        reconstructed_signals (dict): Cache of reconstructed signals
        pole_cache (dict): Cache of pole extraction results
    """
    
    def __init__(self, dt: float = 1.0):
        """
        Initialize the HankelSVDAnalyzer.
        
        Args:
            dt (float): Time step between signal samples (default: 1.0 second)
        """
        self.dt = dt
        self.signal = None
        self.hankel_matrix = None
        self.U = None
        self.s = None
        self.Vt = None
        self.reconstructed_signals = {}
        self.pole_cache = {}
        self._hankel_size = None
        
    def analyze_signal(self, signal: np.ndarray, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze a 1D signal using Hankel matrix SVD decomposition.
        
        Args:
            signal (np.ndarray): 1D time series signal to analyze
            dt (float, optional): Time step. If provided, updates self.dt
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (U, s, Vt) from SVD
        """
        if dt is not None:
            self.dt = dt
            
        self.signal = np.array(signal)
        n = len(self.signal)
        
        if n < 4:
            raise ValueError(f"Signal too short ({n} points). Need at least 4 points.")
            
        # Create Hankel matrix
        m = n // 2  # Half length for Hankel matrix
        self._hankel_size = m
        
        try:
            self.hankel_matrix = hankel(self.signal[:m], self.signal[m-1:])
        except Exception as e:
            raise ValueError(f"Failed to construct Hankel matrix: {e}")
            
        # Perform SVD
        try:
            self.U, self.s, self.Vt = svd(self.hankel_matrix, full_matrices=False)
        except Exception as e:
            raise ValueError(f"SVD decomposition failed: {e}")
            
        # Clear caches since we have new data
        self.reconstructed_signals.clear()
        self.pole_cache.clear()
        
        return self.U, self.s, self.Vt
    
    def reconstruct_signal(self, n_modes: int) -> np.ndarray:
        """
        Reconstruct signal using specified number of SVD modes.
        
        Args:
            n_modes (int): Number of SVD modes to use for reconstruction
            
        Returns:
            np.ndarray: Reconstructed signal
        """
        if self.U is None or self.s is None or self.Vt is None:
            raise ValueError("Must call analyze_signal() first")
            
        if n_modes in self.reconstructed_signals:
            return self.reconstructed_signals[n_modes]
            
        n_modes = min(n_modes, len(self.s))
        if n_modes <= 0:
            raise ValueError("n_modes must be positive")
            
        # Reconstruct Hankel matrix using first n_modes
        H_reconstructed = self.U[:, :n_modes] @ np.diag(self.s[:n_modes]) @ self.Vt[:n_modes, :]
        
        # Extract signal from anti-diagonals
        n = len(self.signal)
        reconstructed_signal = np.zeros(n)
        
        for k in range(n):
            diagonal_vals = []
            
            # Collect values from anti-diagonal k
            for i in range(H_reconstructed.shape[0]):
                for j in range(H_reconstructed.shape[1]):
                    if i + j == k and k < n:
                        diagonal_vals.append(H_reconstructed[i, j])
                        break
                        
            if diagonal_vals:
                reconstructed_signal[k] = np.mean(diagonal_vals)
            else:
                # Fallback for edge cases
                if k < len(self.U[:, 0]):
                    reconstructed_signal[k] = self.U[k, 0] * self.s[0] / len(self.U[:, 0])
        
        # Cache result
        self.reconstructed_signals[n_modes] = reconstructed_signal
        return reconstructed_signal
    
    def extract_poles(self, n_components: int) -> List[Dict]:
        """
        Extract complex poles using ESPRIT/HSVD method.
        
        Args:
            n_components (int): Number of SVD components to use
            
        Returns:
            List[Dict]: List of pole information dictionaries
        """
        if self.U is None:
            raise ValueError("Must call analyze_signal() first")
            
        if n_components in self.pole_cache:
            return self.pole_cache[n_components]
            
        n_components = min(n_components, len(self.s))
        if n_components <= 0:
            raise ValueError("n_components must be positive")
            
        try:
            # Truncated SVD
            Uk = self.U[:, :n_components]
            
            if Uk.shape[0] <= 1:
                warnings.warn("Insufficient data points for pole extraction")
                return []
            
            # Shift invariance
            U_up = Uk[:-1, :]
            U_down = Uk[1:, :]
            
            if U_up.shape[0] == 0:
                warnings.warn("Empty U_up matrix")
                return []
            
            # Solve for companion matrix Phi
            Phi, _, _, _ = lstsq(U_up, U_down)
            
            # Extract poles (eigenvalues of Phi)
            z_poles = eigvals(Phi)
            
            # Convert to physical parameters
            pole_results = []
            for z in z_poles:
                if np.abs(z) < 1e-10:  # Skip numerically insignificant poles
                    continue
                    
                # Extract physical parameters
                kappa = -np.log(np.abs(z)) / self.dt  # Decay rate (1/s)
                w = np.angle(z) / self.dt  # Angular frequency (rad/s)
                freq_hz = w / (2 * np.pi)  # Frequency in Hz
                
                # Only keep positive or near-zero frequencies
                if freq_hz >= -1e-5:
                    is_oscillating = abs(freq_hz) > 1e-4
                    tau_s = 1 / kappa if kappa > 1e-10 else np.inf  # Time constant
                    
                    pole_results.append({
                        "z_pole": complex(z),
                        "freq_hz": float(freq_hz),
                        "decay_rate": float(kappa),
                        "time_constant": float(tau_s),
                        "is_oscillating": bool(is_oscillating),
                        "magnitude": float(np.abs(z)),
                        "phase": float(np.angle(z))
                    })
            
            # Sort by decay rate magnitude (most significant first)
            pole_results.sort(key=lambda x: abs(x["decay_rate"]), reverse=True)
            
            # Cache result
            self.pole_cache[n_components] = pole_results
            return pole_results
            
        except Exception as e:
            warnings.warn(f"Pole extraction failed for n_components={n_components}: {e}")
            return []
    
    def sweep_components(self, n_component_max: int) -> Dict[int, List[Dict]]:
        """
        Extract poles for all component counts from 1 to n_component_max.
        
        Args:
            n_component_max (int): Maximum number of components to analyze
            
        Returns:
            Dict[int, List[Dict]]: Dictionary mapping n_components to pole results
        """
        if self.U is None:
            raise ValueError("Must call analyze_signal() first")
            
        n_component_max = min(n_component_max, len(self.s))
        
        results = {}
        for n_comp in range(1, n_component_max + 1):
            try:
                poles = self.extract_poles(n_comp)
                results[n_comp] = poles
            except Exception as e:
                warnings.warn(f"Failed to extract poles for n_components={n_comp}: {e}")
                results[n_comp] = []
                
        return results
    
    def estimate_n_components(self, method: str = 'energy', threshold: float = 0.95, 
                             noise_factor: float = 1e-10) -> int:
        """
        Estimate suitable number of components based on singular values.
        
        Args:
            method (str): Method to use for estimation
                - 'energy': Components that capture threshold fraction of total energy
                - 'elbow': Elbow method using second derivative
                - 'gap': Largest gap in singular values
                - 'noise': Components above noise_factor * s[0]
                - 'knee': Knee detection using maximum curvature
            threshold (float): Threshold for 'energy' method (default: 0.95)
            noise_factor (float): Noise threshold factor (default: 1e-10)
            
        Returns:
            int: Estimated number of components
        """
        if self.s is None:
            raise ValueError("Must call analyze_signal() first")
            
        s = self.s
        n_total = len(s)
        
        if method == 'energy':
            # Energy-based: components that capture threshold fraction of total energy
            energy_cumsum = np.cumsum(s**2) / np.sum(s**2)
            n_comp_energy = np.argmax(energy_cumsum >= threshold)
            
            # Additionally constrain by ratio criterion: find max i where (s[i]/s[i+1]) > 2
            # but i must be < n_comp_energy
            if n_total > 1:
                ratios = s[:-1] / s[1:]  # s[i] / s[i+1]
                valid_indices = np.where((ratios > np.e) & (np.arange(len(ratios)) < n_comp_energy))[0]
                if len(valid_indices) > 0:
                    n_comp = valid_indices[-1] + 1  # +1 because we want number of components
                else:
                    n_comp = min(1, n_comp_energy + 1)  # fallback to energy method
            else:
                n_comp = n_comp_energy + 1
            
        elif method == 'elbow':
            # Elbow method using second derivative of log singular values
            if n_total < 3:
                return min(2, n_total)
            log_s = np.log(s + 1e-15)  # Add small value to avoid log(0)
            # Second derivative approximation
            second_deriv = log_s[:-2] - 2*log_s[1:-1] + log_s[2:]
            # Find maximum curvature (elbow point)
            elbow_idx = np.argmax(second_deriv) + 1  # +1 because of array slicing
            n_comp = min(elbow_idx + 1, n_total)
            
        elif method == 'gap':
            # Gap method: largest relative gap in singular values
            if n_total < 2:
                return n_total
            ratios = s[:-1] / s[1:]  # Ratio between consecutive singular values
            gap_idx = np.argmax(ratios)
            n_comp = gap_idx + 1
            
        elif method == 'noise':
            # Noise threshold: components above noise_factor * largest singular value
            threshold_val = noise_factor * s[0]
            n_comp = np.sum(s > threshold_val)
            
        elif method == 'knee':
            # Knee detection using maximum distance from line connecting first and last points
            if n_total < 2:
                return n_total
            # Normalize indices and singular values
            x = np.arange(n_total) / (n_total - 1)
            y = s / s[0]
            # Line from first to last point
            line_y = y[0] + (y[-1] - y[0]) * x
            # Distance from each point to the line
            distances = np.abs(y - line_y)
            knee_idx = np.argmax(distances)
            n_comp = knee_idx + 1
            
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: 'energy', 'elbow', 'gap', 'noise', 'knee'")
        
        return max(1, min(n_comp, n_total))  # Ensure valid range [1, n_total]
    
    def plot_n_components_analysis(self, methods: Optional[List[str]] = None, 
                                  threshold: float = 0.95, noise_factor: float = 1e-10) -> Dict[str, int]:
        """
        Plot singular values with different n_components estimates for visual inspection.
        
        Args:
            methods (List[str], optional): Methods to compare. If None, uses all methods.
            threshold (float): Threshold for 'energy' method
            noise_factor (float): Noise threshold factor
            
        Returns:
            Dict[str, int]: Dictionary of method names to estimated n_components
        """
        if self.s is None:
            raise ValueError("Must call analyze_signal() first")
            
        if methods is None:
            methods = ['energy', 'elbow', 'gap', 'noise', 'knee']
            
        # Calculate estimates
        estimates = {}
        for method in methods:
            try:
                estimates[method] = self.estimate_n_components(method, threshold, noise_factor)
            except Exception as e:
                print(f"Warning: {method} estimation failed: {e}")
                continue
        
        # Create plot
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Singular values with estimates
        x = np.arange(len(self.s))
        ax1.semilogy(x, self.s, 'bo-', linewidth=2, markersize=4, label='Singular values')
        
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (method, n_comp) in enumerate(estimates.items()):
            color = colors[i % len(colors)]
            ax1.axvline(x=n_comp-1, color=color, linestyle='--', alpha=0.7, 
                       label=f'{method}: {n_comp}')
        
        ax1.set_xlabel('Component Index')
        ax1.set_ylabel('Singular Value')
        ax1.set_title('Singular Values with n_components Estimates')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative energy
        energy_cumsum = np.cumsum(self.s**2) / np.sum(self.s**2)
        ax2.plot(x, energy_cumsum, 'go-', linewidth=2, markersize=4)
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold: {threshold:.1%}')
        
        if 'energy' in estimates:
            ax2.axvline(x=estimates['energy']-1, color='red', linestyle='--', alpha=0.7,
                       label=f'Energy method: {estimates["energy"]}')
        
        ax2.set_xlabel('Component Index')
        ax2.set_ylabel('Cumulative Energy Fraction')
        ax2.set_title('Cumulative Energy vs Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return estimates
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the analysis results.
        
        Returns:
            Dict: Summary information about the analysis
        """
        if self.signal is None:
            return {"status": "No analysis performed"}
            
        summary = {
            "signal_length": len(self.signal),
            "dt": self.dt,
            "hankel_size": self.hankel_matrix.shape if self.hankel_matrix is not None else None,
            "n_singular_values": len(self.s) if self.s is not None else 0,
            "dominant_singular_values": self.s[:5].tolist() if self.s is not None else [],
            "energy_in_first_mode": float(self.s[0]**2 / np.sum(self.s**2) * 100) if self.s is not None else 0,
            "effective_rank": int(np.sum(self.s > 1e-10 * self.s[0])) if self.s is not None else 0,
            "condition_number": float(self.s[0] / self.s[-1]) if self.s is not None and self.s[-1] > 0 else np.inf,
            "cached_reconstructions": list(self.reconstructed_signals.keys()),
            "cached_pole_extractions": list(self.pole_cache.keys())
        }
        
        return summary
    
    def clear_cache(self):
        """Clear all cached results to free memory."""
        self.reconstructed_signals.clear()
        self.pole_cache.clear()
        
    def reset(self):
        """Reset the analyzer to initial state."""
        self.signal = None
        self.hankel_matrix = None
        self.U = None
        self.s = None
        self.Vt = None
        self.clear_cache()
        self._hankel_size = None


# Convenience functions for quick analysis
def quick_svd_analysis(signal: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick SVD analysis of a signal.
    
    Args:
        signal (np.ndarray): Input signal
        dt (float): Time step
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (U, s, Vt)
    """
    analyzer = HankelSVDAnalyzer(dt=dt)
    return analyzer.analyze_signal(signal)

def quick_pole_extraction(signal: np.ndarray, n_components: int, dt: float = 1.0) -> List[Dict]:
    """
    Quick pole extraction from a signal.
    
    Args:
        signal (np.ndarray): Input signal
        n_components (int): Number of components to use
        dt (float): Time step
        
    Returns:
        List[Dict]: Pole information
    """
    analyzer = HankelSVDAnalyzer(dt=dt)
    analyzer.analyze_signal(signal)
    return analyzer.extract_poles(n_components)

def quick_sweep_analysis(signal: np.ndarray, max_components: int, dt: float = 1.0) -> Dict[int, List[Dict]]:
    """
    Quick component sweep analysis.
    
    Args:
        signal (np.ndarray): Input signal
        max_components (int): Maximum number of components
        dt (float): Time step
        
    Returns:
        Dict[int, List[Dict]]: Component count to pole results mapping
    """
    analyzer = HankelSVDAnalyzer(dt=dt)
    analyzer.analyze_signal(signal)
    return analyzer.sweep_components(max_components)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing HankelSVDAnalyzer...")
    
    # Create test signal: damped oscillation + noise
    t = np.linspace(0, 10, 100)
    dt = t[1] - t[0]
    signal = 2.0 * np.exp(-0.5 * t) * np.cos(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(len(t))
    
    # Initialize analyzer
    analyzer = HankelSVDAnalyzer(dt=dt)
    
    # Analyze signal
    U, s, Vt = analyzer.analyze_signal(signal)
    print(f"SVD completed: U shape={U.shape}, s shape={s.shape}, Vt shape={Vt.shape}")
    
    # Reconstruct with different numbers of modes
    for n_modes in [1, 2, 3]:
        reconstructed = analyzer.reconstruct_signal(n_modes)
        error = np.mean((signal - reconstructed)**2)
        print(f"Reconstruction with {n_modes} modes: MSE={error:.6f}")
    
    # Extract poles
    poles = analyzer.extract_poles(3)
    print(f"Extracted {len(poles)} poles:")
    for i, pole in enumerate(poles[:3]):
        print(f"  Pole {i+1}: freq={pole['freq_hz']:.3f} Hz, decay={pole['decay_rate']:.3f} 1/s")
    
    # Component sweep
    sweep_results = analyzer.sweep_components(5)
    print(f"Sweep analysis completed for {len(sweep_results)} component counts")
    
    # Summary
    summary = analyzer.get_summary()
    print(f"Analysis summary: {summary['signal_length']} points, {summary['effective_rank']} effective rank")
    
    print("Testing completed successfully!")