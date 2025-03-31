import numpy as np
import matplotlib.pyplot as plt

# Parameters for the transmon
phi = np.linspace(-np.pi, np.pi, 500)  # Phase values
# Parameters for the updated transmon plot with E_C = 0.02
E_C = 0.025  # Small charging energy
E_J = 1.0   # Josephson energy remains the same

# Recompute potential and energy levels
potential = -E_J * np.cos(phi) +E_J
quadratic_potential = 0.5 * E_J * phi**2  # Quadratic (x^2) approximation
E_levels = [
     np.sqrt(8 * E_J * E_C) * (n + 0.5) - E_C / 12 * (6*n**2 +6*n + 3)
    for n in range(4)
]

# Plotting the updated potential and energy levels
plt.figure(figsize=(8, 6))

# Plot the potential
plt.plot(phi, potential, label="Cosine Potential Energy", color="Black", lw=2)
plt.plot(phi, quadratic_potential, label="Quadratic Approximation ($\\phi^2$)", color="gray", alpha=0.5, lw=2)
# Highlight computational subspace and energy levels, ending at the potential curve
for n, E in enumerate(E_levels):
    # Calculate the endpoints of the horizontal lines where they intersect the potential curve
    phi_left = -np.arccos( (E-E_J) / -E_J)
    phi_right = np.arccos( (E-E_J) / -E_J)
    plt.hlines(E, phi_left, phi_right, color="black", linestyle="--", label=f"$|{n}\\rangle$" if n < 2 else None, alpha=1-n*0.15, lw=2)
    # plt.text(phi_right +0.1, E, f"$\\ket{n}$", color="red", fontsize=10, ha="center")

# Set labels, limits, and titles
plt.xlabel("Superconducting phase $\\phi$", fontsize=20)
plt.ylabel(f"Junction Energy ($E_j$)", fontsize=20)
plt.xticks([-np.pi, 0, np.pi], ["$-\\pi$", "$0$", "$\\pi$"], fontsize=14)
plt.yticks([0, 1, 2], fontsize=14)

# plt.title("Energy of Transmon Qubit (E_C = 0.02)")
plt.xlim(-np.pi, np.pi)
plt.ylim(-0.05, 2.05)
# plt.legend(loc="upper center")
plt.grid(alpha=0.4, linestyle="--")

# Show the plot
plt.tight_layout()
plt.show()
