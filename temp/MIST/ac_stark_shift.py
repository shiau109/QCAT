import numpy as np
import matplotlib.pyplot as plt

# === User inputs (edit these) ===
# All frequency-like quantities should use the same units (e.g. MHz or GHz).
# Then the AC Stark shift will come out in those units too.
chi_01 = -0.08    # dispersive shift for 0-1 transition (per photon / 2), e.g. in MHz
g      = 25   # coupling strength g, same units as chi_01 (e.g. MHz)
Delta  = 1955  # detuning Δ = ω_q - ω_r, same units
alpha  = -242  # anharmonicity α (ω_12 - ω_01), same units

# === Derived quantities ===

# Critical photon numbers for 0-1 and 1-2 transitions (approximate)
n_crit_01 = (Delta**2) / (4 * g**2)


# Take the most restrictive n_crit
n_crit = n_crit_01

# Range of photon numbers to plot (a bit beyond n_crit)
n_max = int(np.ceil(3 * n_crit))
n = np.arange(0, n_max + 1)

# Linear AC Stark shift vs n: Δω_AC(n) = 2 * χ_01 * n
ac_stark = 2 * chi_01 * n

# AC Stark shift at the critical photon number
ac_stark_crit = 2 * chi_01 * n_crit

# === Plot ===
fig, ax = plt.subplots()

ax.plot(n, ac_stark, label="AC Stark shift (linear dispersive)")

# Vertical dashed line at n_crit
ax.axvline(n_crit, linestyle="--", linewidth=1, label=r"$n_{\mathrm{crit}}$")

# Horizontal dashed line at AC Stark shift at n_crit
ax.axhline(ac_stark_crit, linestyle="--", linewidth=1)

# Mark and annotate the crossing point
ax.plot(n_crit, ac_stark_crit, "o")
ax.annotate(
    fr"$n_{{\rm crit}} \approx {n_crit:.1f}$",
    xy=(n_crit, ac_stark_crit),
    xytext=(0.05, 0.9),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", lw=1),
)

ax.set_xlabel("Photon number $n$")
ax.set_ylabel("AC Stark shift $\\Delta\\omega_{AC}(n)$ [same units as $\\chi_{01}$]")
ax.set_title("AC Stark shift vs photon number in a dispersive transmon–resonator system")
ax.legend()
ax.grid(True, which="both", linestyle=":")

plt.show()
