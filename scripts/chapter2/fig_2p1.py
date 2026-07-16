import numpy as np
from aptapy.plotting import plt
from thesis import FIGURES_DIR
from scipy.constants import Avogadro


proton_density = np.logspace(-10, 20, 1000)
proton_mass = 1.6726219E-24 # g

hydrogen_cross_section_1kev = 7.214E+00 * proton_mass # cm^2 / proton
nitrogen_cross_section_1kev = 3.310E+03 * proton_mass # cm^2 / proton

hydrogen_mean_free_path_1kev = 1 / (hydrogen_cross_section_1kev * proton_density)
nitrogen_mean_free_path_1kev = 1 / (nitrogen_cross_section_1kev * proton_density)

OUTER_ATMOSPHERE = (1e17, 1 / (nitrogen_cross_section_1kev * 1e17))
INNER_ATMOSPHERE = (1e19, 1 / (nitrogen_cross_section_1kev * 1e19))
ISM = (1, 1 / (hydrogen_cross_section_1kev * 1))
IGM = (1e-6, 1 / (hydrogen_cross_section_1kev * 1e-6))

ax1 = plt.gca() 

# Configura l'asse X e l'asse Y di SINISTRA (ax1)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Density [atoms cm$^{-3}$]")
ax1.set_ylabel("Mean free path [cm]")
ax1.set_xlim(1e-10, 1e20)

# Disegna i grafici usando ax1 invece di plt (per sicurezza)
ax1.plot(proton_density, hydrogen_mean_free_path_1kev, "-k", label="Hydrogen")
ax1.plot(proton_density, nitrogen_mean_free_path_1kev, "--k", label="Nitrogen")

ax1.scatter(OUTER_ATMOSPHERE[0], OUTER_ATMOSPHERE[1], color="k", s=20, edgecolors='black', zorder=3)
ax1.annotate('Outer atmosphere', (OUTER_ATMOSPHERE[0], OUTER_ATMOSPHERE[1]), xytext=(-55, -5), textcoords='offset points', fontsize=10, ha='center')
ax1.scatter(INNER_ATMOSPHERE[0], INNER_ATMOSPHERE[1], color="k", s=20, edgecolors='black', zorder=3)
ax1.annotate('Inner atmosphere', (INNER_ATMOSPHERE[0], INNER_ATMOSPHERE[1]), xytext=(-55, -5), textcoords='offset points', fontsize=10, ha='center')
ax1.scatter(ISM[0], ISM[1], color="k", s=20, edgecolors='black', zorder=3)
ax1.annotate('ISM', (ISM[0], ISM[1]), xytext=(0, 8), textcoords='offset points', fontsize=10, ha='center')
ax1.scatter(IGM[0], IGM[1], color="k", s=20, edgecolors='black', zorder=3)
ax1.annotate('IGM', (IGM[0], IGM[1]), xytext=(0, 8), textcoords='offset points', fontsize=10, ha='center')

# --- CONFIGURAZIONE ASSE DESTRO (ax2) ---
ax2 = ax1.twinx()
ax2.set_yscale("log")
ax2.set_ylim(ax1.get_ylim()) # Devono avere gli stessi identici limiti!
ax2.grid(False)
# Definisci i tuoi ticks personalizzati (in cm) ed etichette
ticks_y = [1e1, 1e3, 1e5, 3.086e18, 3.086e21, 3.086e24, 3.086e27]
labels_y = ["1 cm", "1 m", "1 km", "1 pc", "1 kpc", "1 Mpc", "1 Gpc"]

ax2.set_yticks(ticks_y)
ax2.set_yticklabels(labels_y)

# --- LEGEND & SALVATAGGIO ---
ax1.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "chapter2/mean_free_path.pdf", dpi=300, bbox_inches="tight")

plt.show()
plt.show()