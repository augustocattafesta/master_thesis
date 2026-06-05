import numpy as np
from aptapy.plotting import plt
from thesis import FIGURES_DIR
from scipy.constants import Avogadro

# Load data for Si and CdTe
si_energy, _, _, si_photo, _, _, _, _ = np.loadtxt("/home/augusto/Thesis/master_thesis/data/nist/si.txt", delimiter=" ", unpack=True)
cdte_energy, _, _, cdte_photo, _, _, _, _ = np.loadtxt("/home/augusto/Thesis/master_thesis/data/nist/cdte.txt", delimiter=" ", unpack=True)

limit = 1
si_photo = si_photo[si_energy <= limit] * 28.085 / Avogadro
si_energy = si_energy[si_energy <= limit] * 1e3
cdte_photo = cdte_photo[cdte_energy <= limit] * 240.01 / Avogadro
cdte_energy = cdte_energy[cdte_energy <= limit] * 1e3

plt.figure()
plt.plot(si_energy, si_photo, "k-", label="Si (Z=14)")
plt.plot(cdte_energy, cdte_photo, "k--", label="CdTe (Z=48,52)")
plt.text(1.3, 7e-21, f"Si K-edge", fontdict={"size": 12, "family": "serif"})
plt.text(19, 4e-20, f"CdTe K-edge", fontdict={"size": 12, "family": "serif"})
plt.text(3, 6e-19, r"CdTe L-edge", fontdict={"size": 12, "family": "serif"})
plt.xscale("log")
plt.yscale("log")
plt.xlim(si_energy[0], si_energy[-1])
plt.xlabel("Energy [keV]")
plt.ylabel(r"Photoelectric cross section [cm$^2$/atom]")
plt.legend(frameon=False)


plt.savefig(FIGURES_DIR / "detector/photo_cross_section.png", dpi=300)
plt.show()