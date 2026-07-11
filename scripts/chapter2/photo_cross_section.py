import numpy as np
from aptapy.plotting import plt
from thesis import FIGURES_DIR
import matplotlib.ticker as ticker
from scipy.constants import Avogadro

# Load data for Si and CdTe
si_energy, _, si_comp, si_photo, si_pair, _, _, _ = np.loadtxt("/home/augusto/Thesis/master_thesis/data/nist/si.txt", delimiter=" ", unpack=True)
cdte_energy, _, cdte_comp, cdte_photo, cdte_pair, _, _, _ = np.loadtxt("/home/augusto/Thesis/master_thesis/data/nist/cdte.txt", delimiter=" ", unpack=True)

si_photo = si_photo * 28.085 / Avogadro * 1e24
si_comp = si_comp * 28.085 / Avogadro * 1e24
si_pair = si_pair * 28.085 / Avogadro * 1e24
si_energy = si_energy * 1e3
cdte_photo = cdte_photo * 240.01 / Avogadro * 1e24
cdte_comp = cdte_comp * 240.01 / Avogadro * 1e24
cdte_pair = cdte_pair * 240.01 / Avogadro * 1e24
cdte_energy = cdte_energy * 1e3

def selective_formatter(x, pos):
    if x == 1e0:
        return "1 keV"
    elif x == 1e3:
        return "1 MeV"
    elif x == 1e6:
        return "1 GeV"
    else:
        return "" # Lascia il tick grafico ma cancella il testo (per 10, 100, 10000...)


si_fig = plt.figure("si_cross_section", figsize=(6, 6))
plt.plot(si_energy, si_photo, linestyle="--", color="dimgray", linewidth=1.5, label="Photoelectric")
plt.plot(si_energy, si_comp, linestyle=":", color="black", linewidth=1.5, label="Compton scattering")
plt.plot(si_energy, si_pair, linestyle="-.", color="gray", linewidth=1.5, label="Pair production")
plt.plot(si_energy, si_photo + si_pair + si_comp, linestyle="-", color="black", linewidth=2, label="Total")
plt.xlabel("Photon energy")
plt.ylabel(r"Cross section [barns/atom]")
plt.xscale("log")
plt.yscale("log")
plt.xlim(si_energy[0], 1e7)
plt.ylim(1e-3, 1e7)
plt.text(5e3, 3e4, "Silicon (Z=14)", fontsize=12, color="black")
plt.text(11.6, 1.7e3, r"$\sigma_{photo}$", fontsize=12, color="black")
plt.text(1.4, 2.2e5, "K-edge", fontsize=10, color="black")
plt.text(2, 0.8, r"$\sigma_{Compton}$", fontsize=12, color="black")
plt.text(3.2e5, 0.6, r"$\sigma_{pair}$", fontsize=12, color="black")
plt.legend(frameon=False)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(selective_formatter))
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
plt.tight_layout()

cdte_fig = plt.figure("cdte_cross_section", figsize=(6, 6))
plt.plot(cdte_energy, cdte_photo, linestyle="--", color="dimgray", linewidth=1.5, label="Photoelectric")
plt.plot(cdte_energy, cdte_pair, linestyle=":", color="black", linewidth=1.5, label="Pair production")
plt.plot(cdte_energy, cdte_comp, linestyle="-.", color="gray", linewidth=1.5, label="Compton scattering")
plt.plot(cdte_energy, cdte_photo + cdte_pair + cdte_comp, linestyle="-", color="black", linewidth=2, label="Total")
plt.xlabel("Photon energy")
plt.ylabel(r"Cross section [barns/atom]")
plt.xscale("log")
plt.yscale("log")
plt.xlim(cdte_energy[0], 1e7)
plt.ylim(1e-3, 1e7)
plt.text(5e3, 3e4, "CdTe (Z=48, 52)", fontsize=12, color="black")
plt.text(100, 1e3, r"$\sigma_{photo}$", fontsize=12, color="black")
plt.text(24, 1.8e4, "K-edges", fontsize=10, color="black")
plt.text(2.8, 5.8e5, "L-edges", fontsize=10, color="black")
plt.text(10, 10, r"$\sigma_{Compton}$", fontsize=12, color="black")
plt.text(3.2e5, 11, r"$\sigma_{pair}$", fontsize=12, color="black")
plt.legend(frameon=False)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(selective_formatter))
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
plt.tight_layout()

si_fig.savefig(FIGURES_DIR / "chapter2/si_cross_section.png", dpi=300, bbox_inches="tight")
cdte_fig.savefig(FIGURES_DIR / "chapter2/cdte_cross_section.png", dpi=300, bbox_inches="tight")
plt.show()