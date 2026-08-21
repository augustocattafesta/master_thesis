import numpy as np
from aptapy.plotting import plt
from thesis import FIGURES_DIR
from scipy.special import dawsn, expm1
import xraydb

# Funzione per calcolare la profonda media di deriva / diffusione media
def compute_mean_z(mu, thickness):
    x = np.sqrt(thickness * mu)
    numeratore = np.sqrt(thickness) - (dawsn(x) / np.sqrt(mu))
    denominatore = -expm1(-thickness * mu)
    return numeratore / denominatore

e = np.logspace(3, 5, 1000)

si_mu = xraydb.material_mu("Si", e, density=2.33)
cdte_mu = xraydb.material_mu("CdTe", e, density=5.85)
ar_mu = xraydb.material_mu("Ar", e, density=0.00176)

D_semi = 0.03
D_gas = 1

photo_abs_si = 1 - np.exp(-si_mu * D_semi)
photo_abs_cdte = 1 - np.exp(-cdte_mu * D_semi)
photo_abs_ar = 1 - np.exp(-ar_mu * D_gas)

plt.plot(e*1e-3, photo_abs_si, "-k", label=r"300 $\mu$m Si")
plt.plot(e*1e-3, photo_abs_cdte, "--k", label=r"300 $\mu$m CdTe")
plt.plot(e*1e-3, photo_abs_ar, "-.k", label=r"1 cm Ar (T=273 K, P=1 bar)")

plt.xlabel("Energy [keV]")
plt.ylabel("Photoabsorption efficiency")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1, 100)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "chapter2/photoabs_eff.pdf", dpi=300, bbox_inches="tight")

e = np.logspace(3, 5.5, 1000)
si_mu = xraydb.material_mu("Si", e, density=2.33)
cdte_mu = xraydb.material_mu("CdTe", e, density=5.85)

# Uso della funzione per calcolare mean_z
mean_z_si = compute_mean_z(si_mu, D_semi)

D_cdte = 0.075
mean_z_cdte = compute_mean_z(cdte_mu, D_cdte)

plt.figure()
plt.plot(e*1e-3, mean_z_si * 50, "-k", label=r"300 $\mu$m Si")
# plt.plot(e*1e-3, mean_z_cdte * 50, "--k", label=r"750 $\mu$m CdTe")

D_semi_alt = 0.02
mean_z_si_alt = compute_mean_z(si_mu, D_semi_alt)
plt.plot(e*1e-3, mean_z_si_alt * 50, "--k", label=r"200 $\mu$m Si")

D_semi_alt = 0.01
mean_z_si_alt = compute_mean_z(si_mu, D_semi_alt)
plt.plot(e*1e-3, mean_z_si_alt * 50, ":k", label=r"100 $\mu$m Si")


plt.xscale("log")
plt.xlabel("Energy [keV]")
plt.ylabel(r"Mean diffusion sigma $\langle \sigma \rangle$ [$\mu$m]")
plt.xlim(1, 200)
plt.ylim(0)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "chapter4/design/mean_diffusion.pdf", dpi=300, bbox_inches="tight")

plt.show()