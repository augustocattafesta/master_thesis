import numpy as np
from aptapy.plotting import plt
from thesis import FIGURES_DIR
from scipy.special import erfi
import xraydb

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
# Soluzione stabile che non rompe Python ad alte energie o grandi spessori
arg_erfi = np.sqrt(D_semi * si_mu)
exp_neg = np.exp(-D_semi * si_mu)
mean_z_si = (1 / (1 - exp_neg)) * (np.sqrt(D_semi) - (np.sqrt(np.pi / si_mu) / 2) * exp_neg * erfi(arg_erfi))

cdte_mu = xraydb.material_mu("CdTe", e, density=5.85)
D_cdte = 0.075

# Argomento per la funzione di Dawson: x = sqrt(mu * D)
x = np.sqrt(D_cdte * cdte_mu)
from scipy.special import dawsn, expm1
# Il termine dawsn(x) sostituisce in toto l'uso combinato di exp e erfi
numeratore = np.sqrt(D_cdte) - (dawsn(x) / np.sqrt(cdte_mu))

# expm1(x) calcola exp(x) - 1 in modo stabile per x vicini a zero. 
# Quindi 1 - exp(-x) diventa -expm1(-x)
denominatore = -expm1(-D_cdte * cdte_mu)

mean_z_cdte = numeratore / denominatore

plt.figure()
plt.plot(e*1e-3, mean_z_si * 50, "-k", label=r"300 $\mu$m Si")
plt.plot(e*1e-3, mean_z_cdte * 50, "--k", label=r"750 $\mu$m CdTe")
plt.xscale("log")
plt.xlabel("Energy [keV]")
plt.ylabel(r"Mean diffusion sigma $\langle \sigma \rangle$ [$\mu$m]")
plt.xlim(1, 200)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "chapter4/design/mean_diffusion.pdf", dpi=300, bbox_inches="tight")








plt.show()
