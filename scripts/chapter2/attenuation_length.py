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

e = np.linspace(1e3, 3e5, 1000)
si_mu = xraydb.material_mu("Si", e, density=2.33)
# Soluzione stabile che non rompe Python ad alte energie o grandi spessori
arg_erfi = np.sqrt(D_semi * si_mu)
exp_neg = np.exp(-D_semi * si_mu)

mean_z = (1 / (1 - exp_neg)) * (np.sqrt(D_semi) - (np.sqrt(np.pi / si_mu) / 2) * exp_neg * erfi(arg_erfi))
plt.figure()
plt.plot(e*1e-3, mean_z * 40 /50, "-k")
plt.xlabel("Energy [keV]")
plt.ylabel("Mean depth of absorption")
plt.tight_layout()








plt.show()
