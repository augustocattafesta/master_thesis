import numpy as np
from aptapy.plotting import plt

from thesis import FIGURES_DIR


def fano_limit(E, F=0.12, w=3.6):
    return 2* np.sqrt(2 * np.log(2)) * np.sqrt(F * w / E)


SI = (0.12, 3.6)
CDTE = (0.15, 4.4)
AR = (0.2, 26.3)

e = np.logspace(3, 4.3, 1000)
fano_si = fano_limit(e, *SI)
fano_cdte = fano_limit(e, *CDTE)
fano_ar = fano_limit(e, *AR)

plt.plot(e*1e-3, fano_si, "-k", label="Si")
plt.plot(e*1e-3, fano_cdte, "--k", label="CdTe")
plt.plot(e*1e-3, fano_ar, "-.k", label="Ar")

E_Fe55 = 5.9
E_Cu = 8.0

plt.axvline(E_Fe55, color="gray", linestyle="--")
plt.text(E_Fe55*1.02, 0.1, r"Mn K$\alpha$", rotation=90, color="gray", fontsize=12, va="bottom")
plt.axvline(E_Cu, color="gray", linestyle="--")
plt.text(E_Cu*1.02, 0.1, r"Cu K$\alpha$", rotation=90, color="gray", fontsize=12, va="bottom")


plt.xlabel("Energy [keV]")
plt.ylabel(r"$\Delta$E/E (FWHM)")
plt.legend(frameon=False)
plt.xlim(1, 2e1)
plt.xscale("log")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "chapter2/fano_limit.pdf", dpi=300, bbox_inches="tight")

plt.show()
