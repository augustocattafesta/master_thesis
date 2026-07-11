import numpy as np
from aptapy.plotting import plt

from thesis import FIGURES_DIR


def enc_limit(enc, E=5900, F=0.12, w=3.6):
    fano = F * E / w
    noise = enc**2
    variance = fano + noise
    return 2* np.sqrt(2 * np.log(2)) * np.sqrt(variance) / (E / w)


def fano_limit(E, F=0.12, w=3.6):
    return 2* np.sqrt(2 * np.log(2)) * np.sqrt(F * w / E)

def enc_noise(enc, E, w=3.6):
    return 2* np.sqrt(2 * np.log(2)) * enc / (E / w)

SI = (0.12, 3.6)
CDTE = (0.15, 4.4)
AR = (0.2, 26.3/200)

E = 5900
fano_si = fano_limit(E, *SI)
fano_cdte = fano_limit(E, *CDTE)
fano_ar = fano_limit(E, *AR)

ENC = np.linspace(0, 200, 100)

res = enc_limit(ENC, E, *SI)

# plt.plot(ENC, np.sqrt(enc_limit(ENC, E, *AR)**2 + 0.17**2), "-k", label="Ar")
plt.plot(ENC, res, "-k", label="Si")
plt.plot(ENC, enc_limit(ENC, E, *CDTE), "--k", label="CdTe")

plt.xlabel("ENC [e$^-$]")
plt.ylabel(r"$\Delta$E/E@5.9 keV (FWHM)")
plt.legend(frameon=False)
plt.xlim(0, 200)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "chapter2/enc_limit.pdf", dpi=300, bbox_inches="tight")

plt.show()
