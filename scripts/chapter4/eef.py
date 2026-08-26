import numpy as np
from aptapy.plotting import plt
from thesis import FIGURES_DIR

x = np.array([12.2, 13.7, 15.2, 16.7, 18.3])

plt.rcParams.update(
    {
        "axes.labelsize": 20,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

eef50_bary = np.array([0.19282288, 0.17442371, 0.15203162, 0.1310705, 0.11182421])
eef90_bary = np.array([0.26782728, 0.24196365, 0.21728173, 0.19740588, 0.17806149])
eef50_lh = np.array([0.07091249, 0.04638995, 0.03735167, 0.0335959, 0.03217421])
eef90_lh = np.array([0.35266752, 0.26078726, 0.18874121, 0.14399409, 0.11970873])
eef50_eta_nomc = np.array([0.09058491, 0.06039499, 0.04137398, 0.03179842, 0.02889425])
eef90_eta_nomc = np.array([0.20878612, 0.17174537, 0.1303046, 0.11308122, 0.09284782])
eef50_eta_mc = np.array([0.04001468, 0.03154003, 0.02761279, 0.0265479, 0.02640179])
eef90_eta_mc = np.array([0.23075983, 0.17630119, 0.12758964, 0.11160627, 0.09802927])

fig50 = plt.figure()
plt.plot(x, eef50_bary, "-k", label="Barycenter")
plt.plot(x, eef50_lh, "--k", label="Likelihood")
plt.plot(x, eef50_eta_nomc, "-.k", label=r"$\eta$-function unmodeled")
plt.plot(x, eef50_eta_mc, ":k", label=r"$\eta$-function modeled")
plt.xlabel(r"$\langle \sigma \rangle / p$ [%]")
plt.ylabel("EEF@50% [pitch]")
plt.ylim(0, 0.4)
plt.xlim(min(x), max(x))
plt.legend(frameon=False)

fig90 = plt.figure()
plt.plot(x, eef90_bary, "-k", label="Barycenter")
plt.plot(x, eef90_lh, "--k", label="Likelihood")
plt.plot(x, eef90_eta_nomc, "-.k", label=r"$\eta$-function unmodeled")
plt.plot(x, eef90_eta_mc, ":k", label=r"$\eta$-function modeled")
plt.xlabel(r"$\langle \sigma \rangle / p$ [%]")
plt.ylabel("EEF@90% [pitch]")
plt.ylim(0, 0.4)
plt.xlim(min(x), max(x))
plt.legend(frameon=False)

fig50.savefig(FIGURES_DIR / "chapter4/position/eef50.png", bbox_inches="tight")
fig90.savefig(FIGURES_DIR / "chapter4/position/eef90.png", bbox_inches="tight")

plt.show()