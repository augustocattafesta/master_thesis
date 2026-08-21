import numpy as np
from aptapy.plotting import plt
from thesis import FIGURES_DIR

x = np.array([12.2, 13.7, 15.2, 16.7, 18.3])

eef50_bary = np.array([0.19, 0.17, 0.15, 0.13, 0.11])
eef90_bary = np.array([0.27, 0.24, 0.22, 0.20, 0.18])
eef50_lh = np.array([0.07, 0.05, 0.04, 0.03, 0.03])
eef90_lh = np.array([0.35, 0.26, 0.19, 0.14, 0.12])
eef50_eta_nomc = np.array([0.09, 0.06, 0.04, 0.03, 0.03])
eef90_eta_nomc = np.array([0.21, 0.17, 0.13, 0.11, 0.09])
eef50_eta_mc = np.array([0.04, 0.03, 0.03, 0.03, 0.03])
eef90_eta_mc = np.array([0.23, 0.18, 0.13, 0.11, 0.10])

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