import numpy as np
from aptapy.plotting import plt
from hexsample.resolution import SlantedEdgeResolution, SlitsAligner
from hexsample.source import Source, SlitBeam
from hexsample.fileio import ReconInputFile


from thesis import DATA_DIR, FIGURES_DIR
from thesis.utils import (
    generate_dataset,
    reconstruct_dataset)
from thesis.defaults import (
    DEFAULT_SIMULATION,
    DEFAULT_SPECTRUM,
    DEFAULT_NOISE_READOUT,
    DEFAULT_ENC,
    DEFAULT_RECON_PARS)


NUM_EVENTS = 1000000
OVERWRITE = False

EDGE_DIR = DATA_DIR / "edge"
PAPER_DIR = DATA_DIR / "paper"

enc20 = PAPER_DIR / "enc20_recon.h5"
enc100 = PAPER_DIR / "enc100_recon.h5"

recon20 = ReconInputFile(enc20)
recon100 = ReconInputFile(enc100)

e20 = recon20.column("energy")
e100 = recon100.column("energy")

from aptapy.hist import Histogram1d
import xraydb
from aptapy.plotting import plt

# edges = np.arange(1600, 3000, 10) * xraydb.ionization_potential("Si")
edges = np.linspace(min(e100), max(e100), 201)
hist20 = Histogram1d(edges)
hist100 = Histogram1d(edges, xlabel="Energia [eV]", ylabel="Eventi/bin")

hist20.fill(e20)
hist100.fill(e100)

fig = plt.figure()
hist20.plot(label=r"20 $e^-$ ENC")
hist100.plot(label=r"100 $e^-$ ENC", fill=None)
plt.xlim(6000, 10900)
plt.text(8200, 7000, r"K$\alpha$")
plt.text(9100, 1000, r"K$\beta$")
plt.legend()

fig.savefig("spettro.png", dpi=500)

plt.show()