import numpy as np
from aptapy.plotting import plt
from hexsample.resolution import SlantedEdgeResolution, SlitsAligner
from hexsample.source import Source, SlitBeam

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
simulation_path = PAPER_DIR / "simulation.h5"
source = Source(spectrum=DEFAULT_SPECTRUM,
                beam=SlitBeam(theta=1, height=0.014))


DEFAULT_SIMULATION["source"] = source
DEFAULT_NOISE_READOUT.enc = 20
DEFAULT_SIMULATION["readout"] = DEFAULT_NOISE_READOUT
generate_dataset(
    simulation_path,
    NUM_EVENTS,
    OVERWRITE,
    **DEFAULT_SIMULATION
    ).close()

ZERO_SUP_THRESHOLD = 2 * 20

mc_eta = reconstruct_dataset(
    simulation_path,
    recon_method="eta",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=OVERWRITE,
    **DEFAULT_RECON_PARS
    )

x_mc_eta = mc_eta.column("posx") * 10000
y_mc_eta = mc_eta.column("posy") * 10000

ALIGNER_BIN_SIZE = 10    # microns
SIGMA_ALIGNER = 3     # bins
mc_aligner = SlitsAligner(ALIGNER_BIN_SIZE, SIGMA_ALIGNER, np.deg2rad(-1.))
x_rot_mc_eta, y_rot_mc_eta = mc_aligner.align(x_mc_eta, y_mc_eta)

EDGE_BIN_SIZE = 2.5
SIGMA_EDGE = 1.5
resolution_mc_eta = SlantedEdgeResolution(y_rot_mc_eta, EDGE_BIN_SIZE, SIGMA_EDGE)

mtf_mc_eta, freq_mc_eta = resolution_mc_eta.mtf()


# Data

data_path = EDGE_DIR / "020_0006531_data_all.h5"
data_eta = reconstruct_dataset(
    data_path,
    recon_method="eta",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=False,
    **DEFAULT_RECON_PARS
    )

x_data_eta = data_eta.column("posx") * 10000
y_data_eta = data_eta.column("posy") * 10000

aligner_data_eta = SlitsAligner(7, 6)
x_rot_data_eta, y_rot_data_eta = aligner_data_eta.align(x_data_eta, y_data_eta)
print(f"Angle of the slanted edge ETA: {np.rad2deg(aligner_data_eta.angle)} degrees")


Y_MASK_SLIT = (-4070, -3850)
X_MASK = -3400
mask_data_eta = (y_rot_data_eta > Y_MASK_SLIT[0]) & (y_rot_data_eta < Y_MASK_SLIT[1]) & (x_rot_data_eta > X_MASK)
y_data_eta_mask = y_rot_data_eta[mask_data_eta]

resolution_data_eta = SlantedEdgeResolution(y_data_eta_mask, EDGE_BIN_SIZE, SIGMA_EDGE)

from aptapy.hist import Histogram2d

fig, ax = plt.subplots()

mtf_data_eta, freq_data_eta = resolution_data_eta.mtf()
plt.plot(freq_mc_eta*1000, mtf_mc_eta, label=r"Monte Carlo (20 $e^-$ ENC)")
plt.plot(freq_data_eta*1000, mtf_data_eta, label="Dati reali")
plt.xlim(0, 150)
plt.hlines(0.1, 0, plt.xlim()[1], colors="gray", linestyles="dashed")
plt.xlabel(r"Frequenza spaziale [lp/mm]")
plt.ylabel("Modulation Transfer Function")
plt.legend()



xedges = np.arange(-3800, 5300, 10)
yedges = np.arange(-5600, 620, 10)

axins = ax.inset_axes([0.45, 0.3, 0.55, 0.55])

hist = Histogram2d(xedges, yedges)
hist.fill(x_rot_data_eta, y_rot_data_eta)
hist.plot(axins, vmax=25)
axins.set_xticks([])
axins.set_yticks([])

fig.savefig("mtf.png", dpi=500)

plt.show()