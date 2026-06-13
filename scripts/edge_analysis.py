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


NUM_EVENTS = 10000
OVERWRITE = False
EDGE_DIR = DATA_DIR / "edge"

simulation_path = EDGE_DIR / "simulation.h5"
source = Source(spectrum=DEFAULT_SPECTRUM,
                beam=SlitBeam(theta=1, height=0.014))

DEFAULT_SIMULATION["source"] = source
DEFAULT_SIMULATION["readout"] = DEFAULT_NOISE_READOUT
generate_dataset(
    simulation_path,
    NUM_EVENTS,
    OVERWRITE,
    **DEFAULT_SIMULATION
    ).close()

ZERO_SUP_THRESHOLD = 2 * DEFAULT_ENC

mc_centroid = reconstruct_dataset(
    simulation_path,
    recon_method="centroid",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=OVERWRITE,
    )

mc_eta = reconstruct_dataset(
    simulation_path,
    recon_method="eta",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=OVERWRITE,
    **DEFAULT_RECON_PARS
    )

x_mc_eta = mc_eta.column("posx") * 10000
y_mc_eta = mc_eta.column("posy") * 10000
x_mc_centroid = mc_centroid.column("posx") * 10000
y_mc_centroid = mc_centroid.column("posy") * 10000


ALIGNER_BIN_SIZE = 10    # microns
SIGMA_ALIGNER = 3     # bins
mc_aligner = SlitsAligner(ALIGNER_BIN_SIZE, SIGMA_ALIGNER, np.deg2rad(-1.))
x_rot_mc_eta, y_rot_mc_eta = mc_aligner.align(x_mc_eta, y_mc_eta)
x_rot_mc_centroid, y_rot_mc_centroid = mc_aligner.align(x_mc_centroid, y_mc_centroid)

EDGE_BIN_SIZE = 2.5
SIGMA_EDGE = 1.5
resolution_mc_eta = SlantedEdgeResolution(y_rot_mc_eta, EDGE_BIN_SIZE, SIGMA_EDGE)
resolution_mc_centroid = SlantedEdgeResolution(y_rot_mc_centroid, EDGE_BIN_SIZE, SIGMA_EDGE)

esf_mc = plt.figure()
esf_mc_eta = resolution_mc_eta.esf
esf_mc_centroid = resolution_mc_centroid.esf
esf_mc_eta.plot(label="Eta")
esf_mc_centroid.plot(label="Centroid")
plt.xlabel(r"Position [$\mu$m]")
plt.legend()

lsf_mc = plt.figure()
lsf_mc_eta = resolution_mc_eta.lsf
lsf_mc_centroid = resolution_mc_centroid.lsf
lsf_mc_eta.plot(label="Eta")
lsf_mc_centroid.plot(label="Centroid")
plt.xlabel(r"Position [$\mu$m]")
plt.legend()

mtf_mc = plt.figure()
mtf_mc_eta, freq_mc_eta = resolution_mc_eta.mtf()
mtf_mc_centroid, freq_mc_centroid = resolution_mc_centroid.mtf()
plt.plot(freq_mc_eta*1000, mtf_mc_eta, label="Eta")
plt.plot(freq_mc_centroid*1000, mtf_mc_centroid, label="Centroid")
plt.xlabel(r"Spatial frequency [cycles/mm]")
plt.ylabel("MTF")
plt.legend()

data_path = EDGE_DIR / "020_0006531_data_all.h5"
data_eta = reconstruct_dataset(
    data_path,
    recon_method="eta",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=False,
    **DEFAULT_RECON_PARS
    )

data_centroid = reconstruct_dataset(
    data_path,
    recon_method="centroid",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=False,
    )

x_data_eta = data_eta.column("posx") * 10000
y_data_eta = data_eta.column("posy") * 10000
x_data_centroid = data_centroid.column("posx") * 10000
y_data_centroid = data_centroid.column("posy") * 10000

aligner_data_eta = SlitsAligner(7, 6)
x_rot_data_eta, y_rot_data_eta = aligner_data_eta.align(x_data_eta, y_data_eta)
print(f"Angle of the slanted edge ETA: {np.rad2deg(aligner_data_eta.angle)} degrees")
aligner_data_centroid = SlitsAligner(7, 6)
x_rot_data_centroid, y_rot_data_centroid = aligner_data_centroid.align(x_data_centroid, y_data_centroid)
print(f"Angle of the slanted edge Centroid: {np.rad2deg(aligner_data_centroid.angle)} degrees")

Y_MASK_SLIT = (-4070, -3850)
X_MASK = -3400
mask_data_eta = (y_rot_data_eta > Y_MASK_SLIT[0]) & (y_rot_data_eta < Y_MASK_SLIT[1]) & (x_rot_data_eta > X_MASK)
mask_data_centroid = (y_rot_data_centroid > Y_MASK_SLIT[0]) & (y_rot_data_centroid < Y_MASK_SLIT[1]) & (x_rot_data_centroid > X_MASK)
y_data_eta_mask = y_rot_data_eta[mask_data_eta]
y_data_centroid_mask = y_rot_data_centroid[mask_data_centroid]

resolution_data_eta = SlantedEdgeResolution(y_data_eta_mask, EDGE_BIN_SIZE, SIGMA_EDGE)
resolution_data_centroid = SlantedEdgeResolution(y_data_centroid_mask, EDGE_BIN_SIZE, SIGMA_EDGE)

esf_data = plt.figure()
esf_data_eta = resolution_data_eta.esf
esf_data_centroid = resolution_data_centroid.esf
esf_data_eta.plot(label="Eta")
esf_data_centroid.plot(label="Centroid")
plt.xlabel(r"Position [$\mu$m]")
plt.legend()

lsf_data = plt.figure()
lsf_data_eta = resolution_data_eta.lsf
lsf_data_centroid = resolution_data_centroid.lsf
lsf_data_eta.plot(label="Eta")
lsf_data_centroid.plot(label="Centroid")
plt.xlabel(r"Position [$\mu$m]")
plt.legend()

mtf_data = plt.figure()
mtf_data_eta, freq_data_eta = resolution_data_eta.mtf()
mtf_data_centroid, freq_data_centroid = resolution_data_centroid.mtf()
plt.plot(freq_data_eta*1000, mtf_data_eta, label="Eta")
plt.plot(freq_data_centroid*1000, mtf_data_centroid, label="Centroid")
plt.xlabel(r"Spatial frequency [cycles/mm]")
plt.ylabel("MTF")
plt.legend()

plt.show()

