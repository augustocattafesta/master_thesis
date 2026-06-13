import numpy as np
from aptapy.hist import Histogram1d
from aptapy.models import Gaussian
from aptapy.plotting import plt

from hexsample.source import DiskBeam, Line, Source
from hexsample.calibration import CalibrationMatrixGain
from hexsample.tasks import calibrate


from thesis import DATA_DIR, FIGURES_DIR
from thesis.utils import generate_dataset, reconstruct_dataset
from thesis.defaults import (
    DEFAULT_ENERGY,
    DEFAULT_SIMULATION,
    DEFAULT_NOISE_READOUT,
    DEFAULT_NUM_ROWS,
    DEFAULT_NUM_COLS,
    DEFAULT_ENC,
    DEFAULT_RECON_PARS
    )


GAIN_ANALYSIS_DIR = DATA_DIR / "gain_analysis"
if not GAIN_ANALYSIS_DIR.exists():
    GAIN_ANALYSIS_DIR.mkdir(parents=True)
HDF_DIR = DATA_DIR / "hdf"


NUM_EVENTS = 500000
OVERWRITE = False
GAIN_LOC = 0.08

rng = np.random.default_rng(seed=0)
mc_matrix = rng.normal(loc=GAIN_LOC,
                       scale=GAIN_LOC * 0.1,
                       size=(DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS))
DEFAULT_NOISE_READOUT.gain = mc_matrix
DEFAULT_SIMULATION["readout"] = DEFAULT_NOISE_READOUT
DEFAULT_SIMULATION["source"] = Source(
    spectrum=Line(DEFAULT_ENERGY),
    beam=DiskBeam(radius=0.15)
)

file_path = GAIN_ANALYSIS_DIR / "simulation_gain.h5"
generate_dataset(
    file_path,
    NUM_EVENTS,
    OVERWRITE,
    **DEFAULT_SIMULATION
    ).close()

# Calculate gain matrix for MC simulation
gain_matrix_path = GAIN_ANALYSIS_DIR / f"{file_path.stem}_matrix_gain.h5"
if not gain_matrix_path.exists() or OVERWRITE:
    calibrate(
        file_path,
        DEFAULT_ENERGY,
        200000,
        quantity="gain"
        )
mc_matrix_gain = CalibrationMatrixGain.from_hdf5(str(gain_matrix_path))
fig_matrix, fig_hist = mc_matrix_gain.plot(200, label="Gain [ADC/e-]")

ZERO_SUP_THRESHOLD = 2 * int(DEFAULT_ENC * GAIN_LOC)
# Reconstruct spectra with and without gain correction
mc_spectrum = reconstruct_dataset(
    file_path,
    suffix="recon_mc_gain",
    recon_method="centroid",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    gain_map=mc_matrix,
    overwrite=OVERWRITE,
    **DEFAULT_RECON_PARS)

mc_recon_spectrum = reconstruct_dataset(
    file_path,
    suffix="recon_calibrated_gain",
    recon_method="centroid",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    gain_map=mc_matrix_gain.matrix,
    overwrite=OVERWRITE,
    **DEFAULT_RECON_PARS)

const_gain_spectrum = reconstruct_dataset(
    file_path,
    suffix="recon_constant_gain",
    recon_method="centroid",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    gain_map=GAIN_LOC * np.ones_like(mc_matrix),
    overwrite=OVERWRITE,
    **DEFAULT_RECON_PARS)

mc_spectrum_energy = mc_spectrum.column("energy")
mc_recon_spectrum_energy = mc_recon_spectrum.column("energy")
const_gain_spectrum_energy = const_gain_spectrum.column("energy")

mc_spectrum.close()
mc_recon_spectrum.close()
const_gain_spectrum.close()


edges = np.linspace(min(const_gain_spectrum_energy), max(const_gain_spectrum_energy), 100)
hist_mc = Histogram1d(edges, xlabel="Energy [eV]")
hist_mc.fill(mc_spectrum_energy)
hist_recon_cal = Histogram1d(edges, xlabel="Energy [eV]")
hist_recon_cal.fill(mc_recon_spectrum_energy)
hist_recon_const = Histogram1d(edges, xlabel="Energy [eV]")
hist_recon_const.fill(const_gain_spectrum_energy)

mc_model = Gaussian()
mc_model.fit(hist_mc)
cal_model = Gaussian()
cal_model.fit(hist_recon_cal)
const_model = Gaussian()
const_model.fit(hist_recon_const)

spectra = plt.figure()
hist_mc.plot(label="MC gain")
mc_model.plot(fit_output=True, color="C0")
hist_recon_cal.plot(label="Calibrated gain")
cal_model.plot(fit_output=True, color="C1")
hist_recon_const.plot(label=r"Fixed gain ($g=0.08$ ADC/e$^-$)")
const_model.plot(fit_output=True, color="C2")
plt.legend()


plt.show()