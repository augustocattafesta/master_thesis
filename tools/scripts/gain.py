import numpy as np
import xraydb
from aptapy.hist import Histogram1d
from aptapy.models import Gaussian
from aptapy.plotting import plt, last_line_color

from hexsample.calibration import CalibrationMatrixGain
from hexsample.fileio import ReconInputFile
from hexsample.hexagon import HexagonalLayout
from hexsample.pipeline import calibrate, reconstruct, simulate
from hexsample.roi import Padding

from config import DATA_DIR, FIGURES_DIR, RNG, SEED

GAIN_DIR = DATA_DIR / "gain"
GAIN_DIR.mkdir(exist_ok=True)
FIGURES_GAIN = FIGURES_DIR / "gain"
FIGURES_GAIN.mkdir(exist_ok=True)

ENERGY = 1e4
SIGNAL_E = int(ENERGY / xraydb.ionization_potential("Si"))
SNR = 20
NOISE_E = int(SIGNAL_E / SNR)
NUM_EVENTS = 500000

print(f"Number of events: {NUM_EVENTS}")
print(f"Number of signal electrons: {SIGNAL_E}")
print(f"Signal-to-noise ratio: {SNR}")
print(f"Number of noise electrons: {NOISE_E}")

# Create the gain matrix file.
gain_path = GAIN_DIR / "gain_gaussian.h5"
gain = CalibrationMatrixGain(304, 352)
gain.matrix = RNG.normal(loc=0.08, scale=0.01, size=gain.matrix.shape)
gain.to_hdf5(gain_path)

NOISE_ADC = int(NOISE_E * np.mean(gain.matrix))
print(f"Number of noise ADC counts: {NOISE_ADC}")

# Create simulation file.
sim_path = GAIN_DIR / "sim_gaussian.h5"
if not sim_path.exists():
    simulate(
            num_events=NUM_EVENTS,
            output_file=str(sim_path),
            beam="disk",
            energy=ENERGY,
            radius=0.15,    # With 0.15 cm we have about 3000 pixels
            enc=NOISE_ADC, # We need to scale the ENC by the mean gain to get the same S/N
            zero_sup_threshold=0,
            readout_mode="rectangular",
            pitch=0.005,
            layout=HexagonalLayout.ODD_R,
            num_cols=304,
            num_rows=352,
            map_gain_file=gain_path,
            padding=Padding(2, 2, 2, 2),
            seed=SEED)

# Run the calibration pipeline.
cal_gain_path = GAIN_DIR / "sim_gaussian_matrix_gain.h5"
if not cal_gain_path.exists():
    calibrate(
        input_file=str(sim_path),
        energy=ENERGY,
        num_events=200000,
        zero_sup_threshold=3 * NOISE_ADC,
    )

cal_gain = CalibrationMatrixGain.from_hdf5(cal_gain_path)

# Plot the gain distribution (only for pixels with non-zero hits).
mask = cal_gain.hits > 0
edges = np.linspace(min(gain.matrix[mask]), max(gain.matrix[mask]), 50)

cal_matrix = cal_gain.matrix[mask]
hist_cal = Histogram1d(edges, xlabel="Gain (ADC/e-)")
hist_cal.fill(cal_matrix)

hist_mc = Histogram1d(edges, xlabel="Gain (ADC/e-)")
hist_mc.fill(gain.matrix[mask])

gain_distr = plt.figure()
hist_cal.plot(label="Calibrated gain")
hist_mc.plot(label="MC gain")
plt.legend()

# Print the mean of the number of hits per pixel.
hits = cal_gain.hits[mask]
print(f"Mean number of hits per pixel: {hits.mean():.2f}")


# Now run the reconstruction pipeline.
ZSUP = 3 * NOISE_ADC

mc = GAIN_DIR / "sim_gaussian_recon_mc.h5"
if not mc.exists():
    reconstruct(
            input_file=str(sim_path),
            suffix="recon_mc",
            zero_sup_threshold=ZSUP,
            max_neighbors=6,
            pos_recon_algorithm="centroid",
            map_gain_file=gain_path,
            padding=Padding(2, 2, 2, 2),
            )

cal = GAIN_DIR / "sim_gaussian_recon_cal.h5"
if not cal.exists():
    reconstruct(
            input_file=str(sim_path),
            suffix="recon_cal",
            zero_sup_threshold=ZSUP,
            max_neighbors=6,
            pos_recon_algorithm="centroid",
            map_gain_file=cal_gain_path,
            padding=Padding(2, 2, 2, 2),
            )

no_cal = GAIN_DIR / "sim_gaussian_recon_mean_corr.h5"
if not no_cal.exists():
    reconstruct(
            input_file=str(sim_path),
            suffix="recon_mean_corr",
            zero_sup_threshold=ZSUP,
            max_neighbors=6,
            pos_recon_algorithm="centroid",
            map_gain_file=None,
            padding=Padding(2, 2, 2, 2),
            )

mc_file = ReconInputFile(mc)
cal_file = ReconInputFile(cal)
no_cal_file = ReconInputFile(no_cal)

mc_energy = mc_file.column("energy")
cal_energy = cal_file.column("energy")
no_cal_energy = no_cal_file.column("energy")

mc_file.close()
cal_file.close()
no_cal_file.close()

edges = np.linspace(min(mc_energy), max(mc_energy), 100)
no_cal_edges = np.arange(min(no_cal_energy), max(no_cal_energy), 3.68/0.08)
hist_mc_energy = Histogram1d(edges, xlabel="Energy (eV)")
hist_mc_energy.fill(mc_energy)
hist_cal_energy = Histogram1d(edges, xlabel="Energy (eV)")
hist_cal_energy.fill(cal_energy)
hist_no_cal_energy = Histogram1d(no_cal_edges, xlabel="Energy (eV)")
hist_no_cal_energy.fill(no_cal_energy)
cal_model = Gaussian()
cal_model.fit(hist_cal_energy)
mc_model = Gaussian()
mc_model.fit(hist_mc_energy)


spectra = plt.figure()
hist_mc_energy.plot(label="MC gain")
hist_cal_energy.plot(label="Calibration")
hist_no_cal_energy.plot(label=r"Fixed gain ($g=0.08$ ADC/e$^-$)")
mc_model.plot(label=rf"MC fit: $\mu=${mc_model.mu.ufloat()} eV", color="C0")
cal_model.plot(label=rf"Cal fit: $\mu=${cal_model.mu.ufloat()} eV", color="C1")
plt.xlim(hist_mc_energy.bin_edges()[0]*0.9, hist_mc_energy.bin_edges()[-1]*1.1)
plt.legend()


gain_distr.savefig(FIGURES_GAIN / "gain_distribution.pdf", format="pdf")
spectra.savefig(FIGURES_GAIN / "spectra.pdf", format="pdf")

