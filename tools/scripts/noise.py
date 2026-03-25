import numpy as np
import xraydb
from aptapy.hist import Histogram1d
from aptapy.models import Gaussian
from aptapy.plotting import plt, last_line_color

from hexsample.calibration import CalibrationMatrixNoise
from hexsample.fileio import ReconInputFile, DigiInputFileRectangular
from hexsample.hexagon import HexagonalLayout, HexagonalGrid
from hexsample.display import HexagonalGridDisplay
from hexsample.pipeline import calibrate, reconstruct, simulate
from hexsample.roi import Padding

from config import DATA_DIR, FIGURES_DIR, RNG, SEED

NOISE_DIR = DATA_DIR / "noise"
NOISE_DIR.mkdir(exist_ok=True)
FIGURES_NOISE = FIGURES_DIR / "noise"
FIGURES_NOISE.mkdir(exist_ok=True)

ENERGY = 1e4
SIGNAL_E = int(ENERGY / xraydb.ionization_potential("Si"))
SNR = 20
NOISE_E = int(SIGNAL_E / SNR)
EXP_NOISE = NOISE_E * np.sqrt(2 / np.pi)
NUM_EVENTS = 1000000

print(f"Number of events: {NUM_EVENTS}")
print(f"Number of signal electrons: {SIGNAL_E}")
print(f"Signal-to-noise ratio: {SNR}")
print(f"Number of noise electrons: {NOISE_E}")

file_name = "simulation"
sim_path = NOISE_DIR / f"{file_name}.h5"
if not sim_path.exists():
    simulate(
            num_events=NUM_EVENTS,
            output_file=str(sim_path),
            beam="disk",
            energy=ENERGY,
            radius=0.5,    # With 0.5 cm we have about 40000 pixels
            enc=NOISE_E,    # We need the number of noise electrons here
            zero_sup_threshold=0,
            readout_mode="rectangular",
            pitch=0.005,
            layout=HexagonalLayout.ODD_R,
            num_cols=304,
            num_rows=352,
            padding=Padding(7, 4, 4, 4),
            random_seed=SEED)

cal_noise_path = NOISE_DIR / f"{file_name}_matrix_noise.h5"
if not cal_noise_path.exists():
    calibrate(
        input_file=str(sim_path),
        energy=ENERGY,
        num_events=100,
    )


matrix_noise = CalibrationMatrixNoise.from_hdf5(cal_noise_path)
plt.figure(figsize=(8, 6))
plt.imshow(matrix_noise.matrix, cmap="viridis", vmin=0)
plt.colorbar(label="Noise electrons")
sim_file = DigiInputFileRectangular(sim_path)
noise_event = sim_file.digi_event(0)
sim_file.close()
display = HexagonalGridDisplay(HexagonalGrid())
display.draw_digi_event(noise_event, 0)
display.setup_gca()
# plt.savefig(FIGURES_NOISE / "noise_event.pdf")
plt.close()

with np.errstate(divide='ignore', invalid='ignore'):
    err_cut = np.where(matrix_noise.hits > 0, 0.707 / matrix_noise.hits, 1000)

mask = (matrix_noise.hits > 1200) #& (err_cut < 0.02)
res = (matrix_noise.matrix[mask] - NOISE_E) / NOISE_E


print(f"Number of pixels with noise measurement: {np.sum(matrix_noise.hits > 0)}")
print(f"Number of pixels with noise measurement and error < 2%: {np.sum(mask)}")
print(f"Mean number of hits per pixel: {np.mean(matrix_noise.hits[mask]):.0f}")
print(f"Std residual: {np.std(res, where=res<0.05):.4f}")

edges = np.linspace(-0.3, 0.3, 100)
hist = Histogram1d(edges)
hist.fill(res)
model = Gaussian()
model.fit(hist)
plt.figure()
hist.plot()
model.plot(fit_output=True)
plt.legend()

plt.figure()
with np.errstate(divide='ignore', invalid='ignore'):
    m = np.where(matrix_noise.hits > 0, 0.707 / np.sqrt(matrix_noise.hits), 0)
plt.imshow(m, cmap="viridis", vmin=0)
plt.colorbar(label="Noise electrons")



mask_few_hits = (matrix_noise.hits > 0) & (matrix_noise.hits < 10)
few_hits = matrix_noise.matrix[mask_few_hits]
plt.figure()
plt.hist(few_hits, bins=50, density=True)
print(f"Number of pixels with few hits: {len(few_hits)}")
plt.figure()
plt.hist(matrix_noise.hits[mask_few_hits], bins=50, density=True)


plt.show()


