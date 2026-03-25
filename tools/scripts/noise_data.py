from hexsample.calibration import CalibrationMatrixNoise
from aptapy.plotting import plt
from aptapy.hist import Histogram1d
from config import DATA_DIR, FIGURES_DIR, RNG, SEED
import numpy as np

MAP_DIR = DATA_DIR / "reconstruction"
map_path = MAP_DIR / "020_0006531_data_all_matrix_noise.h5"

noise_cal = CalibrationMatrixNoise.from_hdf5(map_path)
matrix = noise_cal.matrix
hits = noise_cal.hits


m = matrix
m[hits == 0] = 0

plt.figure()
plt.imshow(m, origin="lower", cmap="viridis", vmin=4, vmax=10)
plt.colorbar(label="Noise electrons")
plt.xlabel("Column")
plt.ylabel("Row")

plt.figure()
plt.imshow(hits, origin="lower", cmap="viridis")
plt.colorbar(label="Number of hits")
plt.xlabel("Column")
plt.ylabel("Row")

print(f"Mean hits per pixel: {hits[hits > 0].mean():.2f}")

plt.figure()
plt.hist(hits[hits > 0], bins=100, log=True)
plt.xlabel("Number of hits")
plt.ylabel("Number of pixels")

HITS = 5000
print(f"Number of pixels with hits: {(hits > 0).sum()}")
print(f"Number of pixels with more than {HITS} hits: {(hits > HITS).sum()}")
h = matrix[hits > HITS]
edges = np.linspace(min(h), max(h), 100)
hist  = Histogram1d(edges, xlabel="Noise ADC", label="Data")
hist.fill(h)
plt.figure()
hist.plot(statistics=True)
plt.legend()
print(f"Mean noise electrons for pixels with more than {HITS} hits: {h.mean():.2f}")



plt.show()