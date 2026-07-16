import numpy as np
from aptapy.plotting import plt
from hexsample import rng
from hexsample.resolution import SlitsAligner
from hexsample.source import SlitBeam

rng.initialize(seed=0)

HEIGHT = 1.
WIDTH = 100.
THETA = 1.
AREA = HEIGHT * WIDTH
BEAM = SlitBeam(0., 0., HEIGHT, WIDTH, THETA)
# Study the dependence of the fitted angle on the number of events density
density = np.logspace(1, 5, 10)
sigma_ar = np.array([1, 2, 4, 10])
theta_fit = np.zeros((len(density), len(sigma_ar)))
X, Y = BEAM.rvs(int(density[-1] * AREA))
for i, d in enumerate(density):
    n_events = int(d * AREA)
    x = X[:n_events]
    y = Y[:n_events]
    print(f"Density: {d:.1f} events/mm^2, number of events: {n_events}")
    bin_size = 2 * np.sqrt(1 / d)
    print(f"Bin size: {bin_size:.3f} mm")
    for j, sigma in enumerate(sigma_ar):
        aligner = SlitsAligner(bin_size, sigma)
        aligner.align(x, y)
        theta_fit[i, j] = np.abs(np.rad2deg(aligner.angle))
        print(f"Fitted angle: {theta_fit[i, j]:.2f} degrees\n")

plt.plot(density, theta_fit, marker="o", label=[f"sigma={s}" for s in sigma_ar])
plt.xscale("log")
plt.legend()
plt.show()