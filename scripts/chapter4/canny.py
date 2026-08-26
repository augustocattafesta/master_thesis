import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform
from aptapy.hist import Histogram1d

from thesis import FIGURES_DIR

np.random.seed(0)

SIZE = (200, 200)
EDGES = (np.arange(-0.5, SIZE[0] + 0.5, 1), np.arange(-0.5, SIZE[1] + 0.5, 1))

N = 10000
ANGLE = -5
SCALE = 5
x_ = np.random.uniform(20., 180., N)
y_ = 100 + np.random.normal(loc=0, scale=SCALE, size=N)
x = x_ * np.cos(np.deg2rad(ANGLE)) + y_ * np.sin(np.deg2rad(ANGLE))
y = y_ * np.cos(np.deg2rad(ANGLE)) - x_ * np.sin(np.deg2rad(ANGLE))


N = 0
NOISE_X = np.random.uniform(0., SIZE[0], N)
NOISE_Y = np.random.uniform(0., SIZE[1], N)

x = np.concatenate((x, NOISE_X))
y = np.concatenate((y, NOISE_Y))

threshold = 0
image = np.histogram2d(x, y, bins=EDGES)[0]
image[image < threshold] = 0

fig_canny, axs = plt.subplots(2, 3)
axs[0, 0].set_title('a. Original')
axs[0, 0].imshow(image.T, origin='lower', cmap='gray')

SIGMA = (1, 2, 3, 5, 10)

edge0 = feature.canny(image.T, sigma=SIGMA[0])
edge1 = feature.canny(image.T, sigma=SIGMA[1])
edge2 = feature.canny(image.T, sigma=SIGMA[2])
edge3 = feature.canny(image.T, sigma=SIGMA[3])
edge4 = feature.canny(image.T, sigma=SIGMA[4])

axs[0, 1].imshow(edge0, origin='lower', cmap='gray')
axs[0, 1].set_title(rf'b. $\sigma$ = {SIGMA[0]}')
axs[0, 2].imshow(edge1, origin='lower', cmap='gray')
axs[0, 2].set_title(rf'c. $\sigma$ = {SIGMA[1]}')
axs[1, 0].imshow(edge2, origin='lower', cmap='gray')
axs[1, 0].set_title(rf'd. $\sigma$ = {SIGMA[2]}')
axs[1, 1].imshow(edge3, origin='lower', cmap='gray')
axs[1, 1].set_title(rf'e. $\sigma$ = {SIGMA[3]}')
axs[1, 2].imshow(edge4, origin='lower', cmap='gray')
axs[1, 2].set_title(rf'f. $\sigma$ = {SIGMA[4]}')

for ax in axs:
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
plt.tight_layout()

plt.rcParams.update(
    {
        "axes.labelsize": 20,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

test_angles = np.deg2rad(np.linspace(90, 100, 1000))
hspace, angles, distances = transform.hough_line(edge3, theta=test_angles)
_, peaks_angles, peaks_dist = transform.hough_line_peaks(hspace, angles, distances, num_peaks=1)
hough_fig = plt.figure()
angles_deg = np.rad2deg(angles)
extent = [angles_deg[0], angles_deg[-1], distances[-1], distances[0]]
plt.imshow(hspace, extent=extent, cmap='gray', aspect='auto')
plt.colorbar(label='Entries / Bin')
plt.xlabel(r"$\theta$ [degrees]")
plt.ylabel(r"$\rho$ [pixels]")
plt.ylim(60, 140)
plt.tight_layout()


angle = peaks_angles[0] - np.pi / 2
x_rot = x * np.cos(angle) + y * np.sin(angle)
y_rot = y * np.cos(angle) - x * np.sin(angle)

lsf, lsf_edge = np.histogram(y_rot, bins=50) 
lsf_fig = plt.figure()
lsf_hist = Histogram1d(lsf_edge, xlabel="y [pixels]")
lsf_hist.set_content(lsf)
from aptapy.models import Gaussian
model = Gaussian()
model.fit(lsf_hist)
lsf_hist.plot()
model.plot(label="Gaussian\n"+rf"$\sigma={model.sigma.ufloat()}$ pix")
plt.legend()


mtf_fft = np.abs(np.fft.fft(lsf))
freq_fft = np.fft.fftfreq(len(lsf), d=lsf_edge[1] - lsf_edge[0])
# Consider only the first half of the MTF, which corresponds to the positive spatial
# frequencies. The MTF is symmetric for real signals, so we can ignore the second half.
mtf = mtf_fft[:len(lsf) // 2]
freqs = freq_fft[:len(lsf) // 2]
# Normalize the MTF so that its value at zero frequency is 1.
mtf /= mtf[0]


mtf_fig = plt.figure()
plt.plot(freqs, mtf, '-k')
plt.hlines(0.1, 0, 0.5, colors="0.6", linestyles="dashed", linewidth=1.0)
plt.annotate("10%", xy=(0.15, 0.1), xytext=(0.15, 0.11), fontsize=10)

plt.xlim(0, 0.5)
plt.ylim(0, 1)
plt.xlabel('Spatial frequency [lp/pixel]')
plt.ylabel('MTF')


fig_canny.savefig(FIGURES_DIR / "chapter4/position/mtf_canny.pdf", format="pdf", bbox_inches="tight")
hough_fig.savefig(FIGURES_DIR / "chapter4/position/mtf_hough.pdf", format="pdf", bbox_inches="tight")
lsf_fig.savefig(FIGURES_DIR / "chapter4/position/mtf_lsf.pdf", format="pdf", bbox_inches="tight")
mtf_fig.savefig(FIGURES_DIR / "chapter4/position/mtf.pdf", format="pdf", bbox_inches="tight")

plt.show()