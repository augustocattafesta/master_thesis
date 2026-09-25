import numpy as np
import matplotlib.pyplot as plt

N = 100000
NBINS = 100


dx = np.random.normal(0, 1, N)
dy = np.random.normal(0, 1, N)

dr = np.sqrt(dx**2 + dy**2)
dr_hist, dr_edges = np.histogram(dr, bins=NBINS)
eef = np.cumsum(dr_hist) / np.sum(dr_hist)

residuals, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(dx, bins=NBINS, alpha=0.5, label='dx')
axs[0].set_xlabel('dx')
axs[0].set_ylabel('Entries / bin')
axs[1].hist(dy, bins=NBINS, alpha=0.5, label='dy')
axs[1].set_xlabel('dy')
axs[1].set_ylabel('Entries / bin')

eef_fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(dr, bins=NBINS, alpha=0.5, label='dr')
axs[0].set_xlabel(r'dr$ = \sqrt{dx^2 + dy^2}$')
axs[0].set_ylabel('Entries / bin')

dr_centers = 0.5 * (dr_edges[1:] + dr_edges[:-1])
axs[1].plot(dr_centers, eef, label='EEF')
axs[1].set_xlabel('dr')
axs[1].set_ylabel('EEF')

plt.tight_layout()

output_path = "/home/augusto/Thesis/master_thesis/slides/residuals.png"
output_path_eef = "/home/augusto/Thesis/master_thesis/slides/eef.png"
residuals.savefig(output_path, dpi=300, bbox_inches='tight')
eef_fig.savefig(output_path_eef, dpi=300, bbox_inches='tight')

plt.show()