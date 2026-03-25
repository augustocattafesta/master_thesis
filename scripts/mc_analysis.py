import numpy as np
from aptapy.plotting import plt
from hexsample.resolution import dist_residual, eef_size_scan

from thesis import DATA_DIR, FIGURES_DIR
from thesis.utils import (
    cluster_size_hist,
    generate_dataset,
    reconstruct_dataset)
from thesis.defaults import (
    DEFAULT_SIMULATION,
    DEFAULT_NOISE_READOUT,
    DEFAULT_ENC,
    DEFAULT_PITCH,
    DEFAULT_RECON_PARS)


NUM_EVENTS = 10000
OVERWRITE = False
MC_ANALYSIS_DIR = DATA_DIR / "mc_analysis"

no_noise_path = MC_ANALYSIS_DIR / "simulation.h5"
noise_path = MC_ANALYSIS_DIR / "simulation_noise.h5"

generate_dataset(
    no_noise_path,
    NUM_EVENTS,
    OVERWRITE,
    **DEFAULT_SIMULATION
    ).close()

DEFAULT_SIMULATION["readout"] = DEFAULT_NOISE_READOUT
generate_dataset(
    noise_path,
    NUM_EVENTS,
    OVERWRITE,
    **DEFAULT_SIMULATION
    ).close()

ZERO_SUP_THRESHOLD = 2 * DEFAULT_ENC

no_noise_recon_centroid = reconstruct_dataset(
    no_noise_path,
    recon_method="centroid",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=OVERWRITE,
    )

no_noise_recon_eta = reconstruct_dataset(
    no_noise_path,
    recon_method="eta",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=OVERWRITE,
    **DEFAULT_RECON_PARS
    )

noise_recon_centroid = reconstruct_dataset(
    noise_path,
    recon_method="centroid",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=OVERWRITE,
    )

noise_recon_eta = reconstruct_dataset(
    noise_path,
    recon_method="eta",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=OVERWRITE,
    **DEFAULT_RECON_PARS
    )

# Cluster size analysis
hist_no_noise = cluster_size_hist(no_noise_recon_centroid)
hist_noise = cluster_size_hist(noise_recon_centroid)
plt.figure()
hist_no_noise.plot(label="No noise")
hist_noise.plot(label="With noise")
plt.legend()



# Residual analysis
dr_eta = dist_residual(noise_recon_eta) / DEFAULT_PITCH
dr_centroid = dist_residual(noise_recon_centroid) / DEFAULT_PITCH

noise_cluster_size = noise_recon_centroid.column("cluster_size")
dr_eta_size1 = dr_eta[noise_cluster_size == 1]
dr_eta_size2 = dr_eta[noise_cluster_size == 2]
dr_eta_size3 = dr_eta[noise_cluster_size == 3]
dr_centroid_size1 = dr_centroid[noise_cluster_size == 1]
dr_centroid_size2 = dr_centroid[noise_cluster_size == 2]
dr_centroid_size3 = dr_centroid[noise_cluster_size == 3]

edges = np.linspace(0, 0.6, 50)
plt.figure("Distance residual for eta")
plt.hist([dr_eta_size1, dr_eta_size2, dr_eta_size3],
         bins=edges, label=["1 Pixel", "2 pixels", "3 pixels"], stacked=True,
         density=True, alpha=0.5, histtype="stepfilled")
plt.xlabel("Residual / Pitch")
plt.legend()

plt.figure("Distance residual for centroid")
plt.hist([dr_centroid_size1, dr_centroid_size2, dr_centroid_size3],
         bins=edges, label=["1 Pixel", "2 pixels", "3 pixels"], stacked=True,
         density=True, alpha=0.5, histtype="stepfilled")
plt.xlabel("Residual / Pitch")
plt.legend()


# EEF analysis
x = np.linspace(0, 0.6, 100)
eef_centroid = plt.figure("EEF 100 ENC noise centroid")
eef_size_scan(x, noise_recon_centroid)
eef_eta = plt.figure("EEF 100 ENC noise eta")
eef_size_scan(x, noise_recon_eta)

# No noise EEF analysis
plt.figure("EEF 0 ENC noise centroid")
eef_size_scan(x, no_noise_recon_centroid)
plt.figure("EEF 0 ENC noise eta")
eef_size_scan(x, no_noise_recon_eta)


no_noise_recon_centroid.close()
no_noise_recon_eta.close()
noise_recon_centroid.close()
noise_recon_eta.close()

plt.show()