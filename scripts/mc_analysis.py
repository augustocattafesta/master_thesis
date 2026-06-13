import numpy as np
from aptapy.plotting import plt
from hexsample.resolution import dist_residual, eef_size_scan, eef
from hexsample.calibration import ChargeFractionMatrices

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
    ETA_60UM_RECON_PARS)


NUM_EVENTS = 10000
OVERWRITE = False
MC_ANALYSIS_DIR = DATA_DIR / "mc_analysis"
MLE_CALIBRATION_DIR = DATA_DIR / "mle_calibration"

no_noise_path = MC_ANALYSIS_DIR / "simulation.h5"
noise_path = MC_ANALYSIS_DIR / "simulation_noise.h5"
low_noise_path = MC_ANALYSIS_DIR / "simulation_low_noise.h5"
mle_tables_path = MLE_CALIBRATION_DIR / "simulation_mle_matrices.h5"
mle_tables = ChargeFractionMatrices.from_hdf5(str(mle_tables_path))


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

DEFAULT_NOISE_READOUT.enc = 30
DEFAULT_SIMULATION["readout"] = DEFAULT_NOISE_READOUT
generate_dataset(
    low_noise_path,
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
    recon_pars=ETA_60UM_RECON_PARS
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
    recon_pars=ETA_60UM_RECON_PARS
    )

noise_recon_mle = reconstruct_dataset(
    noise_path,
    recon_method="mle",
    zero_sup_threshold=ZERO_SUP_THRESHOLD,
    overwrite=True,
    charge_fraction_matrices=mle_tables
    )

low_noise_recon_centroid = reconstruct_dataset(
    low_noise_path,
    recon_method="centroid",
    zero_sup_threshold=60,
    overwrite=OVERWRITE,
    )

low_noise_recon_eta = reconstruct_dataset(
    low_noise_path,
    recon_method="eta",
    zero_sup_threshold=60,
    overwrite=OVERWRITE,
    recon_pars=ETA_60UM_RECON_PARS
    )

low_noise_recon_mle = reconstruct_dataset(
    low_noise_path,
    recon_method="mle",
    zero_sup_threshold=0,
    overwrite=True,
    charge_fraction_matrices=mle_tables
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
dr_mle = dist_residual(noise_recon_mle) / DEFAULT_PITCH

# mask = dr_mle > 0.7
# trig_id = noise_recon_mle.column("trigger_id")[mask]
# print(f"Triggers with large residuals: {trig_id}")

noise_cluster_size = noise_recon_centroid.column("cluster_size")
dr_eta_size1 = dr_eta[noise_cluster_size == 1]
dr_eta_size2 = dr_eta[noise_cluster_size == 2]
dr_eta_size3 = dr_eta[noise_cluster_size == 3]
dr_centroid_size1 = dr_centroid[noise_cluster_size == 1]
dr_centroid_size2 = dr_centroid[noise_cluster_size == 2]
dr_centroid_size3 = dr_centroid[noise_cluster_size == 3]

edges = np.linspace(0, 1.2, 100)
plt.figure("Distance residual for eta")
plt.hist([dr_eta_size1, dr_eta_size2, dr_eta_size3],
         bins=edges, label=["1 Pixel", "2 pixels", "3 pixels"], stacked=True,
         density=True, alpha=0.5, histtype="stepfilled")
# empty step
plt.hist(dr_mle, bins=edges, density=True, alpha=0.5, histtype="step", label="MLE")
plt.xlabel("Residual / Pitch")
plt.legend()

plt.figure("Distance residual for centroid")
plt.hist([dr_centroid_size1, dr_centroid_size2, dr_centroid_size3],
         bins=edges, label=["1 Pixel", "2 pixels", "3 pixels"], stacked=True,
         density=True, alpha=0.5, histtype="stepfilled")
plt.xlabel("Residual / Pitch")
plt.legend()




# # EEF analysis
x = np.linspace(0, 0.6, 100)
# eef_centroid = plt.figure("EEF 100 ENC noise centroid")
# eef_size_scan(x, noise_recon_centroid)
# eef_eta = plt.figure("EEF 100 ENC noise eta")
# eef_size_scan(x, noise_recon_eta)

# Low noise EEF analysis
eef_centroid_low_noise = eef(x, low_noise_recon_centroid, max_neighbors=6)
eef_eta_low_noise = eef(x, low_noise_recon_eta, max_neighbors=6)
eef_mle_low_noise = eef(x, low_noise_recon_mle, max_neighbors=6)
plt.figure("EEF 30 ENC noise centroid vs eta vs mle")
plt.plot(x, eef_centroid_low_noise, label="Centroid", color="blue")
plt.plot(x, eef_eta_low_noise, label="Eta", color="red")
plt.plot(x, eef_mle_low_noise, label="MLE", color="black")
plt.xlabel("Distance Residual / Pitch")
plt.ylabel("EEF")
plt.legend()

# # High noise EEF analysis
all_eef_centroid = eef(x, noise_recon_centroid, max_neighbors=6)
all_eef_eta = eef(x, noise_recon_eta, max_neighbors=6)
all_eef_mle = eef(x, noise_recon_mle, max_neighbors=6)
plt.figure("EEF 100 ENC noise centroid vs eta vs mle")
plt.plot(x, all_eef_centroid, label="Centroid", color="blue")
plt.plot(x, all_eef_eta, label="Eta", color="red")
plt.plot(x, all_eef_mle, label="MLE", color="black")
plt.xlabel("Distance Residual / Pitch")
plt.ylabel("EEF")
plt.legend()

# # No noise EEF analysis
# plt.figure("EEF 0 ENC noise centroid")
# eef_size_scan(x, no_noise_recon_centroid)
# plt.figure("EEF 0 ENC noise eta")
# eef_size_scan(x, no_noise_recon_eta)


no_noise_recon_centroid.close()
no_noise_recon_eta.close()
noise_recon_centroid.close()
noise_recon_eta.close()
low_noise_recon_centroid.close()
low_noise_recon_eta.close()
low_noise_recon_mle.close()




# Show pull
# absx = noise_recon_mle.mc_column("absx")
# posx = noise_recon_mle.column("posx")
# errx_low = noise_recon_mle.column("errx_low")
# errx_up = noise_recon_mle.column("errx_high")
# mask = (abs(errx_low) > 0) & (abs(errx_up) > 0)
# errx = (abs(errx_low) + abs(errx_up)) / 2
# pullx = (posx[mask] - absx[mask]) / errx[mask]

# std = np.std(pullx[abs(pullx) < 5])
# plt.figure("Pull distribution")
# plt.hist(pullx, bins=np.linspace(-5, 5, 100), label=f"Std: {std:.2f}")
# plt.xlabel("Pull")
# plt.legend()



plt.show()