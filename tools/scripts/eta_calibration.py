import sys
from aptapy.hist import Histogram1d

from hexsample.pipeline import reconstruct
from hexsample.fileio import ReconInputFile
from hexsample.resolution import eef, eef_size_scan, resolution_spatial_dependence

from config import DATA_DIR, FIGURES_DIR

SIZE_DIR = DATA_DIR / "size"
SIZE_DIR.mkdir(exist_ok=True)
RECON_DIR = DATA_DIR / "reconstruction"
RESOLUTION_DIR = DATA_DIR / "resolution"
FIGURES_RESOLUTION = FIGURES_DIR / "resolution"
FIGURES_RESOLUTION.mkdir(exist_ok=True)

NOISE_E = 8

path_scripts = "/home/augusto/Thesis/hexsample/scripts"
sys.path.append(path_scripts)
import calibration


cal_file = SIZE_DIR / "simulation.h5"
args = dict(zero_sup_threshold=200, input_file=cal_file, save=True)
# sigma_2pix, offset_rad_3pix, sigma_rad_3pix, sigma_theta_3pix = calibration.hxeta(**args)

recon_pars = dict(
    eta_2pix_rad = 0.172,
    eta_2pix_pivot = 0.,
    eta_3pix_rad0 = 0.491,
    eta_3pix_rad1 = 0.1999,
    eta_3pix_rad_pivot = 0.,
    eta_3pix_theta0 = 0.1516,
)

data_path = RECON_DIR / "020_0006531_data_all.h5"
data_recon_centroid = RECON_DIR / "020_0006531_data_all_recon_centroid.h5"
if not data_recon_centroid.exists():
    reconstruct(
            input_file=str(data_path),
            zero_sup_threshold=2 * NOISE_E,
            max_neighbors=6,
            suffix="recon_centroid",
            pos_recon_algorithm="centroid",
            **recon_pars
            )

data_recon_eta = RECON_DIR / "020_0006531_data_all_recon_eta.h5"
if not data_recon_eta.exists():
    reconstruct(
            input_file=str(data_path),
            zero_sup_threshold=2 * NOISE_E,
            max_neighbors=6,
            suffix="recon_eta",
            pos_recon_algorithm="eta",
            **recon_pars,
            )

