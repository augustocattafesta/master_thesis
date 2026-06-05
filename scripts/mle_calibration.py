import numpy as np
from aptapy.plotting import plt
from hexsample.tasks import calibrate_mle

from thesis import DATA_DIR, FIGURES_DIR
from thesis.utils import generate_dataset
from thesis.defaults import DEFAULT_SIMULATION


NUM_EVENTS = 200000
OVERWRITE = False
MLE_CALIBRATION_DIR = DATA_DIR / "mle_calibration"
if not MLE_CALIBRATION_DIR.exists():
    MLE_CALIBRATION_DIR.mkdir()

no_noise_path = str(MLE_CALIBRATION_DIR / "simulation.h5")

generate_dataset(
    no_noise_path,
    NUM_EVENTS,
    OVERWRITE,
    **DEFAULT_SIMULATION
    ).close()


mle_matrices_path = calibrate_mle(
    no_noise_path,
    bin_size=0.02
)
