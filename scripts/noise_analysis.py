import numpy as np
from aptapy.plotting import plt
from hexsample.calibration import CalibrationMatrixNoise
from hexsample.tasks import calibrate

from thesis import DATA_DIR, FIGURES_DIR
from thesis.utils import generate_dataset
from thesis.defaults import (
    DEFAULT_SIMULATION,
    DEFAULT_ENERGY,
    DEFAULT_NOISE_READOUT,
    DEFAULT_ENC,
    DEFAULT_NUM_ROWS,
    DEFAULT_NUM_COLS,
    )

NUM_EVENTS = 500000
OVERWRITE = False
NOISE_ANALYSIS_DIR = DATA_DIR / "noise_analysis"
if not NOISE_ANALYSIS_DIR.exists():
    NOISE_ANALYSIS_DIR.mkdir(parents=True)
HDF_DIR = DATA_DIR / "hdf"

file_path = NOISE_ANALYSIS_DIR / "simulation_noise.h5"

rng = np.random.default_rng(seed=0)
DEFAULT_NOISE_READOUT.enc = rng.normal(loc=DEFAULT_ENC,
                                      scale=0.1 * DEFAULT_ENC,
                                      size=(DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS))
DEFAULT_SIMULATION["readout"] = DEFAULT_NOISE_READOUT

generate_dataset(
    file_path,
    NUM_EVENTS,
    OVERWRITE,
    **DEFAULT_SIMULATION
    ).close()


# Calculate noise matrix for MC simulation
matrix_noise_path = NOISE_ANALYSIS_DIR / f"{file_path.stem}_matrix_noise.h5"
if not matrix_noise_path.exists() or OVERWRITE:
    calibrate(
        file_path,
        DEFAULT_ENERGY,
        0,
        quantity="noise"
        )
mc_matrix_noise = CalibrationMatrixNoise.from_hdf5(str(matrix_noise_path))

# Calculate noise matrix for data
data_path = HDF_DIR / "020_0006531_data_all.h5"
data_matrix_noise_path = HDF_DIR / "020_0006531_data_all_matrix_noise.h5"
if not data_matrix_noise_path.exists() or OVERWRITE:
    calibrate(
        data_path,
        DEFAULT_ENERGY,
        0,
        quantity="noise"
        )
data_matrix_noise = CalibrationMatrixNoise.from_hdf5(str(data_matrix_noise_path))

mc_matrix_noise.plot(100)
data_matrix_noise.plot(1000)
plt.show()