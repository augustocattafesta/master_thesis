"""Study the size distribution of the clusters.
"""

import numpy as np
from aptapy.plotting import plt

from thesis import CALDB, DATA_DIR
from thesis.utils import (
    generate_dataset,
    create_sim_readout,
    reconstruct_dataset
)
from thesis.defaults import (
    DEFAULT_SENSOR,
    DEFAULT_SOURCE,
    HEADER_KWARGS
)

NUM_EVENTS = 10000
OVERWRITE = True
SIZE_ANALYSIS_DIR = DATA_DIR / "size"
if not SIZE_ANALYSIS_DIR.exists():
    SIZE_ANALYSIS_DIR.mkdir()

ENC_FILE = CALDB / "enc/sim_xpol3_enc-0_uniform_v001.h5"
GAIN_FILE = CALDB / "gain/sim_xpol3_gain-1_gauss-p10_v001.h5"
PEDESTAL_FILE = CALDB / "pedestal/sim_xpol3_pedestal-1000_gauss-p10_v001.h5"

readout = create_sim_readout(
    enc=str(ENC_FILE),
    gain=str(GAIN_FILE),
    pedestal=str(PEDESTAL_FILE)
)

dataset_path = SIZE_ANALYSIS_DIR / "simulation.h5"

data = generate_dataset(
    dataset_path,
    NUM_EVENTS,
    overwrite=OVERWRITE,
    sensor=DEFAULT_SENSOR,
    source=DEFAULT_SOURCE,
    readout=readout,
    header_kwargs=HEADER_KWARGS
)

