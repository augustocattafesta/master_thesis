from pathlib import Path

import numpy as np

SEED = 0
RNG = np.random.default_rng(0)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)