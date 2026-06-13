from hexsample.readout import HexagonalReadoutRectangular, Padding
from hexsample.sensor import Sensor
from hexsample.source import DiskBeam, Line, Source
from hexsample.xpol import XPOL3

from . import CALDB

# Default calibration files
DEFAULT_ENC_0 = CALDB / "sim_xpol3_enc-0_uniform_v001.h5"
DEFAULT_ENC_20 = CALDB / "sim_xpol3_enc-20_gauss-p10_v001.h5"
DEFAULT_ENC_100 = CALDB / "sim_xpol3_enc-100_gauss-p10_v001.h5"

DEFAULT_NOISE_0 = CALDB / "sim_xpol3_noise-0_uniform_v001.h5"
DEFAULT_NOISE_8 = CALDB / "sim_xpol3_noise-8_gauss-p10_v001.h5"
DEFAULT_NOISE_20 = CALDB / "sim_xpol3_noise-20_gauss-p10_v001.h5"
DEFAULT_NOISE_100 = CALDB / "sim_xpol3_noise-100_gauss-p10_v001.h5"

DEFAULT_GAIN_ONE = CALDB / "sim_xpol3_gain-1_uniform_v001.h5"
DEFAULT_GAIN_SMALL = CALDB / "sim_xpol3_gain-0p08_gauss-p10_v001.h5"

DEFAULT_PEDESTAL_0 = CALDB / "sim_xpol3_pedestal-0_uniform_v001.h5"
DEFAULT_PEDESTAL_1000 = CALDB / "sim_xpol3_pedestal-1000_gauss-p10_v001.h5" 

# Default chip configuration parameters
DEFAULT_LAYOUT = XPOL3.layout
DEFAULT_NUM_COLS, DEFAULT_NUM_ROWS = XPOL3.size
DEFAULT_PITCH = XPOL3.pitch
DEFAULT_READOUT_MODE = "rectangular"
DEFAULT_PADDING = Padding(7, 4, 4, 4)

# Default sensor parameters
DEFAULT_TRANS_SIGMA = 60
DEFAULT_MATERIAL_SYMBOL = "Si"
DEFAULT_THICKNESS = 0.03
DEFAULT_SENSOR = Sensor(
    material_symbol=DEFAULT_MATERIAL_SYMBOL,
    thickness=DEFAULT_THICKNESS,
    diffusion_sigma=DEFAULT_TRANS_SIGMA
    )

# Default source parameters
DEFAULT_ENERGY = 1e4
DEFAULT_SPECTRUM = Line(DEFAULT_ENERGY)
DEFAULT_BEAM = DiskBeam(radius=0.5)
DEFAULT_SOURCE = Source(
    spectrum=DEFAULT_SPECTRUM,
    beam=DEFAULT_BEAM
    )

HEADER_KWARGS = dict(
    readout_mode=DEFAULT_READOUT_MODE,
    layout=DEFAULT_LAYOUT,
    num_cols=DEFAULT_NUM_COLS,
    num_rows=DEFAULT_NUM_ROWS,
    pitch=DEFAULT_PITCH,
    padding=DEFAULT_PADDING,
)
