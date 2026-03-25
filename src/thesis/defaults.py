from hexsample.hexagon import HexagonalLayout
from hexsample.readout import HexagonalReadoutRectangular, Padding
from hexsample.sensor import Sensor
from hexsample.source import DiskBeam, Line, Source

DEFAULT_ENERGY = 1e4
DEFAULT_READOUT_MODE = "rectangular"
DEFAULT_LAYOUT = HexagonalLayout.ODD_R
DEFAULT_NUM_COLS = 304
DEFAULT_NUM_ROWS = 352
DEFAULT_PITCH = 0.005
DEFAULT_TRANS_SIGMA = 60
DEFAULT_PADDING = Padding(7, 4, 4, 4)
DEFAULT_GAIN = 1.
DEFAULT_ENC = 100

DEFAULT_SPECTRUM = Line(DEFAULT_ENERGY)
DEFAULT_BEAM = DiskBeam(radius=0.5)
DEFAULT_SOURCE = Source(spectrum=DEFAULT_SPECTRUM, beam=DEFAULT_BEAM)

DEFAULT_SENSOR = Sensor(diffusion_sigma=DEFAULT_TRANS_SIGMA)

DEFAULT_NO_NOISE_READOUT = HexagonalReadoutRectangular(
    layout=DEFAULT_LAYOUT,
    num_cols=DEFAULT_NUM_COLS,
    num_rows=DEFAULT_NUM_ROWS,
    pitch=DEFAULT_PITCH,
    enc=0,
    gain=DEFAULT_GAIN,
    padding=DEFAULT_PADDING
)

DEFAULT_NOISE_READOUT = HexagonalReadoutRectangular(
    layout=DEFAULT_LAYOUT,
    num_cols=DEFAULT_NUM_COLS,
    num_rows=DEFAULT_NUM_ROWS,
    pitch=DEFAULT_PITCH,
    enc=DEFAULT_ENC,
    gain=DEFAULT_GAIN,
    padding=DEFAULT_PADDING
)

HEADER_KWARGS = dict(
    readout_mode=DEFAULT_READOUT_MODE,
    gain=DEFAULT_GAIN,
    enc=DEFAULT_ENC,
    layout=DEFAULT_LAYOUT,
    num_cols=DEFAULT_NUM_COLS,
    num_rows=DEFAULT_NUM_ROWS,
    pitch=DEFAULT_PITCH,
    padding=DEFAULT_PADDING,
)


DEFAULT_SIMULATION = dict(
    source=DEFAULT_SOURCE,
    sensor=DEFAULT_SENSOR,
    readout=DEFAULT_NO_NOISE_READOUT,
    header_kwargs=HEADER_KWARGS
)


# Eta recon parameters for 60 um/cm^0.5 diffusion sigma
DEFAULT_RECON_PARS = dict(
    eta_2pix_rad = 0.172,
    eta_2pix_pivot = 0.,
    eta_3pix_rad0 = 0.491,
    eta_3pix_rad1 = 0.1999,
    eta_3pix_rad_pivot = 0.,
    eta_3pix_theta0 = 0.1516,
)

