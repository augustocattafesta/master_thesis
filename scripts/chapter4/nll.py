import numpy as np
from aptapy.plotting import plt

from hexsample.caldb import CalDB
from hexsample.likelihood import nll_numba

from thesis import FIGURES_DIR

def test_nll_numba():
    """Test the nll_numba function.
    """
    # Test a simple case, this pha is taken from an event with 20 ENC noise, offset set to 512.
    pha = np.array([1023, -2, 601, 17, -23, 39, 55])
    x = np.linspace(-0.5, 0.5, 100)
    y = np.linspace(-1/np.sqrt(3), 1/np.sqrt(3), 100)
    # Load the charge fraction matrices and extract the relevant attributes
    position_cal = CalDB.open_position(
        "sim_xpol3_position_layout-oddr_pitch-50_diff-40_thick-300_v001")
    f = position_cal.values
    x_bins = position_cal.x_bins
    y_bins = position_cal.y_bins
    xbin0 = x_bins[0]
    ybin0 = y_bins[0]
    bin_size = x_bins[1] - x_bins[0]

    sigma = np.array([20.0] * 7)
    nll = np.zeros((len(x), len(y)))
    for i_x, _x in enumerate(x):
        for i_y, _y in enumerate(y):
            nll[i_x, i_y] = nll_numba(_x, _y, np.sum(pha),pha, f, xbin0, ybin0, bin_size, sigma)
    fig = plt.figure("test_negative_log_likelihood")
    plt.imshow(nll.T, extent=(x[0], x[-1], y[0], y[-1]), origin="lower")
    plt.colorbar(label="NLL")
    plt.xlabel("x [pitch]")
    plt.ylabel("y [pitch]")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "chapter4/position/nll.pdf", format="pdf", bbox_inches="tight")
    plt.show()
test_nll_numba()