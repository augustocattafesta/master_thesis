"""Utils
"""

from typing import Tuple

import numpy as np
import scipy.signal
import xraydb
import aptapy.models
from aptapy.models import Line
from aptapy.plotting import plt
from aptapy.typing_ import ArrayLike
from uncertainties import unumpy

ELEMENTARY_CHARGE = 1.609e-19   # Coulomb


def weighted_energy(element: str, *lines: str) -> float:
    """Compute the intensity-weighted mean energy of specified X-ray lines of an element.

    The data are retrieved from the X-ray database at https://xraydb.seescience.org/.

    Arguments
    ----------
    element : str
        Atomic symbol of the element.

    lines : str
        Lines to consider in the weighted mean.
    
    Returns
    -------
    energy : float
        Intensity-weighted mean energy of the lines.
    """
    line_data = [xraydb.xray_line(element=element, line=line) for line in lines]
    total_intensity = sum(line.intensity for line in line_data)

    return sum(line.energy * line.intensity for line in line_data) / total_intensity



KALPHA = weighted_energy('Mn', 'Ka1', 'Ka2', 'Ka3') * 1e-3


class PeakAnalyzer:

    @staticmethod
    def find_peaks_iterative(xdata: ArrayLike, ydata: ArrayLike,
                              npeaks: int) -> Tuple[ArrayLike, ArrayLike]:
        """Find the position and height of a fixed number of peaks in a sample of data

        Arguments
        ---------
        xdata : ArrayLike,
            The x values of the sample.

        ydata : ArrayLike,
            The y values of the sample.

        nlines : int,
            Maximum number of peaks to find in the sample.

        Returns
        -------
        xpeaks : ArrayLike
            The position of the peaks on the x axis.

        ypeaks : ArrayLike
            The height of the peaks.
        """
        min_width, max_width = 0, len(ydata)
        peaks, properties = scipy.signal.find_peaks(ydata, width=(min_width, max_width))
        widths = properties['widths']
        while len(peaks) > npeaks:
            min_width = min(widths)*1.1
            peaks, properties = scipy.signal.find_peaks(ydata, width=(min_width, max_width))
            widths = properties['widths']

        return xdata[peaks], ydata[peaks]


def gain(W: float, capacity: float, voltages: ArrayLike, line_adc: ArrayLike, line_pars: ArrayLike,
         **kwargs) -> ArrayLike:
    """_summary_

    Arguments
    ----------
    W : float
        _description_
    capacity : float
        _description_
    voltages : ArrayLike
        _description_
    line_adc : ArrayLike
        _description_
    line_model : Line
        _description_

    Returns
    -------
    ArrayLike
        _description_
    """
    exp_electrons = KALPHA / (W * 1e-3)
    slope = line_pars[0] * 1e3 / capacity * ELEMENTARY_CHARGE
    meas_electrons = (line_adc - line_pars[1]) / slope
    gain = meas_electrons / exp_electrons
    plt.figure("Gain")
    plt.errorbar(voltages, unumpy.nominal_values(gain), unumpy.std_devs(gain), fmt='ko')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")

    if kwargs.get("fit", False):
        model = aptapy.models.Exponential()
        model.fit(voltages, unumpy.nominal_values(gain), sigma=unumpy.std_devs(gain), absolute_sigma=True)
        model.plot(label=f"scale: {-model.scale.ufloat()} V")

    return gain

def energy_resolution(voltages: ArrayLike, line_adc: ArrayLike, sigma: ArrayLike) -> ArrayLike:
    """_summary_

    Arguments
    ----------
    voltages : ArrayLike
        _description_
    line_adc : ArrayLike
        _description_
    sigma : ArrayLike
        _description_

    Returns
    -------
    ArrayLike
        _description_
    """
    plt.figure("Energy Resolution")
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    resolution = fwhm / line_adc
    plt.errorbar(voltages, unumpy.nominal_values(resolution), unumpy.std_devs(resolution),
                    fmt='ko')
    plt.xlabel("Voltage [V]")
    plt.ylabel("FWHM / E")

    return resolution
