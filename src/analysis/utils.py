"""Module containing various methods to estimate physical quantities or find properties of a
signal.
"""

import numpy as np
import scipy.signal
import xraydb

ELEMENTARY_CHARGE = 1.609e-19   # Coulomb
ELECTRONS_IN_1FC = 1e-15 / ELEMENTARY_CHARGE  # Number of electrons in 1 fC
W_ARGON = 26.   # eV


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


KALPHA = weighted_energy('Mn', 'Ka1', 'Ka2', 'Ka3') * 1e-3  # keV
KBETA = weighted_energy('Mn', 'Kb1', 'Kb3', 'Kb5') * 1e-3   # keV
AR_ESCAPE = weighted_energy("Ar", "Ka1", "Ka2", "Ka3", "Kb1", "Kb3") * 1e-3 # keV


def find_peaks_iterative(xdata: np.ndarray, ydata: np.ndarray,
                            npeaks: int) -> tuple[np.ndarray, np.ndarray]:
    """Find the position and height of a fixed number of peaks in a sample of data

    Arguments
    ---------
    xdata : ArrayLike,
        The x values of the sample.
    ydata : ArrayLike,
        The y values of the sample.
    npeaks : int,
        Maximum number of peaks to find in the sample.

    Returns
    -------
    xpeaks : ArrayLike
        The position of the peaks on the x axis.
    ypeaks : ArrayLike
        The height of the peaks.
    """
    min_width, max_width = 0., len(ydata)
    peaks, properties = scipy.signal.find_peaks(ydata, width=(min_width, max_width))
    widths = properties['widths']
    while len(peaks) > npeaks:
        min_width = min(widths)*1.1
        peaks, properties = scipy.signal.find_peaks(ydata, width=(min_width, max_width))
        widths = properties['widths']

    return xdata[peaks], ydata[peaks]


def gain(w: float, line_val: np.ndarray, energy: float) -> np.ndarray:
    """Estimate the gain of the detector from the analysis of a spectral emission line.

    Arguments
    ----------
    w : float
        W-value of the gas inside the detector.
    line_val : ArrayLike
        Position of the peak of the emission line in charge (fC).
    energy : float
        Energy of the emission line (in keV).
    """
    exp_electrons = energy / (w * 1e-3)
    meas_electrons = line_val * ELECTRONS_IN_1FC
    return meas_electrons / exp_electrons


def energy_resolution(line_val: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Estimate the energy resolution of the detector, using the position and the sigma of a
    spectral emission line.

    Arguments
    ----------
    line_val : np.ndarray
        Position of the peak of the emission line in charge (fC).
    sigma : np.ndarray
        Sigma of the emission line.
    """
    return sigma * 2 * np.sqrt(2 * np.log(2)) / line_val * 100


def amptek_accumulate_time(start_times: np.ndarray, real_times: np.ndarray) -> np.ndarray:
    """Compute the accumulated acquisition time for Amptek MCA data, taking into account the
    start times of each acquisition and integration times.

    Arguments
    ---------
    start_times : np.ndarray
        Array of datetime objects representing the start time of each acquisition.
    real_times : np.ndarray
        Array of real (integration) times for each acquisition.
    
    Returns 
    -------
    accumulated_times : np.ndarray
        Array of accumulated acquisition times.
    """
    t = np.zeros(len(start_times))
    t[0] = real_times[0]
    t_acc = 0.0
    for i in range(1, len(start_times)):
        if start_times[i] == start_times[i-1]:
            t_acc += real_times[i]
        else:
            dt_gap = (start_times[i] - start_times[0]).total_seconds()
            t_acc = dt_gap + real_times[i]
        t[i] = t_acc

    return t - real_times / 2
