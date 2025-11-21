import pathlib
import re

import numpy as np
import scipy.signal
import xraydb
from aptapy.hist import Histogram1d
from aptapy.modeling import line_forest
from aptapy.models import Gaussian, GaussianForest, Line
from aptapy.plotting import plt


from analysis import ANALYSIS_DATA

NUM_SIGMA = 2

ka = xraydb.xray_line('Mn', 'Ka1').energy
kb = xraydb.xray_line('Mn', 'Kb1').energy

@line_forest(ka, kb)
class Fe55Forest(GaussianForest):
    pass

def source_file_parser(input_file: pathlib.Path):
    assert "D1000" in input_file.name

    i = input_file.name.index("B")
    voltage = int(input_file.name[i+1:i+4])

    return voltage

def pulse_file_parser(input_file: pathlib.Path):
    if "ci" in input_file.name:
        i = input_file.name.index("ci")
        name = input_file.name[i+2:]
        
        return np.array(list(map(int, re.findall(r'\d+', name))))
    else:
        raise ValueError    # change it
    
    
def _find_peaks_iterative(xdata, ydata, npeaks: int):
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


def _find_e_res(input_file: pathlib.Path):
    plt.figure(input_file.name)
    hist = Histogram1d.from_amptek_file(input_file)
    hist.plot()
    mu = hist.bin_centers()[hist.content.argmax()]
    xmin = mu - 2 * np.sqrt(mu)
    xmax = mu + 2 * np.sqrt(mu)

    model = Gaussian(xlabel="ADC Counts")
    model.fit_iterative(hist, num_iterations=3, xmin=xmin, xmax=xmax, num_sigma_left=NUM_SIGMA, num_sigma_right=NUM_SIGMA)
    model.plot(fit_output=True)

    fwhm = model.sigma.ufloat() * 2*np.sqrt(2*np.log(2))
    e_res = fwhm / model.mu.ufloat()

    plt.xlim(model.default_plotting_range())
    plt.legend()

    return e_res

def _find_e_res_lineforest(input_file: pathlib.Path):
    plt.figure(input_file.name)
    hist = Histogram1d.from_amptek_file(input_file)
    hist.plot()
    mu = hist.bin_centers()[hist.content.argmax()]
    xmin = mu - 1.5 * np.sqrt(mu)
    xmax = mu + 3 * np.sqrt(mu)

    model = Fe55Forest()
    model.fit(hist, xmin=xmin, xmax=xmax)
    model.plot(fit_output=True)

    fwhm = model.sigma.ufloat() * 2*np.sqrt(2*np.log(2))
    e_res = fwhm / (model.energies[0] / model.energy_scale.ufloat())
    plt.xlim(model.default_plotting_range())
    plt.legend()

    return e_res, model.energy_scale.ufloat()

def _analyze_pulses(input_file):
    plt.figure(input_file.name)
    hist = Histogram1d.from_amptek_file(input_file)
    hist.plot()

    x_peaks, y_peaks = _find_peaks_iterative(hist.bin_centers(), hist._sumw, 3)
    mu = []
    for i, x_peak in enumerate(x_peaks):
        xmin = x_peak - 1 * np.sqrt(x_peak)
        xmax = x_peak + 1 * np.sqrt(x_peak)
        model = Gaussian()
        model.set_parameters(y_peaks[i], x_peak, np.sqrt(x_peak))
        model.fit_iterative(hist, xmin=xmin, xmax=xmax, num_sigma_left=NUM_SIGMA, num_sigma_right=NUM_SIGMA)
        model.plot(fit_output=True)
        
        mu.append(model.mu.ufloat())
    
    plt.legend()

    return mu


def analyze_source_files(folder_path: pathlib.Path):
    input_files = [f for f in folder_path.iterdir()]

    voltage = []
    ka_mu = []
    e_res = []
    sigma_e_res = []
    for input_file in input_files:
        try:
            voltage.append(source_file_parser(input_file))
            _e_res, e_scale = _find_e_res_lineforest(input_file)
            e_res.append(_e_res.n)
            sigma_e_res.append(_e_res.s)

            ka_mu.append(ka / e_scale)

        except AssertionError:
            pass

    plt.figure()
    plt.errorbar(voltage, e_res, yerr=sigma_e_res, fmt=".k")
    plt.xlabel("Voltage [V]")
    plt.ylabel(r"$FWHM / E$")

    return voltage, np.array(ka_mu)


def analyze_pulse_files(folder_path: pathlib.Path, c=1e-12):
    e = 1.602e-19

    input_files = [f for f in folder_path.iterdir()]

    conversion_factors = []
    intercept = []
    for input_file in input_files:
        try:
            voltages = pulse_file_parser(input_file)*1e-3 * c / e
            mu_list =_analyze_pulses(input_file)
            y = [_mu.n for _mu in mu_list]
            sy = [_mu.s for _mu in mu_list]
            plt.figure(str(voltages))
            plt.errorbar(voltages, y, yerr=sy, fmt='o')
            plt.xlabel("Voltage [mV]")
            plt.ylabel("ADC Channel")

            model = Line()
            model.fit(voltages, y, sigma=sy)
            model.plot(fit_output=True)
            conversion_factors.append(model.slope.ufloat()) # in ADC/e-
            intercept.append(model.intercept.ufloat()) # ADC

            plt.legend()


        except ValueError:
            pass

    return np.array(conversion_factors), np.array(intercept)

def gain_study(folder_path: pathlib.Path, c=1e-12):
    conversion_factors, intercept = analyze_pulse_files(folder_path)

    voltages, ka_mu = analyze_source_files(folder_path)

    for conv, interc in zip(conversion_factors, intercept):
        plt.figure(str(conv))
        measured_electrons = (ka_mu - interc) / conv
        expected_electron = ka / xraydb.ionization_potential('Ar')

        gain = measured_electrons / expected_electron
        gain_n = [_g.n for _g in gain]
        gain_s = [_g.s for _g in gain]
        plt.errorbar(voltages, gain_n, gain_s, fmt='ko')
        plt.xlabel("Voltage [V]")
        plt.ylabel("Gain")




data_path = ANALYSIS_DATA / "251118"

# analyze_source_files(data_path)
# analyze_pulse_files(data_path)
# _find_e_res_lineforest(data_path / "live_data_chip18112025_D1000_B360.mca")
# pulse_file_parser(data_path / "live_data_chip18112025_ci5-10-20.mca")
gain_study(data_path)
plt.show()
