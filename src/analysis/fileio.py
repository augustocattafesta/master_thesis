"""FileIO
"""

import re
from pathlib import Path

import numpy as np
from aptapy.hist import Histogram1d
from aptapy.modeling import AbstractFitModel
from aptapy.models import Fe55Forest, Gaussian, Line
from aptapy.plotting import plt
from uncertainties import unumpy

from .utils import PeakAnalyzer


class FileBase:
    def __init__(self, file_path: Path):
        """Class constructor.
        """
        self.file_path = file_path
        self.hist = Histogram1d.from_amptek_file(file_path)


class DataFolder:
    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.input_files = list(folder_path.iterdir())

    @property
    def source_files(self):
        return [_f for _f in self.input_files if re.search(r"_B(\d+)", _f.name) is not None]

    @property
    def pulse_files(self):
        return [_f for _f in self.input_files if re.search(r"ci([\d\-]+)", _f.name) is not None]

class SourceFile(FileBase):

    @property
    def voltage(self) -> float:
        match = re.search(r"_B(\d+)", self.file_path.name)
        if match is not None:
            _voltage = float(match.group(1))
        else:
            raise ValueError("Incorrect file type or different name used.")

        return _voltage

    def fit_line_forest(self, num_sigma_left: float = 2., num_sigma_right: float = 2.):
        plt.figure(f"{self.voltage} Fe55Forest")
        self.hist.plot()
        mu0 = self.hist.bin_centers()[self.hist.content.argmax()]
        xmin = mu0 - num_sigma_left * np.sqrt(mu0)
        xmax = mu0 + num_sigma_right * np.sqrt(mu0)
        model = Fe55Forest()
        model.fit(self.hist, xmin=xmin, xmax=xmax)
        model.plot(fit_output=True)
        plt.xlim(model.default_plotting_range())
        plt.legend()

        return model

    def fit_line(self, **kwargs):
        plt.figure(f"{self.voltage} Gaussian")
        self.hist.plot()
        mu0 = self.hist.bin_centers()[self.hist.content.argmax()]
        xmin = mu0 - kwargs.get('num_sigma_left', 2.) * np.sqrt(mu0)
        xmax = mu0 + kwargs.get('num_sigma_right', 2.) * np.sqrt(mu0)
        model = Gaussian()
        model.fit_iterative(self.hist, xmin=xmin, xmax=xmax, **kwargs)
        model.plot(fit_output=True)
        plt.xlim(model.default_plotting_range())
        plt.legend()

        return model


class PulsatorFile(FileBase):

    @property
    def voltage(self) -> np.ndarray:
        match = re.search(r"ci([\d\-]+)", self.file_path.name).group(1)
        if match is not None:
            _voltage = np.array([int(n) for n in match.split("-")])
        else:
            raise ValueError("Incorrect file type or different name used.")

        return _voltage

    @property
    def num_pulses(self) -> int:
        return len(self.voltage)

    def fit_pulse(self, xpeak: float, num_sigma: float = 2.) -> "AbstractFitModel":
        xmin = xpeak - np.sqrt(xpeak)
        xmax = xpeak + np.sqrt(xpeak)
        model = Gaussian()
        model.fit_iterative(self.hist, xmin=xmin, xmax=xmax, num_sigma_left=num_sigma,
                            num_sigma_right=num_sigma)
        model.plot(fit_output=True)

        return model

    def analyze_pulse(self, **kwargs):
        plt.figure(self.file_path.name)
        self.hist.plot()
        xpeaks, _ = PeakAnalyzer.find_peaks_iterative(self.hist.bin_centers(),
                                                             self.hist.content, self.num_pulses)
        models = [self.fit_pulse(xpeak, **kwargs) for xpeak in xpeaks]
        plt.legend()
        mu = np.array([model.mu.ufloat() for model in models])

        plt.figure()
        plt.errorbar(self.voltage, unumpy.nominal_values(mu), unumpy.std_devs(mu), fmt='o')
        line_model = Line('Conversion factor', "Voltage [mV]", "ADC Channel")
        line_model.fit(self.voltage, unumpy.nominal_values(mu), sigma=unumpy.std_devs(mu))
        line_model.plot(fit_output=True)
        plt.legend()

        return line_model
