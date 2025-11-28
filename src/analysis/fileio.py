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
import uncertainties
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
        # Keep files containing _B<number>
        filtered = [_f for _f in self.input_files if re.search(r"_B\d+", _f.name)]

        # Sort by the last number in the filename
        def numeric_sort_key(p: Path):
            numbers = [int(x) for x in re.findall(r"\d+", p.stem)]
            return numbers[-1]  # sort by the last number

        return sorted(filtered, key=numeric_sort_key)


    @property
    def pulse_files(self):
        return [_f for _f in self.input_files if re.search(r"ci([\d\-_]+)", _f.name, re.IGNORECASE) is not None]

class SourceFile(FileBase):

    @property
    def voltage(self) -> float:
        match = re.search(r"_B(\d+)", self.file_path.name)
        if match is not None:
            _voltage = float(match.group(1))
        else:
            raise ValueError("Incorrect file type or different name used.")

        return _voltage

    @property
    def real_time(self):
        with open(self.file_path, encoding="UTF-8") as input_file:
            real_time_str = input_file.readlines()[8]
        if real_time_str.split('-')[0].strip() == 'REAL_TIME':
            return float(real_time_str.split('-')[1].strip())
        else:
            raise ValueError("Not reading REAL_TIME")

    def fit_line_forest(self, num_sigma_left: float = 2., num_sigma_right: float = 2.):
        plt.figure(f"{self.voltage} Fe55Forest")
        plt.title(f"{self.voltage} V Fe55Forest")
        self.hist.plot()
        mu0 = self.hist.bin_centers()[self.hist.content.argmax()]
        xmin = mu0 - num_sigma_left * np.sqrt(mu0)
        xmax = mu0 + num_sigma_right * np.sqrt(mu0)
        model = Fe55Forest()
        for i in range(2):
            fitstatus = model.fit(self.hist, xmin=xmin, xmax=xmax, absolute_sigma=True)
            xmin = mu0 - num_sigma_left * model.sigma.value
            xmax = mu0 + num_sigma_right * model.sigma.value
        model.plot(fit_output=True)
        plt.xlim(model.default_plotting_range())
        plt.legend()
        # print(f"Voltage: {self.voltage}")
        # print(f"Pars:")
        # print(model.parameter_values())
        # print(f"PCOV:")
        # print(fitstatus.pcov)
        # print("------------------------")
        corr_pars = uncertainties.correlated_values(model.parameter_values(), fitstatus.pcov)
        # variance = np.diag(fitstatus.pcov)
        # corr_pars = unumpy.uarray(model.parameter_values(), np.sqrt(variance))
        return corr_pars

    def fit_line(self, **kwargs):
        plt.figure(f"{self.voltage} Gaussian")
        plt.title(f"{self.voltage} V Gaussian")
        self.hist.plot()
        mu0 = self.hist.bin_centers()[self.hist.content.argmax()]
        xmin = mu0 - kwargs.get('num_sigma_left', 2.) * np.sqrt(mu0)
        xmax = mu0 + kwargs.get('num_sigma_right', 2.) * np.sqrt(mu0)
        model = Gaussian()
        fitstatus = model.fit_iterative(self.hist, xmin=xmin, xmax=xmax, absolute_sigma=True, **kwargs)
        model.plot(fit_output=True)
        plt.xlim(model.default_plotting_range())
        plt.legend()

        corr_pars = uncertainties.correlated_values(model.parameter_values(), fitstatus.pcov)

        return corr_pars


class PulsatorFile(FileBase):

    @property
    def voltage(self) -> np.ndarray:
        match = re.search(r"ci([\d_-]+)(?=[^\d_-])", self.file_path.name, re.IGNORECASE)
        if match is not None:
            _voltage = np.array([int(n) for n in re.split(r"[-_]", match.group(1))])
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
                            num_sigma_right=num_sigma, absolute_sigma=True)
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
        fitstatus = line_model.fit(self.voltage, unumpy.nominal_values(mu), sigma=unumpy.std_devs(mu))
        line_model.plot(fit_output=True)
        plt.legend()

        corr_pars = uncertainties.correlated_values(line_model.parameter_values(), fitstatus.pcov)

        return corr_pars
