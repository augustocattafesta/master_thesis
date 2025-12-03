"""FileIO
"""

import importlib
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import uncertainties
from aptapy.hist import Histogram1d
from aptapy.modeling import AbstractFitModel
from aptapy.models import Fe55Forest, Gaussian, GaussianForestBase, Line
from aptapy.plotting import plt
from aptapy.typing_ import ArrayLike
from uncertainties import unumpy

from .log import logger
from .utils import find_peaks_iterative


def load_class(class_path: str):
    """
    Load a class from a string.
    Supports:
      - "ClassName" (local or global)
      - "module.ClassName"
      - "package.module.ClassName"
    """

    # Case 1: "ClassName" – search locals, globals, and all loaded modules
    if "." not in class_path:
        frame = inspect.currentframe().f_back
        # Search locals first
        if class_path in frame.f_locals:
            return frame.f_locals[class_path]
        # Then globals
        if class_path in frame.f_globals:
            return frame.f_globals[class_path]
        # Search through all loaded modules
        for module in sys.modules.values():
            if module and hasattr(module, class_path):
                return getattr(module, class_path)
        raise ImportError(f"Class '{class_path}' not found in locals, globals, or loaded modules.")
    # Case 2: dotted path → module + class
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


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
        filtered = [_f for _f in self.input_files if re.search(r"B\d+", _f.name)]

        # Sort by the last number in the filename
        def numeric_sort_key(p: Path):
            numbers = [int(x) for x in re.findall(r"\d+", p.stem)]
            return numbers[-1]  # sort by the last number

        def custom_sort_key(p: Path):
            if "trend" in p.stem:
                return numeric_sort_key(p)
            return p.name

        return sorted(filtered, key=custom_sort_key)


    @property
    def pulse_files(self):
        return [_f for _f in self.input_files if re.search(r"ci([\d\-_]+)",
                                                           _f.name,re.IGNORECASE) is not None]

class SourceFile(FileBase):

    @property
    def voltage(self) -> float:
        match = re.search(r"B(\d+)", self.file_path.name)
        if match is not None:
            _voltage = float(match.group(1))
        else:
            raise ValueError("Incorrect file type or different name used.")

        return _voltage

    @property
    def drift_voltage(self) -> float:
        match = re.search(r"D(\d+)", self.file_path.name)
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
        raise ValueError("Not reading REAL_TIME")

    def fit_line_forest(self, **kwargs):
        if kwargs.get('xmin', float("-inf")) == float("-inf"):
            mu0 = self.hist.bin_centers()[self.hist.content.argmax()]
            xmin = mu0 - kwargs.get('num_sigma_left', 2.) * np.sqrt(mu0)
            xmax = mu0 + kwargs.get('num_sigma_right', 2.) * np.sqrt(mu0)
        else:
            xmin = kwargs["xmin"]
            kwargs.pop("xmin")
            xmax = kwargs["xmax"]
            kwargs.pop("xmax")
        model = Fe55Forest()
        for _i in range(2):
            fitstatus = model.fit(self.hist, xmin=xmin, xmax=xmax, absolute_sigma=True)
            xmin = mu0 - kwargs.get("num_sigma_left", 2.) * model.sigma.value
            xmax = mu0 + kwargs.get("num_sigma_right", 2.) * model.sigma.value
        plt.figure(f"{self.voltage} Fe55Forest")
        plt.title(f"{self.voltage} V Fe55Forest")
        self.hist.plot()
        # model.plot(fit_output=True)
        label = f"Fe55Forest\nFWHM@5.9keV: {model.fwhm()}"
        model.plot(label=label)
        plt.xlim(model.default_plotting_range())
        plt.legend()
        corr_pars = uncertainties.correlated_values(model.parameter_values(), fitstatus.pcov)
        return corr_pars

    def fit_line(self, **kwargs):
        if kwargs.get('xmin') is None:
            mu0 = self.hist.bin_centers()[self.hist.content.argmax()]
            xmin = mu0 - kwargs.get('num_sigma_left', 2.) * np.sqrt(mu0)
            xmax = mu0 + kwargs.get('num_sigma_right', 2.) * np.sqrt(mu0)
        else:
            xmin = kwargs["xmin"]
            kwargs.pop("xmin")
            xmax = kwargs["xmax"]
            kwargs.pop("xmax")
        model = Gaussian()
        fitstatus = model.fit_iterative(self.hist, xmin=xmin, xmax=xmax, absolute_sigma=True,
                                        **kwargs)

        plt.figure(f"{self.voltage} Gaussian")
        plt.title(f"{self.voltage} V Gaussian")
        self.hist.plot()
        model.plot(fit_output=True)
        plt.xlim(model.default_plotting_range())
        plt.legend()

        corr_pars = uncertainties.correlated_values(model.parameter_values(), fitstatus.pcov)
        return corr_pars

    def fit(self, model, **kwargs):
        if issubclass(model, Gaussian) or issubclass(model, GaussianForestBase):
            model_instance = model()
            fitstatus = model_instance.fit_iterative(self.hist, **kwargs)
        else:
            raise TypeError("Choose between Gaussian or GaussianForestBase child class")

        logger.info(self.file_path.name)
        corr_pars = uncertainties.correlated_values(model_instance.parameter_values(),
                                                    fitstatus.pcov)
        return corr_pars, model_instance


class PulsatorFile(FileBase):

    @property
    def voltage(self) -> np.ndarray:
        match = re.search(r"ci([\d_-]+)(?=[^\d_-])", self.file_path.name, re.IGNORECASE)
        if match is not None:
            parts = [n for n in re.split(r"[-_]", match.group(1)) if n]  # <-- filter empty strings
            _voltage = np.array([int(n) for n in parts])
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

    def analyze_pulse(self, **kwargs) -> ArrayLike:
        logger.info("PULSE FILE ANALYZED:")
        logger.info(f"{self.file_path.name}\n")

        fig = plt.figure(self.file_path.name)
        plt.title("Calibration pulses")
        self.hist.plot()
        xpeaks, _ = find_peaks_iterative(self.hist.bin_centers(),
                                                             self.hist.content, self.num_pulses)
        models = [self.fit_pulse(xpeak, **kwargs) for xpeak in xpeaks]
        mu = np.array([model.mu.ufloat() for model in models])
        plt.legend()

        line_fig = plt.figure("Calibration fit")
        plt.errorbar(self.voltage, unumpy.nominal_values(mu), unumpy.std_devs(mu), fmt='o')
        line_model = Line("Calibration fit", "Voltage [mV]", "ADC Channel")
        fitstatus = line_model.fit(self.voltage, unumpy.nominal_values(mu),
                                   sigma=unumpy.std_devs(mu))
        line_model.plot(fit_output=True)
        plt.legend()

        corr_pars = uncertainties.correlated_values(line_model.parameter_values(), fitstatus.pcov)

        return corr_pars, fig, line_fig
