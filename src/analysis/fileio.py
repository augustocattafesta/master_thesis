"""Module to handle reading of different types of file and analyze specific type of signals.
"""

import inspect
import pathlib
import re

import aptapy.modeling
import numpy as np
import yaml
from aptapy.hist import Histogram1d
from aptapy.modeling import FitParameter
from aptapy.models import Fe55Forest, Gaussian, GaussianForestBase, Line
from aptapy.plotting import plt
from uncertainties import unumpy

from . import ANALYSIS_RESOURCES
from .utils import find_peaks_iterative


class FileBase:
    """Load data from a file and define the path and the histogram.
    """
    def __init__(self, file_path: pathlib.Path) -> None:
        """Class constructor.

        Arguments
        ----------
        file_path : pathlib.Path
            Path of the file to open.
        """
        self.file_path = file_path
        self.hist = Histogram1d.from_amptek_file(file_path)


class DataFolder:
    """Load source files and calibration pulse files from a folder.
    """
    def __init__(self, folder_path: pathlib.Path) -> None:
        """Class constructor.

        Parameters
        ----------
        folder_path : pathlib.Path
            Path of the folder to open.
        """
        self.folder_path = folder_path
        if not self.folder_path.exists():
            raise FileExistsError(f"Folder {str(self.folder_path)} does not exist.\
                                  Verify the path.")
        self.input_files = list(folder_path.iterdir())
        self.pulse_data = [PulsatorFile(pulse_file_path) for pulse_file_path in self.pulse_files]
        self.source_data = [SourceFile(source_file_path) for source_file_path in self.source_files]

    @property
    def source_files(self) -> list[pathlib.Path]:
        """Extract the source files from all the files of the directory and return a sorted list
        of the files. If file names contain "trend{i}", the sorting is done numerically according
        to the index {i}, otherwise it's done alphabetically.
        """
        # Keep files containing _B<number>
        filtered = [_f for _f in self.input_files if re.search(r"B\d+", _f.name)]

        def numeric_sort_key(p: pathlib.Path):
            """Sort the files with numerical order.
            """
            numbers = [int(x) for x in re.findall(r"\d+", p.stem)]
            return numbers[-1]  # sort by the last number

        def custom_sort_key(p: pathlib.Path):
            """Define if the sorting is numerical or alphabetical.
            """
            if "trend" in p.stem:
                return numeric_sort_key(p)
            return p.name

        return sorted(filtered, key=custom_sort_key)

    @property
    def pulse_files(self) -> list[pathlib.Path]:
        """Extract the calibration pulse files from all the files of the directory.
        """
        return [_f for _f in self.input_files if re.search(r"ci([\d\-_]+)",
                                                           _f.name,re.IGNORECASE) is not None]

class SourceFile(FileBase):
    """Class to analyze a source file and extract relevant quantities from the name of the file.
    """
    @property
    def voltage(self) -> float:
        """Back voltage of the detector extracted from the file name.
        """
        match = re.search(r"B(\d+)", self.file_path.name)
        if match is not None:
            _voltage = float(match.group(1))
        else:
            raise ValueError("Incorrect file type or different name used.")
        return _voltage

    @property
    def drift_voltage(self) -> float:
        """Drift voltage of the detector extracted from the file name.
        """
        match = re.search(r"D(\d+)", self.file_path.name)
        if match is not None:
            _voltage = float(match.group(1))
        else:
            raise ValueError("Incorrect file type or different name used.")
        return _voltage

    @property
    def real_time(self) -> float:
        """Real integration time of the histogram.
        """
        with open(self.file_path, encoding="UTF-8") as input_file:
            real_time_str = input_file.readlines()[8]
        if real_time_str.split('-')[0].strip() == 'REAL_TIME':
            return float(real_time_str.split('-')[1].strip())
        raise ValueError("Not reading REAL_TIME")

    def fit(self, model_class: type[aptapy.modeling.AbstractFitModel],
            **kwargs) -> aptapy.modeling.AbstractFitModel:
        """Fit the spectrum data.

        Parameters
        ----------
        model_class : aptapy.modeling.AbstractFitModel
            Model class to fit the emission line(s). 

        Returns
        -------
        model: aptapy.modeling.AbstractFitModel
            Returns the model instance containing results of the fit.
        """
        if issubclass(model_class, (Gaussian, GaussianForestBase)):
            model = model_class()
            if isinstance(model, Fe55Forest):
                intensity_par: FitParameter = model.intensity1  # type: ignore [attr-defined]
                intensity_par.freeze(0.16)  # Freeze Mn K-beta / K-alpha ratio
            model.fit_iterative(self.hist, **kwargs)
        else:
            raise TypeError("Choose between Gaussian or GaussianForestBase child class")
        return model


class PulsatorFile(FileBase):
    """Class to analyze a calibration pulse file.
    """
    @property
    def voltage(self) -> np.ndarray:
        """Voltages of the pulses.
        """
        match = re.search(r"ci([\d_-]+)(?=[^\d_-])", self.file_path.name, re.IGNORECASE)
        if match is not None:
            parts = [n for n in re.split(r"[-_]", match.group(1)) if n]  # <-- filter empty strings
            _voltage = np.array([int(n) for n in parts])
        else:
            raise ValueError("Incorrect file type or different name used.")

        return _voltage

    @property
    def num_pulses(self) -> int:
        """Number of pulses in the spectrum.
        """
        return len(self.voltage)

    def analyze_pulses(self) -> tuple[aptapy.modeling.AbstractFitModel, plt.Figure, plt.Figure]:
        """Find pulses in the spectrum and independently fit each of them with a Gaussian model.
        Using the resulting position of the peaks, do a calibration fit with a Line model to
        determine the calibration parameters of the spectrum.

        Returns
        -------
        line_model, pulse_fig, line_fig : tuple[aptapy.modeling.AbstractFitModel, Figure, Figure]
            Returns fit model and figures of the pulses and of the
            calibration fit.
        """
        # log = LOGGER.log_main() or LOGGER.NULL_LOGGER

        pulse_fig = plt.figure(self.file_path.name)
        plt.title("Calibration pulses")
        self.hist.plot()
        xpeaks, _ = find_peaks_iterative(self.hist.bin_centers(),
                                                             self.hist.content, self.num_pulses)
        mu = np.zeros(shape=len(xpeaks), dtype=object)
        for i, xpeak in enumerate(xpeaks):
            peak_model = Gaussian()
            xmin = xpeak - np.sqrt(xpeak)
            xmax = xpeak + np.sqrt(xpeak)
            peak_model.fit_iterative(self.hist, xmin=xmin, xmax=xmax, absolute_sigma=True)
            mu[i] = peak_model.mu.ufloat()
            peak_model.plot(fit_output=True)
        plt.legend()

        line_model = Line("Calibration fit", "Voltage [mV]", "ADC Channel")
        line_model.fit(self.voltage, unumpy.nominal_values(mu), sigma=unumpy.std_devs(mu),
                       absolute_sigma=True)
        line_fig = plt.figure("Calibration fit")
        plt.errorbar(self.voltage, unumpy.nominal_values(mu), unumpy.std_devs(mu), fmt='o')
        line_model.plot(fit_output=True)
        plt.legend()

        return line_model, pulse_fig, line_fig


def load_label(key: str) -> str | None:
    """Load a label from the analysis resources labels.yaml file based on the calling function
    name and on the file name.

    Arguments
    ---------
    key : str
        Key of the label to load.
    """
    label_value = None
    yaml_file_path = ANALYSIS_RESOURCES / "labels.yaml"
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            yaml_file = yaml.safe_load(f)
        functions = yaml_file["function"]
        current_frame = inspect.currentframe()
        if current_frame is not None:
            previous_frame = current_frame.f_back
            if previous_frame is not None:
                label = functions.get(previous_frame.f_code.co_name, None)
                label_value = label[key]
    except (FileNotFoundError, TypeError, KeyError):
        pass
    return label_value
