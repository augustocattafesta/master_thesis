"""Module to handle reading of different types of file and analyze specific type of signals.
"""

import datetime
import inspect
import pathlib
import re

import aptapy.modeling
import numpy as np
import yaml
from aptapy.hist import Histogram1d
from uncertainties import unumpy

from . import ANALYSIS_RESOURCES


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


class Folder:
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

        return sorted(filtered, key=numeric_sort_key)

    @property
    def pulse_file(self) -> list[pathlib.Path]:
        """Extract the calibration pulse files from all the files of the directory.
        """
        return [_f for _f in self.input_files if re.search(r"ci([\d\-_]+)",
                                                           _f.name,re.IGNORECASE) is not None][0]


class SourceFile(FileBase):
    """Class to analyze a source file and extract relevant quantities from the name of the file.
    """
    def __init__(self, file_path: pathlib.Path,
                 charge_conv_model: aptapy.modeling.AbstractFitModel) -> None:
        super().__init__(file_path)
        content = self.hist.content
        old_edges = self.hist.bin_edges()
        slope, offset = charge_conv_model.parameter_values()
        # Converting the binning from ADC channels to charge (fC) using the calibration pulse file
        # This feature could become optional in the future
        new_edges = unumpy.nominal_values(old_edges * slope + offset)
        self.hist = Histogram1d(new_edges, xlabel="Charge [fC]")
        self.hist.set_content(content)

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
        """Real integration time of the histogram extracted from the amptek file..
        """
        with open(self.file_path, encoding="UTF-8") as input_file:
            real_time_str = input_file.readlines()[8]
        if real_time_str.split('-')[0].strip() == "REAL_TIME":
            return float(real_time_str.split('-')[1].strip())
        raise ValueError("Not reading REAL_TIME")

    @property
    def start_time(self) -> datetime.datetime:
        """Start time of the acquisition extracted from the amptek file.
        """
        with open(self.file_path, encoding="UTF-8") as input_file:
            start_time_str = input_file.readlines()[9]
        if start_time_str.split('-')[0].strip() == "START_TIME":
            start = datetime.datetime.strptime(start_time_str, "START_TIME - %m/%d/%Y %H:%M:%S\n")
            return start
        raise ValueError("Not reading START_TIME")


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
