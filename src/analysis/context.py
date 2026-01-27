from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aptapy import modeling, models
from uncertainties import UFloat

from .config import AppConfig
from .fileio import PulsatorFile, SourceFile
from .utils import SIGMA_TO_FWHM


@dataclass
class TargetContext:
    """Container class for target-specific analysis context information. This class holds the
    results of the spectral analysis for a specific target (line) in a source file. 

    Attributes
    ----------
    target : str
        The name of the target line being analyzed.
    line_val : UFloat
        The fitted line position value.
    sigma : UFloat
        The fitted line sigma value.
    voltage : float
        The voltage at which the measurement was taken.
    model : modeling.AbstractFitModel
        The fit model used for the analysis.
    """
    target: str
    line_val: UFloat
    sigma: UFloat
    voltage: float
    model: modeling.AbstractFitModel

    # Internal attributes for storing computed values and labels
    _gain_val: UFloat | None = field(default=None, init=False, repr=False)
    _gain_label: str = field(default="", init=False, repr=False)

    _res_val: UFloat | None = field(default=None, init=False, repr=False)
    _res_label: str = field(default="", init=False, repr=False)

    _res_escape_val: UFloat | None = field(default=None, init=False, repr=False)
    _res_escape_label: str | None = field(default=None, init=False, repr=False)

    # Default energy for resolution labels
    _energy: float = field(default=5.9, init=False, repr=False)

    @property
    def gain_val(self) -> UFloat:
        """The gain value computed in the gain analysis task.
        """
        if self._gain_val is None:
            raise AttributeError("Gain value has not been set yet.")
        return self._gain_val

    @gain_val.setter
    def gain_val(self, value: UFloat):
        self._gain_val = value
        # Set the gain label for the spectral plot
        self._gain_label = f"Gain@{self.voltage:.0f} V: {self.gain_val}"

    @property
    def res_val(self) -> UFloat:
        """The resolution value computed in the resolution analysis task.
        """
        if self._res_val is None:
            raise AttributeError("Resolution value has not been set yet.")
        return self._res_val

    @res_val.setter
    def res_val(self, value: UFloat):
        self._res_val = value
        fwhm = SIGMA_TO_FWHM * self.sigma
        # Set the resolution label for the spectral plot
        self._res_label = f"FWHM@{self._energy:.1f} keV: {fwhm} fC\n"
        self._res_label += fr"$\Delta$E/E: {self.res_val} %"

    @property
    def res_escape_val(self) -> UFloat:
        """The resolution (escape) value computed in the resolution analysis task.
        """
        if self._res_escape_val is None:
            raise AttributeError("Resolution (escape) value has not been set yet.")
        return self._res_escape_val

    @res_escape_val.setter
    def res_escape_val(self, value: UFloat):
        self._res_escape_val = value
        # Set the resolution (escape) label for the spectral plot
        self._res_escape_label = fr"$\Delta$E/E(esc.): {self.res_escape_val} %"

    @property
    def energy(self) -> float:
        """The energy value used for resolution labels.
        """
        return self._energy

    @energy.setter
    def energy(self, value: float):
        self._energy = value

    def task_label(self, task: str) -> str | None:
        """Retrieve the label string for the specified analysis task.

        Arguments
        ----------
        task : str
            The analysis task for which to retrieve the label.
        
        Returns
        -------
        label : str
            The label string corresponding to the specified task.
        """
        label_mapping = {
            "gain": self._gain_label,
            "resolution": self._res_label,
            "resolution_escape": self._res_escape_label
        }
        try:
            return label_mapping[task]
        except KeyError:
            raise ValueError(f"Unknown task '{task}' for label retrieval.") from None


@dataclass
class Context:
    """Container class for analysis context information. This class holds the analysis
    configuration and all the data generated during the analysis pipeline. This class gets
    continually updated during the analysis.
    
    Attributes
    ----------
    config : AppConfig
        The application configuration object.
    """
    config: AppConfig

    # Internal attributes for storing calibration, source files, fit results, and figures
    _calibration: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _sources: dict[str, SourceFile] = field(default_factory=dict, init=False, repr=False)
    _fit: dict = field(default_factory=dict, init=False, repr=False)
    _results: dict = field(default_factory=dict, init=False, repr=False)
    _figures: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def pulse(self) -> PulsatorFile:
        """The PulsatorFile object obtained from the calibration pulse file."""
        _pulse = self._calibration.get("pulse")
        if _pulse is None:
            raise AttributeError("Pulse file has not been set in the context.")
        return _pulse

    @pulse.setter
    def pulse(self, value):
        if not isinstance(value, PulsatorFile):
            raise TypeError("Pulse must be an instance of PulsatorFile")
        self._calibration["pulse"] = value

    @property
    def conversion_model(self) -> models.Line:
        """The calibration conversion model to use for charge calibration of the spectral data."""
        model = self._calibration.get("model")
        if model is None:
            raise AttributeError("Calibration model has not been set in the context.")
        if not isinstance(model, models.Line):
            raise TypeError("Conversion model in context is not an instance of aptapy.models.Line")
        return model

    @conversion_model.setter
    def conversion_model(self, value):
        if not isinstance(value, models.Line):
            raise TypeError("Conversion model must be an instance of aptapy.models.Line")
        self._calibration["model"] = value

    def add_source(self, source: SourceFile) -> None:
        """Add a source file to the private `sources` dictionary."""
        if not isinstance(source, SourceFile):
            raise TypeError("Source must be an instance of SourceFile")
        file_name = source.file_path.stem
        self._sources[file_name] = source

    def source(self, file_name: str) -> SourceFile:
        """Retrieve a source file from the private `sources` dictionary by its file name."""
        if file_name not in self._sources:
            raise KeyError(f"Source file '{file_name}' not found in context.")
        return self._sources[file_name]

    @property
    def last_source(self) -> SourceFile:
        """The last source file added to the private `sources` dictionary."""
        if not self._sources:
            raise AttributeError("No source files have been set in the context.")
        return list(self._sources.values())[-1]

    @property
    def file_names(self) -> list[str]:
        """List of source file names stored in the context."""
        if not self._sources:
            raise AttributeError("No source files have been set in the context.")
        return list(self._sources.keys())

    def add_target_ctx(self, source: SourceFile, target_ctx: TargetContext) -> None:
        """Add a target context for a specific source file to the private `fit` dictionary."""
        if not isinstance(target_ctx, TargetContext):
            raise TypeError("Target context must be an instance of TargetContext")
        if not isinstance(source, SourceFile):
            raise TypeError("Source must be an instance of SourceFile")
        file_name = source.file_path.stem
        if file_name not in self._fit:
            self._fit[file_name] = {}
        self._fit[file_name][target_ctx.target] = target_ctx

    def target_ctx(self, file_name: str, target: str) -> TargetContext:
        """Retrieve a target context for a specific source file and target from the private `fit`
        dictionary."""
        if file_name not in self._fit:
            raise KeyError(f"File '{file_name}' not found in fit results")
        if target not in self._fit[file_name]:
            raise KeyError(f"Target subtask '{target}' not found in fit results")
        return self._fit[file_name][target]

    def add_task_results(self, task: str, target: str, results: dict) -> None:
        """Add results for a specific task and target to the private `results` dictionary."""
        if task not in self._results:
            self._results[task] = {}
        if results is None:
            raise ValueError("Results dictionary cannot be None")
        self._results[task][target] = results

    def add_task_fit_model(self, task: str, target: str, model: modeling.AbstractFitModel) -> None:
        """Add the fit model used in a specific task and target to the private `results`
        dictionary."""
        if task not in self._results:
            self._results[task] = {}
        if target not in self._results[task]:
            self._results[task][target] = {}
        self._results[task][target]["model"] = model

    def add_subtask_fit_model(self, task: str, target: str, subtask: str,
                              model: modeling.AbstractFitModel) -> None:
        """Add the fit model used in a specific subtask of a task and target to the private
        `results` dictionary."""
        if task not in self._results:
            self._results[task] = {}
        if target not in self._results[task]:
            self._results[task][target] = {}
        if subtask not in self._results[task][target]:
            self._results[task][target][subtask] = {}
        self._results[task][target][subtask]["model"] = model

    def task_results(self, task: str, target: str) -> dict:
        """Retrieve results for a specific task and target from the private `results`
        dictionary."""
        if task not in self._results:
            raise KeyError(f"Task '{task}' not found in results.")
        if target not in self._results[task]:
            raise KeyError(f"Target '{target}' not found in results for task '{task}'.")
        return self._results[task][target]

@dataclass
class FoldersContext:
    """Container class for folders-specific analysis context information. This class holds the
    analysis configuration and the results from multiple folders tasks.
    
    Attributes
    ----------
    config : AppConfig
        The application configuration object.
    """
    config: AppConfig

    # Internal attributes for storing folder contexts and results
    _folders: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _results: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def folder_names(self) -> list[str]:
        """List of folder names stored in the context."""
        if not self._folders:
            raise AttributeError("No folders have been set in the context.")
        return list(self._folders.keys())

    def add_folder(self, folder_path: Path, folder_ctx: Context) -> None:
        """Add a folder context to the private `folders` dictionary."""
        if not isinstance(folder_ctx, Context):
            raise TypeError("Folder context must be an instance of Context")
        folder_name = folder_path.stem
        self._folders[folder_name] = folder_ctx

    def folder_ctx(self, folder_name: str) -> Context:
        """Retrieve a folder context from the private `folders` dictionary by its folder name."""
        if folder_name not in self._folders:
            raise KeyError(f"Folder '{folder_name}' not found in context.")
        return self._folders[folder_name]

    def add_task_results(self, task: str, target: str, results: dict) -> None:
        """Add results for a specific task and target to the private `results` dictionary."""
        if task not in self._results:
            self._results[task] = {}
        if results is None:
            raise ValueError("Results dictionary cannot be None")
        self._results[task][target] = results
