"""Module to handle logging interface.
"""

import datetime
import inspect
import os
import pathlib
import re
import sys
from numbers import Real
from typing import Any

import aptapy.modeling
import numpy as np
import yaml
from aptapy.typing_ import ArrayLike
from uncertainties import ufloat

from . import ANALYSIS_RESULTS


def get_subcommand(cmd: str) -> str | None:
    """Extract the subcommand from the entire terminal command line string.

    Arguments
    ---------
    cmd :   str
        Command line to analyze.
    """
    match = re.search(r"\banalysis\b\s+(\S+)", cmd)
    return match.group(1) if match else None


def get_command(cmd: str) -> str:
    """Extract the entire command line from the entire terminal output, removing the execution
    path.

    Arguments
    ---------
    cmd :   str
        Command line to analyze.
    """
    m = re.search(r'\banalysis\b.*', cmd)
    return m.group(0) if m else cmd


class LogYaml:
    """Handle logging of the analysis and save the results as a YAML file.
    """
    _LOG_FOLDER: pathlib.Path = ANALYSIS_RESULTS
    _YAML_PATH: pathlib.Path = ANALYSIS_RESULTS / "log.yaml"
    _YAML_DICT: dict = {}

    @property
    def log_folder(self) -> pathlib.Path:
        """Get the log folder path.
        """
        return self._LOG_FOLDER

    @property
    def yaml_dict(self) -> dict:
        """Get the YAML dictionary containing the logged information.
        """
        return self._YAML_DICT

    @staticmethod
    def _clean_numpy_types(data: Any) -> Any:
        """Recursively convert numpy types to native python types for YAML serialization.
        """
        if isinstance(data, dict):
            return {k: LogYaml._clean_numpy_types(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [LogYaml._clean_numpy_types(item) for item in data]
        if isinstance(data, Real) and not isinstance(data, (int, float)):
            return float(data) if isinstance(data, (np.floating, float)) else int(data)
        return data

    @classmethod
    def start_logging(cls) -> None:
        """Start logging by creating the log folder and preparing the YAML dictionary."""
        if cls._LOG_FOLDER is not None:
            return None
        # Create log folder
        date = datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
        cmd = " ".join(sys.argv)
        log_folder_name = f"{date}_{get_subcommand(cmd)}"
        log_file = f"{date}_{get_subcommand(cmd)}.yaml"
        pathlib.Path.mkdir(ANALYSIS_RESULTS / log_folder_name)
        log_folder = ANALYSIS_RESULTS / log_folder_name
        cls._LOG_FOLDER = log_folder
        # Get caller's frame to extract arguments
        current_frame = inspect.currentframe()
        if current_frame is not None:
            frame = current_frame.f_back
            if frame is not None:
                info = inspect.getargvalues(frame)
                args_dict = {name: info.locals[name] for name in info.args}
                kwargs_dict = info.locals[info.keywords] if info.keywords else {}
                cls._YAML_PATH = cls._LOG_FOLDER / log_file
                # Prepare YAML dictionary
                cls._YAML_DICT["execution_datetime"] = date
                cls._YAML_DICT["command_line"] = get_command(cmd)
                cls._YAML_DICT["positional_arguments"] = cls._clean_numpy_types(args_dict)
                cls._YAML_DICT["keyword_arguments"] = cls._clean_numpy_types(kwargs_dict)

        return None

    @staticmethod
    def _fit_dict(fit_model: aptapy.modeling.AbstractFitModel) -> dict:
        """Create a dictionary from the fit model results.

        Arguments
        ---------
        fit_model : aptapy.modeling.AbstractFitModel
            Fit model containing the results to log.
        """
        pars_dictionary = {}
        for par in fit_model:
            pars_dictionary[par.name] = {"val": par.value, "err": par.error}

        fit_dict = {"model": fit_model.name(),
                "chisquare": fit_model.status.chisquare,
                "dof": fit_model.status.dof,
                "fit_parameters" : pars_dictionary}

        return fit_dict

    @classmethod
    def add_pulse_results(cls, key: str, line_model: aptapy.modeling.AbstractFitModel) -> None:
        """Add pulse calibration results to the YAML dictionary.
        """
        if "calibration" not in cls._YAML_DICT:
            cls._YAML_DICT["calibration"] = {}
        cls._YAML_DICT["calibration"][key] = cls._fit_dict(line_model)

    @classmethod
    def add_source_results(cls, key: str, fit_model: aptapy.modeling.AbstractFitModel) -> None:
        """Add spectra fit results to the YAML dictionary.

        Arguments
        ---------
        key :  str
            Key to identify the source file analyzed.
        fit_model : aptapy.modeling.AbstractFitModel
            Fit model containing the results to log.
        """
        # Ensure the analysis key exists
        if "analysis" not in cls._YAML_DICT:
            cls._YAML_DICT["analysis"] = {}
        cls._YAML_DICT["analysis"][key] = cls._fit_dict(fit_model)

    @classmethod
    def add_source_gain_res(cls, key: str, g: ufloat, res: ufloat) -> None:
        """Add gain and resolution results to the YAML dictionary.
        
        Arguments
        ---------
        key :  str
            Key to identify the source file analyzed.
        g :    ufloat
            Gain value with uncertainty.
        res :  ufloat
            Resolution value with uncertainty.
        """
        # Ensure the analysis and source keys exist
        if "analysis" not in cls._YAML_DICT:
            cls._YAML_DICT["analysis"] = {}
        if key not in cls._YAML_DICT["analysis"]:
            cls._YAML_DICT["analysis"][key] = {}
        # Add gain and resolution results
        res_dict = {"gain":{"val":g.n, "err":g.s},
                    "resolution":{"val":res.n, "err":res.s}}
        cls._YAML_DICT["analysis"][key]["results"] = res_dict

    @classmethod
    def save(cls) -> None:
        """Save the YAML dictionary to a file.
        """
        with open(cls._YAML_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(cls._YAML_DICT, f, sort_keys=False, default_flow_style=False)
