"""Module to handle logging interface.
"""

import datetime
import inspect
import pathlib
import re
import sys
from numbers import Real

import numpy as np
import yaml

from . import ANALYSIS_RESULTS


def get_subcommand(cmd: str) -> str:
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
    _LOG_FOLDER = None
    _YAML_DICT = {}

    @staticmethod
    def _clean_numpy_types(data):
        """Recursively converts NumPy (and similar) numeric types to standard float/int."""
        if isinstance(data, dict):
            return {k: LogYaml._clean_numpy_types(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            # Per le tuple, restituisci una lista o una tupla (qui usiamo una lista per semplicità)
            return [LogYaml._clean_numpy_types(item) for item in data]
        if isinstance(data, Real) and not isinstance(data, (int, float)):
            # Se è un numero ma non un int/float standard Python (è probabilmente un tipo NumPy)
            # numpy.int64 viene convertito in int, numpy.float64 in float.
            return float(data) if isinstance(data, (np.floating, float)) else int(data)
        return None

    @classmethod
    def start_logging(cls):
        if cls._LOG_FOLDER is not None:
            return None

        date = datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
        cmd = " ".join(sys.argv)
        log_folder_name = f"{date}_{get_subcommand(cmd)}"
        log_file = f"{date}_{get_subcommand(cmd)}.yaml"
        pathlib.Path.mkdir(ANALYSIS_RESULTS / log_folder_name)
        log_folder = ANALYSIS_RESULTS / log_folder_name

        cls._LOG_FOLDER = log_folder

        frame = inspect.currentframe().f_back
        info = inspect.getargvalues(frame)
        args_dict = {name: info.locals[name] for name in info.args}
        kwargs_dict = info.locals[info.keywords] if info.keywords else {}
        cls._YAML_PATH = cls._LOG_FOLDER / log_file
        cls._YAML_DICT["execution_datetime"] = date
        cls._YAML_DICT["command_line"] = get_command(cmd)
        cls._YAML_DICT["positional_arguments"] = cls._clean_numpy_types(args_dict)
        cls._YAML_DICT["keyword_arguments"] = cls._clean_numpy_types(kwargs_dict)

        return None

    @property
    def log_folder(self):
        return self._LOG_FOLDER

    @property
    def yaml_dict(self):
        return self._YAML_DICT

    @staticmethod
    def save_par(par_ufloat):
        par_dict = {"val": par_ufloat.n, "err": par_ufloat.s}

        return par_dict

    @classmethod
    def add_pulse_results(cls, key, line_pars):
        cls._YAML_DICT["calibration"] = {"file": key,
                                        "results": {"m":cls.save_par(line_pars[0]),
                                                    "q":cls.save_par(line_pars[1])}
                                        }

    @classmethod
    def add_source_results(cls, key: str, fit_model):
        if "analysis" not in cls._YAML_DICT:
            cls._YAML_DICT["analysis"] = {}
        cls._YAML_DICT["analysis"][key] = cls.fit_dict(fit_model)

    @classmethod
    def add_source_gain_res(cls, key:str, g, res):
        if "analysis" not in cls._YAML_DICT:
            cls._YAML_DICT["analysis"] = {}
        res_dict = {"gain":{"val":g.n, "err":g.s},
                    "resolution":{"val":res.n, "err":res.s}}
        cls._YAML_DICT["analysis"][key]["results"] = res_dict

    @staticmethod
    def fit_dict(fit_model):
        pars_dictionary = {}
        for par in fit_model:
            pars_dictionary[par.name] = {"val": par.value, "err": par.error}

        fit_dict = {"model": fit_model.name(),
                "chisquare": fit_model.status.chisquare,
                "dof": fit_model.status.dof,
                "fit_parameters" : pars_dictionary}

        return fit_dict

    @classmethod
    def save(cls):
        with open(cls._YAML_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(cls._YAML_DICT, f, sort_keys=False, default_flow_style=False)
