import datetime
import inspect
import pathlib
import re
import sys

import aptapy.modeling
import numpy as np
from loguru import logger

from . import ANALYSIS_RESULTS


_LOG_FOLDER = None
log_main = None
log_fit = None
null_logger = logger.bind()


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


def start_logging() -> None:
    """Create two loggers to save info during file and folder analysis. The first logger is for
    info about the current analysis, the second is for the results of the source file fits. All
    the data are saved in the system data folder created at the execution of the program.
    """
    global _LOG_FOLDER
    global log_main, log_fit

    if _LOG_FOLDER is not None:
        return _LOG_FOLDER
    logger.remove()

    date = datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
    cmd = " ".join(sys.argv)
    log_folder_name = f"{date}_{get_subcommand(cmd)}"
    log_file = f"{date}_{get_subcommand(cmd)}.log"
    pathlib.Path.mkdir(ANALYSIS_RESULTS / log_folder_name)
    log_folder = ANALYSIS_RESULTS / log_folder_name
    log_format = "{message}"
    logger.add(log_folder / log_file, level="INFO", format=log_format,
               filter=lambda r: r["extra"].get("tag") == "main")
    logger.add(log_folder / "_fitresults.log", level="INFO", format=log_format,
               filter=lambda r: r["extra"].get("tag") == "fit")

    log_main = logger.bind(tag="main")
    log_fit = logger.bind(tag="fit")
    
    log_main.info("EXECUTION DATETIME:")
    log_main.info(f"{date}\n")
    log_main.info("COMMAND LINE (check path before executing)")
    log_main.info(f"{get_command(cmd)}\n")

    _LOG_FOLDER = log_folder
    return log_folder


def log_args() -> None:
    """Log all the arguments and the keyword arguments of the current method execution in the main
    log file.
    """
    log = log_main or null_logger
    frame = inspect.currentframe().f_back
    info = inspect.getargvalues(frame)
    args_dict = {name: info.locals[name] for name in info.args}
    if info.varargs:
        args_dict[info.varargs] = info.locals[info.varargs]
    if info.keywords:
        args_dict[info.keywords] = info.locals[info.keywords]

    log.info("FUNCTION ARGUMENTS:")
    log.info(f"{args_dict}\n")


def log_pulse_results(line_pars: np.ndarray) -> None:
    """Log the results of the calibration fit in the main log file.

    Parameters
    ----------
    line_pars : np.ndarray
        Parameters obtained from the calibration fit.
    """
    log = log_main or null_logger
    log.info("PULSE CALIBRATION RESULTS:")
    log.info(f"{'m:':<12} {line_pars[0]} ADC/mV")
    log.info(f"{'q:':<12} {line_pars[1]} ADC\n")

def log_fit_results(file_name: str, fit_model: aptapy.modeling.AbstractFitModel) -> None:
    """Log the results of the fit of a spectrum file in the fit log file.

    Parameters
    ----------
    file_name : str
        Name of the spectrum file to log.
    fit_model : aptapy.modeling.AbstractFitModel
        Fit model returned after the analysis of the spectrum file.
    """
    log = log_fit or null_logger
    log.info(f"FILE: {file_name}")
    log.info(f"{str(fit_model)}\n")
