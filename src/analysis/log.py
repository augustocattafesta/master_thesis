"""Module to handle logging interface.
"""

import datetime
import inspect
import pathlib
import re
import sys

import loguru
from loguru import logger

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


class LogManager:
    """Class to handle the logging interface.
    """
    _LOG_FOLDER = None
    _log_main = None
    _log_fit = None
    NULL_LOGGER = logger.bind()

    @classmethod
    def log_main(cls) -> None | loguru.Logger:
        """Return the main logger to log info in the main log file.
        """
        return cls._log_main

    @classmethod
    def log_fit(cls) -> None | loguru.Logger:
        """Return the fit logger to log info in the fit log file.
        """
        return cls._log_fit

    @classmethod
    def start_logging(cls) -> None:
        """Create two loggers to save info during file and folder analysis. The first logger is for
        info about the current analysis, the second is for the results of the source file fits. All
        the data are saved in the system data folder created at the execution of the program.
        """
        if cls._LOG_FOLDER is not None:
            return cls._LOG_FOLDER
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

        cls._log_main = logger.bind(tag="main")
        cls._log_fit = logger.bind(tag="fit")
        cls._LOG_FOLDER = log_folder

        cls._log_main.info("EXECUTION DATETIME:")
        cls._log_main.info(f"{date}\n")
        cls._log_main.info("COMMAND LINE (check path before executing)")
        cls._log_main.info(f"{get_command(cmd)}\n")

        return log_folder

    @classmethod
    def log_args(cls) -> None:
        """Log all the arguments and the keyword arguments of the current method execution in the
        main log file.
        """
        log = cls._log_main or cls.NULL_LOGGER
        frame = inspect.currentframe().f_back
        info = inspect.getargvalues(frame)
        args_dict = {name: info.locals[name] for name in info.args}
        if info.varargs:
            args_dict[info.varargs] = info.locals[info.varargs]
        if info.keywords:
            args_dict[info.keywords] = info.locals[info.keywords]

        log.info("FUNCTION ARGUMENTS:")
        log.info(f"{args_dict}\n")
