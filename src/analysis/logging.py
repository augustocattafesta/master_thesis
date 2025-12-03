import datetime
import inspect
import pathlib
import re
import sys

from loguru import logger

from . import ANALYSIS_RESULTS

_LOG_FOLDER = None

def get_word_after_analysis(cmd: str):
    match = re.search(r'\banalysis\b\s+(\S+)', cmd)
    return match.group(1) if match else None


def strip_before_analysis(cmd: str):
    m = re.search(r'\banalysis\b.*', cmd)
    return m.group(0) if m else cmd

log_main = None
log_fit = None
null_logger = logger.bind()

def start_logging():
    global _LOG_FOLDER
    global log_main, log_fit

    if _LOG_FOLDER is not None:
        return _LOG_FOLDER
    logger.remove()

    date = datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
    cmd = " ".join(sys.argv)
    log_folder_name = f"{date}_{get_word_after_analysis(cmd)}"
    log_file = f"{date}_{get_word_after_analysis(cmd)}.log"
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
    log_main.info(f"{strip_before_analysis(cmd)}\n")

    _LOG_FOLDER = log_folder
    return log_folder

def log_pulse_results(line_pars):
    log = log_main or null_logger
    log.info("PULSE CALIBRATION RESULTS:")
    log.info(f"{'m:':<12} {line_pars[0]} ADC/mV")
    log.info(f"{'q:':<12} {line_pars[1]} ADC\n")

def log_fit_results(file_name, fit_model):
    log = log_fit or null_logger
    log.info(f"FILE: {file_name}")
    log.info(f"{str(fit_model)}\n")

def log_args():
    log = log_main or null_logger
    frame = inspect.currentframe().f_back
    info = inspect.getargvalues(frame)
    args_dict = {name: info.locals[name] for name in info.args}
    # *args and **kwargs if present
    if info.varargs:
        args_dict[info.varargs] = info.locals[info.varargs]
    if info.keywords:
        args_dict[info.keywords] = info.locals[info.keywords]

    log.info("FUNCTION ARGUMENTS:")
    log.info(f"{args_dict}\n")
