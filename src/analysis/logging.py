import re
import inspect
import sys
import datetime
import pathlib

from . import ANALYSIS_RESULTS

from loguru import logger

_LOG_FOLDER = None

def get_word_after_analysis(cmd: str):
    match = re.search(r'\banalysis\b\s+(\S+)', cmd)
    return match.group(1) if match else None


def strip_before_analysis(cmd: str):
    m = re.search(r'\banalysis\b.*', cmd)
    return m.group(0) if m else cmd


def start_logging():
    global _LOG_FOLDER
    
    # Se una cartella di log è già stata creata, restituisci semplicemente il percorso attivo.
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
    logger.add(log_folder / log_file, level="INFO", format=log_format)
    logger.info("EXECUTION DATETIME:")
    logger.info(f"{date}\n")
    logger.info("COMMAND LINE (check path before executing)")
    logger.info(f"{strip_before_analysis(cmd)}\n")

    _LOG_FOLDER = log_folder
    return log_folder

def log_pulse_results(line_pars):
    logger.info("PULSE CALIBRATION RESULTS:")
    logger.info(f"{'m:':<12} {line_pars[0]} ADC/mV")
    logger.info(f"{'q:':<12} {line_pars[1]} ADC\n")


def log_args():
    frame = inspect.currentframe().f_back
    info = inspect.getargvalues(frame)
    args_dict = {name: info.locals[name] for name in info.args}
    # *args and **kwargs if present
    if info.varargs:
        args_dict[info.varargs] = info.locals[info.varargs]
    if info.keywords:
        args_dict[info.keywords] = info.locals[info.keywords]

    logger.info("FUNCTION ARGUMENTS:")
    logger.info(f"{args_dict}\n")