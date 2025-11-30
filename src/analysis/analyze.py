"""Module to analyze data
"""

from pathlib import Path
from typing import Tuple, Union

import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.plotting import plt
from aptapy.typing_ import ArrayLike
from loguru import logger
from uncertainties import unumpy

from . import ANALYSIS_DATA
from .fileio import DataFolder, PulsatorFile, SourceFile
from .utils import KALPHA, energy_resolution, gain


def analyze_file(pulse_file: Union[str, Path], source_file: Union[str, Path],
                 models: Tuple[AbstractFitModel], W: float, capacity: float,
                 **kwargs) -> Union[ArrayLike, Tuple[float, float]]:
    pulse_data = PulsatorFile(Path(pulse_file))
    line_pars = pulse_data.analyze_pulse()
    logger.info(f"Calibration. Slope: {line_pars[0]} ADC/mV. Offset: {line_pars[1]} ADC")
    if source_file is not None:
        source_data = SourceFile(Path(source_file))
        g = np.zeros(shape=len(models), dtype=object)
        res = np.zeros(shape=len(models), dtype=object)
        for i, model in enumerate(models):
            if issubclass(model, aptapy.models.Fe55Forest):
                pars_Fe = source_data.fit_line_forest(**kwargs)
                line_adc = KALPHA / pars_Fe[2]
                sigma = pars_Fe[3]
            elif issubclass(model, aptapy.models.Gaussian):
                pars = source_data.fit_line(**kwargs)
                line_adc = pars[1]
                sigma = pars[2]
            else:
                raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")

            g[i] = gain(W, capacity, line_adc, line_pars)
            res[i] = energy_resolution(line_adc, sigma)

        for _model, _g, _res in zip(models, g, res):
            logger.info(f"Line fit model: {_model.__name__}. Gain: {_g}. \
                        Energy Resolution: {_res} %")

        return res, g
    return line_pars


def analyze_folder(folder_name: str, models: Tuple[AbstractFitModel], W: float, capacity: float,
                   **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    data_folder = DataFolder(ANALYSIS_DATA / folder_name)
    pulse_files = data_folder.pulse_files

    # Take the first pulse file
    pulse_data = PulsatorFile(Path(pulse_files[0]))
    line_pars = pulse_data.analyze_pulse()
    source_data = [SourceFile(_s) for _s in data_folder.source_files]
    voltage = [file.voltage for file in source_data]
    g = np.zeros(shape=len(models), dtype=object)
    res = np.zeros(shape=len(models), dtype=object)
    for i, model in enumerate(models):
        if issubclass(model, aptapy.models.Fe55Forest):
            pars = np.array([source.fit_line_forest(**kwargs) for source in source_data])
            line_adc = KALPHA / pars[:, 2]
            sigma = pars[:, 3]
        elif issubclass(model, aptapy.models.Gaussian):
            pars = np.array([source.fit_line(**kwargs) for source in source_data])
            line_adc = pars[:, 1]
            sigma = pars[:, 2]
        else:
            raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")

        g[i] = gain(W, capacity, line_adc, line_pars)
        res[i] = energy_resolution(line_adc, sigma)

    plt.figure("Gain")
    for i, model in enumerate(models):
        plt.errorbar(voltage, unumpy.nominal_values(g[i]), unumpy.std_devs(g[i]), fmt="o",
                         label=f"{model.__name__}")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.legend()

    plt.figure("Energy resolution")
    for i, model in enumerate(models):
        plt.errorbar(voltage, unumpy.nominal_values(res[i]), unumpy.std_devs(res[i]), fmt="o",
                         label=f"{model.__name__}")
    plt.xlabel("Voltage [V]")
    plt.ylabel("FWHM / E")
    plt.legend()

    return voltage, res, g


def compare_folders(folder_names: Tuple[str], model: AbstractFitModel, W: float,
                    capacity: float, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    voltage = np.zeros(shape=len(folder_names), dtype=object)
    res = np.zeros(shape=len(folder_names), dtype=object)
    g = np.zeros(shape=len(folder_names), dtype=object)
    for i, folder_name in enumerate(folder_names):
        voltage[i], res[i], g[i] = analyze_folder(folder_name, [model], W, capacity, **kwargs)
    plt.close("all")

    plt.figure("Gain")
    for i, _ in enumerate(folder_names):
        plt.errorbar(voltage[i], unumpy.nominal_values(g[i][0]), unumpy.std_devs(g[i][0]), fmt="o",
                         label=f"")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.legend()

    plt.figure("Energy resolution")
    for i, _ in enumerate(folder_names):
        plt.errorbar(voltage[i], unumpy.nominal_values(res[i][0]), unumpy.std_devs(res[i][0]),
                     fmt="o", label=f"")
    plt.xlabel("Voltage [V]")
    plt.ylabel("FWHM / E")
    plt.legend()       


def analyze_trend(folder_name: str, model: AbstractFitModel, W: float, capacity: float,
                  **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    folder_path = ANALYSIS_DATA / folder_name
    folder_data = DataFolder(folder_path)
    pulses = PulsatorFile(folder_data.pulse_files[0])
    line_pars = pulses.analyze_pulse()

    source_files = [SourceFile(_s) for _s in folder_data.source_files]
    voltage = [_source.voltage for _source in source_files]
    real_times = np.array([_source.real_time for _source in source_files])
    if issubclass(model, aptapy.models.Fe55Forest):
        pars_Fe = np.array([source.fit_line_forest(**kwargs) for source in source_files])
        line_adc = KALPHA / pars_Fe[:, 2]
    elif issubclass(model, aptapy.models.Gaussian):
        pars = np.array([source.fit_line(**kwargs) for source in source_files])
        line_adc = pars[:, 1]
    else:
        raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")

    g = gain(W, capacity, voltage, line_adc, line_pars, **kwargs)
    time = real_times.cumsum()
    plt.close('all')
    plt.figure("Gain vs time")
    plt.errorbar(time, unumpy.nominal_values(g), unumpy.std_devs(g), fmt='.k')
    plt.xlabel("Time [s]")
    plt.ylabel("Gain")
    model = aptapy.models.Exponential() + aptapy.models.Constant()
    model.fit(time, unumpy.nominal_values(g), sigma=unumpy.std_devs(g))
    model.plot(fit_output=True)
    plt.legend()

    return time, g
