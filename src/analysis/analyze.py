"""Module to analyze data
"""

from pathlib import Path
from typing import Tuple, Union

import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel, line_forest
from aptapy.plotting import plt, last_line_color
from aptapy.typing_ import ArrayLike
from loguru import logger
from uncertainties import unumpy

from . import ANALYSIS_DATA
from .fileio import DataFolder, PulsatorFile, SourceFile
from .utils import AR_ESCAPE, KALPHA, KBETA, energy_resolution, gain

@line_forest(KALPHA - AR_ESCAPE, KBETA - AR_ESCAPE)
class ArEscape(aptapy.models.GaussianForest):
    pass

def analyze_file(pulse_file: Union[str, Path], source_file: Union[str, Path],
                 models: Tuple[AbstractFitModel], W: float, capacity: float,
                 e_peak: float, plot: bool = False, **kwargs) -> Union[ArrayLike, Tuple[float, float]]:
    pulse_data = PulsatorFile(Path(pulse_file))
    line_pars = pulse_data.analyze_pulse()
    logger.info(f"Calibration. Slope: {line_pars[0]} ADC/mV. Offset: {line_pars[1]} ADC")
    if source_file is not None:
        source_data = SourceFile(Path(source_file))
        g = np.zeros(shape=len(models), dtype=object)
        res = np.zeros(shape=len(models), dtype=object)
        for i, model in enumerate(models):
            pars, fit_model = source_data.fit(model, **kwargs)            
            if issubclass(model, aptapy.models.GaussianForest):
                line_adc = fit_model.energies[0] / pars[2]
                sigma = pars[3]
            elif issubclass(model, aptapy.models.Gaussian):
                line_adc = pars[1]
                sigma = pars[2]
            else:
                raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")
            g[i] = gain(W, capacity, line_adc, line_pars, e_peak)
            res[i] = energy_resolution(line_adc, sigma)

            if plot:
                plt.figure(f"{source_data.file_path.name}")
                plt.title(f"{int(source_data.voltage)} V {fit_model.name()}")
                source_data.hist.plot()
                label = f"{fit_model.name()}\nFWHM/E@{e_peak} keV: {res[i]} %"
                fit_model.plot(label=label)
                plt.xlim(fit_model.default_plotting_range())
                plt.legend()


        for _model, _g, _res in zip(models, g, res):
            logger.info(f"Line fit model: {_model.__name__}. Gain: {_g}. \
                        Energy Resolution: {_res} %")

        return res, g
    return line_pars


def analyze_folder(folder_name: str, models: Tuple[AbstractFitModel], W: float, capacity: float,
                   e_peak: float, plot: bool = False,
                   **kwargs) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    data_folder = DataFolder(ANALYSIS_DATA / folder_name)
    pulse_files = data_folder.pulse_files

    # Take the first pulse file
    pulse_data = PulsatorFile(Path(pulse_files[0]))
    line_pars = pulse_data.analyze_pulse()
    source_data = [SourceFile(_s) for _s in data_folder.source_files]
    voltage = np.array([file.voltage for file in source_data])
    g = np.zeros(shape=len(models), dtype=object)
    res = np.zeros(shape=len(models), dtype=object)
    xmin_init = kwargs["xmin"]
    xmax_init = kwargs["xmax"]
    for i, model in enumerate(models):
        results = []
        for source in source_data:
            x_peak = source.hist.bin_centers()[source.hist.content.argmax()]
            # Without a proper initialization of xmin and xmax the fit doesn't converge
            if xmin_init == float("-inf"):
                kwargs.update(xmin=x_peak - 0.5 * x_peak)
            if xmax_init == float("inf"):
                kwargs.update(xmax=x_peak + 0.5 * x_peak)
            results.append(source.fit(model, **kwargs))
        pars, fit_models = zip(*results)
        pars = np.stack(pars)
        fit_models = list(fit_models)
        if issubclass(model, aptapy.models.GaussianForest):
            line_adc = fit_models[0].energies[0] / pars[:, 2]
            sigma = pars[:, 3]
        elif issubclass(model, aptapy.models.Gaussian):
            line_adc = pars[:, 1]
            sigma = pars[:, 2]
        else:
            raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")

        if plot:
            for j, _s in enumerate(source_data):
                plt.figure(f"{_s.file_path.name}")
                plt.title(f"{int(voltage[j])} V {fit_models[j].name()}")
                _s.hist.plot()
                label = f"{fit_models[j].name()}\nFWHM@{e_peak} keV: {fit_models[j].fwhm()}"
                fit_models[j].plot(label=label)
                plt.legend()

        g[i] = gain(W, capacity, line_adc, line_pars, KALPHA)
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
                    capacity: float, e_peak: float, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    voltage = np.zeros(shape=len(folder_names), dtype=object)
    res = np.zeros(shape=len(folder_names), dtype=object)
    g = np.zeros(shape=len(folder_names), dtype=object)
    for i, folder_name in enumerate(folder_names):
        voltage[i], res[i], g[i] = analyze_folder(folder_name, [model], W, capacity, e_peak, **kwargs)
    plt.close("all")

    plt.figure("Gain")
    labels = {"251118":"W2b 86.6 top-right", "251127":"W8b 86.6 top-left high rate"}
    model = aptapy.models.Exponential()
    for i, folder_name in enumerate(folder_names):
        if folder_name == "251118":
            voltage[i] = np.append(voltage[i], [300, 310, 320])
            g[i][0] = np.append(g[i][0], unumpy.uarray([33.821245688904206, 40.060014190288435, 47.85847481701872],
                          [0.004015431655164531, 0.003950437657169777, 0.0038697164947474323]))
        if folder_name == "251127":
            g_350 = g[i][0][voltage[i] == 350.]
            g[i][0] = g[i][0][voltage[i] != 350.]
            voltage[i] = voltage[i][voltage[i] != 350.]
            voltage[i] = np.append(voltage[i], 350.)
            g[i][0] = np.append(g[i][0], np.min(g_350))

        plt.errorbar(voltage[i], unumpy.nominal_values(g[i][0]), unumpy.std_devs(g[i][0]), fmt="o",
                    label=labels[folder_name])
        model.fit(voltage[i], unumpy.nominal_values(g[i][0]), sigma=unumpy.std_devs(g[i][0]), absolute_sigma=True)
        model.plot(label=f"scale: {-model.scale.ufloat()} V", color=last_line_color())
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.legend()

    # plt.figure("Energy resolution")
    # for i, folder_name in enumerate(folder_names):
    #     if folder_name == "251118":
    #         voltage[i] = voltage[i][:-3]
    #     elif folder_name == "251127":
    #         voltage[i] = np.append(voltage[i], 350.)
    #     plt.errorbar(voltage[i], unumpy.nominal_values(res[i][0]), unumpy.std_devs(res[i][0]),
    #                  fmt="o", label=f"")
    # plt.xlabel("Voltage [V]")
    # plt.ylabel("FWHM / E")
    # plt.legend()       


def analyze_trend(folder_name: str, model: AbstractFitModel, W: float, capacity: float,
                  e_peak, plot: bool = False, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    folder_path = ANALYSIS_DATA / folder_name
    folder_data = DataFolder(folder_path)
    pulse = PulsatorFile(folder_data.pulse_files[0])
    line_pars = pulse.analyze_pulse()
    _, res, g = analyze_folder(folder_name, [model], W, capacity, e_peak, plot, **kwargs)
    res = res[0]
    g = g[0]
    source_files = [SourceFile(_s) for _s in folder_data.source_files]
    real_times = np.array([SourceFile(_s).real_time for _s in folder_data.source_files])
    time = real_times.cumsum()


    results = [source.fit(aptapy.models.Gaussian, xmin=25, xmax=53, num_sigma_left=1.5,
                          num_sigma_right=1.5) for source in source_files]
    pars, _ = zip(*results)
    pars = np.stack(pars)
    line_adc = pars[:, 1]
    g_esc = gain(W, capacity, line_adc, line_pars, 3.)
    plt.figure("Gain vs time")
    plt.errorbar(time, unumpy.nominal_values(g), unumpy.std_devs(g), fmt='.', label=r'K$\alpha$')
    plt.errorbar(time, unumpy.nominal_values(g_esc), unumpy.std_devs(g_esc), fmt='.', label="Esc. Peak")
    plt.xlabel("Time [s]")
    plt.ylabel("Gain")
    plt.legend()

    plt.figure("Resolution vs time")
    plt.errorbar(time, unumpy.nominal_values(res), unumpy.std_devs(res), fmt='.', label=r'K$\alpha$')
    plt.xlabel("Time [s]")
    plt.ylabel("FWHM/E")
    plt.legend()

    return time, g
