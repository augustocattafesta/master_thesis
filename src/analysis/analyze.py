from pathlib import Path

import numpy as np
from aptapy.modeling import AbstractFitModel
import aptapy.models
from aptapy.plotting import plt
from uncertainties import unumpy

from . import ANALYSIS_DATA
from .fileio import DataFolder, PulsatorFile, SourceFile
from .utils import KALPHA, gain, energy_resolution


def analyze_trend(folder_name: str, model: AbstractFitModel, W: float, capacity: float,
                  **kwargs) -> None:
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

    plt.show()


def analyze_file(source_file: str, pulse_file: str,  model: AbstractFitModel, W: float,
                 capacity: float, **kwargs):
    pulse_data = PulsatorFile(Path(pulse_file))
    line_pars = pulse_data.analyze_pulse()
    source_data = SourceFile(Path(source_file))
    voltage = source_data.voltage
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

    g = gain(W, capacity, voltage, line_adc, line_pars, **kwargs)
    res = energy_resolution(voltage, line_adc, sigma)

    plt.show()
