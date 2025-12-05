"""Testing for the analyze module.
"""
import numpy as np
from aptapy.models import Fe55Forest, Gaussian
from uncertainties.unumpy import nominal_values, std_devs

import analysis.analyze
from analysis.fileio import PulsatorFile


def test_analyze_file(datadir):
    pulse_file = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    source_file = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    args = [Gaussian, Fe55Forest], 26., 1e-12, 5.9, False, False
    kwargs = {"xmin":30., "xmax":60., "num_sigma_left":1., "num_sigma_right":1.,
              "absolute_sigma":False}
    # Test analysis of calibration pulses file
    line_pars_script = analysis.analyze.analyze_file(pulse_file, None, *args)
    pulse_data = PulsatorFile(pulse_file)
    line_pars, _, _ = pulse_data.analyze_pulses()
    # Test analysis of source file
    resolution, gain = analysis.analyze.analyze_file(pulse_file, source_file, *args)
    # Test analysis with kwargs
    resolution_kwargs, gain_kwargs = analysis.analyze.analyze_file(pulse_file, source_file,
                                                                   *args, **kwargs)

    assert np.allclose(nominal_values(line_pars_script), nominal_values(line_pars))
    assert np.allclose(std_devs(line_pars_script), std_devs(line_pars))
    assert resolution.shape[0] == len(args[0])
    assert gain.shape[0] == len(args[0])
    assert resolution_kwargs.shape[0] == len(args[0])
    assert gain_kwargs.shape[0] == len(args[0])


def test_analyze_folder(datadir):
    folder = datadir / "folder0"
    args = [Gaussian, Fe55Forest], 26., 1e-12, 5.9, False, False
    kwargs = {"xmin":30., "xmax":60., "num_sigma_left":1., "num_sigma_right":1.,
              "absolute_sigma":False}

    # Test analysis without kwargs
    voltage, resolution, gain = analysis.analyze.analyze_folder(folder, *args)
    # Test analysis with kwargs
    voltage_kwargs, resolution_kwargs, gain_kwargs = analysis.analyze.analyze_folder(folder, *args,
                                                                              **kwargs)

    assert resolution.shape == gain.shape
    assert resolution_kwargs.shape == gain_kwargs.shape
    assert np.array_equal(voltage, voltage_kwargs)


def test_compare_folders(datadir):
    folder0 = datadir / "folder0"
    folder1 = datadir / "folder1"
    args = Gaussian, 26., 1e-12, 5.9, False, False
    kwargs = {"xmin":30., "xmax":60., "num_sigma_left":1., "num_sigma_right":1.,
              "absolute_sigma":False}
    analysis.analyze.compare_folders([folder0, folder1], *args, **kwargs)


def test_analyze_trend(datadir):
    folder = datadir / "folder0"
    args = Gaussian, 26., 1e-12, 5.9, False, False
    kwargs = {"xmin":30., "xmax":60., "num_sigma_left":1., "num_sigma_right":1.,
              "absolute_sigma":False}
    gain, resolution, _, _ = analysis.analyze.analyze_trend(folder, *args, **kwargs)

    print(gain.shape)
    assert resolution.shape == gain.shape
