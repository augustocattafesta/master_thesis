"""Test for the module fileio.
"""

import aptapy.models
import numpy as np
from aptapy.plotting import plt

from analysis.app import load_class
from analysis.fileio import PulsatorFile, SourceFile, LoadYamlLog


def test_source_file(datadir):
    file_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(file_path)
    real_time = source.real_time

    assert source.voltage == 370
    assert np.equal(real_time, 163.800000)

def test_pulse_file(datadir):
    file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    pulses = PulsatorFile(file_path)

    assert np.array_equal(pulses.voltage, [5, 10, 15])

def test_fit(datadir):
    file_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(file_path)
    plt.figure("Test Gaussian")
    model = source.fit(load_class("Gaussian"), xmin=50, xmax=80)
    source.hist.plot()
    model.plot(fit_output=True)
    plt.legend()

    plt.figure("Test Fe55Forest")
    model = source.fit(aptapy.models.Fe55Forest, xmin=50, xmax=80, num_sigma_right=1.5)
    source.hist.plot()
    model.plot(fit_output=True)
    plt.legend()

    plt.figure("Test Escape Peak")
    model = source.fit(aptapy.models.Fe55Forest, xmin=20, xmax=35, num_sigma_right=1.5)
    source.hist.plot()
    model.plot(fit_output=True)
    plt.legend()


def test_load_yaml(datadir):
    file_path = datadir / "2025-12-09_13:50:23_compare.yaml"
    log = LoadYamlLog(file_path)

    assert log.yaml_file != {}
    assert log.command == "analysis compare 251118 251127 --save"
    assert log.gain is not None
    assert log.resolution is not None
    assert log.model == "Gaussian"
    
