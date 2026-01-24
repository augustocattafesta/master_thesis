"""Test for the module fileio.
"""

import aptapy.models
import numpy as np
from analysis.app import load_class
from aptapy.plotting import plt

from analysis.fileio import PulsatorFile, SourceFile


def test_pulse_file(datadir):
    file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    pulses = PulsatorFile(file_path)

    assert np.array_equal(pulses.voltage, [5, 10, 15])


def test_source_file(datadir):
    file_pulse_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    pulses = PulsatorFile(file_pulse_path)
    charge_conv_model, _, _ = pulses.analyze_pulses(fit_charge=True)
    file_source_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(file_source_path, charge_conv_model)
    real_time = source.real_time

    assert source.voltage == 370
    assert np.equal(real_time, 163.800000)


def test_fit(datadir):
    file_pulse_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    pulses = PulsatorFile(file_pulse_path)
    charge_conv_model, _, _ = pulses.analyze_pulses(fit_charge=True)
    
    file_source_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(file_source_path, charge_conv_model)

    xmin, xmax = 3., 5.
    plt.figure("Test Gaussian")
    model = source.fit(load_class("Gaussian"), xmin=xmin, xmax=xmax)
    source.hist.plot()
    model.plot(fit_output=True)
    plt.legend()

    plt.figure("Test Fe55Forest")
    model = source.fit(aptapy.models.Fe55Forest, xmin=xmin, xmax=xmax, num_sigma_right=1.5)
    source.hist.plot()
    model.plot(fit_output=True)
    plt.legend()

    xmin_esc, xmax_esc = 1., 3.
    plt.figure("Test Escape Peak")
    model = source.fit(aptapy.models.Fe55Forest, xmin=xmin_esc, xmax=xmax_esc, num_sigma_right=1.5)
    source.hist.plot()
    model.plot(fit_output=True)
    plt.legend()
