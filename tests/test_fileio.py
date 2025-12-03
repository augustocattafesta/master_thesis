"""Test FileIO
"""

import aptapy.models
import numpy as np
from aptapy.plotting import plt

from analysis.fileio import PulsatorFile, SourceFile, output_log


def test_source_file(datadir):
    file_path = datadir / "live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(file_path)
    real_time = source.real_time

    assert source.voltage == 370
    assert np.equal(real_time, 163.800000)

def test_pulse_file(datadir):
    file_path = datadir / "live_data_chip18112025_ci5-10-15_hvon.mca"
    pulses = PulsatorFile(file_path)

    assert np.array_equal(pulses.voltage, [5, 10, 15])

def test_new_fit(datadir):
    file_path = datadir / "live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(file_path)
    plt.figure("Test Gaussian")
    _, model = source.fit(aptapy.models.Gaussian, xmin=50, xmax=80)
    source.hist.plot()
    model.plot(fit_output=True)
    plt.legend()

    plt.figure("Test Fe55Forest")
    _, model = source.fit(aptapy.models.Fe55Forest, xmin=50, xmax=80, num_sigma_right=1.5)
    source.hist.plot()
    model.plot(fit_output=True)
    plt.legend()

    plt.figure("Test Escape Peak")
    _, model = source.fit(aptapy.models.Fe55Forest, xmin=20, xmax=35, num_sigma_right=1.5)
    source.hist.plot()
    model.plot(fit_output=True)
    plt.legend()


def test_log_caller(a, b, c, **kwargs):
    print(output_log())

test_log_caller(1, 2, 3, x=10)