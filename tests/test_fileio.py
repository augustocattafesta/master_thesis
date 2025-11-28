"""Test FileIO
"""

import numpy as np

from analysis.fileio import PulsatorFile, SourceFile


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
