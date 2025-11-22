"""
"""

import pytest
import numpy as np

from analysis.fileio import SourceFile, PulsatorFile

def test_source_file(datadir):
    file_path = datadir / "live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(file_path)

    assert source.voltage == 370

def test_source_file(datadir):
    file_path = datadir / "live_data_chip18112025_ci5-10-15_hvon.mca"
    pulses = PulsatorFile(file_path)

    assert np.array_equal(pulses.voltage, [5, 10, 15])
