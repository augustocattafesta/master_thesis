from aptapy.plotting import plt

from analysis.fileio import PulsatorFile, SourceFile
from analysis.tasks import calibration, fit_peak

def test_calibration(datadir):
    file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    pulse = PulsatorFile(file_path)

    cal_model, pulse_fig, cal_fig = calibration(pulse)


def test_main_peak(datadir):
    file_pulse_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    pulse = PulsatorFile(file_pulse_path)
    charge_conv_model, _, _ = calibration(pulse)
    file_source_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(file_source_path, charge_conv_model)
    results = fit_peak(source)
    print(results)
