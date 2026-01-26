"""Tests for analysis tasks."""
import aptapy.models
import pytest

import analysis.tasks as tasks
from analysis.fileio import PulsatorFile, SourceFile


def test_calibration(datadir, context):
    """Test the calibration task, checking if the context gets updated correctly."""
    pulse_file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    pulse_data = PulsatorFile(pulse_file_path)
    context["calibration"]["pulse"] = pulse_data
    context = tasks.calibration(context)
    calibration_ctx = context["calibration"]
    
    assert calibration_ctx["pulse"] == pulse_data
    assert "model" in calibration_ctx
    charge_conv_model = calibration_ctx["model"]
    assert isinstance(charge_conv_model, aptapy.models.Line)

def test_fit_peak(datadir, context):
    """Test the fit_peak task, checking if the context gets updated correctly."""
    # Perform calibration first to obtain the charge conversion model
    pulse_file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    context["calibration"]["pulse"] = PulsatorFile(pulse_file_path)
    context = tasks.calibration(context)
    charge_conv_model = context["calibration"]["model"]
    # Load source file
    source_file_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(source_file_path, charge_conv_model)
    context["tmp_source"] = source  # Temporary storage for the source
    # Perform peak fitting
    context = tasks.fit_peak(context, subtask="subtask", model_class=[aptapy.models.Gaussian])
    fit_ctx = context["fit"]
    assert source_file_path.stem in fit_ctx
    target_ctx = fit_ctx[source_file_path.stem]["subtask"]
    assert "line_val" in target_ctx
    assert "voltage" in target_ctx
    assert "model" in target_ctx