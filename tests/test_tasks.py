"""Tests for analysis tasks."""
import aptapy.models
import numpy as np

from analysis import tasks
from analysis.context import Context
from analysis.fileio import PulsatorFile, SourceFile


def test_calibration(datadir, context):
    """Test the calibration task, checking if the context gets updated correctly."""
    pulse_file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    pulse_data = PulsatorFile(pulse_file_path)
    context.pulse = pulse_data
    context = tasks.calibration(context)
    calibration_model = context.conversion_model

    assert context.pulse == pulse_data
    assert isinstance(calibration_model, aptapy.models.Line)


def test_fit_peak(datadir, context: Context):
    """Test the fit_peak task, checking if the context gets updated correctly."""
    # Perform calibration first to obtain the charge conversion model
    pulse_file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    context.pulse = PulsatorFile(pulse_file_path)
    context = tasks.calibration(context)
    charge_conv_model = context.conversion_model
    # Load source file
    source_file_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(source_file_path, charge_conv_model)
    context.add_source(source)
    # Perform peak fitting
    context = tasks.fit_peak(context, subtask="subtask", model_class=[aptapy.models.Gaussian])
    target_ctx = context.target_ctx(source.file_path.stem, "subtask")
    assert target_ctx is not None
    assert target_ctx.line_val is not None
    assert target_ctx.sigma is not None
    assert target_ctx.voltage == 370.0
    assert isinstance(target_ctx.model, aptapy.models.Gaussian)


def test_gain(datadir, context: Context):
    """Test the gain task."""
    target = "subtask"
    # Perform calibration first to obtain the charge conversion model
    pulse_file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    context.pulse = PulsatorFile(pulse_file_path)
    context = tasks.calibration(context)
    charge_conv_model = context.conversion_model
    # Load source file
    source_file_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(source_file_path, charge_conv_model)
    context.add_source(source)
    # Perform peak fitting
    context = tasks.fit_peak(context, subtask=target, model_class=[aptapy.models.Gaussian])
    # Perform gain calculation
    context = tasks.gain_task(context, target=target)

    gain_ctx = context.task_results("gain", target)
    assert gain_ctx is not None
    assert np.array_equal(gain_ctx["voltages"], np.array([370.0]))
    assert "gain_vals" in gain_ctx


def test_resolution(datadir, context: Context):
    """Test the resolution task."""
    target = "subtask"
    # Perform calibration first to obtain the charge conversion model
    pulse_file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    context.pulse = PulsatorFile(pulse_file_path)
    context = tasks.calibration(context)
    charge_conv_model = context.conversion_model
    # Load source file
    source_file_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    source = SourceFile(source_file_path, charge_conv_model)
    context.add_source(source)
    # Perform peak fitting
    context = tasks.fit_peak(context, subtask=target, model_class=[aptapy.models.Gaussian])
    # Perform resolution calculation
    context = tasks.resolution_task(context, target=target)
    resolution_ctx = context.task_results("resolution", target)
    assert resolution_ctx is not None
    assert np.array_equal(resolution_ctx["voltages"], np.array([370.0]))
    assert "res_vals" in resolution_ctx
