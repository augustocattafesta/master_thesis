"""Testing for the logging module.
"""
import datetime
import shutil
from unittest.mock import patch

import numpy as np
from aptapy.models import Gaussian
from uncertainties import ufloat

from analysis.log import LogYaml, get_command, get_subcommand


def test_get_subcommand():
    """Test get_subcommand.
    """
    subcommand = "example_subcommand"
    cmd_line = f"/home/user/test/bin/analysis {subcommand} --options [OPTIONS]"

    assert get_subcommand(cmd_line) == subcommand


def test_get_command():
    """Test get_command.
    """
    command = "analysis example_subcommand --options [OPTIONS]"
    cmd_line = f"/home/user/test/bin/{command}"

    assert get_command(cmd_line) == command


# Use the decorator to tell python datetime is modified
@patch("analysis.log.datetime")
def test_start_logging(mock_datetime):
    """Test start_logging method.
    """
    # Faking the time of the system to check the format
    mock_datetime.datetime.now.return_value = datetime.datetime(2000, 1, 1, 12, 30, 15, 230)
    logyaml = LogYaml()
    logyaml.start_logging()

    assert logyaml.log_folder is not None
    assert logyaml.log_folder.name == "2000-01-01__12:30:15_None"
    assert logyaml.yaml_dict["execution_datetime"] == "2000-01-01__12:30:15"
    # Remove the folder created during the test
    shutil.rmtree(logyaml.log_folder)


def test_add_pulse_results():
    """Test add_pulse_results.
    """
    # Create a LogYaml instance
    logyaml = LogYaml()
    line_pars = np.array([ufloat(1.0, 0.1), ufloat(2.0, 0.2)])
    logyaml.add_pulse_results("calibration_file.txt", line_pars)

    assert "calibration" in logyaml.yaml_dict
    calibration = logyaml.yaml_dict["calibration"]
    assert calibration["file"] == "calibration_file.txt"
    assert calibration["results"]["m"]["val"] == 1.0
    assert calibration["results"]["m"]["err"] == 0.1
    assert calibration["results"]["q"]["val"] == 2.0
    assert calibration["results"]["q"]["err"] == 0.2


def test_add_source_results():
    """Test add_source_results.
    """
    mock_fit_model = Gaussian()
    # Create a LogYaml instance
    logyaml = LogYaml()
    logyaml.add_source_results("source_1", mock_fit_model)

    assert "analysis" in logyaml.yaml_dict
    analysis = logyaml.yaml_dict["analysis"]
    assert "source_1" in analysis
    fit_results = analysis["source_1"]
    assert fit_results["model"] == mock_fit_model.name()
    # Since chisquare and dof are zero for an unfit model, we just check they exist
    assert "chisquare" in fit_results
    assert "dof" in fit_results
    assert "fit_parameters" in fit_results
    params = fit_results["fit_parameters"]
    for par in mock_fit_model:
        assert par.name in params
        assert params[par.name]["val"] == par.value
        assert params[par.name]["err"] == par.error


def test_add_source_gain_res():
    """Test add_source_gain_res.
    """
    # Create a LogYaml instance
    logyaml = LogYaml()
    # First, add source results to ensure the key exists
    mock_fit_model = Gaussian()
    logyaml.add_source_results("source_1", mock_fit_model)

    gain = ufloat(2.5, 0.05)
    resolution = ufloat(0.1, 0.002)
    logyaml.add_source_gain_res("source_1", gain, resolution)

    analysis = logyaml.yaml_dict["analysis"]
    assert "source_1" in analysis
    results = analysis["source_1"]["results"]
    assert results["gain"]["val"] == 2.5
    assert results["gain"]["err"] == 0.05
    assert results["resolution"]["val"] == 0.1
    assert results["resolution"]["err"] == 0.002
