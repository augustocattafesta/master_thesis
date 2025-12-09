"""Testing for the logging module.
"""
import datetime
import shutil
from unittest.mock import patch

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
    """Test start_logging
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
