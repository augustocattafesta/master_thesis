"""Testing for the logging module
"""
import datetime
import shutil
from unittest.mock import patch

from analysis import log


def test_get_subcommand():
    """Test get_subcommand.
    """
    subcommand = "example_subcommand"
    cmd_line = f"/home/user/test/bin/analysis {subcommand} --options [OPTIONS]"

    assert log.get_subcommand(cmd_line) == subcommand


def test_get_command():
    """Test get_command.
    """
    command = "analysis example_subcommand --options [OPTIONS]"
    cmd_line = f"/home/user/test/bin/{command}"

    assert log.get_command(cmd_line) == command


# Use the decorator to tell python datetime is modified 
@patch("analysis.log.datetime")
def test_start_logging(mock_datetime):
    """Test start_logging
    """
    assert log._LOG_FOLDER == None
    assert log.log_main == None
    assert log.log_fit == None
    
    # Faking the time of the system to check the format
    mock_datetime.datetime.now.return_value = datetime.datetime(2000, 1, 1, 12, 30, 15, 230)
    log_folder = log.start_logging()

    assert log_folder.name == "2000-01-01__12:30:15_None"
    assert log._LOG_FOLDER != None
    assert log.log_main != None
    assert log.log_fit != None
    # Remove the folder created during the test
    shutil.rmtree(log_folder)



    
