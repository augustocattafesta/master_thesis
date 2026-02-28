"""Test for the runner module.
"""

from analysis.runner import run, run_folders

def test_run(datadir):
    """Test the run function.
    """
    config_file_path = datadir / "config_default.yaml"
    source_file_path = datadir / "folder0/live_data_chip18112025_D1000_B370.mca"
    pulse_file_path = datadir / "folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
    # Run the analysis pipeline
    run(config_file_path, source_file_path, pulse_file_path)

def test_run_folders(datadir):
    """Test the run_folders function.
    """
    config_file_path = datadir / "config_default.yaml"
    folder_path = datadir / "folder0"
    # Run the analysis pipeline
    run_folders(config_file_path, folder_path)
