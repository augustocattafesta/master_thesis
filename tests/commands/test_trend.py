"""Test for compare CLI command
"""


from unittest.mock import patch

import aptapy.models

from analysis.cli.main import main


def test_trend_cli_runs():
    # Mock analyze_file so we don't run the full analysis
    with patch("analysis.cli.commands.trend.analyze_trend") as mock_analyze, \
         patch("analysis.cli.commands.trend.plt") as mock_plt, \
         patch("sys.argv", [
             "analysis",                # program name
             "trend",                 # subcommand
             "folder1",
             "Gaussian",
             "--sigmaleft", "2.0",
             "--sigmaright", "3.0",
             "--W", "30",
             "--capacity", "2e-12",
         ]):
        # call CLI
        main()

        # assert analyze_file called correctly
        mock_analyze.assert_called_once()

        args, kwargs = mock_analyze.call_args
        # positional args
        assert args[0] == "folder1"
        assert args[1] == aptapy.models.Gaussian
        assert args[2] == 30              # W
        assert args[3] == 2e-12           # capacity
        # keyword args
        assert kwargs["num_sigma_left"] == 2.0
        assert kwargs["num_sigma_right"] == 3.0
        # plt.show should be called too
        mock_plt.show.assert_called_once()