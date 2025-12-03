"""Test for single CLI command
"""


from unittest.mock import patch

import aptapy.models

from analysis.cli.main import main


def test_single_cli_runs():
    # Mock analyze_file so we don't run the full analysis
    with patch("analysis.cli.commands.single.analyze_file") as mock_analyze, \
         patch("analysis.cli.commands.single.plt") as mock_plt, \
         patch("sys.argv", [
             "analysis",                # program name
             "single",                  # subcommand
             "pulse.bin",               # pulsefile (positional)
             "--model", "Gaussian",
             "--sourcefile", "data/",
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
        assert args[0] == "pulse.bin"     # pulsefile
        assert args[1] == "data/"         # sourcefile
        assert args[2] == [aptapy.models.Gaussian]
        assert args[3] == 30              # W
        assert args[4] == 2e-12           # capacity
        # keyword args
        assert kwargs["num_sigma_left"] == 2.0
        assert kwargs["num_sigma_right"] == 3.0
        # plt.show should be called too
        mock_plt.show.assert_called_once()
