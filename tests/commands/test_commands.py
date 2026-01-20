"""Testing for command line commands.
"""

from unittest.mock import patch

from analysis.cli.main import main

COMMON_ARGS = ["--w", "30",
               "--e_peak", "5.9"]

def _test_cli_command(subcommand: str, function_name: str, args: list[str]):
    """Generic test for CLI commands."""
    cmd_prefix = "analysis.cli.commands."
    kwargs = ["--numsigmaleft", "2.", "--numsigmaright", "2.5",
              "--xmin", "0.", "--xmax", "10.", "--absolutesigma", "True"]
    with patch(f"{cmd_prefix}{subcommand}.{function_name}") as mock_analyze, \
         patch(f"{cmd_prefix}{subcommand}.plt") as mock_plt, \
         patch("sys.argv", [
             "analysis",        # program name
             subcommand,       # subcommand
             *args,
             *kwargs
         ]):
        main()

        mock_analyze.assert_called_once()
        _ = mock_analyze.call_args

        mock_plt.show.assert_called_once()

def test_single_command():
    """Test single command."""
    args = ["pulse.bin",
            "--model", "Gaussian",
            "--sourcefile", "data/"] + COMMON_ARGS
    _test_cli_command("single", "analyze_file", args)


def test_folder_command():
    """Test folder command."""
    args = ["data/folder",
            "--model", "Gaussian", "Fe55Forest",] + COMMON_ARGS
    _test_cli_command("folder", "analyze_folder", args)


def test_compare_command():
    """Test compare command."""
    args = ["data/folder0",
            "data/folder1",
            "data/folder2",
            "--model", "Gaussian"] + COMMON_ARGS
    _test_cli_command("compare", "compare_folders", args)


def test_trend_command():
    """Test trend command."""
    args = ["data/folder0",
            "--model", "Gaussian"] + COMMON_ARGS
    _test_cli_command("trend", "analyze_trend", args)
