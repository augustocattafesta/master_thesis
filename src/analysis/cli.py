import argparse
from pathlib import Path

from aptapy.plotting import plt

from analysis import ANALYSIS_DATA, ANALYSIS_RESULTS
from analysis.runner import run, run_folders


__description__ = """
Analysis CLI tool for processing and visualizing data collected with μGPDs. The workflow starts
with a calibration step using pulsed data files. The resulting calibrated data is then used to
perform the analysis of the provided spectra. The results produced by the analysis can be used
to characterize the detector.
The tasks to execute are defined in the provided configuration .yaml file. To see how to
create the configuration file, refer to the documentation: 
https://augustocattafesta.github.io/master_thesis/
"""

def check_source_paths(paths: list[str]) -> tuple[list[Path], str]:
    """Check if source paths exist and are either all files or all folders.
    
    Parameters
    ----------
    paths : list[str]
        List of paths to check.
    
    Returns
    -------
    checked_paths : tuple[list[Path], str]
        A tuple containing the list of checked paths and a string indicating
        whether they are 'file' or 'folder'.
    """
    # Create a list to hold the checked paths
    checked_paths = []
    # Loop through the paths and check if they exist. If the given path does not exist, check
    # if it exists in the ANALYSIS_DATA folder.
    for p in paths:
        path = Path(p)
        if not path.exists():
            file_path = ANALYSIS_DATA / path
            if not file_path.exists():
                # Raise an error if the path does not exist in either location
                raise FileNotFoundError(f"Data path {p} does not exist.")
        checked_paths.append(file_path)
    # Check if all paths are either files or folders
    is_file = [p.is_file() for p in checked_paths]
    is_folder = [p.is_dir() for p in checked_paths]
    # Return the checked paths and their type
    if all(is_file):
        return checked_paths, "file"
    elif all(is_folder):
        return checked_paths, "folder"
    else:
        raise ValueError("All source paths must be either files or folders.")


def check_config_path(path: str) -> Path:
    """Check if config path exists.
    
    Parameters
    ----------
    path : str
        Path to check.

    Returns
    -------
    checked_path : Path
        The checked path.
    """
    # Convert to Path object
    config_path = Path(path)
    # Check if the path exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist.")
    # Check if the path is a .yaml file
    if not config_path.suffix == ".yaml":
        raise ValueError("Config file must be a .yaml file.")
    return config_path


def main():
    """Main function for the analysis CLI."""
    parser = argparse.ArgumentParser(
        prog="analysis",
        description=__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        "config",
        help="Path to the analysis configuration .yaml file.")

    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to source and calibration files or folders. If multiple files are provided, " \
        " the calibration file should always be the last file. If folders are provided, there is" \
        " no need to specify calibration files separately as they will be searched for in each" \
        " folder.")

    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help=f"Save figures and results to the analysis results folder {ANALYSIS_RESULTS}"
    )

    parser.add_argument(
        "-f", "--format",
        default="png",
        choices=["png", "pdf"],
        help="Format of the saved figures"
    )

    args = parser.parse_args()
    # Check source paths and config path and convert to Path objects
    file_paths, path_type = check_source_paths(args.paths)
    config_file_path = check_config_path(args.config)
    # Run analysis based on path type
    if path_type == "file":
        context = run(config_file_path, *file_paths)
    else:
        context = run_folders(config_file_path, *file_paths)
    # Plot results if available
    plt.tight_layout()
    # Save results if specified
    if args.save:
        context.save(ANALYSIS_RESULTS, fig_format=args.format)
    plt.show()


if __name__ == "__main__":
    main()
