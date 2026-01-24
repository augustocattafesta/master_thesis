import argparse
from pathlib import Path

from aptapy.plotting import plt

from analysis import ANALYSIS_DATA
from analysis.runner import run, run_folders


def main():
    parser = argparse.ArgumentParser(prog="analysis")

    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to data files or folders.")

    parser.add_argument(
        "config",
        help="Config path")
    args = parser.parse_args()
    # Think of a good path read logic
    is_file = []
    is_folder = []
    file_paths = []
    for path in args.paths:
        file_path = Path(path)
        if not file_path.exists():
            file_path = ANALYSIS_DATA / file_path
            if not file_path.exists():
                raise FileNotFoundError(f"Data path {path} does not exist.")
        if file_path.is_file():
            is_file.append(True)
            is_folder.append(False)
        elif file_path.is_dir():
            is_folder.append(True)
            is_file.append(False)
        file_paths.append(file_path)

    config_file_path = args.config
    if not Path(config_file_path).exists():
        # Should check if is a .yaml file too
        raise FileNotFoundError(f"Config file {config_file_path} does not exist.")

    if all(is_file):
        run(config_file_path, *file_paths)
    elif all(is_folder):
        run_folders(config_file_path, *file_paths)
    else:
        raise ValueError("All paths must be either files or folders.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
