from pathlib import Path

from analysis.runner import run

source_file_path = Path(__file__).parent.parent / "data/single_file/source_w1a_D1000B340.mca"
calibration_file_path = Path(__file__).parent.parent / "data/single_file/calibration_ci2-4-6-10mv.mca"
config_file_path = Path(__file__).parent.parent / "single_example_config.yaml"
context = run(config_file_path, source_file_path, calibration_file_path)

