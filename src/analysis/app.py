
import importlib
import inspect
import sys
from argparse import ArgumentParser

import aptapy.modeling

from . import ANALYSIS_DATA, ANALYSIS_RESULTS
from .utils import KALPHA


def _load_single_class(class_path: str) -> type[aptapy.modeling.AbstractFitModel]:
    """
    Load a class from a string.
    Supports:
      - "ClassName" (local or global)
      - "module.ClassName"
      - "package.module.ClassName"
    """
    cls = None
    if "." not in class_path:
        if hasattr(aptapy.models, class_path):
            cls = getattr(aptapy.models, class_path)
        else:
            raise ImportError(f"Class '{class_path}' not found in aptapy.modeling.")
    else:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    return cls


def load_class(class_path: str) -> type[aptapy.modeling.AbstractFitModel]:
    class_paths = [p.strip() for p in class_path.split("+")]
    return [_load_single_class(p) for p in class_paths]


def add_pulsefile(parser: ArgumentParser) -> None:
    parser.add_argument("pulsefile", type=str,
                        help="Path of the calibration pulses file.")

def add_sourcefile(parser: ArgumentParser) -> None:
    parser.add_argument("--sourcefile", type=str, default=None,
                        help="Path of the source data file. Default is None.")

def add_foldername(parser: ArgumentParser) -> None:
    parser.add_argument("foldername", type=str,
                        help=f"Name of the folder to analyze. Please note that only the path after\
                            ({ANALYSIS_DATA}) must be given. The path is automatically added\
                             during the analysis.")

def add_foldernames(parser: ArgumentParser) -> None:
    parser.add_argument("foldernames", type=str, nargs="+",
                        help=f"Name of the folders to analyze and compare. Please note that only \
                            the paths after ({ANALYSIS_DATA}) must be given. The paths are \
                            automatically added during the analysis.")

def add_singlemodel(parser: ArgumentParser) -> None:
    parser.add_argument("--model", default="Gaussian", type=str,
                        help="Model to fit the emission line(s). Default is Gaussian.")

def add_multiplemodel(parser: ArgumentParser) -> None:
    parser.add_argument("--model", default="Gaussian", nargs='+', type=str,
                        help="Model to fit the emission line(s). Multiple models can be given.\
                             Default is Gaussian.")

def add_fit_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("fit", "Fit options")
    group.add_argument("--numsigmaleft", type=float, default=1.5,
                       help="The number of sigma on the left of the peak to be used to define the\
             fitting range. Default is 1.5.")
    group.add_argument("--numsigmaright", type=float, default=1.5,
                       help="The number of sigma on the right of the peak to be used to define the\
             fitting range. Default is 1.5.")
    group.add_argument("--xmin", type=float, default=float("-inf"),
                       help="The minimum value of the independent variable to fit. Default is\
                         -inf.")
    group.add_argument("--xmax", type=float, default=float("inf"),
                       help="The maximum value of the independent variable to fit. Default is\
                         +inf.")
    group.add_argument("--absolutesigma", type=bool, default=True,
                       help="See the `curve_fit()` documentation for details. Default is True.")

def add_source_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("source", "Source options")
    group.add_argument("--e_peak", type=float, default=KALPHA,
                       help=f"Energy value of the main line to analyzed. The value is expressed in\
                         keV. Default is K-alpha of the Fe55. Default is {KALPHA:.3f}.")

def add_detector_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("detector", "Detector options")
    group.add_argument("--w", type=float, default=26.,
                       help="W-value of the gas inside the detector. Argon is 26 eV. Default is 26\
                          eV.")

def add_output_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("output", "Output options")
    group.add_argument("--plot", action="store_true",
                       help="Plot all the figures generated during the analysis.")
    group.add_argument("--save", action="store_true",
                       help=f"Save a log file and all the figures generated during the analysis \
                       inside the folder {(ANALYSIS_RESULTS)}. A .csv file with the data produced \
                       is also saved.")
