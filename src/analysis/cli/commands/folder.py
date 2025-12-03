"""Folder CLI
"""

import argparse

from aptapy.plotting import plt

from analysis.analyze import analyze_folder
from analysis.app import (
    add_detector_options,
    add_fit_options,
    add_foldername,
    add_multiplemodel,
    add_output_options,
    add_source_options,
)
from analysis.fileio import load_class

__description__ =     """
    Analyze a folder containing calibration pulse files and source data (spectrum) files. If
    multiple calibration files are present, the first in alphabetical order is taken. For each
    spectrum a fit of the emission line(s) is done using the model(s) specified. If multiple models
    are given, the fit is done using both of them, and for each result the energy resolution and
    the gain is calculated."""

def run(args):
    # call your function with positional + keyword args
    models_arg = args.model
    if isinstance(models_arg, str):
        models_arg = [models_arg]  # make sure it's a list

    models = [load_class(m) for m in models_arg]
    analyze_folder(
        args.foldername,
        models,
        args.W,
        args.capacity,
        args.e_peak,
        # pass additional kwargs here if needed
        num_sigma_left=args.numsigmaleft,
        num_sigma_right=args.numsigmaright,
        xmin=args.xmin,
        xmax=args.xmax,
        absolute_sigma=args.absolutesigma,
        plot=args.plot,
        save=args.save
    )
    plt.show()

def register(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("folder", description=__description__)

    add_foldername(parser)
    add_multiplemodel(parser)
    add_fit_options(parser)
    add_source_options(parser)
    add_detector_options(parser)
    add_output_options(parser)

    parser.set_defaults(func=run)
