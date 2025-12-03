"""Trend CLI
"""
from aptapy.plotting import plt

from analysis.analyze import analyze_trend
from analysis.app import (
    add_detector_options,
    add_fit_options,
    add_foldername,
    add_output_options,
    add_singlemodel,
    add_source_options,
)
from analysis.fileio import load_class

__description__ = """
    Analyze a folder containing calibration pulse files and source data (spectrum) files. If
    multiple calibration files are present, the first in alphabetical order is taken. For each
    spectrum a fit of the emission line(s) is done using the given model. The gain and the
    resolution are calculated with the fit results and their trend with time and drift voltage
    is plotted."""


def run(args):
    # call your function with positional + keyword args
    analyze_trend(
        args.foldername,
        load_class(args.model),
        args.W,
        args.capacity,
        args.e_peak,
        # pass additional kwargs here if needed
        num_sigma_left=args.sigmaleft,
        num_sigma_right=args.sigmaright,
        xmin=args.xmin,
        xmax=args.xmax,
        absolute_sigma=args.absolutesigma,
        plot=args.plot,
        save=args.save
    )
    plt.show()

def register(subparsers):
    parser = subparsers.add_parser("trend", description=__description__)

    add_foldername(parser)
    add_singlemodel(parser)
    add_fit_options(parser)
    add_source_options(parser)
    add_detector_options(parser)
    add_output_options(parser)
    parser.set_defaults(func=run)
