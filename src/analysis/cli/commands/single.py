import argparse

from aptapy.plotting import plt

from analysis.analyze import analyze_file
from analysis.app import (
    add_detector_options,
    add_fit_options,
    add_multiplemodel,
    add_output_options,
    add_pulsefile,
    add_source_options,
    add_sourcefile,
)
from analysis.fileio import load_class

__description__ = """
    Analyze a calibration pulses file to determine the calibration parameters of the readout
    circuit. If a source data file (spectrum) is given, the emission line(s) is fitted using the
    given model. If multiple models are given, the fit is done with each model. """


def run(args):
    models_arg = args.model
    if isinstance(models_arg, str):
        models_arg = [models_arg]
    models = [load_class(m) for m in models_arg]

    analyze_file(
        args.pulsefile,
        args.sourcefile,
        models,
        args.W,
        args.capacity,
        args.e_peak,
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
    parser = subparsers.add_parser("single", description=__description__)

    add_pulsefile(parser)
    add_sourcefile(parser)
    add_multiplemodel(parser)
    add_fit_options(parser)
    add_source_options(parser)
    add_detector_options(parser)
    add_output_options(parser)

    parser.set_defaults(func=run)
