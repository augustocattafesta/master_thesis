"""Folder CLI
"""

from aptapy.plotting import plt

from analysis.analyze import compare_folders
from analysis.app import (
    add_detector_options,
    add_fit_options,
    add_foldernames,
    add_output_options,
    add_singlemodel,
    add_source_options,
    load_class,
)

__description__ = """
    Analyze the files in different folders and compare them. In particular, the gain and the
    energy resolution are calculated and plotted. The gain and the energy resolution are obtained
    with the script `analyze_folder`, using the model given to fit the emission line(s) in the
    spectrum."""


def run(args):
    # call your function with positional + keyword args
    compare_folders(
        args.foldernames,
        load_class(args.model),
        args.w,
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

def register(subparsers):
    parser = subparsers.add_parser("compare", description=__description__)

    add_foldernames(parser)
    add_singlemodel(parser)
    add_fit_options(parser)
    add_source_options(parser)
    add_detector_options(parser)
    add_output_options(parser)

    parser.set_defaults(func=run)
