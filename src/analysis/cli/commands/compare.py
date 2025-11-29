"""Folder CLI
"""

import aptapy.models
from aptapy.plotting import plt

from analysis.analyze import compare_folders


def run(args):
    # call your function with positional + keyword args
    compare_folders(
        args.foldername,
        getattr(aptapy.models, args.model),
        args.W,
        args.capacity,
        # pass additional kwargs here if needed
        num_sigma_left=args.sigmaleft,
        num_sigma_right=args.sigmaright,
    )
    plt.show()

def register(subparsers):
    parser = subparsers.add_parser(
        "compare",
        help="An example subcommand",
    )
    parser.add_argument(
        "foldername",
        nargs="+",
        help="Name of the pulse file")
    parser.add_argument(
        "--model",
        default="Gaussian",
        help="Model to fit lines.")
    parser.add_argument(
        "--sigmaleft",
        type=float,
        default=1.5,
        help="Number of sigma to fit left.")
    parser.add_argument(
        "--sigmaright",
        type=float,
        default=1.5,
        help="Number of sigma to fit right.")
    parser.add_argument(
        "--W",
        type=float,
        default=26.,
        help="W-value of  the gas. Default is 26 eV for Argon.")
    parser.add_argument(
        "--capacity",
        type=float,
        default=1e-12,
        help="Value of the capacity of the circuit. Default to 1e-12 F.")

    parser.set_defaults(func=run)