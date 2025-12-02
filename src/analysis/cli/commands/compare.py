"""Folder CLI
"""

from aptapy.plotting import plt

from analysis.analyze import compare_folders
from analysis.fileio import load_class

def run(args):
    # call your function with positional + keyword args
    compare_folders(
        args.foldernames,
        load_class(args.model),
        args.W,
        args.capacity,
        args.e_peak,
        # pass additional kwargs here if needed
        num_sigma_left=args.sigmaleft,
        num_sigma_right=args.sigmaright,
        xmin=args.xmin,
        xmax=args.xmax,
        save=args.save
    )
    plt.show()

def register(subparsers):
    parser = subparsers.add_parser(
        "compare",
        help="An example subcommand",
    )
    parser.add_argument(
        "foldernames",
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
    parser.add_argument(
        "--e_peak",
        type=float,
        default=5.9,
        help="Energy of the main peak in keV")
    parser.add_argument(
        "--xmin",
        type=float,
        default=float("-inf"),
        help="xmin.")
    parser.add_argument(
        "--xmax",
        type=float,
        default=float("inf"),
        help="xmax.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="")

    parser.set_defaults(func=run)