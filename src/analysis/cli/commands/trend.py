"""Trend CLI
"""
from aptapy.plotting import plt

from analysis.analyze import analyze_trend
from analysis.fileio import load_class

def run(args):
    # call your function with positional + keyword args
    analyze_trend(
        args.dirname,
        load_class(args.model),
        args.W,
        args.capacity,
        args.e_peak,
        # pass additional kwargs here if needed
        num_sigma_left=args.sigmaleft,
        num_sigma_right=args.sigmaright,
        xmin=args.xmin,
        xmax=args.xmax,
        plot=args.plot
    )
    plt.show()

def register(subparsers):
    parser = subparsers.add_parser(
        "trend",
        help="An example subcommand",
    )
    parser.add_argument(
        "dirname",
        help="Name of the directory of data to analyze.")
    parser.add_argument(
        "model",
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
        "--plot",
        action="store_true",
        help="")
    parser.set_defaults(func=run)