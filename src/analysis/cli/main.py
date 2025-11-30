
import argparse

from .commands import compare, folder, single, trend


def main():
    parser = argparse.ArgumentParser(
        prog="analysis",
        description="Command-line interface for pkg",
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        required=True,
    )

    # Register all commands here
    trend.register(subparsers)
    single.register(subparsers)
    folder.register(subparsers)
    compare.register(subparsers)
    # Parse args
    args = parser.parse_args()

    # Call the handler associated with this subcommand
    args.func(args)