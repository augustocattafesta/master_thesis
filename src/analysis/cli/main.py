import argparse
from .commands import trend, single


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
    # cmd_example.register(subparsers)
    trend.register(subparsers)
    single.register(subparsers)

    # Parse args
    args = parser.parse_args()

    # Call the handler associated with this subcommand
    args.func(args)