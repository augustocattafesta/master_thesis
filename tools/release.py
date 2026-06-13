import argparse
import subprocess
from enum import Enum
from packaging.version import Version, parse


class BumpMode(str, Enum):

    """Small enum class describing the bump mode.
    """

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


def _cmd(*args, verbose=False) -> subprocess.CompletedProcess:
    """Run a command in a subprocess.
    """
    # print(f"Executing command \"{' '.join(args)}\"...")
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    if verbose:
        print(result.stdout)
    return result


def _cleanup() -> None:
    """Delete all auxiliary files and the files to ignore generated during the build process.
    """
    _cmd("git", "clean", "-fdX")


def _compile() -> None:
    """Compile the TeX source code.
    """
    _cmd("latexmk", "-pdf", "-interaction=nonstopmode", "latex/main.tex")


def _get_latest_tag() -> Version:
    """Get the latest tag from git.
    """
    try:
        # Fetch the tags to make sure we have the latest ones.
        _cmd("git", "fetch", "--tags")
        # Get the latest tag.
        tag = _cmd("git", "describe", "--tags", "--abbrev=0")
        tag_str = tag.stdout.strip().lstrip('v')
        # Parse the tag string into a Version object.
        return parse(tag_str)
    except (subprocess.CalledProcessError, ValueError):
        return parse("0.0.0")


def _bump_version(version: Version, mode: BumpMode) -> Version:
    """Bump the version string.
    """
    major, minor, micro = version.release
    if mode == BumpMode.MAJOR:
        version_string = f"{major + 1}.0.0"
    elif mode == BumpMode.MINOR:
        version_string = f"{major}.{minor + 1}.0"
    elif mode == BumpMode.PATCH:
        version_string = f"{major}.{minor}.{micro + 1}"
    else:
        raise ValueError(f"Invalid bump mode {mode}")
    return Version(version_string)


def release(mode: BumpMode) -> None:
    print("Cleaning up the build directory...\n")
    _cleanup()
    print("Compiling the TeX source code...\n")
    _compile()

    latest_tag = _get_latest_tag()
    new_tag = _bump_version(latest_tag, mode)
    # Push the code to GitHub.    
    msg = f"Prepare for tag release {new_tag}."
    
    # Push the code to GitHub.
    print(f"Pushing changes to GitHub.\n")
    _cmd("git", "add", ".")
    _cmd("git", "commit", "--allow-empty", "-a", "-m", msg)
    _cmd("git", "push")
    # Publish the release.
    _cmd("gh", "release", "create", f"v{new_tag}", "latex/main.pdf",
         "--title", f"Release v{new_tag}", "--generate-notes")
    print(f"Release v{new_tag} created successfully.\n")
    print("Cleaning up the build directory again...\n")
    _cleanup()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Release a new version of the PDF.")
    parser.add_argument("mode",
                        choices=[mode.value for mode in BumpMode],
                        help="The bump mode to use.")
    args = parser.parse_args()
    release(args.mode)