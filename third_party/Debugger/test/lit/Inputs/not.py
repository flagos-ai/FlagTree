# SPDX-License-Identifier: MIT
"""Run a command and succeed only when that command fails."""

import subprocess
import sys


def main(argv):
    if len(argv) < 2:
        print("usage: not.py COMMAND [ARG ...]", file=sys.stderr)
        return 2
    return 0 if subprocess.run(argv[1:]).returncode != 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
