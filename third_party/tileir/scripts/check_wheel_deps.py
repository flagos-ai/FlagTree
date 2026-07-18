#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Sanity check that wheel contains tileir backend deps (tileiras, ptxas, nvvm).
Usage: python check_wheel_deps.py [path/to/wheel.whl]
  If no path given, uses the first .whl in dist/.
Exits 0 if all required names are found, 1 otherwise.
"""
import sys
import zipfile
from pathlib import Path

REQUIRED = [
    "tileiras",  # backend/bin/tileiras
    "ptxas",  # backend/bin/ptxas
    "libnvvm.so",  # backend/lib/nvvm/.../libnvvm.so
]


def main():
    if len(sys.argv) > 1:
        wheel_path = Path(sys.argv[1])
    else:
        dist = Path(__file__).resolve().parents[2] / ".." / ".." / "dist"
        dist = dist.resolve()
        whls = list(dist.glob("*.whl"))
        if not whls:
            print("No wheel found in dist/ and no path given.", file=sys.stderr)
            return 1
        wheel_path = whls[0]

    if not wheel_path.is_file():
        print(f"Not a file: {wheel_path}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(wheel_path) as z:
        names = z.namelist()

    # Normalize: we only care about the last path component for binaries/libs
    found = {r: False for r in REQUIRED}
    for name in names:
        base = name.rstrip("/").split("/")[-1]
        if base == "tileiras":
            found["tileiras"] = True
        elif base == "ptxas":
            found["ptxas"] = True
        elif "libnvvm.so" in name:
            found["libnvvm.so"] = True

    all_ok = all(found.values())
    for r in REQUIRED:
        status = "ok" if found[r] else "MISSING"
        print(f"  {r}: {status}")
    if not all_ok:
        print("Wheel is missing one or more tileir deps.", file=sys.stderr)
        return 1
    print("All tileir deps present in wheel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
