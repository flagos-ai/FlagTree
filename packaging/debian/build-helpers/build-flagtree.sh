#!/usr/bin/env bash

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

# Build a FlagTree .deb for one backend.
#
# Usage:
#   ./packaging/debian/build-helpers/build-flagtree.sh [backend]
#
# Output: ./dist/output/python3-flagtree-<backend>_*.deb
#
# Run from the FlagTree repo root (the one containing python/setup.py).

set -euo pipefail

BACKEND="${1:-nvidia}"

case "$BACKEND" in
    nvidia)
        ;;
    mthreads|metax|amd|iluvatar|cambricon|hcu|xpu)
        echo "ERROR: backend '$BACKEND' is not yet wired up in Dockerfile.deb"
        echo "       (only 'nvidia' is supported in this revision)"
        exit 1
        ;;
    *)
        echo "ERROR: unknown backend '$BACKEND'"
        exit 1
        ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p dist

echo ">>> Building wheel + .deb for backend=${BACKEND}"
docker build \
    --network=host \
    -f packaging/debian/build-helpers/Dockerfile.deb \
    --target deb-output \
    --output "type=local,dest=${REPO_ROOT}/dist" \
    .

echo ""
echo ">>> Output:"
ls -lh dist/output/ 2>/dev/null || ls -lh dist/
