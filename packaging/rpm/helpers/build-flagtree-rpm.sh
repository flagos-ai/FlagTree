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

# Build a FlagTree .rpm for one backend.
#
# Usage:
#   ./packaging/rpm/helpers/build-flagtree-rpm.sh [backend]

set -euo pipefail

BACKEND="${1:-nvidia}"

case "$BACKEND" in
    nvidia) ;;
    *)
        echo "ERROR: only 'nvidia' is supported in this revision (got '$BACKEND')"
        exit 1
        ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p dist-rpm
rm -rf dist-rpm/output

# Base image override, e.g. openEuler 24.03:
#   RPM_BASE_IMAGE=openeuler/openeuler:24.03-lts ./build-flagtree-rpm.sh
RPM_BASE_IMAGE="${RPM_BASE_IMAGE:-fedora:43}"

echo ">>> Building wheel + .rpm for backend=${BACKEND} on ${RPM_BASE_IMAGE}"
docker build \
    --network=host \
    -f packaging/rpm/helpers/Dockerfile.rpm \
    --build-arg RPM_BASE_IMAGE="${RPM_BASE_IMAGE}" \
    --target rpm-output \
    --output "type=local,dest=${REPO_ROOT}/dist-rpm" \
    .

echo ""
echo ">>> Output:"
ls -lh dist-rpm/output/ 2>/dev/null || ls -lh dist-rpm/
