#!/usr/bin/env bash
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

# Single source of truth for the package version: the RPM spec. Passed into
# the wheel build so the wheel version matches instead of falling back to
# setup.py's hardcoded default.
WHEEL_VERSION="$(awk '/^Version:/{print $2; exit}' packaging/rpm/specs/flagtree.spec)"

echo ">>> Building wheel + .rpm for backend=${BACKEND} on ${RPM_BASE_IMAGE} (version ${WHEEL_VERSION})"
docker build \
    --network=host \
    -f packaging/rpm/helpers/Dockerfile.rpm \
    --build-arg RPM_BASE_IMAGE="${RPM_BASE_IMAGE}" \
    --build-arg FLAGTREE_WHEEL_VERSION="${WHEEL_VERSION}" \
    --target rpm-output \
    --output "type=local,dest=${REPO_ROOT}/dist-rpm" \
    .

echo ""
echo ">>> Output:"
ls -lh dist-rpm/output/ 2>/dev/null || ls -lh dist-rpm/
