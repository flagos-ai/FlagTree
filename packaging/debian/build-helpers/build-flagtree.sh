#!/usr/bin/env bash
# Build a FlagTree .deb for one backend.
#
# Usage:
#   ./packaging/debian/build-helpers/build-flagtree.sh [backend]
#
# The backend must be a row in packaging/backends.conf, which decides whether
# the wheel is compiled from this tree or downloaded from the index where the
# backend's delivery workflow published it.
#
# Output: ./dist/output/python3-flagtree-<backend>_*.deb
#
# Run from the FlagTree repo root (the one containing python/setup.py).

set -euo pipefail

BACKEND="${1:-nvidia}"

HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./backends.sh
. "${HELPERS_DIR}/backends.sh"

# Every per-backend fact comes from packaging/backends.conf; an unlisted
# backend fails here with the known ones named.
WHEEL_SOURCE="$(backend_field "$BACKEND" wheel)"
case "$WHEEL_SOURCE" in
    source) WHEEL_STAGE=wheel-builder ;;
    index)  WHEEL_STAGE=wheel-index ;;
    *)
        echo "ERROR: backend '$BACKEND' has wheel='$WHEEL_SOURCE'; expected source or index"
        exit 1
        ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p dist

# Build-environment knobs (see Dockerfile.deb):
#   DEB_BASE_IMAGE     the backend row's "base" (default) | e.g. ubuntu:22.04
#   PYTHON_VERSION     the backend row's "python" (default); empty = the base
#                      image's python3 | e.g. 3.12 (deadsnakes on 22.04)
#   MAX_JOBS           parallel compile jobs for the wheel (default 4; the MLIR
#                      build needs roughly 2-3 GB RAM per job)
#   DEB_VERSION_SUFFIX auto (+ubuntu<ver>) | "" | explicit suffix
# wheel=index backends only:
#   FLAGTREE_PYPI_INDEX  index to download the published wheel from
#   FLAGTREE_WHEEL_LOCAL exact local version to pin, e.g. metax3.6
# The base image and interpreter default to the backend's row; the CI matrix
# overrides them to build the same backend for another release line.
DEB_BASE_IMAGE="${DEB_BASE_IMAGE:-$(backend_field "$BACKEND" base)}"
PYTHON_VERSION="${PYTHON_VERSION:-$(backend_field "$BACKEND" python)}"
MAX_JOBS="${MAX_JOBS:-4}"
DEB_VERSION_SUFFIX="${DEB_VERSION_SUFFIX:-auto}"
SHLIBDEPS_EXCLUDE="${SHLIBDEPS_EXCLUDE:-$(backend_field "$BACKEND" excludes)}"
# Only consulted for wheel=index backends; see build-helpers/fetch-wheel.sh.
FLAGTREE_PYPI_INDEX="${FLAGTREE_PYPI_INDEX:-}"
FLAGTREE_WHEEL_LOCAL="${FLAGTREE_WHEEL_LOCAL:-}"

# Single source of truth for the package version: the Debian changelog's
# upstream version. Passed into the wheel build so the wheel version matches
# instead of falling back to setup.py's hardcoded default.
WHEEL_VERSION="$(head -n1 packaging/debian/changelog | sed -E 's/^[^(]*\(([0-9][^)-]*)-[^)]*\).*$/\1/')"
# Debian spells a pre-release 0.7.0~rc2; the wheel needs PEP 440, 0.7.0rc2.
WHEEL_VERSION="${WHEEL_VERSION//\~/}"

echo ">>> Building .deb for backend=${BACKEND} version ${WHEEL_VERSION} on ${DEB_BASE_IMAGE}"
echo ">>> wheel from ${WHEEL_SOURCE} (stage ${WHEEL_STAGE}), python ${PYTHON_VERSION:-default}, MAX_JOBS=${MAX_JOBS}"
docker build \
    --network=host \
    --build-arg DEB_BASE_IMAGE="${DEB_BASE_IMAGE}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    --build-arg DEB_VERSION_SUFFIX="${DEB_VERSION_SUFFIX}" \
    --build-arg FLAGTREE_BACKEND="${BACKEND}" \
    --build-arg FLAGTREE_WHEEL_STAGE="${WHEEL_STAGE}" \
    --build-arg FLAGTREE_PYPI_INDEX="${FLAGTREE_PYPI_INDEX}" \
    --build-arg FLAGTREE_WHEEL_LOCAL="${FLAGTREE_WHEEL_LOCAL}" \
    --build-arg SHLIBDEPS_EXCLUDE="${SHLIBDEPS_EXCLUDE}" \
    -f packaging/debian/build-helpers/Dockerfile.deb \
    --build-arg FLAGTREE_WHEEL_VERSION="${WHEEL_VERSION}" \
    --target deb-output \
    --output "type=local,dest=${REPO_ROOT}/dist" \
    .

echo ""
echo ">>> Output:"
ls -lh dist/output/ 2>/dev/null || ls -lh dist/
