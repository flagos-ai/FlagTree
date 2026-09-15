#!/usr/bin/env bash
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

# Build-environment knobs (see Dockerfile.deb):
#   DEB_BASE_IMAGE     ubuntu:24.04 (default) | ubuntu:22.04
#   PYTHON_VERSION     empty = base image default | e.g. 3.12 (deadsnakes on 22.04)
#   MAX_JOBS           parallel compile jobs for the wheel (default 4; the MLIR
#                      build needs roughly 2-3 GB RAM per job)
#   DEB_VERSION_SUFFIX auto (+ubuntu<ver>) | "" | explicit suffix
DEB_BASE_IMAGE="${DEB_BASE_IMAGE:-ubuntu:24.04}"
PYTHON_VERSION="${PYTHON_VERSION:-}"
MAX_JOBS="${MAX_JOBS:-4}"
DEB_VERSION_SUFFIX="${DEB_VERSION_SUFFIX:-auto}"
echo ">>> Building wheel + .deb for backend=${BACKEND} on ${DEB_BASE_IMAGE} (python ${PYTHON_VERSION:-default}, MAX_JOBS=${MAX_JOBS})"
docker build \
    --network=host \
    --build-arg DEB_BASE_IMAGE="${DEB_BASE_IMAGE}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    --build-arg DEB_VERSION_SUFFIX="${DEB_VERSION_SUFFIX}" \
    -f packaging/debian/build-helpers/Dockerfile.deb \
    --target deb-output \
    --output "type=local,dest=${REPO_ROOT}/dist" \
    .

echo ""
echo ">>> Output:"
ls -lh dist/output/ 2>/dev/null || ls -lh dist/
