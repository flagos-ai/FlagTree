#!/usr/bin/env bash
# Build a FlagTree .deb for one backend.
#
# Usage:
#   ./packaging/debian/helpers/build-flagtree.sh [backend]
#
# Output: ./dist/python3-flagtree-<backend>_*.deb
#
# Run from the FlagTree repo root (the one containing python/setup.py).

set -euo pipefail

BACKEND="${1:-nvidia}"

case "$BACKEND" in
    nvidia)
        ;;
    mthreads|metax|amd|iluvatar|cambricon|hcu|xpu)
        echo "WARN: backend '$BACKEND' has not been validated for packaging yet."
        ;;
    *)
        echo "ERROR: unknown backend '$BACKEND'"
        exit 1
        ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p dist

IMG="flagtree-deb-${BACKEND}:local"

echo ">>> Building wheel + .deb for backend=${BACKEND} (image: ${IMG})"
docker build \
    --network=host \
    -f packaging/debian/helpers/Dockerfile.deb \
    --build-arg FLAGTREE_BACKEND="${BACKEND}" \
    --target deb-output \
    --output "type=local,dest=${REPO_ROOT}/dist" \
    .

echo ""
echo ">>> Output:"
ls -lh dist/output/ 2>/dev/null || ls -lh dist/
