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

# Single source of truth for the package version: the Debian changelog's
# upstream version. Passed into the wheel build so the wheel version matches
# instead of falling back to setup.py's hardcoded default.
WHEEL_VERSION="$(head -n1 packaging/debian/changelog | sed -E 's/^[^(]*\(([0-9][^)-]*)-[^)]*\).*$/\1/')"

echo ">>> Building wheel + .deb for backend=${BACKEND} (version ${WHEEL_VERSION})"
docker build \
    --network=host \
    -f packaging/debian/build-helpers/Dockerfile.deb \
    --build-arg FLAGTREE_WHEEL_VERSION="${WHEEL_VERSION}" \
    --target deb-output \
    --output "type=local,dest=${REPO_ROOT}/dist" \
    .

echo ""
echo ">>> Output:"
ls -lh dist/output/ 2>/dev/null || ls -lh dist/
