#!/usr/bin/env bash
# Build a FlagTree .rpm for one backend.
#
# Usage:
#   ./packaging/rpm/helpers/build-flagtree-rpm.sh [backend]

set -euo pipefail

BACKEND="${1:-nvidia}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p dist-rpm

IMG="flagtree-rpm-${BACKEND}:local"

echo ">>> Building wheel + .rpm for backend=${BACKEND} (image: ${IMG})"
docker build \
    --network=host \
    -f packaging/rpm/helpers/Dockerfile.rpm \
    --build-arg FLAGTREE_BACKEND="${BACKEND}" \
    --target rpm-output \
    --output "type=local,dest=${REPO_ROOT}/dist-rpm" \
    .

echo ""
echo ">>> Output:"
ls -lh dist-rpm/output/ 2>/dev/null || ls -lh dist-rpm/
