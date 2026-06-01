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

echo ">>> Building wheel + .rpm for backend=${BACKEND}"
docker build \
    --network=host \
    -f packaging/rpm/helpers/Dockerfile.rpm \
    --target rpm-output \
    --output "type=local,dest=${REPO_ROOT}/dist-rpm" \
    .

echo ""
echo ">>> Output:"
ls -lh dist-rpm/output/ 2>/dev/null || ls -lh dist-rpm/
