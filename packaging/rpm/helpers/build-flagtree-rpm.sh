#!/usr/bin/env bash
# Build a FlagTree .rpm for one backend.
#
# Usage:
#   ./packaging/rpm/helpers/build-flagtree-rpm.sh [backend]
#
# The backend must be a row in packaging/backends.conf with an "rpm_wheel"
# column, which decides whether the wheel is compiled from this tree or
# downloaded from the index where the backend's delivery workflow published it.
#
# Output: ./dist-rpm/output/python3-flagtree-<backend>-*.rpm

set -euo pipefail

BACKEND="${1:-nvidia}"

HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../build-helpers" && pwd)"
# shellcheck source=../../build-helpers/backends.sh
. "${HELPERS_DIR}/backends.sh"

# Every per-backend fact comes from packaging/backends.conf; an unlisted
# backend fails here with the known ones named.
WHEEL_SOURCE="$(backend_field "$BACKEND" rpm_wheel)"
case "$WHEEL_SOURCE" in
    source) WHEEL_STAGE=wheel-builder ;;
    index)  WHEEL_STAGE=wheel-index ;;
    "")
        echo "ERROR: backend '$BACKEND' has an empty rpm_wheel column in packaging/backends.conf:"
        echo "       the rpm line does not package it (no distribution python3 matches its wheel)"
        exit 1
        ;;
    *)
        echo "ERROR: backend '$BACKEND' has rpm_wheel='$WHEEL_SOURCE'; expected source or index"
        exit 1
        ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p dist-rpm
rm -rf dist-rpm/output

# Build-environment knobs (see Dockerfile.rpm):
#   RPM_BASE_IMAGE  the backend row's "rpm_base" (default), e.g.
#                   openeuler/openeuler:24.03-lts
#   RPM_PLATFORM    docker --platform for a cross build, e.g. linux/arm64 to
#                   assemble the aarch64 rpm on an x86_64 host through qemu;
#                   empty = the host's own architecture
# rpm_wheel=index backends only:
#   FLAGTREE_PYPI_INDEX  index to download the published wheel from
#   FLAGTREE_WHEEL_LOCAL exact local version to pin, e.g. ascend3.5
RPM_BASE_IMAGE="${RPM_BASE_IMAGE:-$(backend_field "$BACKEND" rpm_base)}"
RPM_PLATFORM="${RPM_PLATFORM:-}"
FLAGTREE_ARCH="$(backend_field "$BACKEND" rpm_arch)"
FLAGTREE_SMOKE="${FLAGTREE_SMOKE:-$(backend_field "$BACKEND" smoke)}"
FLAGTREE_SMOKE="${FLAGTREE_SMOKE:-import}"
FLAGTREE_PYPI_INDEX="${FLAGTREE_PYPI_INDEX:-}"
FLAGTREE_WHEEL_LOCAL="${FLAGTREE_WHEEL_LOCAL:-}"

# The spec's ExclusiveArch is the row's rpm_arch, so rpmbuild refuses to start
# on a host of another architecture. Say so here instead, where the platform
# can still be named.
if [ -z "${RPM_PLATFORM}" ] && [ "$(uname -m)" != "${FLAGTREE_ARCH}" ]; then
    echo "ERROR: the ${BACKEND} rpm is ${FLAGTREE_ARCH}; this host is $(uname -m)"
    echo "       build it on a ${FLAGTREE_ARCH} machine, or emulate:"
    echo "       RPM_PLATFORM=linux/${FLAGTREE_ARCH/x86_64/amd64} $0 ${BACKEND}"
    exit 1
fi

# Single source of truth for the package version: the RPM spec. Passed into
# the wheel build so the wheel version matches instead of falling back to
# setup.py's hardcoded default, and used to pick the published wheel.
WHEEL_VERSION="$(awk '/^Version:/{print $2; exit}' packaging/rpm/specs/flagtree.spec)"

echo ">>> Building .rpm for backend=${BACKEND} version ${WHEEL_VERSION} on ${RPM_BASE_IMAGE}"
echo ">>> wheel from ${WHEEL_SOURCE} (stage ${WHEEL_STAGE}), arch ${FLAGTREE_ARCH}${RPM_PLATFORM:+, platform ${RPM_PLATFORM}}"
docker build \
    --network=host \
    ${RPM_PLATFORM:+--platform "${RPM_PLATFORM}"} \
    -f packaging/rpm/helpers/Dockerfile.rpm \
    --build-arg RPM_BASE_IMAGE="${RPM_BASE_IMAGE}" \
    --build-arg FLAGTREE_BACKEND="${BACKEND}" \
    --build-arg FLAGTREE_WHEEL_STAGE="${WHEEL_STAGE}" \
    --build-arg FLAGTREE_WHEEL_SOURCE="${WHEEL_SOURCE}" \
    --build-arg FLAGTREE_ARCH="${FLAGTREE_ARCH}" \
    --build-arg FLAGTREE_PYPI_INDEX="${FLAGTREE_PYPI_INDEX}" \
    --build-arg FLAGTREE_WHEEL_LOCAL="${FLAGTREE_WHEEL_LOCAL}" \
    --build-arg FLAGTREE_SMOKE="${FLAGTREE_SMOKE}" \
    --build-arg FLAGTREE_WHEEL_VERSION="${WHEEL_VERSION}" \
    --target rpm-output \
    --output "type=local,dest=${REPO_ROOT}/dist-rpm" \
    .

echo ""
echo ">>> Output:"
ls -lh dist-rpm/output/ 2>/dev/null || ls -lh dist-rpm/
