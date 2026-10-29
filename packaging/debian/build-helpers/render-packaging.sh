#!/usr/bin/env bash
# Render the backend-specific files in packaging/debian from the row for
# <backend> in packaging/backends.conf.
#
# Usage:
#   ./packaging/debian/build-helpers/render-packaging.sh <backend>
#
# Writes, next to this script's debian/ directory:
#   control                                        from control.in
#   python3-flagtree-<backend>.lintian-overrides   from lintian-overrides.in
#
# Both are build products; debian/control is in .gitignore. Dockerfile.deb
# runs this inside the deb-assembler stage, so a plain build never writes
# into the work tree.

set -euo pipefail

HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBIAN_DIR="$(dirname "${HELPERS_DIR}")"
# shellcheck source=./backends.sh
. "${HELPERS_DIR}/backends.sh"

if [ "$#" -ne 1 ]; then
    echo "usage: $(basename "$0") <backend>   (known: $(backend_list))" >&2
    exit 2
fi
BACKEND="$1"

arch="$(backend_field "${BACKEND}" arch)"
vendor="$(backend_field "${BACKEND}" vendor)"
runtime="$(backend_field "${BACKEND}" runtime)"
for field in arch vendor runtime; do
    if [ -z "${!field}" ]; then
        echo "ERROR: backend '${BACKEND}' has an empty '${field}' in ${BACKENDS_CONF}" >&2
        exit 1
    fi
done

# The leading comment block of control.in documents the template, not the
# source package; drop it so the rendered file starts with the Source stanza.
sed -e '/^Source:/,$!d' \
    -e "s|@BACKEND@|${BACKEND}|g" \
    -e "s|@ARCH@|${arch}|g" \
    -e "s|@VENDOR@|${vendor}|g" \
    -e "s|@RUNTIME@|${runtime}|g" \
    "${DEBIAN_DIR}/control.in" > "${DEBIAN_DIR}/control"

if grep -n '@[A-Z_]\+@' "${DEBIAN_DIR}/control"; then
    echo "ERROR: unfilled placeholder left in debian/control" >&2
    exit 1
fi

rm -f "${DEBIAN_DIR}"/python3-flagtree-*.lintian-overrides
cp "${DEBIAN_DIR}/lintian-overrides.in" \
   "${DEBIAN_DIR}/python3-flagtree-${BACKEND}.lintian-overrides"

echo "Rendered debian/control for python3-flagtree-${BACKEND} (${arch}, ${vendor})"
