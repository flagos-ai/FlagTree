#!/usr/bin/env bash
# Download the published FlagTree wheel for one backend from the FlagOS Python
# index, for backends whose "wheel" column in packaging/backends.conf is
# "index". Those wheels are built by the backend's own delivery workflow on
# vendor hardware, so packaging them needs neither a vendor SDK nor a vendor
# machine.
#
# Usage:
#   ./packaging/build-helpers/fetch-wheel.sh <backend> <version> <outdir>
#
# <version> is the upstream version without a local segment, e.g. 0.7.0; the
# published file carries the backend and chip generation as a PEP 440 local
# version, e.g. flagtree-0.7.0+metax3.6-cp312-cp312-linux_x86_64.whl. The
# highest local version matching this interpreter and architecture wins;
# FLAGTREE_WHEEL_LOCAL pins one explicitly (e.g. metax3.6).
#
# Environment:
#   FLAGTREE_PYPI_INDEX    index to read (default: the FlagOS hosted index)
#   FLAGTREE_WHEEL_LOCAL   exact local version to take instead of the highest
#
# The wheel is checked against the sha256 the index publishes in the link
# fragment, and the digest is echoed into the build log.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $(basename "$0") <backend> <version> <outdir>" >&2
    exit 2
fi
BACKEND="$1"
VERSION="$2"
OUTDIR="$3"

INDEX="${FLAGTREE_PYPI_INDEX:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"
INDEX="${INDEX%/}"

# Only a wheel for this interpreter can be packaged: debian/rules installs it
# with pip and derives the package's python dependency from the cp tag, and the
# smoke test imports it with this python3. A mismatch has to fail here, with
# the available tags named, rather than at import time on a user's machine.
py_tag="$(python3 -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
case "$(uname -m)" in
    x86_64)  plat_tag="linux_x86_64" ;;
    aarch64) plat_tag="linux_aarch64" ;;
    *)       echo "ERROR: no FlagTree wheels are published for $(uname -m)" >&2; exit 1 ;;
esac

echo ">>> Looking for flagtree ${VERSION}+${BACKEND}* ${py_tag} ${plat_tag} in ${INDEX}"
listing="$(mktemp)"
trap 'rm -f "${listing}"' EXIT
curl -fsSL --retry 3 --retry-delay 5 "${INDEX}/flagtree/" > "${listing}"

local_re="${FLAGTREE_WHEEL_LOCAL:-${BACKEND}[0-9.]*}"
# Each link is <a href="../../packages/flagtree/<ver>/<file>#sha256=<digest>" ...>.
# Sorting by -V puts the highest chip generation last (tsingmicro3.3 before
# tsingmicro3.6); the version itself is fixed by the caller.
match="$(grep -oE "\.\./\.\./packages/flagtree/[^\"#]+/flagtree-${VERSION}\+${local_re}-${py_tag}-${py_tag}-${plat_tag}\.whl#sha256=[0-9a-f]+" \
            "${listing}" | sort -V | tail -1 || true)"
if [ -z "${match}" ]; then
    echo "ERROR: no flagtree-${VERSION}+${BACKEND}*-${py_tag}-${py_tag}-${plat_tag}.whl in ${INDEX}" >&2
    echo "       published wheels for ${VERSION}:" >&2
    grep -oE "flagtree-${VERSION}[^\"<]*\.whl" "${listing}" | sort -u | sed 's/^/         /' >&2
    exit 1
fi

path="${match%#sha256=*}"
digest="${match##*#sha256=}"
file="$(basename "${path}")"
# The index is served under <index>/flagtree/, the packages one level above it.
url="${INDEX%/simple}/${path#../../}"

mkdir -p "${OUTDIR}"
echo ">>> ${file}"
echo ">>> ${url}"
curl -fsSL --retry 3 --retry-delay 5 -o "${OUTDIR}/${file}" "${url}"
echo "${digest}  ${OUTDIR}/${file}" | sha256sum -c -
ls -lh "${OUTDIR}/${file}"
