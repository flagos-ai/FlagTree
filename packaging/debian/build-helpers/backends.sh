# Reader for packaging/backends.conf. Source it; do not execute it.
#
#   . "$(dirname "$0")/backends.sh"
#   backend_field metax vendor     # -> MetaX
#   backend_list                   # -> nvidia metax
#
# BACKENDS_CONF overrides the table location. The default works both in the
# repo (packaging/backends.conf) and inside the deb-assembler container, where
# debian/ and backends.conf keep the same relative layout under /build.

BACKENDS_CONF="${BACKENDS_CONF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/backends.conf}"

_backend_columns="backend arch wheel base python vendor excludes runtime"

# backend_field <backend> <column> -- print one field, or fail if the backend
# is not in the table. An empty field prints nothing and succeeds.
backend_field() {
    local backend="$1" column="$2" index=0 name
    for name in ${_backend_columns}; do
        index=$((index + 1))
        [ "${name}" = "${column}" ] && break
        name=""
    done
    if [ -z "${name}" ]; then
        echo "backend_field: unknown column '${column}'" >&2
        return 2
    fi
    awk -F'|' -v want="${backend}" -v col="${index}" '
        /^[[:space:]]*(#|$)/ { next }
        $1 == want { print $col; found = 1; exit }
        END { if (!found) exit 1 }
    ' "${BACKENDS_CONF}" || {
        echo "backend '${backend}' is not listed in ${BACKENDS_CONF}" >&2
        echo "known backends: $(backend_list)" >&2
        return 1
    }
}

# backend_list -- print every backend in table order, space separated.
backend_list() {
    awk -F'|' '/^[[:space:]]*(#|$)/ { next } { printf "%s%s", sep, $1; sep = " " } END { print "" }' \
        "${BACKENDS_CONF}"
}
