#!/usr/bin/env bash
set -euo pipefail

: "${PTOAS_BIN:?PTOAS_BIN not set}"
: "${TILEOP_ROOT:?TILEOP_ROOT not set}"

CXX_BIN=${CXX_BIN:-c++}
EXPECTED_TILEOP_TREE=b1dba35bb63251e42c5bcdd8e8dfc9dd90b2fbad
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTUAL_TILEOP_TREE=$(git -C "${TILEOP_ROOT}" rev-parse 'HEAD^{tree}')
if [[ "${ACTUAL_TILEOP_TREE}" != "${EXPECTED_TILEOP_TREE}" ]]; then
  echo "TileOP tree mismatch: expected ${EXPECTED_TILEOP_TREE}, got ${ACTUAL_TILEOP_TREE}" >&2
  exit 1
fi

TILEOP_VARIANTS=${TILEOP_ROOT}/test/tileop_api/src/MXScaleVariants.cpp
for required_call in \
  'TMATMUL_MX(d, f16a, bf16b)' \
  'TMATMUL_MX(d, e4a, sa, f16b)' \
  'TMATMUL_MX(d, bf16a, e5b, sb)' \
  'TMATMUL_MX(d, e4a, sa, e5b, sb)' \
  'TGEMV_MX(d, bf16b, f16a)' \
  'TGEMV_MX(d, f16b, e4a, sa)' \
  'TGEMV_MX(d, e5b, sb, bf16a)' \
  'TGEMV_MX(d, scaled_b, sb, e4a, sa)'; do
  grep -F "${required_call}" "${TILEOP_VARIANTS}" >/dev/null
done

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ptoas-linx-mx-tileop.XXXXXX")
trap 'rm -rf "${TMP_DIR}"' EXIT

"${PTOAS_BIN}" --pto-arch=linx \
  "${SOURCE_ROOT}/test/lit/pto/v0583_linx_tmatmul_mx_variants.pto" \
  >"${TMP_DIR}/tmatmul.cpp"
"${PTOAS_BIN}" --pto-arch=linx \
  "${SOURCE_ROOT}/test/lit/pto/v0583_linx_tgemv_mx_optional_scales.pto" \
  >"${TMP_DIR}/tgemv.cpp"

for generated in "${TMP_DIR}/tmatmul.cpp" "${TMP_DIR}/tgemv.cpp"; do
  "${CXX_BIN}" -std=c++20 -D__linx \
    -include "${TILEOP_ROOT}/test/linx_host_type_shim.hpp" \
    -I "${SOURCE_ROOT}/test/compile_cpp/linx_mx_tileop_overlay" \
    -I "${TILEOP_ROOT}/include" \
    -fsyntax-only "${generated}"
done
