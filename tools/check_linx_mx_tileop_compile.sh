#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

: "${PTOAS_BIN:?PTOAS_BIN not set}"
: "${TILEOP_ROOT:?TILEOP_ROOT not set}"
: "${LINX_LLVM_BUILD:?LINX_LLVM_BUILD not set}"
: "${LINX_SYSROOT:?LINX_SYSROOT not set}"
: "${EXPECTED_LLVM_COMMIT:?EXPECTED_LLVM_COMMIT not set}"
: "${EXPECTED_LLVM_TREE:?EXPECTED_LLVM_TREE not set}"
: "${EXPECTED_TILEOP_COMMIT:?EXPECTED_TILEOP_COMMIT not set}"
: "${EXPECTED_TILEOP_TREE:?EXPECTED_TILEOP_TREE not set}"

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LINX_CXX=${LINX_LLVM_BUILD}/bin/clang++
LLVM_CACHE=${LINX_LLVM_BUILD}/CMakeCache.txt

for required_file in \
  "${PTOAS_BIN}" \
  "${LLVM_CACHE}" \
  "${LINX_SYSROOT}/include/c++/v1/cstdint" \
  "${TILEOP_ROOT}/include/jcore/template_asm.hpp" \
  "${TILEOP_ROOT}/test/tileop_api/verify_target_cxx_frontend.sh" \
  "${TILEOP_ROOT}/test/tileop_api/compile.all" \
  "${TILEOP_ROOT}/test/tileop_api/verify_pto_identity.py"; do
  if [[ ! -e "${required_file}" ]]; then
    echo "required integration input is missing: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -x "${PTOAS_BIN}" ]]; then
  echo "PTOAS_BIN must be executable" >&2
  exit 1
fi

LLVM_SOURCE_DIR=$(sed -n 's/^LLVM_SOURCE_DIR:STATIC=//p' "${LLVM_CACHE}")
if [[ -z "${LLVM_SOURCE_DIR}" ]] ||
   ! LLVM_REPO=$(git -C "${LLVM_SOURCE_DIR}" rev-parse --show-toplevel 2>/dev/null); then
  echo "LINX_LLVM_BUILD does not identify its LLVM source checkout" >&2
  exit 1
fi
ACTUAL_LLVM_COMMIT=$(git -C "${LLVM_REPO}" rev-parse HEAD)
ACTUAL_LLVM_TREE=$(git -C "${LLVM_REPO}" rev-parse 'HEAD^{tree}')
if [[ "${ACTUAL_LLVM_COMMIT}" != "${EXPECTED_LLVM_COMMIT}" ||
      "${ACTUAL_LLVM_TREE}" != "${EXPECTED_LLVM_TREE}" ]]; then
  echo "Linx LLVM identity mismatch: expected ${EXPECTED_LLVM_COMMIT}/${EXPECTED_LLVM_TREE}, got ${ACTUAL_LLVM_COMMIT}/${ACTUAL_LLVM_TREE}" >&2
  exit 1
fi

cmake --build "${LINX_LLVM_BUILD}" --target clang \
  --parallel "${PTOAS_LINX_BUILD_JOBS:-2}"
if [[ ! -x "${LINX_CXX}" ]]; then
  echo "exact Linx clang++ was not built at ${LINX_CXX}" >&2
  exit 1
fi

ACTUAL_TILEOP_COMMIT=$(git -C "${TILEOP_ROOT}" rev-parse HEAD)
ACTUAL_TILEOP_TREE=$(git -C "${TILEOP_ROOT}" rev-parse 'HEAD^{tree}')
if [[ "${ACTUAL_TILEOP_COMMIT}" != "${EXPECTED_TILEOP_COMMIT}" ||
      "${ACTUAL_TILEOP_TREE}" != "${EXPECTED_TILEOP_TREE}" ]]; then
  echo "TileOP identity mismatch: expected ${EXPECTED_TILEOP_COMMIT}/${EXPECTED_TILEOP_TREE}, got ${ACTUAL_TILEOP_COMMIT}/${ACTUAL_TILEOP_TREE}" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ptoas-linx-mx-tileop.XXXXXX")
trap 'rm -rf "${TMP_DIR}"' EXIT

"${PTOAS_BIN}" --pto-arch=linx \
  "${SOURCE_ROOT}/test/lit/pto/v0583_linx_tmatmul_mx_variants.pto" \
  >"${TMP_DIR}/tmatmul.cpp"
"${PTOAS_BIN}" --pto-arch=linx \
  "${SOURCE_ROOT}/test/lit/pto/v0583_linx_tgemv_mx_optional_scales.pto" \
  >"${TMP_DIR}/tgemv.cpp"

for callee in TMATMUL_MX TMATMUL_MX_ACC TMATMUL_MX_BIAS; do
  [[ $(grep -c "  ${callee}(" "${TMP_DIR}/tmatmul.cpp") -eq 4 ]]
done
for callee in TGEMV_MX TGEMV_MX_ACC TGEMV_MX_BIAS; do
  [[ $(grep -c "  ${callee}(" "${TMP_DIR}/tgemv.cpp") -eq 4 ]]
done

FLAGS=(
  --target=linx64-unknown-linux-musl
  --sysroot="${LINX_SYSROOT}"
  -nostdinc++
  -isystem "${LINX_SYSROOT}/include/c++/v1"
  -fenable-matrix
  -O2
  -std=c++20
  -D__linx
  -DENABLE_TENSOR_INSTR
  -Werror
  -I "${TILEOP_ROOT}/include"
)

for generated in "${TMP_DIR}/tmatmul.cpp" "${TMP_DIR}/tgemv.cpp"; do
  "${LINX_CXX}" "${FLAGS[@]}" -fsyntax-only "${generated}"
  object=${generated%.cpp}.o
  "${LINX_CXX}" "${FLAGS[@]}" -c "${generated}" -o "${object}"
  if [[ ! -s "${object}" ]]; then
    echo "Linx object gate produced no object: ${object}" >&2
    exit 1
  fi
done

TC_DIR=${LINX_LLVM_BUILD}/bin LINX_SYSROOT=${LINX_SYSROOT} \
  bash "${TILEOP_ROOT}/test/tileop_api/verify_target_cxx_frontend.sh"

(
  cd "${TILEOP_ROOT}/test/tileop_api"
  COMPILER_DIR=${LINX_LLVM_BUILD}/bin \
    LINX_SYSROOT=${LINX_SYSROOT} \
    LINX_TARGET=linx64-unknown-linux-musl \
    bash ./compile.all link-smoke
)
