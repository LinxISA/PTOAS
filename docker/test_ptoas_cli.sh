#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Test ptoas CLI with sample files.
#
# Usage: ./test_ptoas_cli.sh
#
# Required environment variables:
#   PTO_SOURCE_DIR  - Path to PTO source directory
#   LLVM_BUILD_DIR  - Path to LLVM build directory
#   PTO_INSTALL_DIR - Path to PTO install directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDENTITY_CHECK_ONLY=0
if [[ "${1:-}" == "--check-ptoas-cli-identity" ]]; then
  if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 --check-ptoas-cli-identity <version-output> [product-version]" >&2
    exit 2
  fi
  IDENTITY_CHECK_ONLY=1
  VERSION_OUTPUT=$2
  PTOAS_VERSION=${3:-}
else

# Validate required environment variables
for var in PTO_SOURCE_DIR LLVM_BUILD_DIR PTO_INSTALL_DIR; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

# Setup environment
export PATH="${PTO_SOURCE_DIR}/build/tools/ptoas:${PATH}"
export PYTHONPATH="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${PTO_INSTALL_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${DYLD_LIBRARY_PATH}"

echo "Testing ptoas CLI..."
which ptoas

echo "Checking ptoas version..."
VERSION_OUTPUT="$(ptoas --version | tr -d '\r')"
fi

echo "$VERSION_OUTPUT"
bash "${SCRIPT_DIR}/check_ptoas_cli_identity.sh" "${VERSION_OUTPUT}" "${PTOAS_VERSION:-}"
if [[ "$IDENTITY_CHECK_ONLY" == 1 ]]; then
  exit 0
fi

# Test MatMul sample
echo "Testing MatMul sample..."
cd "${PTO_SOURCE_DIR}/test/samples/MatMul/"
python ./tmatmulk.py > ./tmatmulk.pto
ptoas ./tmatmulk.pto -o ./tmatmulk.cpp
echo "MatMul test passed"

# Test Abs sample
echo "Testing Abs sample..."
cd "${PTO_SOURCE_DIR}/test/samples/Abs/"
python ./abs.py > ./abs.pto
ptoas --enable-insert-sync ./abs.pto -o ./abs.cpp
echo "Abs test passed"

echo "All ptoas CLI tests passed!"
