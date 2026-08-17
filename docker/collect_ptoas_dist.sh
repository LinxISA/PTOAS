#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Collect ptoas binary and its dependencies into a self-contained distribution.
#
# Usage: ./collect_ptoas_dist.sh <output_directory>
#
# Required environment variables:
#   LLVM_BUILD_DIR  - Path to LLVM build directory
#   PTO_INSTALL_DIR - Path to PTO install directory
#   PTO_SOURCE_DIR  - Path to PTO source directory
#
# Output structure:
#   <output_directory>/
#     ptoas           - Wrapper script that sets up LD_LIBRARY_PATH
#     bin/ptoas       - The actual ptoas binary
#     lib/*.so*       - Required shared library dependencies

set -euo pipefail

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

if [ $# -lt 1 ]; then
  echo "Usage: $0 <output_directory>" >&2
  exit 1
fi

PTOAS_DIST_DIR="$1"

# Validate required environment variables
for var in LLVM_BUILD_DIR PTO_INSTALL_DIR PTO_SOURCE_DIR; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

PTOAS_BIN="${PTO_SOURCE_DIR}/build/tools/ptoas/ptoas"
PTOAS_DEPS_DIR="${PTOAS_DIST_DIR}/lib"

if [ ! -f "$PTOAS_BIN" ]; then
  echo "Error: ptoas binary not found at $PTOAS_BIN" >&2
  exit 1
fi

remove_rpath() {
  local path="$1"
  if ! has_rpath "$path"; then
    return
  fi
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --remove-rpath "$path"
  fi
  if has_rpath "$path" && command -v chrpath >/dev/null 2>&1; then
    chrpath -d "$path"
  fi
  if has_rpath "$path"; then
    echo "Error: failed to scrub RPATH/RUNPATH from ${path}" >&2
    exit 1
  fi
}

strip_symbols() {
  local path="$1"
  strip --strip-unneeded "$path"
}

has_rpath() {
  local path="$1"
  if command -v patchelf >/dev/null 2>&1; then
    local rpath_value
    rpath_value="$(patchelf --print-rpath "$path" 2>/dev/null || true)"
    [[ -n "$rpath_value" ]]
    return
  fi
  readelf -d "$path" 2>/dev/null | grep -Eq '(RPATH|RUNPATH)'
}

assert_relro() {
  local path="$1"
  if ! readelf -l "$path" 2>/dev/null | grep -q 'GNU_RELRO'; then
    echo "WARN: RELRO segment missing in ${path}" >&2
    return
  fi
  if ! readelf -d "$path" 2>/dev/null | grep -Eq '(BIND_NOW|FLAGS.*NOW|FLAGS_1.*NOW)'; then
    echo "WARN: NOW binding missing in ${path}" >&2
  fi
}

assert_no_symtab() {
  local path="$1"
  if readelf -S "$path" 2>/dev/null | grep -Eq '[[:space:]]\\.symtab[[:space:]]'; then
    echo "Error: symbol table still present in ${path}" >&2
    exit 1
  fi
}

assert_no_rpath() {
  local path="$1"
  if has_rpath "$path"; then
    echo "Error: runtime search path still present in ${path}" >&2
    exit 1
  fi
}

harden_elf() {
  local path="$1"
  remove_rpath "$path"
  strip_symbols "$path"
  assert_relro "$path"
  assert_no_symtab "$path"
  assert_no_rpath "$path"
}

# Create output directories
mkdir -p "${PTOAS_DIST_DIR}/bin" "${PTOAS_DEPS_DIR}"

# Copy ptoas binary
echo "Copying ptoas binary..."
cp "$PTOAS_BIN" "${PTOAS_DIST_DIR}/bin/"
harden_elf "${PTOAS_DIST_DIR}/bin/ptoas"

# Collect *.so dependencies (transitive closure under /llvm-workspace)
echo "Collecting shared library dependencies..."
copy_so() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local name
  name=$(basename "$f")
  [[ -f "${PTOAS_DEPS_DIR}/${name}" ]] && return 0
  cp -L -n "$f" "${PTOAS_DEPS_DIR}/" 2>/dev/null || true
  harden_elf "${PTOAS_DEPS_DIR}/${name}"
  while read -r res; do
    copy_so "$res"
  done < <(ldd "$f" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')
}

while read -r res; do
  copy_so "$res"
done < <(ldd "$PTOAS_BIN" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')

while read -r packaged; do
  harden_elf "$packaged"
done < <(find "${PTOAS_DIST_DIR}/bin" "${PTOAS_DEPS_DIR}" -type f | sort)

# Create wrapper script
echo "Creating wrapper script..."
cat > "${PTOAS_DIST_DIR}/ptoas" << 'WRAPPER_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH}"
exec "${SCRIPT_DIR}/bin/ptoas" "$@"
WRAPPER_EOF
chmod +x "${PTOAS_DIST_DIR}/ptoas"

echo "Smoke testing packaged ptoas dist..."
VERSION_OUTPUT="$(env -u PYTHONPATH -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  "${PTOAS_DIST_DIR}/ptoas" --version | tr -d '\r')"
fi

echo "$VERSION_OUTPUT"
bash "${SCRIPT_DIR}/check_ptoas_cli_identity.sh" "${VERSION_OUTPUT}" "${PTOAS_VERSION:-}"
if [[ "$IDENTITY_CHECK_ONLY" == 1 ]]; then
  exit 0
fi

# Show collected files
echo ""
echo "=== ptoas distribution contents ==="
ls -la "${PTOAS_DIST_DIR}/"
ls -la "${PTOAS_DIST_DIR}/bin/"
SO_COUNT=$(find "${PTOAS_DEPS_DIR}" -name "*.so*" 2>/dev/null | wc -l)
echo "=== Collected .so dependencies (${SO_COUNT} files) ==="
du -sh "${PTOAS_DEPS_DIR}/"
echo ""
echo "Distribution created at: ${PTOAS_DIST_DIR}"
