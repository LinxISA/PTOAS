#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <ptoas-version-output> [product-version]" >&2
  exit 2
fi

VERSION_OUTPUT=$1
PTOAS_VERSION=${2:-}
PTOAS_RELEASE_VERSION=0.41

if [[ "$VERSION_OUTPUT" == *$'\n'* || "$VERSION_OUTPUT" == *$'\r'* ]]; then
  echo "Error: ptoas --version must emit exactly one line" >&2
  exit 1
fi

if [[ -n "$PTOAS_VERSION" ]]; then
  EXPECTED_VERSION_OUTPUT="ptoas ${PTOAS_VERSION} (PTO ISA 0.58.1)"
  if [[ "$VERSION_OUTPUT" != "$EXPECTED_VERSION_OUTPUT" ]]; then
    echo "Error: expected '${EXPECTED_VERSION_OUTPUT}', got '${VERSION_OUTPUT}'" >&2
    exit 1
  fi
else
  EXPECTED_VERSION_OUTPUT="ptoas ${PTOAS_RELEASE_VERSION} (PTO ISA 0.58.1)"
  if [[ "$VERSION_OUTPUT" != "$EXPECTED_VERSION_OUTPUT" ]]; then
    echo "Error: invalid packaged ptoas identity '${VERSION_OUTPUT}'" >&2
    exit 1
  fi
fi
