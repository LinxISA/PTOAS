#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

: "${PTOBC_BIN:?PTOBC_BIN not set}"
: "${TEST_INPUT_DIR:?TEST_INPUT_DIR not set}"

OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_v0581_contract_out"}
mkdir -p "${OUT_DIR}"

input="${TEST_INPUT_DIR}/v0581_linx_contract.pto"
bytecode="${OUT_DIR}/v0581_linx_contract.ptobc"
decoded="${OUT_DIR}/v0581_linx_contract.roundtrip.pto"
"${PTOBC_BIN}" encode "${input}" -o "${bytecode}"
"${PTOBC_BIN}" decode "${bytecode}" -o "${decoded}"

grep -F "pto.timg2col ins(" "${decoded}" >/dev/null
grep -F "pto.tinsert ins(" "${decoded}" >/dev/null
grep -F "pto.tprefetch ins(" "${decoded}" >/dev/null
