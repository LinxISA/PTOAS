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

OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_v0571_ops_out"}
mkdir -p "${OUT_DIR}"

roundtrip() {
  local stem=$1
  local input="${TEST_INPUT_DIR}/${stem}.pto"
  local bytecode="${OUT_DIR}/${stem}.ptobc"
  local decoded="${OUT_DIR}/${stem}.roundtrip.pto"
  "${PTOBC_BIN}" encode "${input}" -o "${bytecode}"
  "${PTOBC_BIN}" decode "${bytecode}" -o "${decoded}"
}

roundtrip tsort_emitc
roundtrip v0571_direct_ops_emitc

grep -F "pto.tsort ins(" "${OUT_DIR}/tsort_emitc.roundtrip.pto" >/dev/null
grep -F "descending = true" "${OUT_DIR}/tsort_emitc.roundtrip.pto" >/dev/null
grep -F "pto.mgather_mask ins(" "${OUT_DIR}/v0571_direct_ops_emitc.roundtrip.pto" >/dev/null
grep -F "pto.mgather_cas ins(" "${OUT_DIR}/v0571_direct_ops_emitc.roundtrip.pto" >/dev/null
grep -F "pto.mscatter_mask ins(" "${OUT_DIR}/v0571_direct_ops_emitc.roundtrip.pto" >/dev/null
grep -F "pto.acccvt outs(" "${OUT_DIR}/v0571_direct_ops_emitc.roundtrip.pto" >/dev/null
