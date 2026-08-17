#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import re
import sys
from collections import Counter
from pathlib import Path


KNOWN_NON_V0_OPCODE_OPS = {
    "pto.talloc",
    "pto.talloc_to_aic",
    "pto.talloc_to_aiv",
    "pto.tconcatidx",
    "pto.tdeinterleave",
    "pto.tgemv.acc",
    "pto.tgemv.bias",
    "pto.tgemv.mx",
    "pto.tgemv.mx.acc",
    "pto.tgemv.mx.bias",
    "pto.tinterleave",
    "pto.tmatmul.acc",
    "pto.tmatmul.bias",
    "pto.tmatmul.mx.acc",
    "pto.tmatmul.mx.bias",
}

DELETED_OPCODE_ASSIGNMENTS = {
    "pto.taddc": 0x1008,
    "pto.taddsc": 0x100A,
    "pto.tfmod": 0x1026,
    "pto.tfmods": 0x1027,
    "pto.tlrelu": 0x1031,
    "pto.tsort32": 0x1063,
    "pto.tsubc": 0x1068,
    "pto.tsubsc": 0x106A,
}

NEW_V0571_OPCODE_ASSIGNMENTS = {
    "pto.acccvt": (0x109C, 1),
    "pto.mgather_mask": (0x109D, 4),
    "pto.mgather_cas": (0x109E, 5),
    "pto.mscatter_mask": (0x109F, 4),
    # `descending` is a BoolAttr serialized in the attribute dictionary, so
    # the PTO-BC SSA operand count is three even though the ISA has four roles.
    "pto.tsort": (0x10A0, 3),
}

NEW_V0580_OPCODE_ASSIGNMENTS = {
    "pto.gmov": (0x10A1, 3),
    "pto.tfma": (0x10A2, 4),
}

# PTO-BC serializes MLIR SSA operands, not duplicated architectural roles.
# TINSERT therefore has four bytecode operands (src, row, col, dst), while its
# five-role Linx contract and emitted call are dst, dst, src, row, col.
V0581_EXACT_BYTECODE_SSA_ARITY = {
    "pto.trowexpand": 2,
    "pto.tcolexpand": 2,
    "pto.tconcat": 3,
    "pto.timg2col": 4,
    "pto.tinsert": 4,
    "pto.tprefetch": 5,
}

EXACT_ARITY = {
    **{name: arity for name, (_, arity) in NEW_V0571_OPCODE_ASSIGNMENTS.items()},
    **{name: arity for name, (_, arity) in NEW_V0580_OPCODE_ASSIGNMENTS.items()},
    **V0581_EXACT_BYTECODE_SSA_ARITY,
}

PRE_V0571_MAX_PTO_OPCODE = 0x109B
PRE_V0580_MAX_PTO_OPCODE = 0x10A0


def parse_td_mnemonics(td_path: Path):
    td = td_path.read_text(encoding="utf-8", errors="ignore")
    mns = set(re.findall(r"\bmnemonic\s*=\s*\"([^\"]+)\"", td))
    mns.update(re.findall(r'PTO_TOp<"([^"]+)"', td))
    return {f"pto.{m}" for m in mns}


def find_duplicates(values):
    return sorted(value for value, count in Counter(values).items() if count > 1)


def parse_u8_token(token):
    constants = {
        "kHasVariant": 1,
        "kVariantDefault": 0,
        "kVariantAcc": 1,
        "kVariantBias": 2,
        "kVariantMx": 3,
        "kVariantMxAcc": 4,
        "kVariantMxBias": 5,
        "kSectionCubeVariant": 0,
        "kSectionVectorVariant": 1,
    }
    token = token.strip()
    return constants[token] if token in constants else int(token, 0)


def parse_header(h_path: Path):
    text = h_path.read_text(encoding="utf-8", errors="ignore")
    table_body = text.split("inline constexpr OpInfo kOpTable[] = {", 1)[1].split(
        "};", 1
    )[0]
    rows = [
        (int(opcode, 16), name, int(has_variant), int(num_operands))
        for opcode, name, has_variant, num_operands in re.findall(
            r'\{(0x[0-9A-Fa-f]+),\s*"([^"]+)",\s*(\d+),'
            r"\s*0x[0-9A-Fa-f]+,\s*0x[0-9A-Fa-f]+,\s*(\d+),",
            table_body,
        )
    ]

    reserved_body = text.split(
        "inline constexpr uint16_t kReservedPtoOpcodes[] = {", 1
    )[1].split("};", 1)[0]
    reserved = [int(value, 16) for value in re.findall(r"0x[0-9A-Fa-f]+", reserved_body)]

    simple_body = text.split(
        "inline std::optional<uint16_t> lookupOpcodeByName", 1
    )[1].split("inline const OpInfo *lookupByName", 1)[0]
    simple_cases = [
        (name, int(opcode, 16))
        for name, opcode in re.findall(
            r'\.Case\("([^"]+)",\s*(0x[0-9A-Fa-f]+)\)', simple_body
        )
    ]

    full_body = text.split(
        "inline std::optional<OpcodeAndVariant> lookupOpcodeAndVariantByFullName", 1
    )[1].split("inline std::optional<int> lookupOperandCountByFullName", 1)[0]
    full_cases = [
        (name, int(opcode, 16), int(has_variant), int(variant))
        for name, opcode, has_variant, variant in re.findall(
            r'\.Case\("([^"]+)",\s*OpcodeAndVariant\{'
            r"(0x[0-9A-Fa-f]+),\s*([^,}]+),\s*([^}]+)\}\)",
            full_body,
        )
        for has_variant, variant in [
            (parse_u8_token(has_variant), parse_u8_token(variant))
        ]
    ]
    return rows, reserved, simple_cases, full_cases


def check_header_contract(h_path: Path):
    rows, reserved, simple_cases, full_cases = parse_header(h_path)
    errors = []

    for label, values in (
        ("kOpTable opcode", [row[0] for row in rows]),
        ("kOpTable name", [row[1] for row in rows]),
        ("reserved opcode", reserved),
        ("lookupOpcodeByName name", [case[0] for case in simple_cases]),
        ("lookupOpcodeAndVariantByFullName name", [case[0] for case in full_cases]),
        (
            "lookupOpcodeAndVariantByFullName opcode/variant",
            [(case[1], case[2], case[3]) for case in full_cases],
        ),
    ):
        duplicates = find_duplicates(values)
        if duplicates:
            errors.append(f"duplicate {label}: {duplicates}")

    table_by_name = {name: (opcode, arity) for opcode, name, _, arity in rows}
    simple_by_name = dict(simple_cases)
    full_by_name = {name: (opcode, has_variant, variant) for name, opcode, has_variant, variant in full_cases}
    reserved_set = set(reserved)

    if reserved_set & {row[0] for row in rows}:
        errors.append("reserved PTO opcode appears in kOpTable")

    for name, opcode in DELETED_OPCODE_ASSIGNMENTS.items():
        if opcode not in reserved_set:
            errors.append(f"deleted opcode 0x{opcode:04X} ({name}) is not reserved")
        if name in table_by_name or name in simple_by_name or name in full_by_name:
            errors.append(f"deleted operation remains encodable: {name}")

    for name, (opcode, arity) in NEW_V0571_OPCODE_ASSIGNMENTS.items():
        if opcode <= PRE_V0571_MAX_PTO_OPCODE:
            errors.append(f"new operation {name} does not use a fresh opcode")
        if table_by_name.get(name) != (opcode, arity):
            errors.append(
                f"{name}: expected opcode/arity {(opcode, arity)}, "
                f"got {table_by_name.get(name)}"
            )

    for name, (opcode, arity) in NEW_V0580_OPCODE_ASSIGNMENTS.items():
        if opcode <= PRE_V0580_MAX_PTO_OPCODE:
            errors.append(f"new operation {name} does not use a fresh opcode")
        if table_by_name.get(name) != (opcode, arity):
            errors.append(
                f"{name} table contract must be opcode=0x{opcode:04X}, arity={arity}"
            )
        if simple_by_name.get(name) != opcode:
            errors.append(f"{name} lookupOpcodeByName mapping mismatch")
        if full_by_name.get(name) != (opcode, 0, 0):
            errors.append(f"{name} full-name opcode/variant mapping mismatch")

    for name, arity in EXACT_ARITY.items():
        if name not in table_by_name or table_by_name[name][1] != arity:
            errors.append(f"{name} must have exactly {arity} bytecode operands")

    for opcode, name, has_variant, _ in rows:
        if simple_by_name.get(name) != opcode:
            errors.append(f"{name} is inconsistent between kOpTable and name lookup")
        if not has_variant and full_by_name.get(name) != (opcode, 0, 0):
            errors.append(f"{name} is inconsistent in full-name lookup")

    return {name for _, name, _, _ in rows}, errors


def main():
    if len(sys.argv) != 3:
        logging.error("usage: %s <PTOOps.td> <ptobc_opcodes_v0.h>", sys.argv[0])
        return 2

    td_path = Path(sys.argv[1])
    h_path = Path(sys.argv[2])

    td_ops = parse_td_mnemonics(td_path)
    hdr_ops, header_errors = check_header_contract(h_path)

    if header_errors:
        logging.error("ptobc v0 opcode table contract violations:")
        for error in header_errors:
            logging.error("  - %s", error)
        return 1

    missing = sorted(
        op for op in td_ops if op not in hdr_ops and op not in KNOWN_NON_V0_OPCODE_OPS
    )
    if missing:
        logging.error("ptobc v0 opcode table is missing ops present in PTOOps.td:")
        for op in missing:
            logging.error("  - %s", op)
        logging.error(
            "Fix: extend docs/bytecode/tools/gen_v0_tables.py "
            "(or table source) and regenerate ptobc_opcodes_v0.h"
        )
        return 1

    logging.info(
        "OK: opcode coverage check passed (PTOOps.td ops=%d, table ops=%d)",
        len(td_ops),
        len(hdr_ops),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
