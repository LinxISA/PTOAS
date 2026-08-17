#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Apply or audit the PTO-BC v0 schema delta for PTO ISA 0.58.1."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ARITY_UPDATES = {
    "pto.timg2col": (0x102C, 2, 4),
    "pto.tprefetch": (0x1045, 2, 5),
}


def transform(text: str) -> str:
    for name, (opcode, old_arity, new_arity) in ARITY_UPDATES.items():
        old = (
            f'{{0x{opcode:04X}, "{name}", 0, 0x00, 0x00, '
            f'{old_arity}, 0, 0, 0x00}}'
        )
        new = (
            f'{{0x{opcode:04X}, "{name}", 0, 0x00, 0x00, '
            f'{new_arity}, 0, 0, 0x00}}'
        )
        if old in text:
            text = text.replace(old, new, 1)
        elif new not in text:
            raise SystemExit(f"missing audited PTO-BC row for {name}")
    return text


def audit(text: str) -> None:
    rows = {
        name: (int(opcode, 16), int(arity))
        for opcode, name, arity in re.findall(
            r'\{(0x[0-9A-Fa-f]+),\s*"([^"]+)",\s*\d+,\s*'
            r'0x[0-9A-Fa-f]+,\s*0x[0-9A-Fa-f]+,\s*(\d+),',
            text,
        )
    }
    for name, (opcode, _old_arity, new_arity) in ARITY_UPDATES.items():
        if rows.get(name) != (opcode, new_arity):
            raise SystemExit(
                f"{name}: expected opcode/arity {(opcode, new_arity)}, got {rows.get(name)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("header", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    original = args.header.read_text()
    updated = transform(original)
    audit(updated)
    if args.write:
        args.header.write_text(updated)
    elif updated != original:
        raise SystemExit("PTO-BC header needs the audited 0.58.1 schema transformation")
    print("PTO-BC v0 0.58.1 schema audit OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
