#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check PTOAS PTO-op coverage against the LinxISA v0.57 PTO manifest.

PTOAS is an MLIR PTO dialect-to-EmitC compiler, not a Linx scalar assembler.
This check therefore validates dialect/EmitC surface coverage for PTO tile
operations and documents the intentionally external TMA/CUBE selector encoding
families owned by the Linx ISA golden catalog.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CANONICAL_ALIASES = {
    "TTRANSPOSE": "TTRANS",
    "TSORT": "TSORT32",
}

# These are selector-family entries in the Linx ISA manifest. PTOAS lowers the
# high-level dialect to PTO-ISA C++ APIs and does not encode BSTART.TMA/CUBE.
EXTERNAL_ENCODING_FAMILY = {
    "TLOAD",
    "TSTORE",
    "TMOV",
    "TPREFETCH",
    "MGATHER",
    "MSCATTER",
    "TMATMUL",
    "TMATMUL.ACC",
    "TMATMUL.BIAS",
    "TMATMULMX",
    "TMATMULMX.ACC",
    "TMATMULMX.BIAS",
    "TGEMV",
    "TGEMV.ACC",
    "TGEMV.BIAS",
    "TGEMVMX",
    "TGEMVMX.ACC",
    "TGEMVMX.BIAS",
    "ACCCVT",
}


def normalize(name: str) -> str:
    return name.replace(".", "").replace("_", "").upper()


def load_manifest(repo_root: Path) -> list[str]:
    manifest_path = repo_root / "isa/v0.57/state/pto_encoding_map.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["profile"] != "v0.57" or manifest["entry_count"] != 111:
        raise SystemExit(f"unexpected v0.57 manifest header in {manifest_path}")
    names = []
    for entry in manifest["entries"]:
        names.append(entry["canonical_name"])
    return names


def load_ptoas_ops(repo_root: Path) -> set[str]:
    ods_path = repo_root / "compiler/ptoas/include/PTO/IR/PTOOps.td"
    text = ods_path.read_text()
    mnemonics = re.findall(r'PTO_TOp<"([^"]+)"', text)
    return {normalize(mnemonic) for mnemonic in mnemonics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest_names = load_manifest(repo_root)
    ptoas_ops = load_ptoas_ops(repo_root)

    missing = []
    external = []
    covered = []
    for canonical_name in manifest_names:
        alias = CANONICAL_ALIASES.get(canonical_name, canonical_name)
        if normalize(alias) in ptoas_ops:
            covered.append(canonical_name)
        elif canonical_name in EXTERNAL_ENCODING_FAMILY:
            external.append(canonical_name)
        else:
            missing.append(canonical_name)

    if missing:
        print("PTOAS missing v0.57 PTO dialect coverage:")
        for name in missing:
            print(f"  - {name}")
        return 1

    print(
        "PTOAS v0.57 PTO manifest coverage OK: "
        f"{len(covered)} dialect ops, {len(external)} external encoding-family ops, "
        f"{len(manifest_names)} manifest entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
