#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("check_v058_pto_manifest.py")
SPEC = importlib.util.spec_from_file_location("check_v058_pto_manifest", MODULE_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECKER)


class LinxIdentityTest(unittest.TestCase):
    def test_local_0583_identity_mismatch_fails_closed(self):
        local_lock = json.loads(
            Path(__file__).with_name("pto_isa_v0_58_3_lock.json").read_text()
        )
        local_lock["encoding_projection_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            ptoas_root = Path(directory)
            tools = ptoas_root / "tools"
            tools.mkdir()
            (tools / "pto_isa_v0_58_3_lock.json").write_text(json.dumps(local_lock))
            with self.assertRaisesRegex(SystemExit, "unexpected PTO ISA 0.58.3 lock"):
                CHECKER.load_lock(ptoas_root)

    def test_mismatched_release_fails_closed(self):
        local_lock = json.loads(
            Path(__file__).with_name("pto_isa_v0_58_3_lock.json").read_text()
        )
        mismatched = json.loads(json.dumps(local_lock))
        mismatched["release"] = "0.58.0"
        with tempfile.TemporaryDirectory() as directory:
            linx_root = Path(directory)
            lock_path = linx_root / "isa/v0.58/pto-spec.lock.json"
            lock_path.parent.mkdir(parents=True)
            lock_path.write_text(json.dumps(mismatched))
            with self.assertRaisesRegex(SystemExit, "PTOAS/root PTO ISA lock mismatch"):
                CHECKER.validate_linx_identity(linx_root, local_lock)

    def test_effective_sfu_mapping_is_read_from_ods(self):
        ptoas_root = Path(__file__).resolve().parents[1]
        operations = CHECKER.load_ptoas_ops(ptoas_root)
        for mnemonic in ("TDIV", "TDIVS", "TREM", "TREMS"):
            self.assertEqual(operations[mnemonic]["linx_engine"], "SFU")

    def test_pipe_v_only_mapping_is_not_accepted_as_linx_engine(self):
        ptoas_root = Path(__file__).resolve().parents[1]
        ods = (ptoas_root / "include/PTO/IR/PTOOps.td").read_text()
        ods = ods.replace(
            "::mlir::pto::LinxEngine getLinxEngine() { return ::mlir::pto::LinxEngine::SFU; }",
            "",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory)
            fake_ods = fake_root / "include/PTO/IR/PTOOps.td"
            fake_ods.parent.mkdir(parents=True)
            fake_ods.write_text(ods)
            operations = CHECKER.load_ptoas_ops(fake_root)
            self.assertIsNone(operations["TDIV"]["linx_engine"])

    def test_linx_target_surface_has_0583_cube_call_contracts(self):
        ptoas_root = Path(__file__).resolve().parents[1]
        CHECKER.validate_linx_target_surface(ptoas_root)

    def test_mx_scales_are_independently_optional_in_ods(self):
        ptoas_root = Path(__file__).resolve().parents[1]
        operations = CHECKER.load_ptoas_ops(ptoas_root)
        for mnemonic in (
            "TGEMVMX",
            "TGEMVMXACC",
            "TGEMVMXBIAS",
            "TMATMULMX",
            "TMATMULMXACC",
            "TMATMULMXBIAS",
        ):
            self.assertEqual(
                operations[mnemonic]["optional_arguments"],
                ("a_scale", "b_scale"),
            )

    def test_linx_mx_compile_gate_uses_real_target_inputs(self):
        ptoas_root = Path(__file__).resolve().parents[1]
        gate = (ptoas_root / "tools/check_linx_mx_tileop_compile.sh").read_text()
        self.assertNotIn("linx_host_type_shim", gate)
        self.assertNotIn("linx_mx_tileop_overlay", gate)
        for required in (
            'LINX_CXX=${LINX_LLVM_BUILD}/bin/clang++',
            '"${TILEOP_ROOT}/include/jcore/template_asm.hpp"',
            '"${TILEOP_ROOT}/test/tileop_api/verify_pto_identity.py"',
            '--target=linx64-unknown-linux-musl',
            '-fsyntax-only',
            '-c "${generated}"',
            "EXPECTED_LLVM_COMMIT",
            "EXPECTED_LLVM_TREE",
            "EXPECTED_TILEOP_COMMIT",
            "EXPECTED_TILEOP_TREE",
            "bash ./compile.all link-smoke",
        ):
            self.assertIn(required, gate)

    def test_misrouted_matmul_mx_variant_fails_closed(self):
        ptoas_root = Path(__file__).resolve().parents[1]
        lowering = (ptoas_root / "lib/PTO/Transforms/PTOToEmitC.cpp").read_text()
        lowering, count = re.subn(
            r'(struct\s+PTOTMatmulMXToTMATMUL_MX\b.*?\n\s*'
            r'replaceOrEraseWithOpaqueCall\([^\n]*?)"TMATMUL_MX"',
            r'\1"TMATMUL_MX_ACC"',
            lowering,
            count=1,
            flags=re.DOTALL,
        )
        self.assertEqual(count, 1)
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory)
            for relative in (
                "tools/ptoas/ptoas.cpp",
                "tools/pto_isa_v0_58_3_operation_contracts.json",
            ):
                destination = fake_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text((ptoas_root / relative).read_text())
            destination = fake_root / "lib/PTO/Transforms/PTOToEmitC.cpp"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(lowering)
            with self.assertRaisesRegex(SystemExit, "PTOTMatmulMXToTMATMUL_MX"):
                CHECKER.validate_linx_target_surface(fake_root)

    def test_misordered_tgemv_mx_optional_scale_fails_closed(self):
        ptoas_root = Path(__file__).resolve().parents[1]
        lowering = (ptoas_root / "lib/PTO/Transforms/PTOToEmitC.cpp").read_text()
        lowering, count = re.subn(
            r"SmallVector<Value, 5> operands\{dst, b\};\s*"
            r"if \(bScale\)\s*operands\.push_back\(bScale\);\s*"
            r"operands\.push_back\(a\);",
            "SmallVector<Value, 5> operands{dst, b};\n"
            "    operands.push_back(a);\n"
            "    if (bScale) operands.push_back(bScale);",
            lowering,
            count=1,
        )
        self.assertEqual(count, 1)
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory)
            for relative in (
                "tools/ptoas/ptoas.cpp",
                "tools/pto_isa_v0_58_3_operation_contracts.json",
            ):
                destination = fake_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text((ptoas_root / relative).read_text())
            destination = fake_root / "lib/PTO/Transforms/PTOToEmitC.cpp"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(lowering)
            with self.assertRaisesRegex(SystemExit, "PTOTGemvMXToTGEMV_MX"):
                CHECKER.validate_linx_target_surface(fake_root)


if __name__ == "__main__":
    unittest.main()
