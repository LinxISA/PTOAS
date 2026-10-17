#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(__file__).with_name("check_v0587_development_contract.py")
SPEC = importlib.util.spec_from_file_location(
    "check_v0587_development_contract", MODULE_PATH
)
CHECKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECKER)


class DevelopmentCapabilityTest(unittest.TestCase):
    def test_development_lock_is_fail_closed(self) -> None:
        capabilities = CHECKER.load_capabilities(ROOT)
        lock_path = ROOT / "tools" / capabilities["golden_lock"]
        lock = json.loads(lock_path.read_text())
        lock["source"]["commit"] = "0" * 40
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)
            tools = fake / "tools"
            tools.mkdir()
            (tools / capabilities["golden_lock"]).write_text(json.dumps(lock))
            with self.assertRaisesRegex(SystemExit, "unexpected source commit"):
                CHECKER.load_lock(fake, capabilities)

    def test_current_ptoas_boundary_is_explicit(self) -> None:
        capabilities = CHECKER.load_capabilities(ROOT)
        base = CHECKER.load_base_contracts(ROOT, capabilities)
        CHECKER.validate_ptoas_boundary(ROOT, capabilities, base)

    def test_unreviewed_golden_operation_cannot_become_available(self) -> None:
        capabilities = CHECKER.load_capabilities(ROOT)
        base = CHECKER.load_base_contracts(ROOT, capabilities)
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)
            for relative in (
                "include/PTO/IR/PTOOps.td",
                "lib/PTO/IR/PTO.cpp",
                "lib/PTO/Transforms/PTOToEmitC.cpp",
                "tools/ptoas/ptoas.cpp",
                "tools/ptobc/generated/ptobc_opcodes_v0.h",
            ):
                destination = fake / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text((ROOT / relative).read_text())
            ods = fake / "include/PTO/IR/PTOOps.td"
            ods.write_text(
                ods.read_text()
                + '\ndef UnsupportedGoldenOp : PTO_TOp<"tlea"> {\n'
                + '  let arguments = (ins AnyType:$src);\n}\n'
            )
            with self.assertRaisesRegex(SystemExit, "without reviewed semantics"):
                CHECKER.validate_ptoas_boundary(fake, capabilities, base)

    def test_retired_operation_must_be_rejected_by_linx_target(self) -> None:
        capabilities = CHECKER.load_capabilities(ROOT)
        base = CHECKER.load_base_contracts(ROOT, capabilities)
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)
            for relative in (
                "include/PTO/IR/PTOOps.td",
                "lib/PTO/IR/PTO.cpp",
                "lib/PTO/Transforms/PTOToEmitC.cpp",
                "tools/ptoas/ptoas.cpp",
                "tools/ptobc/generated/ptobc_opcodes_v0.h",
            ):
                destination = fake / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                text = (ROOT / relative).read_text()
                if relative == "lib/PTO/IR/PTO.cpp":
                    text = text.replace('"pto.tconcat",', '"pto.not_tconcat",', 1)
                destination.write_text(text)
            with self.assertRaisesRegex(SystemExit, "does not explicitly reject"):
                CHECKER.validate_ptoas_boundary(fake, capabilities, base)


if __name__ == "__main__":
    unittest.main()
