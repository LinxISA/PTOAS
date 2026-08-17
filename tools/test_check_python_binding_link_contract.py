#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for the nanobind shared-runtime packaging contract."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(__file__).with_name("check_python_binding_link_contract.py")
VALID_PRODUCER = """
if(UNIX AND NOT APPLE)
  target_link_options(nanobind-mlir PRIVATE "LINKER:-z,undefs")
  install(TARGETS nanobind-mlir
    LIBRARY DESTINATION lib
  )
endif()
"""
VALID_DIALECT = """
from ._ods_common import get_op_result_or_value as _ods_get_op_result_or_value

def get_op_result_or_value(value):
    return _ods_get_op_result_or_value(value)
"""


class PythonBindingLinkContractTest(unittest.TestCase):
    def run_checker(
        self,
        cmake_source: str,
        workflow: str,
        dialect_source: str = VALID_DIALECT,
        sample_source: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cmake_path = root / "lib/Bindings/Python/CMakeLists.txt"
            cmake_path.parent.mkdir(parents=True)
            cmake_path.write_text(cmake_source)
            workflow_path = root / ".github/workflows/build_wheel.yml"
            workflow_path.parent.mkdir(parents=True)
            workflow_path.write_text(workflow)
            dialect_path = root / "python/pto/dialects/pto.py"
            dialect_path.parent.mkdir(parents=True)
            dialect_path.write_text(dialect_source)
            if sample_source is not None:
                sample_path = root / "test/samples/Example/example.py"
                sample_path.parent.mkdir(parents=True)
                sample_path.write_text(sample_source)
            return subprocess.run(
                ["python3", str(CHECKER), "--ptoas-root", str(root)],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_rejects_shared_runtime_not_installed_in_auditwheel_search_path(
        self,
    ) -> None:
        result = self.run_checker(
            """
if(UNIX AND NOT APPLE)
  target_link_options(nanobind-mlir PRIVATE "LINKER:-z,undefs")
endif()
""",
            "export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH\n",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("install nanobind-mlir into lib", result.stderr)

    def test_accepts_installed_runtime_and_matching_auditwheel_search_path(
        self,
    ) -> None:
        result = self.run_checker(
            VALID_PRODUCER,
            """
jobs:
  build:
    steps:
      - name: Repair wheel with auditwheel
        run: |
          export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
          auditwheel repair dist/ptoas.whl -w wheelhouse
""",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_valid_producer_with_missing_repair_step_search_path(self) -> None:
        result = self.run_checker(
            VALID_PRODUCER,
            """
jobs:
  build:
    steps:
      - name: Repair wheel with auditwheel
        run: auditwheel repair dist/ptoas.whl -w wheelhouse
""",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Repair wheel with auditwheel", result.stderr)

    def test_rejects_search_path_only_in_unrelated_comment_and_step(self) -> None:
        result = self.run_checker(
            VALID_PRODUCER,
            """
# $PTO_INSTALL_DIR/lib is documented here but not exported by repair.
jobs:
  build:
    steps:
      - name: Diagnose installed libraries
        run: export LD_LIBRARY_PATH=$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
      - name: Repair wheel with auditwheel
        run: auditwheel repair dist/ptoas.whl -w wheelhouse
""",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Repair wheel with auditwheel", result.stderr)

    def test_ci_does_not_override_nanobind_python_with_relative_executable(
        self,
    ) -> None:
        workflow = (CHECKER.parents[1] / ".github/workflows/ci.yml").read_text()
        self.assertIn("-DPython3_EXECUTABLE=python3", workflow)
        self.assertNotIn("-DPython_EXECUTABLE=python3", workflow)

    def test_rejects_removed_llvm23_generated_operand_helper(self) -> None:
        result = self.run_checker(
            VALID_PRODUCER,
            """
jobs:
  build:
    steps:
      - name: Repair wheel with auditwheel
        run: |
          export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
          auditwheel repair dist/ptoas.whl -w wheelhouse
""",
            """
def get_op_result_or_value(value):
    return _pto_ops_gen._get_op_result_or_value(value)
""",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stable MLIR operand conversion helper", result.stderr)

    def test_rejects_legacy_scf_if_keyword(self) -> None:
        result = self.run_checker(
            VALID_PRODUCER,
            """
jobs:
  build:
    steps:
      - name: Repair wheel with auditwheel
        run: |
          export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
          auditwheel repair dist/ptoas.whl -w wheelhouse
""",
            sample_source="branch = scf.IfOp(cond, [], hasElse = True)\n",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SCF IfOp uses has_else, not hasElse", result.stderr)


if __name__ == "__main__":
    unittest.main()
