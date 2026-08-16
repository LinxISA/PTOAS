#!/usr/bin/env python3
"""Regression tests for the nanobind shared-runtime packaging contract."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(__file__).with_name("check_python_binding_link_contract.py")


class PythonBindingLinkContractTest(unittest.TestCase):
    def run_checker(self, cmake_source: str, workflow: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cmake_path = root / "lib/Bindings/Python/CMakeLists.txt"
            cmake_path.parent.mkdir(parents=True)
            cmake_path.write_text(cmake_source)
            workflow_path = root / ".github/workflows/build_wheel.yml"
            workflow_path.parent.mkdir(parents=True)
            workflow_path.write_text(workflow)
            return subprocess.run(
                ["python3", str(CHECKER), "--ptoas-root", str(root)],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_rejects_shared_runtime_not_installed_in_auditwheel_search_path(self) -> None:
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

    def test_accepts_installed_runtime_and_matching_auditwheel_search_path(self) -> None:
        result = self.run_checker(
            """
if(UNIX AND NOT APPLE)
  target_link_options(nanobind-mlir PRIVATE "LINKER:-z,undefs")
  install(TARGETS nanobind-mlir
    LIBRARY DESTINATION lib
  )
endif()
""",
            "export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
