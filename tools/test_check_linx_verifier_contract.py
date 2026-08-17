#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for the Linx decoded-representation verifier audit."""

from __future__ import annotations

import unittest

import check_linx_verifier_contract as contract


class LinxVerifierContractTest(unittest.TestCase):
    def test_finds_independent_representation_early_success_lambda(self) -> None:
        scanner = getattr(contract, "find_independent_early_success_bypasses", None)
        self.assertIsNotNone(scanner, "checker must expose the independent bypass audit")
        source = """
LogicalResult pto::ExampleOp::verify() {
  auto acceptDecoded = [&]() -> bool {
    return isa<MemRefType>(getSrc().getType()) ||
           getSrc().getDefiningOp<pto::BindTileOp>();
  };
  if (acceptDecoded())
    return success();
  return emitOpError("Linx legality is not implemented");
}
"""
        self.assertEqual(
            scanner(source),
            ["acceptDecoded"],
        )

    def test_finds_early_success_before_implicit_linx_fail_closed_dispatch(
        self,
    ) -> None:
        scanner = getattr(contract, "find_independent_early_success_bypasses", None)
        self.assertIsNotNone(scanner, "checker must expose the independent bypass audit")
        source = """
LogicalResult pto::TStoreFPOp::verify() {
  auto acceptDecoded = [&]() -> bool {
    return isa<MemRefType>(getDst().getType()) ||
           getDst().getDefiningOp<pto::BindTileOp>();
  };
  if (acceptDecoded())
    return success();
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
"""
        self.assertEqual(
            scanner(source),
            ["acceptDecoded"],
        )

    def test_ignores_representation_predicate_without_early_success(self) -> None:
        scanner = getattr(contract, "find_independent_early_success_bypasses", None)
        self.assertIsNotNone(scanner, "checker must expose the independent bypass audit")
        source = """
LogicalResult pto::ExampleOp::verify() {
  auto isDecoded = [&]() -> bool {
    return isa<MemRefType>(getSrc().getType());
  };
  if (isDecoded())
    return emitOpError("decoded form is unsupported");
  return success();
}
"""
        self.assertEqual(scanner(source), [])

    def test_finds_direct_representation_early_success(self) -> None:
        scanner = getattr(contract, "find_independent_early_success_bypasses", None)
        self.assertIsNotNone(scanner, "checker must expose the independent bypass audit")
        source = """
LogicalResult pto::ExampleOp::verify() {
  if (isa<MemRefType>(getSrc().getType()))
    return success();
  return emitOpError("Linx legality is not implemented");
}
"""
        findings = scanner(source)
        self.assertEqual(len(findings), 1)
        self.assertRegex(findings[0], r"^direct@\d+$")


if __name__ == "__main__":
    unittest.main()
