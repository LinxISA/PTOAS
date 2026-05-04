// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncSolverCodeGen.h - Emit pto sync ops ------------------*- C++ -*-===//
//
// Walks the chosen ConflictPairs and writes back `pto.set_flag`,
// `pto.wait_flag`, and `pto.barrier` ops at the right insertion points
// (PlaceHolders and direct ops). The emission supports a single event id per
// ConflictPair (no multi-buffer logic).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H

#include "PTO/Transforms/GraphSyncSolver/SyncSolver.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir {
namespace pto {
namespace syncsolver {

class CodeGenerator {
public:
  SyncSolverOptions options;
  func::FuncOp funcOp;
  std::unique_ptr<OperationBase> funcIr;
  std::vector<std::unique_ptr<ConflictPair>> chosenConflictedPairs;

  explicit CodeGenerator(std::unique_ptr<Solver> solver);

  // Emit the sync ops into the underlying MLIR func::FuncOp.
  void generateResultOps();

private:
  void emitOne(IRRewriter &rewriter, ConflictPair *cp);

  // Resolve the actual MLIR Operation* and a Location for an OperationBase,
  // accounting for PlaceHolders that point at a parent scope op.
  Operation *resolveSyncAnchor(OperationBase *opBase, bool insertAfter);
  Location resolveSyncLoc(OperationBase *opBase);

  // Insert a single set/wait/barrier op at the given anchor.
  void insertSetFlag(IRRewriter &rewriter, OperationBase *anchor,
                     PIPE setPipe, PIPE waitPipe, int64_t eventId,
                     bool insertAfter);
  void insertWaitFlag(IRRewriter &rewriter, OperationBase *anchor,
                      PIPE setPipe, PIPE waitPipe, int64_t eventId,
                      bool insertAfter);
  void insertBarrier(IRRewriter &rewriter, OperationBase *anchor, PIPE pipe,
                     bool insertAfter);
};

} // namespace syncsolver
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H
