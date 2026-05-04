// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncSolverIRTranslator.h - HIVM-style IR builder ---------*- C++ -*-===//
//
// Builds a hierarchical SyncSolver IR (Function/Scope/Loop/Condition/RW/
// PlaceHolder) from a func::FuncOp and reuses InsertSync's IRTranslator for
// PTO-specific buffer alias tracing (Buffer2MemInfoMap). Then DFS-flattens
// it into a syncIr stream + a list of (occ1, occ2) processing orders.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>
#include <vector>

namespace mlir {
namespace pto {
namespace syncsolver {

class IRTranslator {
public:
  SyncSolverOptions options;
  func::FuncOp funcOp;

  // Hierarchical IR root (a Function wrapping a Scope per func::FuncOp).
  std::unique_ptr<OperationBase> funcIr;

  // Linearized DFS sequence (with each Loop body expanded twice for backward
  // dependency analysis).
  std::vector<std::unique_ptr<Occurrence>> syncIr;

  // op -> list of occurrences (one for each DFS visit).
  llvm::DenseMap<OperationBase *, std::vector<Occurrence *>> opAllOccurrences;

  // Pairs of occurrences the solver will examine.
  std::vector<ProcessingOrder> processingOrders;

  // PTO-specific alias tracking (reused from InsertSync).
  Buffer2MemInfoMap buffer2MemInfoMap;

  IRTranslator(func::FuncOp func, SyncSolverOptions options);

private:
  int64_t globalIndex{0};

  // Phase 1: build PTO Buffer2MemInfoMap by reusing the InsertSync translator
  // logic but only its `buffer2MemInfoMap_` side-effect (we discard its
  // SyncIR output and build our own hierarchical tree below).
  void buildBufferAliasMap();

  // Phase 2: build hierarchical OperationBase tree.
  std::unique_ptr<Scope> buildScopeFromRegion(Region &region,
                                              OperationBase *parentOp);
  std::unique_ptr<OperationBase>
  buildRWFromPipeOp(Operation *op, OperationBase *parentOp);

  // Resolve a Value to its participating BaseMemInfo* entries.
  void collectMemInfo(Value v,
                      llvm::SmallVector<const BaseMemInfo *> &out) const;
  std::pair<llvm::SmallVector<const BaseMemInfo *>,
            llvm::SmallVector<const BaseMemInfo *>>
  getReadWriteMemInfo(Operation *op) const;

  // Phase 3: DFS the hierarchical IR to build syncIr / processingOrders.
  void syncIrBuilder(OperationBase *op, Occurrence *parentOcc, int depth,
                     bool isUseless);

  // Helpers used while emitting processing orders.
  bool skipLaterIterations(Occurrence *occ1, Occurrence *occ2);
  void emitOrders(Occurrence *occ1, Occurrence *occ2, bool isUseless);
  void emitOrdersForList(const llvm::SmallVector<Occurrence *> &occs,
                         bool isUseless);
  void emitOrdersForCross(const llvm::SmallVector<Occurrence *> &occs1,
                          const llvm::SmallVector<Occurrence *> &occs2,
                          bool isUseless);
};

} // namespace syncsolver
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H
