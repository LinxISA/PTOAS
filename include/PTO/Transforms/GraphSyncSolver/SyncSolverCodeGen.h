//===---------- SyncSolverCodeGen.h ---- Graph Sync Solver ----------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H

#include "PTO/Transforms/GraphSyncSolver/SyncSolver.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir::pto::syncsolver {

class CodeGenerator {
public:
  const SyncSolverOptions options;
  func::FuncOp funcOp;
  std::unique_ptr<OperationBase> funcIr;

private:
  SyncMap syncMapBefore, syncMapAfter;

public:
  CodeGenerator() = delete;

  explicit CodeGenerator(std::unique_ptr<Solver> solver)
      : options(solver->options) {
    auto [syncBefore, syncAfter] = solver->getBeforeAfterSyncMaps();
    syncMapBefore = std::move(syncBefore);
    syncMapAfter = std::move(syncAfter);
    funcOp = solver->funcOp;
    funcIr = std::move(solver->funcIr);
  }

  void generateResultOps();

private:
  Operation *resolveSyncAnchor(OperationBase *opBase);
  Location resolveSyncLoc(OperationBase *opBase);
  void setInsertionPoint(IRRewriter &rewriter, OperationBase *opBase,
                         bool insertAfter);
  void emitSyncOp(IRRewriter &rewriter, SyncOp *syncOp);
  void emitSyncMap(IRRewriter &rewriter, SyncMap &syncMap, bool insertAfter);
};

} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H
