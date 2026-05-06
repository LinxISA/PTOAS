//===--------- IRTranslator.h ---- Graph Sync Solver ------------===//
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
#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <utility>

namespace mlir::pto::syncsolver {

class IRTranslator {
public:
  const SyncSolverOptions options;
  func::FuncOp funcOp;
  std::unique_ptr<OperationBase> funcIr;
  std::vector<std::unique_ptr<Occurrence>> syncIr;
  llvm::DenseSet<RWOperation *> unitFlagFeaturedOps;
  llvm::DenseMap<OperationBase *, std::vector<Occurrence *>> opAllOccurrences;
  std::vector<ProcessingOrder> processingOrders;
  llvm::DenseMap<Value, llvm::SmallVector<Value>> blockArgAliases;

  IRTranslator(func::FuncOp func, SyncSolverOptions options)
      : options(options), funcOp(func) {
    auto funcOp = std::make_unique<syncsolver::Function>(func.getOperation());
    auto scopeOp = funcIrBuilder(func.getRegion(), funcOp.get());
    funcOp->body.push_back(std::move(scopeOp));
    funcIr = std::move(funcOp);
    syncIrBuilder(funcIr.get());
  }

  IRTranslator(std::unique_ptr<OperationBase> funcIr, SyncSolverOptions options)
      : options(options), funcIr(std::move(funcIr)) {
    syncIrBuilder(this->funcIr.get());
  }

private:
  int64_t globalIndex{0};

  std::unique_ptr<Scope> funcIrBuilder(Region &region, OperationBase *parentOp,
                                       bool skipEmptyScopes = false);

  void generateProcessingOrders(Occurrence *occ1, Occurrence *occ2,
                                bool isUseless);
  void generateProcessingOrders(Loop *loopOp, Occurrence *occ, bool isUseless);
  void generateProcessingOrders(Scope *scopeOp, Occurrence *occ,
                                bool isUseless);
  void generateProcessingOrders(const llvm::SmallVector<Occurrence *> &occs,
                                bool isUseless);
  void generateProcessingOrders(const llvm::SmallVector<Occurrence *> &occs1,
                                const llvm::SmallVector<Occurrence *> &occs2,
                                bool isUseless);
  void generateProcessingOrders(RWOperation *rwOp1, RWOperation *rwOp2,
                                Occurrence *occ1, Occurrence *occ2,
                                bool isUseless);

  bool skipLaterIterations(Occurrence *occ1, Occurrence *occ2);

  void syncIrBuilder(OperationBase *op, Occurrence *parentOcc = nullptr,
                     int depth = 0, bool isUseless = false);

  llvm::SmallVector<Value> tracebackMemVals(Value val);
  llvm::SmallVector<Value> tracebackMemValsStep(Value val);
  llvm::SmallVector<Value> getMemoryOps(const SmallVector<Value> &vals);

  std::pair<llvm::SmallVector<Value>, llvm::SmallVector<Value>>
  getReadWriteMemoryOps(Operation *op);

  template <typename OP>
  std::unique_ptr<OperationBase> getLoadStoreOp(OP op, OperationBase *parentOp);

  std::unique_ptr<OperationBase> getPipeInterfaceOp(pto::OpPipeInterface op,
                                                    OperationBase *parentOp);

  std::unique_ptr<OperationBase> getTensorExtractOp(tensor::ExtractOp extractOp,
                                                    OperationBase *parentOp);

  std::unique_ptr<OperationBase> getCallOp(func::CallOp callOp,
                                           OperationBase *parentOp);

  void updateBlockArgAliases(Block *block, OperandRange destOperands);
  bool isUnlikelyCondition(Condition *condOp);
  bool isParallelLoop(Loop *loopOp);
  std::optional<int64_t> getLoopMultibufferUnrollNum(Loop *loopOp);
  std::optional<int64_t> getScopePreloadNum(Scope *scopeOp);
  std::optional<int64_t> getScopeMaxPreloadNum(Scope *scopeOp);
};

} // namespace mlir::pto::syncsolver

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H
