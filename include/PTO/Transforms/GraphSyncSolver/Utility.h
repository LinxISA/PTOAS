// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- Utility.h - GraphSyncSolver shared types -----------------*- C++ -*-===//
//
// Minimal port of the AscendNPU-IR HIVM GraphSyncSolver utility layer:
//   * Occurrence       : flattened DFS instance of an OperationBase
//   * ConflictPair     : selected sync candidate (set/wait or barrier)
//   * EventIdNode      : graph-coloring node tracking allocated event ids
//   * CorePipeInfo     : (coreType, pipe) tuple key (intra-core only here)
//   * ProcessingOrder  : (occ1, occ2, rwOp1, rwOp2) item driving the solver
//   * SyncSolverOptions: minimal options struct
//
// Multi-buffer / unit-flag / cross-core fields from the upstream design are
// intentionally omitted to keep the first PTO port small.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_UTILITY_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_UTILITY_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include <climits>
#include <cstdint>
#include <tuple>
#include <utility>

namespace mlir {
namespace pto {
namespace syncsolver {

// PTO has 8 hardware EVENT IDs (EVENT_ID0..EVENT_ID7). We mirror the
// conservative reservation strategy used by PTO InsertSync's
// SyncEventIdAllocation so the two flows can stay numerically comparable.
constexpr int64_t kIntraCoreEventIdNum = 8;

struct CorePipeInfo {
  // Intra-core only: coreType is always CUBE_OR_VECTOR but kept for parity
  // with the upstream HIVM design and to ease future cross-core extension.
  TCoreType coreType{TCoreType::CUBE_OR_VECTOR};
  PIPE pipe{PIPE::PIPE_UNASSIGNED};

  CorePipeInfo() = default;
  CorePipeInfo(TCoreType coreType, PIPE pipe) : coreType(coreType), pipe(pipe) {}
  CorePipeInfo(std::pair<TCoreType, PIPE> p)
      : coreType(p.first), pipe(p.second) {}

  bool operator==(const CorePipeInfo &o) const {
    return coreType == o.coreType && pipe == o.pipe;
  }
  bool operator!=(const CorePipeInfo &o) const { return !(*this == o); }
  bool operator<(const CorePipeInfo &o) const {
    return std::tie(coreType, pipe) < std::tie(o.coreType, o.pipe);
  }
};

} // namespace syncsolver
} // namespace pto
} // namespace mlir

namespace llvm {
template <> struct DenseMapInfo<mlir::pto::syncsolver::CorePipeInfo> {
  using PairTy = std::pair<mlir::pto::TCoreType, mlir::pto::PIPE>;
  static inline mlir::pto::syncsolver::CorePipeInfo getEmptyKey() {
    return DenseMapInfo<PairTy>::getEmptyKey();
  }
  static inline mlir::pto::syncsolver::CorePipeInfo getTombstoneKey() {
    return DenseMapInfo<PairTy>::getTombstoneKey();
  }
  static unsigned getHashValue(const mlir::pto::syncsolver::CorePipeInfo &v) {
    return DenseMapInfo<PairTy>::getHashValue({v.coreType, v.pipe});
  }
  static bool isEqual(const mlir::pto::syncsolver::CorePipeInfo &a,
                      const mlir::pto::syncsolver::CorePipeInfo &b) {
    return a == b;
  }
};
} // namespace llvm

namespace mlir {
namespace pto {
namespace syncsolver {

struct SyncSolverOptions {
  // Currently single-mode: intra-core only. Field kept for future extension.
  bool isIntraCore{true};
  // Maximum coloring budget. For the minimal port we always use the full
  // hardware pool; falling back to barrier-all when this is exceeded.
  int64_t eventIdNumMax{kIntraCoreEventIdNum};
};

struct EventIdNode;
struct ConflictPair;

// One DFS appearance of an OperationBase in the syncIr stream.
struct Occurrence {
  OperationBase *op{nullptr};
  Occurrence *parentOcc{nullptr};
  int depth{-1};

  // Monotonic open/close timestamps used to construct half-open
  // [endIndex(set), startIndex(wait)) intervals for ConflictPair lifetimes.
  int startIndex{-1};
  int endIndex{-1};

  // Position of this Occurrence in the linear `syncIr` vector.
  int syncIrIndex{-1};
  int syncIrEndIndex{-1};

  // Index where the second "iteration" copy of a Loop's children begins;
  // used by skipLaterIterations.
  int loopSplitIndex{-1};

  llvm::SmallVector<Occurrence *> childOccs;

  Occurrence(OperationBase *op, Occurrence *parentOcc, int depth, int startIndex,
             int endIndex)
      : op(op), parentOcc(parentOcc), depth(depth), startIndex(startIndex),
        endIndex(endIndex) {}

  static int getDepth(Occurrence *occ);
  static bool sameScope(Occurrence *a, Occurrence *b);
  static std::pair<Occurrence *, Occurrence *> getLCAPair(Occurrence *a,
                                                          Occurrence *b);
  static Occurrence *getParentLoop(Occurrence *occ);

  Occurrence *getNthParent(int dist);
  Occurrence *getParentWithOp(OperationBase *op, bool assertExists = true);
  bool isProperAncestor(Occurrence *occ);
  llvm::SmallVector<Occurrence *> getAllParents();
};

struct ProcessingOrder {
  Occurrence *occ1{nullptr};
  Occurrence *occ2{nullptr};
  RWOperation *rwOp1{nullptr};
  RWOperation *rwOp2{nullptr};
  bool isUseless{false};
  ProcessingOrder(Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
                  RWOperation *rwOp2, bool isUseless)
      : occ1(occ1), occ2(occ2), rwOp1(rwOp1), rwOp2(rwOp2),
        isUseless(isUseless) {}
};

struct ConflictPair {
  static int globalIdCounter;

  const int id;
  RWOperation *const op1;          // producer RWOperation
  RWOperation *const op2;          // consumer RWOperation

  // Anchors where SetFlagOp / WaitFlagOp will eventually be inserted.
  OperationBase *setOp{nullptr};
  OperationBase *waitOp{nullptr};
  Occurrence *setOcc{nullptr};
  Occurrence *waitOcc{nullptr};

  const CorePipeInfo setCorePipeInfo;
  const CorePipeInfo waitCorePipeInfo;
  int startIndex{-1};
  int endIndex{-1};

  bool isUseless{false};
  bool isBarrierAll{false}; // fallback marker: emit pto.barrier <PIPE_ALL>

  EventIdNode *eventIdNode{nullptr};

  ConflictPair(RWOperation *op1, RWOperation *op2, OperationBase *setOp,
               OperationBase *waitOp, Occurrence *setOcc, Occurrence *waitOcc,
               CorePipeInfo setCorePipeInfo, CorePipeInfo waitCorePipeInfo,
               int startIndex, int endIndex)
      : id(globalIdCounter++), op1(op1), op2(op2), setOp(setOp), waitOp(waitOp),
        setOcc(setOcc), waitOcc(waitOcc), setCorePipeInfo(setCorePipeInfo),
        waitCorePipeInfo(waitCorePipeInfo), startIndex(startIndex),
        endIndex(endIndex) {}

  bool isBarrier() const { return setCorePipeInfo == waitCorePipeInfo; }
};

// Single coloring node in the EventIdSolver's interference graph.
// In the minimal port every node has eventIdNum == 1.
struct EventIdNode {
private:
  static int globalIdCounter;

public:
  const int64_t id{-1};
  ConflictPair *const initConflictPair;
  const int64_t eventIdNum;

private:
  llvm::SmallVector<int64_t> eventIds; // assigned colors
  llvm::DenseMap<ConflictPair *, int64_t> conflictPairs;

public:
  EventIdNode(ConflictPair *initConflictPair, int64_t eventIdNum)
      : id(globalIdCounter++), initConflictPair(initConflictPair),
        eventIdNum(eventIdNum) {
    insertConflictPair(initConflictPair);
  }

  void insertConflictPair(ConflictPair *cp) { conflictPairs[cp] += 1; }
  void eraseConflictPair(ConflictPair *cp) {
    if (--conflictPairs[cp] <= 0)
      conflictPairs.erase(cp);
  }

  const llvm::SmallVector<int64_t> &getEventIds() const { return eventIds; }
  void setEventIds(llvm::SmallVector<int64_t> v) { eventIds = std::move(v); }
};

// Half-open interval intersection helper.
bool checkRangesIntersect(int l1, int r1, int l2, int r2);

} // namespace syncsolver
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_UTILITY_H
