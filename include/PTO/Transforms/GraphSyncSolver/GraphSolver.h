// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- GraphSolver.h - Pipe reachability via Dijkstra -----------*- C++ -*-===//
//
// Lightweight Dijkstra-style search across (corePipeSrc, corePipeDst) edges
// labeled with [startIndex, endIndex] active intervals. Used by the solver to
// decide whether a candidate sync pair is already covered transitively by
// existing chosen pairs. Unit-flag variants from the upstream design are
// omitted.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H

#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
#include <set>

namespace mlir {
namespace pto {
namespace syncsolver {

class GraphSolver {
public:
  struct Edge {
    ConflictPair *const conflictPair;
    const CorePipeInfo corePipeSrc;
    const CorePipeInfo corePipeDst;
    const int startIndex;
    const int endIndex;
    Edge() = delete;
    Edge(ConflictPair *cp, CorePipeInfo s, CorePipeInfo d, int sIdx, int eIdx)
        : conflictPair(cp), corePipeSrc(s), corePipeDst(d), startIndex(sIdx),
          endIndex(eIdx) {}
    bool operator<(const Edge &o) const;
  };

  llvm::DenseMap<CorePipeInfo, llvm::DenseMap<CorePipeInfo, std::set<Edge>>>
      adjacencyList;

  void addPair(ConflictPair *cp, CorePipeInfo src, CorePipeInfo dst, int sIdx,
               int eIdx);
  void addConflictPair(ConflictPair *cp);

  // Returns the minimum reachable endIndex on `corePipeDst` if the search
  // can reach there within [startIndex, endIndex]; std::nullopt otherwise.
  std::optional<int> runDijkstra(CorePipeInfo corePipeSrc,
                                 CorePipeInfo corePipeDst, int startIndex,
                                 int endIndex);
};

} // namespace syncsolver
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H
