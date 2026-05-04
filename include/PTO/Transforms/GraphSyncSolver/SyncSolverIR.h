// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncSolverIR.h - GraphSyncSolver in-memory IR ------------*- C++ -*-===//
//
// In-memory hierarchical representation used by the GraphSyncSolver. It is a
// trimmed port of bishengir's SyncSolverIR: scopes / loops / conditions /
// place-holders + pipe-typed RWOperation leaves and Set/Wait/Barrier sync
// nodes. MmadL1 decomposition and unit-flag fields are intentionally absent.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIR_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <string>

namespace mlir {
namespace pto {
namespace syncsolver {

class OperationBase;
class Scope;

using Body = std::vector<std::unique_ptr<OperationBase>>;

enum struct OpType {
  PLACE_HOLDER,
  SCOPE,
  FUNCTION,
  FUNCTION_BLOCK,
  LOOP,
  LOOP_END,
  CONDITION,
  SCOPE_END,
  SYNC_OP,
  BARRIER_OP,
  SW_FLAG_OP,
  SET_FLAG_OP,
  WAIT_FLAG_OP,
  SW_FLAG_OP_END,
  SYNC_OP_END,
  RW_OPERATION,
  RW_OPERATION_END
};

class OperationBase {
private:
  static int globalIndex;

public:
  int id{-1};
  const OpType opType;
  Operation *op{nullptr};
  OperationBase *parentOp{nullptr};

  OperationBase() = delete;
  OperationBase(OpType opType, Operation *op, OperationBase *parentOp)
      : opType(opType), op(op), parentOp(parentOp) {
    id = globalIndex++;
  }
  virtual ~OperationBase() = default;

  static bool sameScope(OperationBase *a, OperationBase *b);
  int getDepth() const;
  OperationBase *getNthParent(int dist);
  static std::pair<OperationBase *, OperationBase *>
  getLCAPair(OperationBase *a, OperationBase *b);
  static OperationBase *getParentLoop(OperationBase *op);
  bool isProperAncestor(OperationBase *op);
  llvm::SmallVector<OperationBase *> getAllParents();

  template <typename Ty> Ty *getParentOfType() {
    OperationBase *cur = parentOp;
    while (cur != nullptr && !llvm::isa<Ty>(cur))
      cur = cur->parentOp;
    return llvm::dyn_cast_if_present<Ty>(cur);
  }
};

// Position anchor inside the IR tree. Used so the solver can refer to
// the gap before/after a loop, the start/end of a block, etc.
class PlaceHolder : public OperationBase {
public:
  Block *block{nullptr};
  OperationBase *beforeOp{nullptr};
  OperationBase *afterOp{nullptr};
  Scope *scopeBegin{nullptr};
  Scope *scopeEnd{nullptr};

  PlaceHolder(Operation *op, OperationBase *parentOp)
      : OperationBase(OpType::PLACE_HOLDER, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::PLACE_HOLDER;
  }
};

class Scope : public OperationBase {
public:
  Body body;

  Scope(OpType opType = OpType::SCOPE, Operation *op = nullptr,
        OperationBase *parentOp = nullptr)
      : OperationBase(opType, op, parentOp) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::SCOPE && e->opType < OpType::SCOPE_END;
  }
};

class FunctionBlock : public Scope {
public:
  FunctionBlock() : Scope(OpType::FUNCTION_BLOCK) {}
  static bool classof(const OperationBase *e) {
    return e->opType == OpType::FUNCTION_BLOCK;
  }
};

class Function : public Scope {
public:
  explicit Function(Operation *op) : Scope(OpType::FUNCTION, op, nullptr) {}
  static bool classof(const OperationBase *e) {
    return e->opType == OpType::FUNCTION;
  }
};

class Loop : public Scope {
public:
  // Mark an scf.for as parallel: backward-sync is unnecessary for its
  // cross-iteration dependencies (mirrors the HIVM `hivm.parallel_loop` attr).
  bool isParallel{false};

  Loop(Operation *op, OperationBase *parentOp)
      : Scope(OpType::LOOP, op, parentOp) {}
  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::LOOP && e->opType < OpType::LOOP_END;
  }
};

class Condition : public Scope {
public:
  Scope *trueScope{nullptr};
  Scope *falseScope{nullptr};

  Condition(Operation *op, OperationBase *parentOp,
            std::unique_ptr<Scope> trueScope, std::unique_ptr<Scope> falseScope)
      : Scope(OpType::CONDITION, op, parentOp) {
    if (trueScope)
      setTrueScope(std::move(trueScope));
    if (falseScope) {
      assert(this->trueScope != nullptr);
      setFalseScope(std::move(falseScope));
    }
  }

  void setTrueScope(std::unique_ptr<Scope> s) {
    s->parentOp = this;
    trueScope = s.get();
    body.push_back(std::move(s));
  }
  void setFalseScope(std::unique_ptr<Scope> s) {
    s->parentOp = this;
    falseScope = s.get();
    body.push_back(std::move(s));
  }
  bool hasFalseScope() const { return falseScope != nullptr; }
  Scope *getTrueScope() const { return trueScope; }
  Scope *getFalseScope() const { return falseScope; }

  static bool classof(const OperationBase *e) {
    return e->opType == OpType::CONDITION;
  }
};

// Pipe-typed read/write op. Reuses InsertSync's BaseMemInfo for the address
// model so we do not duplicate the alias-tracing logic.
class RWOperation : public OperationBase {
public:
  PIPE pipeRead{PIPE::PIPE_UNASSIGNED};
  PIPE pipeWrite{PIPE::PIPE_UNASSIGNED};

  // Pointers into Buffer2MemInfoMap entries owned by IRTranslator.
  llvm::SmallVector<const BaseMemInfo *> readMemInfo;
  llvm::SmallVector<const BaseMemInfo *> writeMemInfo;

  RWOperation(Operation *op, OperationBase *parentOp, PIPE pipeRead,
              PIPE pipeWrite,
              llvm::SmallVector<const BaseMemInfo *> readMemInfo,
              llvm::SmallVector<const BaseMemInfo *> writeMemInfo)
      : OperationBase(OpType::RW_OPERATION, op, parentOp), pipeRead(pipeRead),
        pipeWrite(pipeWrite), readMemInfo(std::move(readMemInfo)),
        writeMemInfo(std::move(writeMemInfo)) {}

  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::RW_OPERATION &&
           e->opType < OpType::RW_OPERATION_END;
  }
};

class SyncOp : public OperationBase {
public:
  SyncOp(OpType opType, Operation *op, OperationBase *parentOp)
      : OperationBase(opType, op, parentOp) {}
  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::SYNC_OP && e->opType < OpType::SYNC_OP_END;
  }
};

class SetWaitOp : public SyncOp {
public:
  llvm::SmallVector<int64_t> eventIds;
  PIPE pipeSrc{PIPE::PIPE_UNASSIGNED};
  PIPE pipeDst{PIPE::PIPE_UNASSIGNED};

  SetWaitOp(OpType t, Operation *op, OperationBase *parent,
            llvm::SmallVector<int64_t> eids, PIPE pSrc, PIPE pDst)
      : SyncOp(t, op, parent), eventIds(std::move(eids)), pipeSrc(pSrc),
        pipeDst(pDst) {}
  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::SW_FLAG_OP &&
           e->opType < OpType::SW_FLAG_OP_END;
  }
};

class SetFlagOp : public SetWaitOp {
public:
  SetFlagOp(Operation *op, OperationBase *parent,
            llvm::SmallVector<int64_t> eids, PIPE pSrc, PIPE pDst)
      : SetWaitOp(OpType::SET_FLAG_OP, op, parent, std::move(eids), pSrc,
                  pDst) {}
  static bool classof(const OperationBase *e) {
    return e->opType == OpType::SET_FLAG_OP;
  }
};

class WaitFlagOp : public SetWaitOp {
public:
  WaitFlagOp(Operation *op, OperationBase *parent,
             llvm::SmallVector<int64_t> eids, PIPE pSrc, PIPE pDst)
      : SetWaitOp(OpType::WAIT_FLAG_OP, op, parent, std::move(eids), pSrc,
                  pDst) {}
  static bool classof(const OperationBase *e) {
    return e->opType == OpType::WAIT_FLAG_OP;
  }
};

class BarrierOp : public SyncOp {
public:
  PIPE pipe{PIPE::PIPE_UNASSIGNED};
  BarrierOp(Operation *op, OperationBase *parent, PIPE pipe)
      : SyncOp(OpType::BARRIER_OP, op, parent), pipe(pipe) {}
  static bool classof(const OperationBase *e) {
    return e->opType == OpType::BARRIER_OP;
  }
};

} // namespace syncsolver
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIR_H
