// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/MultiBuffer.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCVAUTOMARKMULTIBUFFER
#define GEN_PASS_DEF_PTOCVMARKPRELOADSCOPES
#define GEN_PASS_DEF_PTOINLINECVPRELOADSCOPES
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr llvm::StringLiteral kScopeIdAttr = "pto.cv.scope_id";
constexpr llvm::StringLiteral kGroupIdAttr = "pto.cv.group_id";
constexpr llvm::StringLiteral kRoleAttr = "pto.cv.role";
constexpr llvm::StringLiteral kCoreAttr = "pto.cv.core";
constexpr llvm::StringLiteral kPreloadNumAttr = "pto.cv.preload_num";
constexpr llvm::StringLiteral kMaxPreloadNumAttr = "pto.cv.max_preload_num";
constexpr llvm::StringLiteral kInputPipeAttr = "pto.cv.input_pipe";
constexpr llvm::StringLiteral kOutputPipeAttr = "pto.cv.output_pipe";

enum class CoreKind { Cube, Vector };
enum class PipeDir { C2V, V2C };

struct PipeKey {
  int64_t id = 0;
  PipeDir dir = PipeDir::C2V;

  bool operator<(const PipeKey &other) const {
    return std::tie(dir, id) < std::tie(other.dir, other.id);
  }

  bool operator==(const PipeKey &other) const {
    return id == other.id && dir == other.dir;
  }
};

static std::string stringify(CoreKind core) {
  return core == CoreKind::Cube ? "cube" : "vector";
}

static std::string stringify(PipeDir dir) {
  return dir == PipeDir::C2V ? "c2v" : "v2c";
}

static std::string stringify(const PipeKey &key) {
  return (llvm::Twine(stringify(key.dir)) + ":" + llvm::Twine(key.id)).str();
}

static std::string stringify(std::optional<PipeKey> key) {
  if (!key)
    return "";
  return stringify(*key);
}

struct PipeAction {
  Operation *op = nullptr;
  bool isTAlloc = false;
  bool isTPush = false;
  bool isTPop = false;
  bool isTFree = false;
  std::optional<PipeKey> input;
  std::optional<PipeKey> output;

  explicit operator bool() const { return op != nullptr; }
};

static PipeAction getPipeAction(Operation *op) {
  PipeAction action;
  action.op = op;

  if (auto alloc = dyn_cast<TAllocToAivOp>(op)) {
    action.isTAlloc = true;
    action.output = PipeKey{alloc.getId(), PipeDir::C2V};
    return action;
  }
  if (auto alloc = dyn_cast<TAllocToAicOp>(op)) {
    action.isTAlloc = true;
    action.output = PipeKey{alloc.getId(), PipeDir::V2C};
    return action;
  }
  if (auto push = dyn_cast<TPushToAivOp>(op)) {
    action.isTPush = true;
    action.output = PipeKey{push.getId(), PipeDir::C2V};
    return action;
  }
  if (auto push = dyn_cast<TPushToAicOp>(op)) {
    action.isTPush = true;
    action.output = PipeKey{push.getId(), PipeDir::V2C};
    return action;
  }
  if (auto pop = dyn_cast<TPopFromAicOp>(op)) {
    action.isTPop = true;
    action.input = PipeKey{pop.getId(), PipeDir::C2V};
    return action;
  }
  if (auto pop = dyn_cast<TPopFromAivOp>(op)) {
    action.isTPop = true;
    action.input = PipeKey{pop.getId(), PipeDir::V2C};
    return action;
  }
  if (auto free = dyn_cast<TFreeFromAicOp>(op)) {
    action.isTFree = true;
    action.input = PipeKey{free.getId(), PipeDir::C2V};
    return action;
  }
  if (auto free = dyn_cast<TFreeFromAivOp>(op)) {
    action.isTFree = true;
    action.input = PipeKey{free.getId(), PipeDir::V2C};
    return action;
  }

  action.op = nullptr;
  return action;
}

struct PendingScope {
  bool active = false;
  std::optional<PipeKey> input;
  std::optional<PipeKey> output;
  bool inputReleased = false;
  bool outputCommitted = false;
  SmallVector<Operation *> ops;
  Operation *firstOp = nullptr;
  Operation *lastOp = nullptr;

  void reset() {
    active = false;
    input.reset();
    output.reset();
    inputReleased = false;
    outputCommitted = false;
    ops.clear();
    firstOp = nullptr;
    lastOp = nullptr;
  }

  void start() {
    if (!active)
      active = true;
  }

  void include(Operation *op) {
    start();
    if (!firstOp)
      firstOp = op;
    lastOp = op;
    ops.push_back(op);
  }
};

struct ScopeInfo {
  int64_t id = -1;
  CoreKind core = CoreKind::Cube;
  std::optional<PipeKey> input;
  std::optional<PipeKey> output;
  SmallVector<Operation *> ops;
  Operation *firstOp = nullptr;
  Operation *lastOp = nullptr;
  int64_t classId = -1;
};

struct ScopeClass {
  CoreKind core = CoreKind::Cube;
  std::optional<PipeKey> input;
  std::optional<PipeKey> output;
  SmallVector<int64_t> scopeIds;
  int64_t groupId = -1;
  int64_t preloadNum = -1;
  int64_t maxPreloadNum = -1;
};

static std::string getRole(const ScopeInfo &scope) {
  if (scope.input && scope.output)
    return "relay";
  if (scope.output)
    return "producer";
  return "consumer";
}

static std::string getClassKey(CoreKind core, std::optional<PipeKey> input,
                               std::optional<PipeKey> output) {
  return (llvm::Twine(stringify(core)) + "|" + llvm::Twine(stringify(input)) +
          "|" + llvm::Twine(stringify(output)))
      .str();
}

static std::optional<CoreKind> getKernelCore(func::FuncOp funcOp) {
  auto kernelKindAttr = funcOp->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;

  switch (kernelKindAttr.getKernelKind()) {
  case FunctionKernelKind::Cube:
    return CoreKind::Cube;
  case FunctionKernelKind::Vector:
    return CoreKind::Vector;
  }
  llvm_unreachable("unexpected kernel kind");
}

static void collectScopeIfValid(PendingScope &pending, CoreKind core,
                                SmallVectorImpl<ScopeInfo> &scopes) {
  std::optional<PipeKey> committedOutput =
      pending.outputCommitted ? pending.output : std::nullopt;
  if (!pending.active || (!pending.input && !committedOutput)) {
    pending.reset();
    return;
  }

  ScopeInfo scope;
  scope.id = static_cast<int64_t>(scopes.size());
  scope.core = core;
  scope.input = pending.input;
  scope.output = committedOutput;
  scope.ops = pending.ops;
  scope.firstOp = pending.firstOp;
  scope.lastOp = pending.lastOp;
  scopes.push_back(std::move(scope));
  pending.reset();
}

static void includeRange(PendingScope &pending, Operation *first,
                         Operation *last) {
  if (!first || !last)
    return;

  for (Operation *op = first; op; op = op->getNextNode()) {
    if (pending.lastOp == op)
      return;
    pending.include(op);
    if (op == last)
      return;
  }
}

static void includeMissingThrough(PendingScope &pending, Operation *last) {
  if (!pending.active || !last || pending.lastOp == last)
    return;
  includeRange(pending, pending.lastOp ? pending.lastOp->getNextNode() : last,
               last);
}

static void collectScopesInFor(scf::ForOp forOp, CoreKind core,
                               SmallVectorImpl<ScopeInfo> &scopes) {
  PendingScope pending;
  Operation *segmentStart = &forOp.getBody()->front();
  Operation *terminator = forOp.getBody()->getTerminator();

  for (Operation &op : forOp.getBody()->without_terminator()) {
    PipeAction action = getPipeAction(&op);
    if (!action)
      continue;

    if (action.isTPop) {
      if (pending.active)
        includeMissingThrough(pending, op.getPrevNode());
      collectScopeIfValid(pending, core, scopes);
      pending.input = action.input;
      includeRange(pending, &op, &op);
      segmentStart = &op;
      continue;
    }

    if (action.isTAlloc) {
      if (pending.active && pending.output)
        collectScopeIfValid(pending, core, scopes);
      Operation *start =
          pending.active
              ? (pending.lastOp ? pending.lastOp->getNextNode() : &op)
              : (segmentStart && segmentStart != terminator ? segmentStart
                                                            : &op);
      includeRange(pending, start, &op);
      pending.output = action.output;
      continue;
    }

    if (action.isTPush) {
      pending.output = action.output;
      pending.outputCommitted = true;
      if (!pending.active) {
        Operation *start =
            segmentStart && segmentStart != terminator ? segmentStart : &op;
        includeRange(pending, start, &op);
      } else {
        includeMissingThrough(pending, &op);
      }
      if (!pending.input || pending.inputReleased) {
        collectScopeIfValid(pending, core, scopes);
        segmentStart = op.getNextNode();
      }
      continue;
    }

    if (action.isTFree && pending.active) {
      includeMissingThrough(pending, &op);
      pending.inputReleased = true;
      if (pending.outputCommitted) {
        collectScopeIfValid(pending, core, scopes);
        segmentStart = op.getNextNode();
      }
      continue;
    }
  }

  if (pending.active)
    includeMissingThrough(pending, terminator ? terminator->getPrevNode()
                                              : nullptr);
  collectScopeIfValid(pending, core, scopes);
}

static void buildScopeClasses(SmallVectorImpl<ScopeInfo> &scopes,
                              SmallVectorImpl<ScopeClass> &classes) {
  llvm::StringMap<int64_t> classByKey;
  for (ScopeInfo &scope : scopes) {
    std::string key = getClassKey(scope.core, scope.input, scope.output);
    auto [it, inserted] = classByKey.try_emplace(key, classes.size());
    if (inserted) {
      ScopeClass klass;
      klass.core = scope.core;
      klass.input = scope.input;
      klass.output = scope.output;
      classes.push_back(std::move(klass));
    }
    scope.classId = it->second;
    classes[scope.classId].scopeIds.push_back(scope.id);
  }
}

static void assignPreloadNumbers(SmallVectorImpl<ScopeClass> &classes) {
  std::map<PipeKey, int64_t> inputClass;
  for (auto [idx, klass] : llvm::enumerate(classes)) {
    if (klass.input)
      inputClass.try_emplace(*klass.input, static_cast<int64_t>(idx));
  }

  SmallVector<int64_t> next(classes.size(), -1);
  for (auto [idx, klass] : llvm::enumerate(classes)) {
    if (!klass.output)
      continue;
    auto it = inputClass.find(*klass.output);
    if (it == inputClass.end())
      continue;
    next[idx] = it->second;
  }

  int64_t nextGroupId = 0;
  for (auto [startIdx, klass] : llvm::enumerate(classes)) {
    if (klass.input || !klass.output)
      continue;

    SmallVector<int64_t> chain;
    std::set<int64_t> seen;
    int64_t cur = static_cast<int64_t>(startIdx);
    while (cur >= 0 && seen.insert(cur).second) {
      chain.push_back(cur);
      cur = next[cur];
    }

    if (chain.size() < 2)
      continue;
    if (classes[chain.back()].output)
      continue;

    int64_t maxPreloadNum = static_cast<int64_t>(chain.size());
    int64_t groupId = nextGroupId++;
    for (auto [stageIdx, classIdx] : llvm::enumerate(chain)) {
      ScopeClass &stage = classes[classIdx];
      stage.groupId = groupId;
      stage.maxPreloadNum = maxPreloadNum;
      stage.preloadNum = maxPreloadNum - 1 - static_cast<int64_t>(stageIdx);
    }
  }
}

static void collectCVScopes(ModuleOp moduleOp,
                            SmallVectorImpl<ScopeInfo> &scopes) {
  moduleOp.walk([&](func::FuncOp funcOp) {
    std::optional<CoreKind> core = getKernelCore(funcOp);
    if (!core)
      return;
    funcOp.walk(
        [&](scf::ForOp forOp) { collectScopesInFor(forOp, *core, scopes); });
  });
}

static int64_t clampMultiBufferNum(Operation *anchor, int64_t num) {
  if (num <= 1)
    return 0;
  if (num <= static_cast<int64_t>(kPtoMultiBufferMaxNum))
    return num;

  anchor->emitWarning()
      << "auto CV multi-buffer depth " << num << " exceeds maximum "
      << kPtoMultiBufferMaxNum << "; clamping";
  return static_cast<int64_t>(kPtoMultiBufferMaxNum);
}

static bool markMultiBuffer(Operation *op, int64_t num) {
  if (!op)
    return false;
  num = clampMultiBufferNum(op, num);
  if (num <= 1)
    return false;
  if (op->getAttr(kPtoMultiBufferAttrName))
    return false;

  OpBuilder builder(op->getContext());
  op->setAttr(kPtoMultiBufferAttrName, builder.getI32IntegerAttr(num));
  return true;
}

static void collectRootAllocLikeOps(Value value,
                                    llvm::SmallPtrSetImpl<Operation *> &roots) {
  SmallVector<Value> worklist{value};
  llvm::DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    Operation *def = current.getDefiningOp();
    if (!def)
      continue;

    if (isa<AllocTileOp, memref::AllocOp>(def)) {
      roots.insert(def);
      continue;
    }

    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(def)) {
      worklist.push_back(viewLike.getViewSource());
      continue;
    }

    if (auto bind = dyn_cast<BindTileOp>(def)) {
      worklist.push_back(bind.getSource());
      continue;
    }
  }
}

static void markScopeLocalBuffers(const ScopeInfo &scope,
                                  const ScopeClass &klass) {
  if (klass.maxPreloadNum <= 1)
    return;

  llvm::SmallPtrSet<Operation *, 16> roots;
  for (Operation *op : scope.ops) {
    for (Value operand : op->getOperands())
      collectRootAllocLikeOps(operand, roots);
  }

  for (Operation *root : roots)
    markMultiBuffer(root, klass.maxPreloadNum);
}

static void markCVPreloadMultiBuffers(MutableArrayRef<ScopeInfo> scopes,
                                      ArrayRef<ScopeClass> classes) {
  for (const ScopeInfo &scope : scopes) {
    if (scope.classId < 0)
      continue;
    const ScopeClass &klass = classes[scope.classId];
    markScopeLocalBuffers(scope, klass);
  }
}

static bool isInsideMovedRange(Operation *op,
                               const llvm::DenseSet<Operation *> &moved) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (moved.contains(cur))
      return true;
  }
  return false;
}

static bool canWrapNoResultScope(const ScopeInfo &scope) {
  if (!scope.firstOp || !scope.lastOp)
    return false;
  if (scope.firstOp->getBlock() != scope.lastOp->getBlock())
    return false;

  llvm::DenseSet<Operation *> moved;
  for (Operation *op = scope.firstOp;; op = op->getNextNode()) {
    moved.insert(op);
    if (op == scope.lastOp)
      break;
  }

  for (Operation *op : moved) {
    for (Value result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (!isInsideMovedRange(use.getOwner(), moved))
          return false;
      }
    }
  }
  return true;
}

static CVScopeOp wrapScope(ScopeInfo &scope, const ScopeClass *klass,
                           MLIRContext *ctx) {
  Builder attrBuilder(ctx);
  OpBuilder builder(scope.firstOp);
  auto cvScope = builder.create<CVScopeOp>(scope.firstOp->getLoc());
  cvScope.getBody().push_back(new Block());

  cvScope->setAttr(kScopeIdAttr, attrBuilder.getI64IntegerAttr(scope.id));
  cvScope->setAttr(kRoleAttr, attrBuilder.getStringAttr(getRole(scope)));
  cvScope->setAttr(kCoreAttr, attrBuilder.getStringAttr(stringify(scope.core)));
  cvScope->setAttr(kInputPipeAttr,
                   attrBuilder.getStringAttr(stringify(scope.input)));
  cvScope->setAttr(kOutputPipeAttr,
                   attrBuilder.getStringAttr(stringify(scope.output)));

  if (klass && klass->groupId >= 0) {
    cvScope->setAttr(kGroupIdAttr,
                     attrBuilder.getI64IntegerAttr(klass->groupId));
    cvScope->setAttr(kPreloadNumAttr,
                     attrBuilder.getI64IntegerAttr(klass->preloadNum));
    cvScope->setAttr(kMaxPreloadNumAttr,
                     attrBuilder.getI64IntegerAttr(klass->maxPreloadNum));
  }

  Block &scopeBlock = cvScope.getBody().front();
  Block *parentBlock = cvScope->getBlock();
  auto begin = Block::iterator(scope.firstOp);
  auto end = std::next(Block::iterator(scope.lastOp));
  scopeBlock.getOperations().splice(scopeBlock.end(),
                                    parentBlock->getOperations(), begin, end);
  return cvScope;
}

static void createScopeOps(MutableArrayRef<ScopeInfo> scopes,
                           ArrayRef<ScopeClass> classes, MLIRContext *ctx) {
  for (ScopeInfo &scope : scopes) {
    const ScopeClass *klass =
        scope.classId >= 0 ? &classes[scope.classId] : nullptr;
    if (!canWrapNoResultScope(scope)) {
      if (scope.firstOp)
        scope.firstOp->emitWarning(
            "cannot wrap CV preload scope without results because a value "
            "defined in the scope is used outside; leave it unwrapped");
      continue;
    }
    wrapScope(scope, klass, ctx);
  }
}

static void inlineCVScope(CVScopeOp scopeOp) {
  Block &scopeBlock = scopeOp.getBody().front();
  Block *parentBlock = scopeOp->getBlock();
  parentBlock->getOperations().splice(Block::iterator(scopeOp),
                                      scopeBlock.getOperations(),
                                      scopeBlock.begin(), scopeBlock.end());
  scopeOp.erase();
}

struct PTOCVMarkPreloadScopesPass
    : public mlir::pto::impl::PTOCVMarkPreloadScopesBase<
          PTOCVMarkPreloadScopesPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<ScopeInfo> scopes;
    collectCVScopes(moduleOp, scopes);

    if (scopes.empty())
      return;

    SmallVector<ScopeClass> classes;
    buildScopeClasses(scopes, classes);
    assignPreloadNumbers(classes);
    createScopeOps(scopes, classes, moduleOp.getContext());
  }
};

struct PTOCVAutoMarkMultiBufferPass
    : public mlir::pto::impl::PTOCVAutoMarkMultiBufferBase<
          PTOCVAutoMarkMultiBufferPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<ScopeInfo> scopes;
    collectCVScopes(moduleOp, scopes);

    if (scopes.empty())
      return;

    SmallVector<ScopeClass> classes;
    buildScopeClasses(scopes, classes);
    assignPreloadNumbers(classes);
    markCVPreloadMultiBuffers(scopes, classes);
  }
};

struct PTOInlineCVPreloadScopesPass
    : public mlir::pto::impl::PTOInlineCVPreloadScopesBase<
          PTOInlineCVPreloadScopesPass> {
  void runOnOperation() override {
    SmallVector<CVScopeOp> scopes;
    getOperation()->walk([&](CVScopeOp scopeOp) {
      scopes.push_back(scopeOp);
    });

    for (CVScopeOp scopeOp : llvm::reverse(scopes))
      inlineCVScope(scopeOp);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOCVAutoMarkMultiBufferPass() {
  return std::make_unique<PTOCVAutoMarkMultiBufferPass>();
}

std::unique_ptr<Pass> mlir::pto::createPTOCVMarkPreloadScopesPass() {
  return std::make_unique<PTOCVMarkPreloadScopesPass>();
}

std::unique_ptr<Pass> mlir::pto::createPTOInlineCVPreloadScopesPass() {
  return std::make_unique<PTOInlineCVPreloadScopesPass>();
}
