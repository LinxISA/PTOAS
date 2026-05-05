// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCVCREATEPRELOAD
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr llvm::StringLiteral kPreloadNumAttr = "pto.cv.preload_num";
constexpr llvm::StringLiteral kMaxPreloadNumAttr = "pto.cv.max_preload_num";
constexpr llvm::StringLiteral kGroupIdAttr = "pto.cv.group_id";

struct LocalBufferBinding {
  Value value;
  PointerCastOp pointerCast;
  BindTileOp bindTile;
};

static std::optional<int64_t> getI64Attr(Operation *op,
                                         llvm::StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static bool isLocalMemRefType(Type type) {
  auto memrefTy = dyn_cast<MemRefType>(type);
  if (!memrefTy)
    return false;
  auto as = dyn_cast_or_null<AddressSpaceAttr>(memrefTy.getMemorySpace());
  if (!as)
    return false;
  AddressSpace space = as.getAddressSpace();
  return space != AddressSpace::GM && space != AddressSpace::Zero;
}

static Value createIntegerLikeConstant(OpBuilder &builder, Location loc,
                                       Type type, int64_t value) {
  if (type.isIndex())
    return builder.create<arith::ConstantIndexOp>(loc, value);
  return builder.create<arith::ConstantOp>(
      loc, type, builder.getIntegerAttr(type, value));
}

static Value createStageIV(OpBuilder &builder, Location loc, Value physicalIV,
                           Value step, int64_t maxPreloadNum,
                           int64_t preloadNum) {
  int64_t distance = maxPreloadNum - 1 - preloadNum;
  if (distance == 0)
    return physicalIV;

  Value distanceValue =
      createIntegerLikeConstant(builder, loc, step.getType(), distance);
  Value delta = builder.create<arith::MulIOp>(loc, step, distanceValue);
  return builder.create<arith::SubIOp>(loc, physicalIV, delta);
}

static Value createShiftedUpperBound(OpBuilder &builder, Location loc, Value ub,
                                     Value step, int64_t maxPreloadNum) {
  Value stages =
      createIntegerLikeConstant(builder, loc, step.getType(), maxPreloadNum);
  Value delta = builder.create<arith::MulIOp>(loc, step, stages);
  return builder.create<arith::AddIOp>(loc, ub, delta);
}

static Value createStageCondition(OpBuilder &builder, Location loc, Value lb,
                                  Value ub, Value stageIV) {
  Value lowerOk = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sle, lb, stageIV);
  Value upperOk = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, stageIV, ub);
  return builder.create<arith::AndIOp>(loc, lowerOk, upperOk);
}

static SmallVector<Value> rotateAddrs(PointerCastOp pointerCast,
                                      int64_t maxPreloadNum,
                                      int64_t preloadNum) {
  ValueRange oldAddrs = pointerCast.getAddrs();
  SmallVector<Value> rotated(maxPreloadNum);
  for (auto [idx, addr] : llvm::enumerate(oldAddrs.take_front(maxPreloadNum))) {
    int64_t shifted =
        (maxPreloadNum - preloadNum - 1 + static_cast<int64_t>(idx)) %
        maxPreloadNum;
    rotated[shifted] = addr;
  }
  return rotated;
}

static LogicalResult collectLocalBufferBindings(
    CVScopeOp scope, int64_t maxPreloadNum, scf::ForOp forOp,
    DenseMap<Value, LocalBufferBinding> &bindings) {
  auto walkResult = scope.walk([&](Operation *op) -> WalkResult {
    for (Value operand : op->getOperands()) {
      if (!operand || !forOp.isDefinedOutsideOfLoop(operand))
        continue;

      if (auto bind = operand.getDefiningOp<BindTileOp>()) {
        auto pointerCast = bind.getSource().getDefiningOp<PointerCastOp>();
        if (!pointerCast || !isLocalMemRefType(pointerCast.getType()))
          continue;
        if (static_cast<int64_t>(pointerCast.getAddrs().size()) <
            maxPreloadNum) {
          scope.emitOpError()
              << "uses preload local buffer with only "
              << pointerCast.getAddrs().size() << " planned address(es); "
              << "expected at least " << maxPreloadNum;
          return WalkResult::interrupt();
        }
        bindings.try_emplace(bind.getResult(),
                             LocalBufferBinding{bind.getResult(), pointerCast,
                                                bind});
        continue;
      }

      if (auto pointerCast = operand.getDefiningOp<PointerCastOp>()) {
        if (!isLocalMemRefType(pointerCast.getType()))
          continue;
        if (static_cast<int64_t>(pointerCast.getAddrs().size()) <
            maxPreloadNum) {
          scope.emitOpError()
              << "uses preload local buffer with only "
              << pointerCast.getAddrs().size() << " planned address(es); "
              << "expected at least " << maxPreloadNum;
          return WalkResult::interrupt();
        }
        bindings.try_emplace(
            pointerCast.getResult(),
            LocalBufferBinding{pointerCast.getResult(), pointerCast,
                               BindTileOp()});
      }
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

static LogicalResult validateNoUnsupportedLoopLocalUses(CVScopeOp scope,
                                                        scf::ForOp forOp) {
  Operation *scopeOp = scope.getOperation();
  LogicalResult result = success();
  scope.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (operand == forOp.getInductionVar())
        continue;
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || scopeOp->isAncestor(defOp))
        continue;
      if (defOp->getParentOfType<scf::ForOp>() == forOp &&
          !forOp.isDefinedOutsideOfLoop(operand)) {
        scope.emitOpError()
            << "contains a use of value defined in the original loop but "
               "outside the CV scope; hoist it into the scope or outside the "
               "loop before create-preload";
        result = failure();
      }
    }
  });
  return result;
}

static LogicalResult collectDirectScopes(scf::ForOp forOp,
                                         SmallVectorImpl<CVScopeOp> &scopes,
                                         int64_t &maxPreloadNum) {
  maxPreloadNum = -1;
  int64_t groupId = -1;
  DenseSet<int64_t> seenPreloadNums;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto scope = dyn_cast<CVScopeOp>(&op);
    if (!scope)
      continue;

    std::optional<int64_t> preload = getI64Attr(scope, kPreloadNumAttr);
    std::optional<int64_t> maxPreload = getI64Attr(scope, kMaxPreloadNumAttr);
    std::optional<int64_t> group = getI64Attr(scope, kGroupIdAttr);
    if (!preload || !maxPreload || !group)
      return scope.emitOpError("is missing required CV preload attributes");
    if (*preload < 0 || *maxPreload <= 1 || *preload >= *maxPreload)
      return scope.emitOpError("has invalid CV preload attributes");
    if (!seenPreloadNums.insert(*preload).second)
      return scope.emitOpError("duplicates preload_num in the same loop");

    if (maxPreloadNum < 0)
      maxPreloadNum = *maxPreload;
    else if (maxPreloadNum != *maxPreload)
      return scope.emitOpError(
          "has a different max_preload_num than sibling CV scopes");
    if (groupId < 0)
      groupId = *group;
    else if (groupId != *group)
      return scope.emitOpError(
          "has a different group_id than sibling CV scopes");

    if (failed(validateNoUnsupportedLoopLocalUses(scope, forOp)))
      return failure();
    scopes.push_back(scope);
  }
  return success();
}

static void clonePointerCastAndBind(OpBuilder &builder,
                                    LocalBufferBinding &binding,
                                    int64_t maxPreloadNum, int64_t preloadNum,
                                    IRMapping &mapping) {
  SmallVector<Value> addrs =
      rotateAddrs(binding.pointerCast, maxPreloadNum, preloadNum);
  std::optional<TileBufConfigAttr> config = binding.pointerCast.getConfig();
  auto newPointerCast = builder.create<PointerCastOp>(
      binding.pointerCast.getLoc(), binding.pointerCast.getType(), addrs,
      binding.pointerCast.getValidRow() ? binding.pointerCast.getValidRow()
                                        : Value(),
      binding.pointerCast.getValidCol() ? binding.pointerCast.getValidCol()
                                        : Value(),
      config ? static_cast<Attribute>(*config) : Attribute());
  mapping.map(binding.pointerCast.getResult(), newPointerCast.getResult());

  if (binding.bindTile) {
    auto newBind = builder.create<BindTileOp>(
        binding.bindTile.getLoc(), binding.bindTile.getResult().getType(),
        newPointerCast.getResult(),
        binding.bindTile.getValidRow() ? binding.bindTile.getValidRow()
                                       : Value(),
        binding.bindTile.getValidCol() ? binding.bindTile.getValidCol()
                                       : Value(),
        binding.bindTile.getConfig());
    mapping.map(binding.bindTile.getResult(), newBind.getResult());
  }
}

static CVScopeOp cloneCVScope(OpBuilder &builder, CVScopeOp scope,
                              IRMapping &mapping) {
  auto cloned = builder.create<CVScopeOp>(scope.getLoc());
  cloned->setAttrs(scope->getAttrDictionary());
  cloned.getBody().push_back(new Block());

  OpBuilder bodyBuilder =
      OpBuilder::atBlockEnd(&cloned.getBody().front());
  for (Operation &op : scope.getBody().front().getOperations())
    bodyBuilder.clone(op, mapping);
  return cloned;
}

static LogicalResult rewritePreloadLoop(IRRewriter &rewriter, scf::ForOp forOp,
                                        ArrayRef<CVScopeOp> scopes,
                                        int64_t maxPreloadNum) {
  if (!forOp.getInitArgs().empty()) {
    return forOp.emitOpError(
        "pto-cv-create-preload currently supports scf.for without iter_args");
  }

  DenseMap<Value, LocalBufferBinding> localBindings;
  for (CVScopeOp scope : scopes) {
    if (failed(
            collectLocalBufferBindings(scope, maxPreloadNum, forOp,
                                       localBindings)))
      return failure();
  }

  SmallVector<LocalBufferBinding> orderedBindings;
  orderedBindings.reserve(localBindings.size());
  for (const auto &it : localBindings)
    orderedBindings.push_back(it.second);

  Location loc = forOp.getLoc();
  rewriter.setInsertionPoint(forOp);
  Value newUpperBound = createShiftedUpperBound(
      rewriter, loc, forOp.getUpperBound(), forOp.getStep(), maxPreloadNum);
  auto newForOp =
      rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(), newUpperBound,
                                  forOp.getStep());

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newForOp.getBody());
  SmallVector<IRMapping> stageMappings(maxPreloadNum);
  SmallVector<Value> stageIVs(maxPreloadNum);
  for (int64_t preloadNum = 0; preloadNum < maxPreloadNum; ++preloadNum) {
    Value stageIV = createStageIV(bodyBuilder, loc, newForOp.getInductionVar(),
                                  newForOp.getStep(), maxPreloadNum,
                                  preloadNum);
    stageIVs[preloadNum] = stageIV;
    stageMappings[preloadNum].map(forOp.getInductionVar(), stageIV);
    for (LocalBufferBinding &binding : orderedBindings) {
      clonePointerCastAndBind(bodyBuilder, binding, maxPreloadNum, preloadNum,
                              stageMappings[preloadNum]);
    }
  }

  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto scope = dyn_cast<CVScopeOp>(&op);
    if (!scope) {
      return op.emitOpError(
          "non-CV-scope op cannot be preserved by pto-cv-create-preload yet");
    }

    int64_t preloadNum = *getI64Attr(scope, kPreloadNumAttr);
    Value cond = createStageCondition(bodyBuilder, scope.getLoc(),
                                      forOp.getLowerBound(),
                                      forOp.getUpperBound(),
                                      stageIVs[preloadNum]);
    auto ifOp = bodyBuilder.create<scf::IfOp>(scope.getLoc(), cond,
                                              /*withElseRegion=*/false);
    OpBuilder thenBuilder(ifOp.thenBlock(), ifOp.thenBlock()->begin());
    cloneCVScope(thenBuilder, scope, stageMappings[preloadNum]);
  }

  rewriter.eraseOp(forOp);
  for (LocalBufferBinding &binding : orderedBindings) {
    if (binding.bindTile && binding.bindTile.getResult().use_empty())
      rewriter.eraseOp(binding.bindTile);
    if (binding.pointerCast && binding.pointerCast.getResult().use_empty())
      rewriter.eraseOp(binding.pointerCast);
  }
  return success();
}

struct PTOCVCreatePreloadPass
    : public mlir::pto::impl::PTOCVCreatePreloadBase<
          PTOCVCreatePreloadPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<scf::ForOp> loops;
    moduleOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    IRRewriter rewriter(moduleOp.getContext());
    for (scf::ForOp forOp : llvm::reverse(loops)) {
      SmallVector<CVScopeOp> scopes;
      int64_t maxPreloadNum = -1;
      if (failed(collectDirectScopes(forOp, scopes, maxPreloadNum))) {
        signalPassFailure();
        return;
      }
      if (scopes.empty())
        continue;
      if (failed(rewritePreloadLoop(rewriter, forOp, scopes, maxPreloadNum))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOCVCreatePreloadPass() {
  return std::make_unique<PTOCVCreatePreloadPass>();
}
