// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOMaterializeTileHandles.cpp -------------------------------------===//
//===----------------------------------------------------------------------===//
//
// Reintroduce explicit tile_buf handles after memory planning/sync have used
// memref IR. EmitC can then lower tile operations from tile-typed operands
// instead of rediscovering tile metadata from every memref use.

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseMap.h"

#include <memory>
#include <optional>
#include <utility>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMATERIALIZETILEHANDLES
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace mlir {
namespace pto {
namespace {

static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";

struct TileHandleMetadata {
  Value source;
  Value validRow;
  Value validCol;
  TileBufConfigAttr config;
  bool explicitConfig = false;
  SmallVector<NamedAttribute, 2> attrs;
};

static bool isLocalTileMemRef(Type type) {
  auto memTy = dyn_cast<MemRefType>(type);
  if (!memTy || memTy.getRank() != 2)
    return false;

  auto asAttr = dyn_cast_or_null<AddressSpaceAttr>(memTy.getMemorySpace());
  if (!asAttr)
    return false;

  switch (asAttr.getAddressSpace()) {
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return false;
  case AddressSpace::VEC:
  case AddressSpace::MAT:
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return true;
  }
  return false;
}

static bool shouldMaterializeOperand(Operation *owner) {
  if (isa<AllocTileOp, MaterializeTileOp, BindTileOp, PointerCastOp>(owner))
    return false;

  StringRef name = owner->getName().getStringRef();
  if (name == "pto.set_validshape")
    return true;
  if (name == "pto.build_async_session")
    return true;
  if (!name.consume_front("pto."))
    return false;
  return name.starts_with("t");
}

static bool shouldMaterializeYieldOperand(Operation *owner) {
  return isa<scf::YieldOp>(owner);
}

static bool hasStringAttr(ArrayRef<NamedAttribute> attrs, StringRef name,
                          StringRef value) {
  return llvm::any_of(attrs, [&](NamedAttribute attr) {
    if (attr.getName().getValue() != name)
      return false;
    auto strAttr = dyn_cast<StringAttr>(attr.getValue());
    return strAttr && strAttr.getValue() == value;
  });
}

static bool hasAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  return llvm::any_of(attrs, [&](NamedAttribute attr) {
    return attr.getName().getValue() == name;
  });
}

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static void copyTileHandleAttrs(Operation *from,
                                SmallVectorImpl<NamedAttribute> &attrs) {
  StringRef names[] = {"pto.view_semantics", kForceDynamicValidShapeAttrName};
  for (StringRef name : names) {
    if (Attribute attr = from->getAttr(name))
      attrs.push_back(NamedAttribute(StringAttr::get(from->getContext(), name),
                                     attr));
  }
}

static std::optional<AddressSpace> getAddressSpace(Type type) {
  if (auto memTy = dyn_cast<MemRefType>(type)) {
    if (auto asAttr =
            dyn_cast_or_null<AddressSpaceAttr>(memTy.getMemorySpace()))
      return asAttr.getAddressSpace();
    return std::nullopt;
  }
  if (auto tileTy = dyn_cast<TileBufType>(type)) {
    if (auto asAttr =
            dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace()))
      return asAttr.getAddressSpace();
  }
  return std::nullopt;
}

static bool isA5Target(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return false;

  if (auto arch = module->getAttrOfType<StringAttr>("pto.target_arch")) {
    if (arch.getValue().equals_insensitive("a5"))
      return true;
  }
  if (auto spec = module->getAttrOfType<StringAttr>("pto.device-spec")) {
    StringRef value = spec.getValue();
    if (value.starts_with("Ascend950") || value.starts_with("Ascend910_95"))
      return true;
  }
  return false;
}

static TileBufConfigAttr makeTileConfig(MLIRContext *ctx, BLayout bl,
                                        SLayout sl) {
  Builder builder(ctx);
  return TileBufConfigAttr::get(
      ctx, BLayoutAttr::get(ctx, bl), SLayoutAttr::get(ctx, sl),
      builder.getI32IntegerAttr(512), PadValueAttr::get(ctx, PadValue::Null),
      CompactModeAttr::get(ctx, CompactMode::Null));
}

static void inferConfigForMaterializedUse(Operation *owner, unsigned operandNo,
                                          Type operandType,
                                          TileHandleMetadata &meta,
                                          MLIRContext *ctx) {
  if (meta.explicitConfig)
    return;

  auto colRow = [&]() {
    return makeTileConfig(ctx, BLayout::ColMajor, SLayout::RowMajor);
  };
  auto rowCol = [&]() {
    return makeTileConfig(ctx, BLayout::RowMajor, SLayout::ColMajor);
  };

  if (isa<TMatmulOp>(owner)) {
    if (!isA5Target(owner))
      return;
    if (operandNo == 0 || operandNo == 2)
      meta.config = colRow();
    else if (operandNo == 1)
      meta.config = rowCol();
    return;
  }

  if (isa<TMatmulAccOp>(owner)) {
    if (!isA5Target(owner))
      return;
    if (operandNo == 0 || operandNo == 1 || operandNo == 3)
      meta.config = colRow();
    else if (operandNo == 2)
      meta.config = rowCol();
    return;
  }

  if (isa<TInsertOp>(owner)) {
    if (operandNo != 0 && operandNo != 3)
      return;
    auto as = getAddressSpace(operandType);
    if (!as)
      return;
    if (*as == AddressSpace::ACC || *as == AddressSpace::MAT)
      meta.config = colRow();
  }
}

static TileHandleMetadata getTileHandleMetadata(Value value,
                                                MLIRContext *ctx) {
  TileHandleMetadata meta;
  meta.source = value;
  meta.config = TileBufConfigAttr::getDefault(ctx);

  if (auto bind = value.getDefiningOp<BindTileOp>()) {
    meta.source = bind.getSource();
    meta.validRow = bind.getValidRow();
    meta.validCol = bind.getValidCol();
    meta.config = bind.getConfig();
    meta.explicitConfig = true;
    copyTileHandleAttrs(bind, meta.attrs);
    return meta;
  }

  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    meta.validRow = cast.getValidRow();
    meta.validCol = cast.getValidCol();
    if (auto config = cast.getConfig()) {
      meta.config = *config;
      meta.explicitConfig = true;
    }
    copyTileHandleAttrs(cast, meta.attrs);
    return meta;
  }

  return meta;
}

static bool getTilePointerStrides(TileBufConfigAttr configAttr, Type elemTy,
                                  int64_t rows, int64_t cols,
                                  int64_t &rowStride, int64_t &colStride) {
  if (rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
    return false;

  int32_t blVal = 0;
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
    blVal = static_cast<int32_t>(blAttr.getValue());
  else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getBLayout()))
    blVal = static_cast<int32_t>(intAttr.getInt());

  int32_t slVal = 0;
  if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout()))
    slVal = static_cast<int32_t>(slAttr.getValue());
  else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getSLayout()))
    slVal = static_cast<int32_t>(intAttr.getInt());

  bool boxed = slVal != 0;
  int64_t innerRows = 1;
  int64_t innerCols = 1;
  if (boxed) {
    int32_t fractal = 512;
    if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
      fractal = static_cast<int32_t>(frAttr.getInt());

    unsigned elemBytes = pto::getPTOStorageElemByteSize(elemTy);
    if (elemBytes == 0)
      return false;

    switch (fractal) {
    case 1024:
      innerRows = 16;
      innerCols = 16;
      break;
    case 32:
      innerRows = 16;
      innerCols = 2;
      break;
    case 512:
      if (slVal == 1) {
        innerRows = 16;
        innerCols = 32 / elemBytes;
      } else if (slVal == 2) {
        innerRows = 32 / elemBytes;
        innerCols = 16;
      } else {
        return false;
      }
      break;
    default:
      return false;
    }
    if (innerRows <= 0 || innerCols <= 0)
      return false;
  }

  if (!boxed) {
    if (blVal == 1) {
      rowStride = 1;
      colStride = rows;
    } else {
      rowStride = cols;
      colStride = 1;
    }
    return true;
  }

  if (blVal == 1) {
    if (slVal != 1)
      return false;
    rowStride = innerCols;
    colStride = rows;
    return true;
  }

  rowStride = cols;
  colStride = innerRows;
  return true;
}

static SmallVector<int64_t, 2>
getMaterializedTileShape(MemRefType memTy, const TileHandleMetadata &meta) {
  SmallVector<int64_t, 2> shape(memTy.getShape().begin(),
                                memTy.getShape().end());
  if (!hasStringAttr(meta.attrs, "pto.view_semantics", "subview"))
    return shape;

  auto sourceMrTy = dyn_cast_or_null<MemRefType>(meta.source.getType());
  if (!sourceMrTy || sourceMrTy.getRank() < 2 ||
      !meta.source.getDefiningOp<memref::SubViewOp>())
    return shape;

  int64_t subRows = sourceMrTy.getDimSize(0);
  int64_t subCols = sourceMrTy.getDimSize(1);
  if (pto::isPTOFloat4PackedType(sourceMrTy.getElementType()) ||
      subRows == ShapedType::kDynamic || subCols == ShapedType::kDynamic)
    return shape;

  SmallVector<int64_t> inheritedStrides;
  int64_t inheritedOffset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(sourceMrTy, inheritedStrides,
                                 inheritedOffset)) ||
      inheritedStrides.size() < 2)
    return shape;

  int64_t childRowStride = 0;
  int64_t childColStride = 0;
  if (!getTilePointerStrides(meta.config, sourceMrTy.getElementType(), subRows,
                             subCols, childRowStride, childColStride))
    return shape;

  if (inheritedStrides[0] == childRowStride &&
      inheritedStrides[1] == childColStride) {
    shape[0] = subRows;
    shape[1] = subCols;
  }

  return shape;
}

static TileBufType buildTileTypeFromMemRef(MemRefType memTy,
                                           const TileHandleMetadata &meta,
                                           MLIRContext *ctx) {
  SmallVector<int64_t, 2> shape = getMaterializedTileShape(memTy, meta);
  SmallVector<int64_t, 2> validShape(shape.begin(), shape.end());
  bool forceDynamic = hasAttr(meta.attrs, kForceDynamicValidShapeAttrName);
  if (forceDynamic) {
    validShape[0] = ShapedType::kDynamic;
    validShape[1] = ShapedType::kDynamic;
  } else {
    if (meta.validRow)
      validShape[0] = getConstantIndexOrDynamic(meta.validRow);
    if (meta.validCol)
      validShape[1] = getConstantIndexOrDynamic(meta.validCol);
  }

  return TileBufType::get(ctx, shape, memTy.getElementType(),
                          memTy.getMemorySpace(), validShape, meta.config);
}

static bool isMaterializedTileAnchor(Operation *op) {
  return isa<BindTileOp, PointerCastOp>(op);
}

static Value makeI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIntOp>(loc, value, 64);
}

static Value ensureI64(Value value, OpBuilder &builder, Location loc) {
  if (!value)
    return Value();

  auto i64Ty = builder.getI64Type();
  if (value.getType() == i64Ty)
    return value;
  if (isa<IndexType>(value.getType()))
    return builder.create<arith::IndexCastOp>(loc, i64Ty, value);
  if (auto intTy = dyn_cast<IntegerType>(value.getType())) {
    if (intTy.getWidth() == 64)
      return value;
    if (intTy.getWidth() < 64)
      return builder.create<arith::ExtSIOp>(loc, i64Ty, value);
    return builder.create<arith::TruncIOp>(loc, i64Ty, value);
  }
  return Value();
}

static Value materializeOffset(OpFoldResult ofr, OpBuilder &builder,
                               Location loc) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return makeI64Constant(builder, loc, intAttr.getInt());
    return Value();
  }
  return ensureI64(ofr.get<Value>(), builder, loc);
}

static Value addI64(Value lhs, Value rhs, OpBuilder &builder, Location loc) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return builder.create<arith::AddIOp>(loc, lhs, rhs);
}

static Value mulI64(Value lhs, int64_t rhs, OpBuilder &builder, Location loc) {
  if (!lhs)
    return Value();
  if (rhs == 0)
    return makeI64Constant(builder, loc, 0);
  if (rhs == 1)
    return lhs;
  return builder.create<arith::MulIOp>(loc, lhs,
                                       makeI64Constant(builder, loc, rhs));
}

static Value computeExplicitAddress(Value value, OpBuilder &builder,
                                    Location loc);

static Value computeSubviewAddress(memref::SubViewOp subview,
                                   OpBuilder &builder, Location loc) {
  Value base = computeExplicitAddress(subview.getSource(), builder, loc);
  if (!base)
    return Value();

  auto sourceTy = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!sourceTy)
    return Value();
  unsigned elemBytes = getPTOStorageElemByteSize(sourceTy.getElementType());
  if (elemBytes == 0)
    return Value();

  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(sourceTy, sourceStrides, sourceOffset)))
    return Value();

  auto mixedOffsets = subview.getMixedOffsets();
  if (sourceStrides.size() < mixedOffsets.size())
    return Value();

  Value linearOffset;
  for (auto [offsetOfr, stride] :
       llvm::zip_equal(mixedOffsets, ArrayRef<int64_t>(sourceStrides).take_front(
                                         mixedOffsets.size()))) {
    if (stride == ShapedType::kDynamic)
      return Value();
    Value offset = materializeOffset(offsetOfr, builder, loc);
    if (!offset)
      return Value();
    linearOffset = addI64(linearOffset, mulI64(offset, stride, builder, loc),
                          builder, loc);
  }

  if (!linearOffset)
    return base;
  linearOffset = mulI64(linearOffset, elemBytes, builder, loc);
  return builder.create<arith::AddIOp>(loc, base, linearOffset);
}

static Value computeExplicitAddress(Value value, OpBuilder &builder,
                                    Location loc) {
  if (auto bind = value.getDefiningOp<BindTileOp>())
    return computeExplicitAddress(bind.getSource(), builder, loc);

  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    if (cast.getAddrs().empty())
      return Value();
    return ensureI64(cast.getAddrs().front(), builder, loc);
  }

  if (auto subview = value.getDefiningOp<memref::SubViewOp>())
    return computeSubviewAddress(subview, builder, loc);

  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return computeExplicitAddress(cast.getSource(), builder, loc);

  return Value();
}

static bool isControlFlowAddressProducer(Operation *op) {
  if (!op)
    return false;

  StringRef name = op->getName().getStringRef();
  return name == "scf.if" || name == "scf.for" || name == "scf.while" ||
         name == "scf.execute_region" || name == "scf.index_switch";
}

static Value peelAddressSource(Value value) {
  while (true) {
    if (auto bind = value.getDefiningOp<BindTileOp>()) {
      value = bind.getSource();
      continue;
    }

    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }

    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }

    return value;
  }
}

static bool isFunctionEntryBlockArgument(BlockArgument arg) {
  Operation *parent = arg.getOwner()->getParentOp();
  auto func = dyn_cast_or_null<func::FuncOp>(parent);
  return func && arg.getOwner() == &func.getBody().front();
}

static bool isUnsupportedControlFlowAddress(Value value) {
  value = peelAddressSource(value);

  if (auto arg = dyn_cast<BlockArgument>(value))
    return !isFunctionEntryBlockArgument(arg);

  return isControlFlowAddressProducer(value.getDefiningOp());
}

static void emitMissingExplicitAddressError(Operation *owner, Value value) {
  value = peelAddressSource(value);
  auto diag = owner->emitOpError()
              << "cannot materialize tile handle for local memref because its "
                 "explicit byte address cannot be recovered";

  if (isa<BlockArgument>(value)) {
    diag << "; region block arguments and loop-carried memref values are "
            "unsupported here";
    return;
  }

  Operation *def = value.getDefiningOp();
  if (!def) {
    diag << "; value has no defining op";
    return;
  }

  if (isControlFlowAddressProducer(def)) {
    diag << "; control-flow result '" << def->getName()
         << "' cannot carry a local memref into tile materialization";
    return;
  }

  diag << "; unsupported defining op is '" << def->getName() << "'";
}

static Value lookupMaterializedTileHandle(
    Value value, DenseMap<Value, Value> &tileHandles) {
  if (isa<TileBufType>(value.getType()))
    return value;

  auto it = tileHandles.find(value);
  if (it == tileHandles.end())
    return Value();
  return it->second;
}

static FailureOr<bool>
materializeSCFIfResults(ModuleOp module, DenseMap<Value, Value> &tileHandles) {
  bool changed = false;

  SmallVector<scf::IfOp, 8> ifOps;
  module.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });

  for (scf::IfOp ifOp : llvm::reverse(ifOps)) {
    if (ifOp.getNumResults() == 0)
      continue;

    auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    if (!thenYield || !elseYield)
      continue;

    for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
      if (!isLocalTileMemRef(result.getType()))
        continue;

      Value thenTile =
          lookupMaterializedTileHandle(thenYield.getOperand(idx), tileHandles);
      Value elseTile =
          lookupMaterializedTileHandle(elseYield.getOperand(idx), tileHandles);
      if (!thenTile || !elseTile)
        continue;

      if (thenTile.getType() != elseTile.getType()) {
        ifOp.emitOpError()
            << "cannot materialize tile result #" << idx
            << " because branch tile types differ: " << thenTile.getType()
            << " vs " << elseTile.getType();
        return failure();
      }

      Type tileTy = thenTile.getType();
      thenYield->setOperand(idx, thenTile);
      elseYield->setOperand(idx, elseTile);
      result.setType(tileTy);
      tileHandles[result] = result;
      changed = true;
    }
  }

  return changed;
}

static FailureOr<bool>
materializeSCFForResults(ModuleOp module, DenseMap<Value, Value> &tileHandles) {
  bool changed = false;

  SmallVector<scf::ForOp, 8> forOps;
  module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

  for (scf::ForOp forOp : llvm::reverse(forOps)) {
    if (forOp.getNumResults() == 0)
      continue;

    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield)
      continue;

    for (auto [idx, result] : llvm::enumerate(forOp.getResults())) {
      if (!isLocalTileMemRef(result.getType()))
        continue;

      Value initTile =
          lookupMaterializedTileHandle(forOp.getInitArgs()[idx], tileHandles);
      if (!initTile)
        continue;

      BlockArgument iterArg = forOp.getRegionIterArg(idx);
      Value yieldValue = yield.getOperand(idx);
      Value yieldTile = lookupMaterializedTileHandle(yieldValue, tileHandles);
      bool yieldIsIterArg = !yieldTile && yieldValue == iterArg;
      if (yieldIsIterArg)
        yieldTile = iterArg;
      if (!yieldTile)
        continue;

      Type yieldTy = yieldIsIterArg ? initTile.getType() : yieldTile.getType();
      if (initTile.getType() != yieldTy) {
        forOp.emitOpError()
            << "cannot materialize tile result #" << idx
            << " because init/yield tile types differ: " << initTile.getType()
            << " vs " << yieldTy;
        return failure();
      }

      Type tileTy = initTile.getType();
      forOp->setOperand(forOp.getNumControlOperands() + idx, initTile);
      iterArg.setType(tileTy);
      yield->setOperand(idx, yieldTile);
      result.setType(tileTy);
      tileHandles[iterArg] = iterArg;
      tileHandles[result] = result;
      changed = true;
    }
  }

  return changed;
}

static LogicalResult
materializeControlFlowTileResults(ModuleOp module,
                                  DenseMap<Value, Value> &tileHandles) {
  bool changed = false;
  do {
    changed = false;

    FailureOr<bool> ifChanged =
        materializeSCFIfResults(module, tileHandles);
    if (failed(ifChanged))
      return failure();
    changed |= *ifChanged;

    FailureOr<bool> forChanged =
        materializeSCFForResults(module, tileHandles);
    if (failed(forChanged))
      return failure();
    changed |= *forChanged;
  } while (changed);

  return success();
}

static Value getAllocValidOperand(TileBufType tileTy, Value operand,
                                  unsigned dim, OpBuilder &builder,
                                  Location loc) {
  auto validShape = tileTy.getValidShape();
  if (validShape.size() <= dim || validShape[dim] >= 0)
    return Value();
  if (operand)
    return operand;

  auto shape = tileTy.getShape();
  if (shape.size() > dim && shape[dim] != ShapedType::kDynamic)
    return builder.create<arith::ConstantIndexOp>(loc, shape[dim]);
  return Value();
}

static Attribute getAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  for (NamedAttribute attr : attrs) {
    if (attr.getName().getValue() == name)
      return attr.getValue();
  }
  return {};
}

static void copyMaterializedTileAttrs(ArrayRef<NamedAttribute> attrs,
                                      Operation *to) {
  if (Attribute attr = getAttr(attrs, kForceDynamicValidShapeAttrName))
    to->setAttr(kForceDynamicValidShapeAttrName, attr);
}

static void updateResultTypesAfterMaterializingOperand(Operation *op,
                                                       unsigned operandNo,
                                                       Type tileTy) {
  if (auto tassign = dyn_cast<TAssignOp>(op)) {
    if (operandNo == 0)
      tassign.getResult().setType(tileTy);
  }
}

static Value materializeAnchorResult(Operation *anchor, Value anchoredValue,
                                     OpBuilder &builder, MLIRContext *ctx,
                                     DenseMap<Value, Value> &tileHandles,
                                     bool &failedMaterialization) {
  auto memTy = dyn_cast<MemRefType>(anchoredValue.getType());
  if (!memTy || !isLocalTileMemRef(memTy))
    return Value();

  SmallVector<OpOperand *> usesToRewrite;
  for (OpOperand &use : anchoredValue.getUses()) {
    if (shouldMaterializeOperand(use.getOwner()) ||
        shouldMaterializeYieldOperand(use.getOwner()))
      usesToRewrite.push_back(&use);
  }

  TileHandleMetadata meta = getTileHandleMetadata(anchoredValue, ctx);
  auto viewSemantics = dyn_cast_or_null<StringAttr>(
      getAttr(meta.attrs, "pto.view_semantics"));
  bool isTileView =
      viewSemantics && (viewSemantics.getValue() == "treshape" ||
                        viewSemantics.getValue() == "bitcast");
  if (usesToRewrite.empty() && !isTileView)
    return Value();

  for (OpOperand *use : usesToRewrite)
    inferConfigForMaterializedUse(use->getOwner(), use->getOperandNumber(),
                                  anchoredValue.getType(), meta, ctx);
  auto tileTy = buildTileTypeFromMemRef(memTy, meta, ctx);

  builder.setInsertionPointAfter(anchor);
  Value materialized;
  Value sourceTile = meta.source ? tileHandles.lookup(meta.source) : Value();
  if (sourceTile && viewSemantics &&
      viewSemantics.getValue() == "treshape") {
    materialized =
        builder.create<TReshapeOp>(anchor->getLoc(), tileTy, sourceTile)
            .getResult();
  } else if (sourceTile && viewSemantics &&
             viewSemantics.getValue() == "bitcast") {
    materialized =
        builder.create<BitcastOp>(anchor->getLoc(), tileTy, sourceTile)
            .getResult();
  } else {
    Value addr = computeExplicitAddress(anchoredValue, builder, anchor->getLoc());
    if (!addr && isUnsupportedControlFlowAddress(anchoredValue)) {
      emitMissingExplicitAddressError(anchor, anchoredValue);
      failedMaterialization = true;
      return Value();
    }
    auto alloc = builder.create<AllocTileOp>(
        anchor->getLoc(), tileTy, addr ? addr : Value(),
        getAllocValidOperand(tileTy, meta.validRow, 0, builder,
                             anchor->getLoc()),
        getAllocValidOperand(tileTy, meta.validCol, 1, builder,
                             anchor->getLoc()));
    copyMaterializedTileAttrs(meta.attrs, alloc);
    materialized = alloc.getResult();
  }

  for (OpOperand *use : usesToRewrite) {
    Operation *owner = use->getOwner();
    unsigned operandNo = use->getOperandNumber();
    use->set(materialized);
    updateResultTypesAfterMaterializingOperand(owner, operandNo, tileTy);
  }

  tileHandles[anchoredValue] = materialized;
  return materialized;
}

struct PTOMaterializeTileHandlesPass
    : public impl::PTOMaterializeTileHandlesBase<
          PTOMaterializeTileHandlesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    OpBuilder builder(ctx);
    DenseMap<Value, Value> tileHandles;
    bool failedMaterialization = false;

    SmallVector<Operation *, 32> anchors;
    module.walk([&](Operation *op) {
      if (!isMaterializedTileAnchor(op))
        return;
      anchors.push_back(op);
    });

    for (Operation *anchor : anchors) {
      if (anchor->getNumResults() != 1)
        continue;
      materializeAnchorResult(anchor, anchor->getResult(0), builder, ctx,
                              tileHandles, failedMaterialization);
    }

    if (failedMaterialization) {
      signalPassFailure();
      return;
    }

    if (failed(materializeControlFlowTileResults(module, tileHandles))) {
      signalPassFailure();
      return;
    }

    SmallVector<std::pair<Operation *, unsigned>, 32> operandsToRewrite;
    module.walk([&](Operation *op) {
      if (!shouldMaterializeOperand(op))
        return;
      for (OpOperand &operand : op->getOpOperands()) {
        if (isLocalTileMemRef(operand.get().getType()))
          operandsToRewrite.push_back({op, operand.getOperandNumber()});
      }
    });

    for (auto [op, operandNo] : operandsToRewrite) {
      Value oldValue = op->getOperand(operandNo);
      if (!isa<MemRefType>(oldValue.getType()))
        continue;
      if (op->getName().getStringRef() == "pto.tassign" && operandNo == 0)
        continue;
      auto memTy = cast<MemRefType>(oldValue.getType());
      TileHandleMetadata meta = getTileHandleMetadata(oldValue, ctx);
      inferConfigForMaterializedUse(op, operandNo, oldValue.getType(), meta,
                                    ctx);
      auto tileTy = buildTileTypeFromMemRef(memTy, meta, ctx);

      builder.setInsertionPoint(op);
      Value materialized;
      Value addr = computeExplicitAddress(oldValue, builder, op->getLoc());
      if (!addr && isUnsupportedControlFlowAddress(oldValue)) {
        emitMissingExplicitAddressError(op, oldValue);
        failedMaterialization = true;
        continue;
      }
      auto alloc = builder.create<AllocTileOp>(
          op->getLoc(), tileTy, addr ? addr : Value(),
          getAllocValidOperand(tileTy, meta.validRow, 0, builder,
                               op->getLoc()),
          getAllocValidOperand(tileTy, meta.validCol, 1, builder,
                               op->getLoc()));
      copyMaterializedTileAttrs(meta.attrs, alloc);
      materialized = alloc.getResult();
      tileHandles[oldValue] = materialized;
      op->setOperand(operandNo, materialized);
      updateResultTypesAfterMaterializingOperand(op, operandNo, tileTy);
    }

    if (failedMaterialization) {
      signalPassFailure();
      return;
    }

    bool erasedBind = true;
    while (erasedBind) {
      erasedBind = false;
      SmallVector<Operation *, 16> deadBinds;
      module.walk([&](BindTileOp op) {
        if (op.getResult().use_empty())
          deadBinds.push_back(op);
      });
      for (Operation *op : deadBinds) {
        op->erase();
        erasedBind = true;
      }
    }

  }
};

} // namespace

std::unique_ptr<Pass> createPTOMaterializeTileHandlesPass() {
  return std::make_unique<PTOMaterializeTileHandlesPass>();
}

} // namespace pto
} // namespace mlir
