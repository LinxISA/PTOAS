// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

struct PTOBindTileToEmitC : public OpConversionPattern<pto::BindTileOp> {
  using OpConversionPattern::OpConversionPattern;

  struct TileBuildSpec {
    std::string tileTypeStr;
    bool useConstructor = false;
    SmallVector<Value> constructorArgs;
  };

  static bool getIndexConst(Value v, int64_t &out) {
    if (!v)
      return false;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        out = ia.getValue().getSExtValue();
        return true;
      }
    }
    return false;
  }

  static bool getTilePointerStrides(pto::TileBufConfigAttr configAttr,
                                    Type elemTy, int64_t rows, int64_t cols,
                                    int64_t &rowStride,
                                    int64_t &colStride) {
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

  LogicalResult matchAndRewrite(pto::BindTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto configAttr = op.getConfigAttr();
    auto viewSemantics = op->getAttrOfType<StringAttr>("pto.view_semantics");
    bool isSubView = viewSemantics && viewSemantics.getValue() == "subview";

    auto peelAllCasts = [](Value v) {
      while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
        v = castOp.getOperand(0);
      if (auto castOp = v.getDefiningOp<emitc::CastOp>())
        v = castOp.getOperand();
      return v;
    };
    auto isTileLike = [](Value v) -> bool {
      auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
      if (!ot)
        return false;
      StringRef s = ot.getValue();
      return s.contains("Tile<") || s.contains("ConvTile<");
    };
    auto buildTileSpec = [&]() -> FailureOr<TileBuildSpec> {
      auto resMrTy = dyn_cast<MemRefType>(op.getType());
      if (!resMrTy)
        return failure();

      const char *roleTok = "TileType::Vec";
      if (auto asAttr =
              dyn_cast_or_null<pto::AddressSpaceAttr>(resMrTy.getMemorySpace())) {
        switch (asAttr.getAddressSpace()) {
        case pto::AddressSpace::VEC:
          roleTok = "TileType::Vec";
          break;
        case pto::AddressSpace::MAT:
          roleTok = "TileType::Mat";
          break;
        case pto::AddressSpace::LEFT:
          roleTok = "TileType::Left";
          break;
        case pto::AddressSpace::RIGHT:
          roleTok = "TileType::Right";
          break;
        case pto::AddressSpace::ACC:
          roleTok = "TileType::Acc";
          break;
        case pto::AddressSpace::BIAS:
          roleTok = "TileType::Bias";
          break;
        case pto::AddressSpace::SCALING:
          roleTok = "TileType::Scaling";
          break;
        case pto::AddressSpace::GM:
        case pto::AddressSpace::Zero:
          roleTok = "TileType::Vec";
          break;
        }
      }

      Type elemTy = resMrTy.getElementType();
      Type emitElemTy = getTypeConverter()->convertType(elemTy);
      if (!emitElemTy)
        return failure();
      auto emitElemOpaque = dyn_cast<emitc::OpaqueType>(emitElemTy);
      if (!emitElemOpaque)
        return failure();
      std::string elemTypeStr = emitElemOpaque.getValue().str();

      if (resMrTy.getRank() < 2)
        return failure();
      int64_t rows = resMrTy.getDimSize(0);
      int64_t cols = resMrTy.getDimSize(1);
      if (rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
        return failure();

      std::string blTok = "BLayout::RowMajor";
      if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout())) {
        if (static_cast<int32_t>(blAttr.getValue()) == 1)
          blTok = "BLayout::ColMajor";
      }
      pto::BLayout blayout = getTileBufBLayoutValue(configAttr);

      if (isSubView) {
        auto subMrTy = dyn_cast<MemRefType>(op.getSource().getType());
        auto subViewOp = op.getSource().getDefiningOp<memref::SubViewOp>();
        if (subMrTy && subMrTy.getRank() >= 2 && subViewOp) {
          int64_t subRows = subMrTy.getDimSize(0);
          int64_t subCols = subMrTy.getDimSize(1);
          SmallVector<int64_t> inheritedStrides;
          int64_t inheritedOffset = ShapedType::kDynamic;

          if (!pto::isPTOFloat4PackedType(elemTy) &&
              subRows != ShapedType::kDynamic &&
              subCols != ShapedType::kDynamic &&
              succeeded(getStridesAndOffset(subMrTy, inheritedStrides,
                                            inheritedOffset)) &&
              inheritedStrides.size() >= 2) {
            int64_t childRowStride = 0;
            int64_t childColStride = 0;
            bool sameStrides = getTilePointerStrides(
                configAttr, elemTy, subRows, subCols, childRowStride,
                childColStride);
            sameStrides = sameStrides &&
                          inheritedStrides[0] == childRowStride &&
                          inheritedStrides[1] == childColStride;
            if (sameStrides) {
              rows = subRows;
              cols = subCols;
            }
          }
        }
      }

      std::string slTok = "SLayout::NoneBox";
      if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout())) {
        int32_t slVal = static_cast<int32_t>(slAttr.getValue());
        slTok = (slVal == 1) ? "SLayout::RowMajor"
                             : (slVal == 2) ? "SLayout::ColMajor"
                                            : "SLayout::NoneBox";
      }

      int32_t fractal = 512;
      if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
        fractal = frAttr.getInt();

      std::string padTok = "PadValue::Null";
      if (auto padAttr = dyn_cast<PadValueAttr>(configAttr.getPad())) {
        switch (static_cast<int32_t>(padAttr.getValue())) {
        case 1:
          padTok = "PadValue::Zero";
          break;
        case 2:
          padTok = "PadValue::Max";
          break;
        case 3:
          padTok = "PadValue::Min";
          break;
        default:
          padTok = "PadValue::Null";
          break;
        }
      }

      std::string compactTok = "CompactMode::Null";
      if (auto compactAttr = dyn_cast<CompactModeAttr>(configAttr.getCompactMode())) {
        switch (static_cast<int32_t>(compactAttr.getValue())) {
        case 1:
          compactTok = "CompactMode::Normal";
          break;
        case 2:
          compactTok = "CompactMode::RowPlusOne";
          break;
        default:
          compactTok = "CompactMode::Null";
          break;
        }
      }

      std::string vrowTok, vcolTok;
      bool useConstructor = false;
      bool rowIsDynamic = false;
      bool colIsDynamic = false;
      SmallVector<Value> constructorArgs;

      Value vRow = op.getValidRow();
      Value vCol = op.getValidCol();
      Value vRowEmitC = adaptor.getValidRow();
      Value vColEmitC = adaptor.getValidCol();
      bool forceDynamicValid = op->hasAttr(kForceDynamicValidShapeAttrName);
      int64_t cRow = 0, cCol = 0;
      bool rowIsConst = vRow && getIndexConst(vRow, cRow);
      bool colIsConst = vCol && getIndexConst(vCol, cCol);

      auto makeCtorDimValue = [&](Value emitted, int64_t fallback) -> Value {
        if (emitted)
          return emitted;
        return makeEmitCIntConstant(
            rewriter, loc, emitc::OpaqueType::get(ctx, "int32_t"), fallback);
      };
      auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
        if (!emitted || !pto::isPTOFloat4PackedType(elemTy))
          return emitted;
        int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
        if (dimIdx != packedDim)
          return emitted;
        auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
        Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
        return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two).getResult();
      };

      if (forceDynamicValid) {
        vrowTok = "-1";
        vcolTok = "-1";
        useConstructor = true;
        constructorArgs.push_back(
            makeCtorDimValue(maybeScaleDynamicValid(vRowEmitC, 0),
                             renderTileTemplateDim(rowIsConst ? cRow : rows,
                                                   elemTy, blayout, 0)));
        constructorArgs.push_back(
            makeCtorDimValue(maybeScaleDynamicValid(vColEmitC, 1),
                             renderTileTemplateDim(colIsConst ? cCol : cols,
                                                   elemTy, blayout, 1)));
      } else {
        if (rowIsConst) {
          vrowTok = std::to_string(
              renderTileTemplateDim(cRow, elemTy, blayout, 0));
        } else if (vRow) {
          vrowTok = "-1";
          rowIsDynamic = true;
          useConstructor = true;
        } else {
          vrowTok = std::to_string(
              renderTileTemplateDim(rows, elemTy, blayout, 0));
        }

        if (colIsConst) {
          vcolTok = std::to_string(
              renderTileTemplateDim(cCol, elemTy, blayout, 1));
        } else if (vCol) {
          vcolTok = "-1";
          colIsDynamic = true;
          useConstructor = true;
        } else {
          vcolTok = std::to_string(
              renderTileTemplateDim(cols, elemTy, blayout, 1));
        }

        if (useConstructor) {
          if (rowIsDynamic && vRowEmitC)
            constructorArgs.push_back(maybeScaleDynamicValid(vRowEmitC, 0));
          if (colIsDynamic && vColEmitC)
            constructorArgs.push_back(maybeScaleDynamicValid(vColEmitC, 1));
        }
      }

      std::string tileTypeStr = std::string("Tile<") + roleTok + ", " +
                                elemTypeStr + ", " +
                                std::to_string(renderTileTemplateDim(
                                    rows, elemTy, blayout, 0)) +
                                ", " +
                                std::to_string(renderTileTemplateDim(
                                    cols, elemTy, blayout, 1)) +
                                ", " + blTok +
                                ", " + vrowTok + ", " + vcolTok + ", " + slTok +
                                ", " + std::to_string(fractal) + ", " + padTok +
                                ", " + compactTok +
                                ">";
      return TileBuildSpec{tileTypeStr, useConstructor, constructorArgs};
    };

    auto buildTileValue = [&](const TileBuildSpec &spec) -> Value {
      auto tileType = emitc::OpaqueType::get(ctx, spec.tileTypeStr);
      if (spec.useConstructor) {
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, tileType, spec.tileTypeStr,
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange(spec.constructorArgs))
            .getResult(0);
      }

      return rewriter
          .create<emitc::VariableOp>(loc, tileType, emitc::OpaqueAttr::get(ctx, ""))
          .getResult();
    };

    auto emitElemTypeToString = [&](Type elemTy) -> std::string {
      return getEmitCScalarTypeToken(elemTy);
    };

    auto buildIntegralAddress = [&](Value sourceValue) -> FailureOr<Value> {
      auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});

      Value rawPtr = sourceValue;
      if (auto ot = dyn_cast<emitc::OpaqueType>(sourceValue.getType())) {
        StringRef tyStr = ot.getValue();
        if (tyStr.contains("Tile<") || tyStr.contains("ConvTile<")) {
          auto srcMrTy = dyn_cast<MemRefType>(op.getSource().getType());
          if (!srcMrTy)
            return failure();
          std::string elemTok = emitElemTypeToString(srcMrTy.getElementType());
          pto::AddressSpace as = pto::AddressSpace::GM;
          if (auto asAttr =
                  dyn_cast_or_null<pto::AddressSpaceAttr>(srcMrTy.getMemorySpace()))
            as = asAttr.getAddressSpace();
          rawPtr = materializeTileDataValue(rewriter, loc, sourceValue, as,
                                            elemTok);
        }
      }

      if (isSetFFTsPointerLikeType(rawPtr.getType())) {
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                         ArrayAttr{}, rcU64, ValueRange{rawPtr})
            .getResult(0);
      }

      if (rawPtr.getType() == u64Ty)
        return rawPtr;
      return rewriter.create<emitc::CastOp>(loc, u64Ty, rawPtr).getResult();
    };

    if (op.getSource().getDefiningOp<pto::DeclareTileMemRefOp>()) {
      auto hasFollowingSetValidShape = [&]() {
        for (Operation *user : op->getUsers()) {
          auto setValidShape = dyn_cast<pto::SetValidShapeOp>(user);
          if (!setValidShape)
            continue;
          if (setValidShape.getSource() != op.getResult())
            continue;
          return true;
        }
        return false;
      };

      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      TileBuildSpec declSpec = *tileSpec;
      if (op->hasAttr(kForceDynamicValidShapeAttrName) &&
          hasFollowingSetValidShape()) {
        declSpec.useConstructor = false;
        declSpec.constructorArgs.clear();
      }
      rewriter.replaceOp(op, buildTileValue(declSpec));
      return success();
    }

    Value tileCandidate = peelAllCasts(adaptor.getSource());
    if (viewSemantics && viewSemantics.getValue() == "bitcast" &&
        isTileLike(tileCandidate)) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      Value dstTile = buildTileValue(*tileSpec);
      FailureOr<Value> addr = buildIntegralAddress(tileCandidate);
      if (failed(addr))
        return failure();

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, *addr});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    if (viewSemantics && viewSemantics.getValue() == "treshape" &&
        isTileLike(tileCandidate)) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      Value dstTile = buildTileValue(*tileSpec);

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TRESHAPE",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, tileCandidate});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    // Subview origins are kept distinct from generic tile rebinding:
    // even when source/destination C++ tile types match, subview may carry
    // shifted base address semantics and should materialize a fresh handle.
    if (isSubView) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      Value dstTile = buildTileValue(*tileSpec);
      FailureOr<Value> addr = buildIntegralAddress(tileCandidate);
      if (failed(addr))
        return failure();

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, *addr});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    // Generic tile-to-tile rebind path: preserve the same backing storage and
    // rebuild a sibling tile with updated metadata/valid dims.
    if (isTileLike(tileCandidate)) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();

      if (!tileSpec->useConstructor) {
        if (auto srcTy = dyn_cast<emitc::OpaqueType>(tileCandidate.getType())) {
          if (srcTy.getValue() == tileSpec->tileTypeStr) {
            rewriter.replaceOp(op, tileCandidate);
            return success();
          }
        }
      }

      Value dstTile = buildTileValue(*tileSpec);
      FailureOr<Value> addr = buildIntegralAddress(tileCandidate);
      if (failed(addr))
        return failure();

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, *addr});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    SmallVector<Value> physAddrs;
    Value source = op.getSource();

    while (auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>())
      source = castOp.getOperand(0);

    if (auto upstreamCast = source.getDefiningOp<pto::PointerCastOp>()) {
      auto upstreamOperands = upstreamCast.getAddrs();
      physAddrs.append(upstreamOperands.begin(), upstreamOperands.end());
    } else {
      physAddrs.push_back(adaptor.getSource());
    }

    Value vRow = op.getValidRow();
    Value vCol = op.getValidCol();

    auto newCast = rewriter.create<pto::PointerCastOp>(
        loc, op.getType(), physAddrs, vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    if (viewSemantics)
      newCast->setAttr("pto.view_semantics", viewSemantics);
    if (op->hasAttr(kForceDynamicValidShapeAttrName))
      newCast->setAttr(kForceDynamicValidShapeAttrName,
                       op->getAttr(kForceDynamicValidShapeAttrName));
    rewriter.replaceOp(op, newCast.getResult());

    return success();
  }
};

// =============================================================================
// Arith CmpI -> EmitC Cmp
// =============================================================================
class ArithCmpIToEmitC : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 将 arith.cmpi 转换为 emitc.cmp
    // 映射 Predicate: eq -> equal, slt -> less, etc.
    emitc::CmpPredicate emitcPred;
    const bool isUnsignedPred =
        op.getPredicate() == arith::CmpIPredicate::ult ||
        op.getPredicate() == arith::CmpIPredicate::ule ||
        op.getPredicate() == arith::CmpIPredicate::ugt ||
        op.getPredicate() == arith::CmpIPredicate::uge;
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:  emitcPred = emitc::CmpPredicate::eq; break;
      case arith::CmpIPredicate::ne:  emitcPred = emitc::CmpPredicate::ne; break;
      case arith::CmpIPredicate::slt: emitcPred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::sle: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::sgt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::sge: emitcPred = emitc::CmpPredicate::ge; break;
      // ... 处理无符号比较 (ult, ule 等) ...
      case arith::CmpIPredicate::ult: emitcPred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::ule: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::ugt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::uge: emitcPred = emitc::CmpPredicate::ge; break;
      default: return failure();
    }

    Type resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (isUnsignedPred) {
      Type opTy = op.getLhs().getType();
      auto intTy = dyn_cast<IntegerType>(opTy);
      const bool isIndex = isa<IndexType>(opTy);
      if (!intTy && !isIndex)
        return rewriter.notifyMatchFailure(
            op, "expected scalar integer or index operands");

      const unsigned bitWidth =
          intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);
      if (bitWidth != 1) {
        lhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth);
        rhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth);
      }
    }

    rewriter.replaceOpWithNewOp<emitc::CmpOp>(
        op, 
        /*resultType=*/resTy, // i1 -> bool/i1
        emitcPred,
        lhs,
        rhs
    );
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Section Op Lowering
//===----------------------------------------------------------------------===//
static bool isA5NoSplitPipeOp(Operation *op) {
  if (auto tpush = dyn_cast<pto::TPushOp>(op))
    return tpush.getSplit() == 0;
  if (auto tpop = dyn_cast<pto::TPopOp>(op))
    return tpop.getSplit() == 0;
  if (auto tfree = dyn_cast<pto::TFreeOp>(op))
    return tfree.getSplit() == 0;
  if (auto tpush = dyn_cast<pto::TPushToAivOp>(op))
    return tpush.getSplit() == 0;
  if (auto tpush = dyn_cast<pto::TPushToAicOp>(op))
    return tpush.getSplit() == 0;
  if (auto tpop = dyn_cast<pto::TPopFromAicOp>(op))
    return tpop.getSplit() == 0;
  if (auto tpop = dyn_cast<pto::TPopFromAivOp>(op))
    return tpop.getSplit() == 0;
  if (auto tfree = dyn_cast<pto::TFreeFromAicOp>(op))
    return tfree.getSplit() == 0;
  if (auto tfree = dyn_cast<pto::TFreeFromAivOp>(op))
    return tfree.getSplit() == 0;
  return false;
}

static bool hasExplicitSubblockControl(Operation *op) {
  bool hasControl = false;
  op->walk([&](Operation *nested) {
    if (isa<pto::GetSubBlockIdxOp, pto::GetSubBlockNumOp>(nested)) {
      hasControl = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasControl;
}

static bool needsA5NoSplitVectorGuard(Operation *op) {
  auto arch = getTargetArch(op);
  if (arch != PTOArch::A5)
    return false;
  bool isVectorScope = isa<pto::SectionVectorOp>(op);
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    if (auto kernelKindAttr =
            func->getAttrOfType<FunctionKernelKindAttr>(
                FunctionKernelKindAttr::name)) {
      isVectorScope =
          kernelKindAttr.getKernelKind() == FunctionKernelKind::Vector;
    }
  }
  if (!isVectorScope)
    return false;
  if (hasExplicitSubblockControl(op))
    return false;

  bool hasNoSplitPipe = false;
  op->walk([&](Operation *nested) {
    if (!isA5NoSplitPipeOp(nested))
      return WalkResult::advance();
    hasNoSplitPipe = true;
    return WalkResult::interrupt();
  });
  return hasNoSplitPipe;
}

template <typename SectionOpTy>
struct SectionToEmitC : public OpConversionPattern<SectionOpTy> {
  using OpConversionPattern<SectionOpTy>::OpConversionPattern;

  std::string getMacroName() const {
    if (std::is_same<SectionOpTy, pto::SectionCubeOp>::value)
      return "__DAV_CUBE__";
    if (std::is_same<SectionOpTy, pto::SectionVectorOp>::value)
      return "__DAV_VEC__";
    return "UNKNOWN_MACRO";
  }

  LogicalResult
  matchAndRewrite(SectionOpTy op, typename SectionOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    std::string startMacro = "\n#if defined(" + getMacroName() + ")";
    rewriter.create<emitc::VerbatimOp>(loc, startMacro);

    if constexpr (std::is_same_v<SectionOpTy, pto::SectionVectorOp>) {
      // Vector mask is a global HW state and may be modified by previous kernels
      // (or earlier sections). Reset it to a well-defined state for deterministic
      // execution of VEC ops.
      rewriter.create<emitc::VerbatimOp>(loc, "set_mask_norm();");
      rewriter.create<emitc::VerbatimOp>(loc, "set_vector_mask(-1, -1);");
    }

    if (needsNoSplitGuard) {
      rewriter.create<emitc::VerbatimOp>(
          loc, "if (get_subblockid() == 0) {");
    }

    Block &innerBlock = op.getBody().front();
    if (!innerBlock.empty()) {
      rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});
    }

    if (needsNoSplitGuard)
      rewriter.create<emitc::VerbatimOp>(loc, "}");

    std::string endMacro = "#endif // " + getMacroName() + "\n";
    rewriter.create<emitc::VerbatimOp>(loc, endMacro);

    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SCF Control-Flow Pre-Lowering
//
// EmitC translation supports `emitc.for`/`emitc.if` plus CFG-style
// `cf.br`/`cf.cond_br`. Upstream SCFToEmitC patterns only cover `scf.for` and
// `scf.if`, so we pre-lower some SCF ops into those supported forms.
//===----------------------------------------------------------------------===//

namespace {

static bool isTriviallyInlineableExecuteRegion(scf::ExecuteRegionOp op) {
  Region &r = op.getRegion();
  if (!r.hasOneBlock())
    return false;
  Block &b = r.front();
  return isa_and_nonnull<scf::YieldOp>(b.getTerminator());
}

static bool needsWholeFunctionSCFToCF(func::FuncOp func) {
  bool needs = false;
  func.walk([&](Operation *op) {
    if (!isa<scf::WhileOp, scf::IndexSwitchOp, scf::ExecuteRegionOp>(op))
      return WalkResult::advance();
    Operation *parentOp = op->getParentOp();

    // `scf.execute_region` can legally appear in single-block parents. Only
    // require whole-function SCFToCF if we need to lower it into CFG blocks
    // (multi-block region / non-trivial terminators).
    if (auto exec = dyn_cast<scf::ExecuteRegionOp>(op)) {
      if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>() &&
          !isTriviallyInlineableExecuteRegion(exec)) {
        needs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      needs = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return needs;
}

// scf.execute_region is semantically just an inlined region producing results
// via scf.yield. Inline it to the parent block to avoid extra lowering needs.
struct SCFExecuteRegionInline
    : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRegion().empty())
      return rewriter.notifyMatchFailure(op, "expected non-empty region");

    Block &innerBlock = op.getRegion().front();
    auto yield = dyn_cast<scf::YieldOp>(innerBlock.getTerminator());
    if (!yield)
      return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");

    // Move the body operations before the execute_region op.
    rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});

    // Replace execute_region results with yielded values, then erase the yield.
    rewriter.replaceOp(op, yield.getOperands());
    rewriter.eraseOp(yield);
    return success();
  }
};

// Lower scf.execute_region into CFG blocks with cf.br/cf.cond_br by inlining the
// region blocks into the parent region and rewriting scf.yield to branch into a
// continuation block carrying results.
//
// Note: This requires the parent region to allow multiple blocks (e.g. the
// function body CFG region). For execute_region nested in single-block regions
// (scf.for/scf.if), run SCFToCF first to eliminate the single-block constraint.
struct SCFExecuteRegionToCF : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (isTriviallyInlineableExecuteRegion(op))
      return rewriter.notifyMatchFailure(op, "trivially inlineable");

    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.execute_region inside a single-block parent region");
    }

    if (op.getRegion().empty())
      return rewriter.notifyMatchFailure(op, "expected non-empty region");

    Location loc = op.getLoc();
    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();

    // Split the parent block so we can branch to a continuation block with phi
    // arguments for the execute_region results.
    auto execIt = Block::iterator(op.getOperation());
    Block *continueBlock = rewriter.splitBlock(curBlock, std::next(execIt));

    SmallVector<BlockArgument> contArgs;
    contArgs.reserve(op.getNumResults());
    for (Type t : op.getResultTypes())
      contArgs.push_back(continueBlock->addArgument(t, loc));

    for (auto it : llvm::enumerate(op.getResults()))
      it.value().replaceAllUsesWith(contArgs[it.index()]);

    // Capture blocks before moving the region.
    SmallVector<Block *> movedBlocks;
    movedBlocks.reserve(op.getRegion().getBlocks().size());
    for (Block &b : op.getRegion())
      movedBlocks.push_back(&b);
    Block *entryBlock = &op.getRegion().front();

    // Inline the execute_region blocks into the parent region right before the
    // continuation block.
    rewriter.inlineRegionBefore(op.getRegion(), *parentRegion,
                                continueBlock->getIterator());

    // Replace all scf.yield terminators with a branch to the continuation.
    for (Block *b : movedBlocks) {
      auto yield = dyn_cast<scf::YieldOp>(b->getTerminator());
      if (!yield)
        continue;
      rewriter.setInsertionPoint(yield);
      rewriter.create<cf::BranchOp>(loc, continueBlock, yield.getOperands());
      rewriter.eraseOp(yield);
    }

    // Replace execute_region itself with a branch to the inlined entry block.
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(loc, entryBlock, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower scf.index_switch into CFG blocks with cf.cond_br/cf.br so that we can
// avoid `scf.if` result materialization quirks (and avoid relying on cf.switch,
// which is not supported by EmitC C++ translation).
struct SCFIndexSwitchToCF : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  static LogicalResult cloneYieldingBlockAndBranchTo(
      PatternRewriter &rewriter, Location loc, Block &srcBlock, Block *destBlock,
      Block *continueBlock) {
    rewriter.setInsertionPointToEnd(destBlock);

    IRMapping mapping;
    for (Operation &inner : srcBlock.without_terminator())
      rewriter.clone(inner, mapping);

    auto yield = dyn_cast<scf::YieldOp>(srcBlock.getTerminator());
    if (!yield)
      return failure();

    SmallVector<Value> yieldOperands;
    yieldOperands.reserve(yield.getNumOperands());
    for (Value v : yield.getOperands())
      yieldOperands.push_back(mapping.lookupOrDefault(v));

    rewriter.create<cf::BranchOp>(loc, continueBlock, yieldOperands);
    return success();
  }

  static Block *splitBlockForContinuation(PatternRewriter &rewriter,
                                          scf::IndexSwitchOp op) {
    auto switchIt = Block::iterator(op.getOperation());
    return rewriter.splitBlock(op->getBlock(), std::next(switchIt));
  }

  static void addContinuationArguments(PatternRewriter &rewriter,
                                       scf::IndexSwitchOp op, Location loc,
                                       Block *continueBlock) {
    SmallVector<BlockArgument> contArgs;
    contArgs.reserve(op.getNumResults());
    for (Type type : op.getResultTypes())
      contArgs.push_back(continueBlock->addArgument(type, loc));
    for (auto result : llvm::enumerate(op.getResults()))
      result.value().replaceAllUsesWith(contArgs[result.index()]);
  }

  static void createIndexSwitchBlocks(PatternRewriter &rewriter,
                                      Region *parentRegion,
                                      Region::iterator insertPt,
                                      unsigned numCases,
                                      SmallVectorImpl<Block *> &checkBlocks,
                                      Block *&defaultBlock,
                                      SmallVectorImpl<Block *> &caseBlocks) {
    checkBlocks.reserve(numCases);
    caseBlocks.reserve(numCases);
    for (unsigned i = 0; i < numCases; ++i)
      checkBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));
    defaultBlock = rewriter.createBlock(parentRegion, insertPt);
    for (unsigned i = 0; i < numCases; ++i)
      caseBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));
  }

  static void populateIndexSwitchCheckBlocks(
      PatternRewriter &rewriter, Location loc, Value selector,
      ArrayRef<int64_t> cases, ArrayRef<Block *> checkBlocks,
      ArrayRef<Block *> caseBlocks, Block *defaultBlock) {
    for (unsigned i = 0; i < checkBlocks.size(); ++i) {
      rewriter.setInsertionPointToEnd(checkBlocks[i]);
      Value caseVal = rewriter.create<arith::ConstantIndexOp>(loc, cases[i]);
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, selector, caseVal);
      Block *falseDest =
          (i + 1 < checkBlocks.size()) ? checkBlocks[i + 1] : defaultBlock;
      rewriter.create<cf::CondBranchOp>(loc, cond, caseBlocks[i], ValueRange{},
                                        falseDest, ValueRange{});
    }
  }

  LogicalResult matchAndRewrite(scf::IndexSwitchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.index_switch inside a single-block parent region");
    }

    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();
    Block *continueBlock = splitBlockForContinuation(rewriter, op);
    addContinuationArguments(rewriter, op, loc, continueBlock);

    unsigned numCases = op.getCases().size();
    auto insertPt = continueBlock->getIterator();

    SmallVector<Block *> checkBlocks;
    SmallVector<Block *> caseBlocks;
    Block *defaultBlock = nullptr;
    createIndexSwitchBlocks(rewriter, parentRegion, insertPt, numCases,
                            checkBlocks, defaultBlock, caseBlocks);

    Value selector = op.getArg();
    auto cases = op.getCases();
    populateIndexSwitchCheckBlocks(rewriter, loc, selector, cases, checkBlocks,
                                   caseBlocks, defaultBlock);

    // Fill case blocks and default block with cloned bodies + branch to cont.
    for (unsigned i = 0; i < numCases; ++i) {
      if (failed(cloneYieldingBlockAndBranchTo(
              rewriter, loc, op.getCaseBlock(i), caseBlocks[i], continueBlock)))
        return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");
    }
    if (failed(cloneYieldingBlockAndBranchTo(rewriter, loc, op.getDefaultBlock(),
                                             defaultBlock, continueBlock)))
      return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");

    // Replace the original switch op with a branch into the check chain.
    Block *entryDest = numCases ? checkBlocks[0] : defaultBlock;
    rewriter.setInsertionPointAfter(op);
    rewriter.create<cf::BranchOp>(loc, entryDest, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower scf.while into CFG blocks with cf.br/cf.cond_br.
//
// Note: This requires the parent region to allow multiple blocks. In
// particular, scf.if/scf.for regions are single-block and cannot contain this
// lowering.
struct SCFWhileToCF : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  static LogicalResult validateWhileResultUses(scf::WhileOp op) {
    Block *parentBlock = op->getBlock();
    for (Value result : op.getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner()->getBlock() != parentBlock)
          return failure();
      }
    }
    return success();
  }

  static Block *splitAfterWhileBlock(PatternRewriter &rewriter,
                                     scf::WhileOp op) {
    auto whileIt = Block::iterator(op.getOperation());
    return rewriter.splitBlock(op->getBlock(), std::next(whileIt));
  }

  static void addWhileExitArguments(PatternRewriter &rewriter, scf::WhileOp op,
                                    Location loc, Block *afterWhileBlock) {
    SmallVector<Value> exitArgs;
    exitArgs.reserve(op.getNumResults());
    for (Type type : op.getResultTypes())
      exitArgs.push_back(afterWhileBlock->addArgument(type, loc));
    for (auto result : llvm::enumerate(op.getResults()))
      result.value().replaceAllUsesWith(exitArgs[result.index()]);
  }

  static Block *createWhileHeaderBlock(PatternRewriter &rewriter,
                                       scf::WhileOp op, Location loc,
                                       Block *afterWhileBlock) {
    SmallVector<Type> headerArgTypes;
    for (Value init : op.getInits())
      headerArgTypes.push_back(init.getType());
    SmallVector<Location> headerArgLocs(headerArgTypes.size(), loc);
    return rewriter.createBlock(afterWhileBlock->getParent(),
                                afterWhileBlock->getIterator(), headerArgTypes,
                                headerArgLocs);
  }

  static Block *createWhileBodyBlock(PatternRewriter &rewriter, scf::WhileOp op,
                                     Location loc, Block *afterWhileBlock) {
    Block &afterRegionBlock = op.getAfter().front();
    SmallVector<Type> bodyArgTypes(afterRegionBlock.getArgumentTypes().begin(),
                                   afterRegionBlock.getArgumentTypes().end());
    SmallVector<Location> bodyArgLocs(bodyArgTypes.size(), loc);
    return rewriter.createBlock(afterWhileBlock->getParent(),
                                afterWhileBlock->getIterator(), bodyArgTypes,
                                bodyArgLocs);
  }

  static void rewriteWhileTerminators(PatternRewriter &rewriter, Location loc,
                                      Block *headerBlock, Block *bodyBlock,
                                      Block *afterWhileBlock) {
    auto condOp = cast<scf::ConditionOp>(headerBlock->getTerminator());
    rewriter.setInsertionPoint(condOp);
    rewriter.create<cf::CondBranchOp>(loc, condOp.getCondition(),
                                      /*trueDest=*/bodyBlock,
                                      /*trueOperands=*/condOp.getArgs(),
                                      /*falseDest=*/afterWhileBlock,
                                      /*falseOperands=*/condOp.getArgs());
    rewriter.eraseOp(condOp);

    auto yieldOp = cast<scf::YieldOp>(bodyBlock->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    rewriter.create<cf::BranchOp>(loc, headerBlock, yieldOp.getOperands());
    rewriter.eraseOp(yieldOp);
  }

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.while inside a single-block parent region");
    }

    if (failed(validateWhileResultUses(op)))
      return rewriter.notifyMatchFailure(
          op, "unsupported: while results used outside the parent block");

    auto loc = op.getLoc();
    Block *afterWhileBlock = splitAfterWhileBlock(rewriter, op);
    addWhileExitArguments(rewriter, op, loc, afterWhileBlock);
    Block *headerBlock = createWhileHeaderBlock(rewriter, op, loc,
                                                afterWhileBlock);
    Block *bodyBlock = createWhileBodyBlock(rewriter, op, loc, afterWhileBlock);

    // Move the before/after region bodies into the new CFG blocks.
    Block &afterRegionBlock = op.getAfter().front();
    rewriter.mergeBlocks(&op.getBefore().front(), headerBlock,
                         headerBlock->getArguments());
    rewriter.mergeBlocks(&afterRegionBlock, bodyBlock, bodyBlock->getArguments());
    rewriteWhileTerminators(rewriter, loc, headerBlock, bodyBlock,
                            afterWhileBlock);

    // Replace scf.while itself with a branch to the header.
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(loc, headerBlock, op.getInits());
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower cf.switch into chained comparisons and cf.cond_br/cf.br.
//
// EmitC C++ translation currently supports cf.br/cf.cond_br, but not cf.switch.
struct CFSwitchToCondBr : public OpRewritePattern<cf::SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  static SmallVector<SmallVector<Value>>
  collectSwitchCaseOperands(cf::SwitchOp op) {
    SmallVector<SmallVector<Value>> caseOperands;
    caseOperands.reserve(op.getCaseDestinations().size());
    for (auto range : op.getCaseOperands())
      caseOperands.emplace_back(range.begin(), range.end());
    return caseOperands;
  }

  static SmallVector<APInt> getSwitchCaseValues(cf::SwitchOp op) {
    SmallVector<APInt> caseValues;
    if (auto caseValuesAttr = op.getCaseValues()) {
      for (APInt value : caseValuesAttr->getValues<APInt>())
        caseValues.push_back(value);
    }
    return caseValues;
  }

  static SmallVector<Block *> createSwitchCheckBlocks(PatternRewriter &rewriter,
                                                      Region *parentRegion,
                                                      Block *curBlock,
                                                      size_t numCases) {
    auto insertPt = std::next(curBlock->getIterator());
    SmallVector<Block *> checkBlocks;
    checkBlocks.reserve(numCases);
    for (size_t i = 0; i < numCases; ++i)
      checkBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));
    return checkBlocks;
  }

  static LogicalResult populateSwitchCheckBlocks(
      PatternRewriter &rewriter, Location loc, Value flag, IntegerType flagTy,
      ArrayRef<APInt> caseValues, ArrayRef<Block *> caseDests,
      ArrayRef<SmallVector<Value>> caseOperands, Block *defaultDest,
      ValueRange defaultOperands, ArrayRef<Block *> checkBlocks,
      cf::SwitchOp op) {
    for (size_t i = 0; i < caseDests.size(); ++i) {
      rewriter.setInsertionPointToEnd(checkBlocks[i]);
      APInt caseVal = caseValues[i];
      if (caseVal.getBitWidth() != flagTy.getWidth()) {
        return rewriter.notifyMatchFailure(
            op, "case value bitwidth doesn't match flag type");
      }

      Value caseConst = rewriter.create<arith::ConstantOp>(
          loc, flagTy, rewriter.getIntegerAttr(flagTy, caseVal));
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, flag, caseConst);
      Block *falseDest =
          (i + 1 < checkBlocks.size()) ? checkBlocks[i + 1] : defaultDest;
      ValueRange falseOperands =
          (i + 1 < checkBlocks.size()) ? ValueRange{} : defaultOperands;
      rewriter.create<cf::CondBranchOp>(loc, cond, caseDests[i],
                                        caseOperands[i], falseDest,
                                        falseOperands);
    }
    return success();
  }

  LogicalResult matchAndRewrite(cf::SwitchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower cf.switch inside a single-block parent region");
    }

    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();

    Value flag = op.getFlag();
    auto flagTy = dyn_cast<IntegerType>(flag.getType());
    if (!flagTy)
      return rewriter.notifyMatchFailure(op, "expected integer switch flag");

    SmallVector<Value> defaultOperands(op.getDefaultOperands().begin(),
                                       op.getDefaultOperands().end());
    Block *defaultDest = op.getDefaultDestination();

    SmallVector<Block *> caseDests(op.getCaseDestinations().begin(),
                                   op.getCaseDestinations().end());
    SmallVector<SmallVector<Value>> caseOperands = collectSwitchCaseOperands(op);

    if (caseDests.empty()) {
      rewriter.replaceOpWithNewOp<cf::BranchOp>(op, defaultDest, defaultOperands);
      return success();
    }

    if (!op.getCaseValues())
      return rewriter.notifyMatchFailure(op, "missing case_values");
    SmallVector<APInt> caseValues = getSwitchCaseValues(op);

    if (caseValues.size() != caseDests.size())
      return rewriter.notifyMatchFailure(op, "case_values/destinations mismatch");
    if (caseOperands.size() != caseDests.size())
      return rewriter.notifyMatchFailure(op, "case_operands/destinations mismatch");

    SmallVector<Block *> checkBlocks =
        createSwitchCheckBlocks(rewriter, parentRegion, curBlock,
                                caseDests.size());
    if (failed(populateSwitchCheckBlocks(rewriter, loc, flag, flagTy,
                                         caseValues, caseDests, caseOperands,
                                         defaultDest, defaultOperands,
                                         checkBlocks, op))) {
      return failure();
    }

    // Replace the switch terminator with a branch into the first check block.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, checkBlocks.front(),
                                              ValueRange{});
    return success();
  }
};

} // namespace

static void populatePTOToEmitCPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       MLIRContext *ctx,
                                       DataFlowSolver &solver,
                                       PTOArch targetArch) {
  (void)solver;
  patterns.add<ArithCmpIToEmitC>(typeConverter, ctx);
  patterns.add<PTOBindTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetFlagToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.set_flag_dyn",
                                      "set_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.wait_flag_dyn",
                                      "wait_flag");
  // Backward-compatible aliases used in some downstream branches.
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.set_flag_d",
                                      "set_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.wait_flag_d",
                                      "wait_flag");
  patterns.add<PTOSubSCToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubCSToEmitC>(typeConverter, ctx);
  patterns.add<PTOWaitFlagToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetBufToEmitC>(typeConverter, ctx);
  patterns.add<PTORlsBufToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetFFTsToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORToEmitC>(typeConverter, ctx);
  patterns.add<PTOReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOScatterToEmitC>(typeConverter, ctx);
  patterns.add<PTOStoreFPSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSqrtSToEmitC>(typeConverter, ctx);
  patterns.add<PTOTTransToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelSToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandAddToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandExpdifToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandSubToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOColProdToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandAddToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandExpdifToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMinToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandSubToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOSORT32SToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTORsqrtToEmitC>(typeConverter, ctx);
  patterns.add<PTORowMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMulToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTORowProdToEmitC>(typeConverter, ctx);
  patterns.add<PTORowSumToEmitC>(typeConverter, ctx);
  patterns.add<PTORowMinToEmitC>(typeConverter, ctx);
  patterns.add<PTORowArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTODivSToEmitC>(typeConverter, ctx);
  patterns.add<PTOTDivSToEmitC>(typeConverter, ctx);
  patterns.add<PTOFModToEmitC>(typeConverter, ctx);
  patterns.add<PTORemToEmitC>(typeConverter, ctx);
  patterns.add<PTOConcatToEmitC, PTOConcatidxToEmitC>(typeConverter, ctx);
  patterns.add<PTORecipToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulsToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpToEmitC>(typeConverter, ctx);
  patterns.add<PTOPreluToEmitC>(typeConverter, ctx);
  patterns.add<PTOFModSToEmitC>(typeConverter, ctx);
  patterns.add<PTORemSToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTONotToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpandsToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartAddToEmitC>(typeConverter, ctx);
  patterns.add<PTOExtractToEmitC, PTOExtractFPToEmitC, PTOInsertToEmitC,
               PTOInsertFPToEmitC>(typeConverter, ctx);
  patterns.add<PTOFillPadToEmitC, PTOFillPadInplaceToEmitC, PTOFillPadExpandToEmitC>(
      typeConverter, ctx);
  patterns.add<PTOGatherToEmitC>(typeConverter, ctx);
  patterns.add<PTOGatherbToEmitC>(typeConverter, ctx);
  patterns.add<PTOMovFPToEmitC>(typeConverter, ctx);
  patterns.add<PTOQuantToEmitC>(typeConverter, ctx);
  patterns.add<PTODequantToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrsToEmitC>(typeConverter, ctx);
  patterns.add<PTOLogToEmitC>(typeConverter, ctx);
  patterns.add<FuncToEmitC>(typeConverter, ctx);
  patterns.add<PTOMovToEmitC>(typeConverter, ctx);
  patterns.add<ArithConstantToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulSIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<AffineApplyMulConstToEmitC>(typeConverter, ctx);
  patterns.add<PTONegToEmitC>(typeConverter, ctx);
  patterns.add<PTOTCIToEmitC>(typeConverter, ctx);
  patterns.add<PTOTTriToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpSToEmitC>(typeConverter, ctx);
  patterns.add<PTOColSumToEmitC>(typeConverter, ctx);
  patterns.add<PTOLReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOMrgSortToEmitC>(typeConverter, ctx);
  patterns.add<PTORandomToEmitC>(typeConverter, ctx);
  patterns.add<SubviewToEmitCPattern>(typeConverter, ctx);
  patterns.add<PointerCastConversion>(typeConverter, ctx);
  patterns.add<PTOSetValToSETVAL, PTOGetValToGETVAL, PTOSetValidShapeToEmitC,
               PTOTAssignToEmitC, PTOLoadScalarToEmitC,
               PTOStoreScalarToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAxpyToEmitC, PTOHistogramToEmitC, PTOGetScaleAddrToEmitC>(
      typeConverter, ctx);
  patterns.add<PTOTAndToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOAndSToEmitC>(typeConverter, ctx);
  patterns.add<PTOCvtToEmitC>(typeConverter, ctx);
  patterns.add<PTODivToTDIV>(typeConverter, ctx);
  patterns.add<PTOMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaxSToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulIToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddIToEmitC>(typeConverter, ctx);
  patterns.add<ArithSubIToEmitC>(typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::AndIOp, emitc::BitwiseAndOp>>(
      typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::OrIOp, emitc::BitwiseOrOp>>(
      typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::XOrIOp, emitc::BitwiseXorOp>>(
      typeConverter, ctx);
  patterns.add<ArithShiftLeftToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithDivUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCeilDivUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCeilDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithFloorDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithNegFToEmitC>(typeConverter, ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::SubFOp, emitc::SubOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::MulFOp, emitc::MulOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::DivFOp, emitc::DivOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithRemFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaximumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinimumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxNumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinNumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithSelectToEmitC>(typeConverter, ctx);
  patterns.add<ArithCmpFToEmitC>(typeConverter, ctx);
  patterns.add<ArithExtUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithExtSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::ExtFOp>>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::TruncFOp>>(typeConverter, ctx);
  patterns.add<ArithUIToFPToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::SIToFPOp>>(typeConverter, ctx);
  patterns.add<ArithFPToUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::FPToSIOp>>(typeConverter, ctx);
  patterns.add<ArithIndexCastUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithBitcastToEmitC>(typeConverter, ctx);
  patterns.add<PTOAddSToTADDS>(typeConverter, ctx);
  patterns.add<PTOColExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOTLoadToTLOAD>(typeConverter, ctx);
  patterns.add<PTOTPrefetchToTPREFETCH>(typeConverter, ctx);
  patterns.add<PTOTPackToTPACK>(typeConverter, ctx);
  patterns.add<PTOTStoreToTSTORE>(typeConverter, ctx);
  patterns.add<PTOMScatterToMSCATTER>(typeConverter, ctx);
  patterns.add<PTOTAddCToTADDC>(typeConverter, ctx);
  patterns.add<PTOMinsToEmitC>(typeConverter, ctx);
  patterns.add<PTOMGatherToMGATHER>(typeConverter, ctx);
  patterns.add<PTOTMatmulToTMATMUL>(typeConverter, ctx);
  patterns.add<PTOTMatmulAccToTMATMULACC>(typeConverter, ctx);
  patterns.add<PTOTGemvToTGEMV>(typeConverter, ctx);
  patterns.add<PTOTGemvAccToTGEMVACC>(typeConverter, ctx);
  patterns.add<ReinterpretCastToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAbsToTABS>(typeConverter, ctx);
  patterns.add<PTOTAddToTADD>(typeConverter, ctx);
  patterns.add<PTOAddSCToTADDSC>(typeConverter, ctx);
  patterns.add<ArithCastOPToEmitC>(typeConverter, ctx);
  patterns.add<ArithTruncIToEmitC>(typeConverter, ctx);
  patterns.add<PTOBuildAsyncSessionToEmitC>(typeConverter, ctx);
  patterns.add<PTOAsyncTransferToEmitC<pto::TPutAsyncOp>>(
      typeConverter, ctx,
      "pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOAsyncTransferToEmitC<pto::TGetAsyncOp>>(
      typeConverter, ctx,
      "pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOAsyncEventToEmitC<pto::WaitAsyncEventOp>>(
      typeConverter, ctx, "PTOAS__ASYNC_EVENT_WAIT");
  patterns.add<PTOAsyncEventToEmitC<pto::TestAsyncEventOp>>(
      typeConverter, ctx, "PTOAS__ASYNC_EVENT_TEST");
  patterns.add<PTOInitializeL2G2LPipeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOInitializeL2LPipeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTODeclareTileMemRefToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareEventIdArrayToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArrayGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArraySetToEmitC>(typeConverter, ctx);
  patterns.add<PTOTPushToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTPopToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTFreeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOSyncSetToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOSyncWaitToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<SectionToEmitC<pto::SectionCubeOp>>(typeConverter, ctx);
  patterns.add<SectionToEmitC<pto::SectionVectorOp>>(typeConverter, ctx);
  patterns.add<PTOGetBlockIdxToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetBlockNumToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetSubBlockIdxToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetSubBlockNumToEmitC>(typeConverter, ctx);
  patterns.add<PTOPrintToTPRINT>(typeConverter, ctx);
  patterns.add<PTOPrintOpToEmitC>(typeConverter, ctx);
  patterns.add<PTOTrapOpToEmitC>(typeConverter, ctx);
  patterns.add<
    PTOTMatmulBiasToTMATMUL_BIAS,
    PTOTMatmulMXToTMATMUL_MX,
    PTOTMatmulMXAccToTMATMUL_MX_ACC,
    PTOTMatmulMXBiasToTMATMUL_MX_BIAS,
    PTOTMatmulBiasToTMATMUL_BIAS,
    PTOTMatmulMXToTMATMUL_MX,
    PTOTMatmulMXAccToTMATMUL_MX_ACC,
    PTOTMatmulMXBiasToTMATMUL_MX_BIAS,
    PTOTGemvBiasToTGEMV_BIAS,
    PTOTGemvMXToTGEMV_MX,
    PTOTGemvMXAccToTGEMV_MX,
    PTOTGemvMXBiasToTGEMV_MX,
    PTOBarrierToEmitC
  >(typeConverter, ctx);

  patterns.add<CallToEmitC, ReturnToEmitC>(typeConverter, ctx);

  populateSCFToEmitCConversionPatterns(patterns);
  // Keep CFG-style branches type-consistent when block argument types are
  // converted (e.g. after lowering scf.while to cf.br/cf.cond_br).
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct EmitPTOManualPass
    : public PassWrapper<EmitPTOManualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitPTOManualPass)

  PTOArch targetArch;

  EmitPTOManualPass() : targetArch(PTOArch::A3) {}

  explicit EmitPTOManualPass(PTOArch arch) : targetArch(arch) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect,
                    memref::MemRefDialect, affine::AffineDialect,
                    mlir::cf::ControlFlowDialect, mlir::pto::PTODialect>();
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "DEBUG: Start PTOToEmitC Pass\n");
    MLIRContext *ctx = &getContext();
    ModuleOp mop = getOperation();

    if (failed(pto::validatePTOEntryFunctions(mop)))
      return signalPassFailure();
    pto::annotatePTOEntryFunctions(mop);

    // A3 requires explicit FFTS base setup for inter-core sync ops.
    if (targetArch == PTOArch::A3) {
      bool hasMissingSetFFTs = false;
      for (auto func : mop.getOps<func::FuncOp>()) {
        if (!hasInterCoreSyncOp(func))
          continue;
        if (hasSetFFTsOp(func))
          continue;
        hasMissingSetFFTs = true;
        func.emitError()
            << "A3 inter-core sync requires explicit `pto.set_ffts` in the "
               "same function when using `pto.sync.set`/`pto.sync.wait`";
      }
      if (hasMissingSetFFTs)
        return signalPassFailure();
    }

        bool needsEventIdArrayHelper = false;
        bool needsTRandomHelper = false;
        mop.walk([&](Operation *op) {
          if (isa<mlir::pto::DeclareEventIdArrayOp>(op))
            needsEventIdArrayHelper = true;
          if (isa<mlir::pto::TRandomOp>(op))
            needsTRandomHelper = true;
        });

		    // 1. 插入头文件
	    auto loc = mop->getLoc();
	    OpBuilder builder(ctx);
	    builder.setInsertionPointToStart(mop.getBody());
	    builder.create<emitc::IncludeOp>(
	        loc, "pto/pto-inst.hpp", /*is_standard_include=*/false);
	    builder.create<emitc::VerbatimOp>(
	        loc, builder.getStringAttr("using namespace pto;"));
        if (needsEventIdArrayHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
template <int N>
struct PTOAS_EventIdArray {
  static_assert(N > 0, "PTOAS_EventIdArray requires a positive static size");
  int32_t data[N] = {};

  AICORE inline int32_t &operator[](int32_t idx) { return data[idx]; }
  AICORE inline const int32_t &operator[](int32_t idx) const { return data[idx]; }
};
)cpp"));
        }
        if (needsTRandomHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
template <uint16_t Rounds, typename DstTile>
static AICORE inline void PTOAS__TRANDOM(
    DstTile &dst, uint32_t key0, uint32_t key1, uint32_t counter0,
    uint32_t counter1, uint32_t counter2, uint32_t counter3) {
  TRandomKey key = {key0, key1};
  TRandomCounter counter = {counter0, counter1, counter2, counter3};
  TRANDOM<Rounds>(dst, key, counter);
}
)cpp"));
        }
	    builder.create<emitc::VerbatimOp>(
	        loc, builder.getStringAttr(R"cpp(
enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}
)cpp"));
	    // Only inject the bitcast helper when we actually lower ops that need it
	    // (e.g. arith.bitcast or arith.maximumf/minimumf tie-breaking on zeros).
	    bool needsBitcastHelper = false;
	    mop.walk([&](Operation *op) {
	      if (isa<arith::BitcastOp, arith::MaximumFOp, arith::MinimumFOp>(op)) {
	        needsBitcastHelper = true;
	        return WalkResult::interrupt();
	      }
	      return WalkResult::advance();
	    });
	    if (needsBitcastHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
		template <typename To, typename From>
		static inline To ptoas_bitcast(From from) {
		  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
		  To to;
		  __builtin_memcpy(&to, &from, sizeof(To));
		  return to;
		}
		)cpp"));
	    }

	    // 1.5 Pre-lower SCF constructs not handled by SCFToEmitC.
	    {
	      // scf.while / scf.index_switch are lowered via CFG blocks. This is not
      // possible inside ops that require single-block regions (e.g. scf.for /
      // scf.if). If we see such nesting, lower the entire function to the
      // ControlFlow dialect first.
      bool needsAnySCFToCF = false;
      for (auto func : mop.getOps<func::FuncOp>()) {
        if (needsWholeFunctionSCFToCF(func)) {
          needsAnySCFToCF = true;
          break;
        }
      }
      if (needsAnySCFToCF) {
        RewritePatternSet scfToCfPatterns(ctx);
        populateSCFToControlFlowConversionPatterns(scfToCfPatterns);
        FrozenRewritePatternSet frozenSCFToCF(std::move(scfToCfPatterns));

        ConversionTarget scfToCfTarget(*ctx);
        // Only eliminate the single-block SCF constructs; we'll pre-lower
        // scf.while/index_switch/execute_region ourselves afterwards.
        scfToCfTarget.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp,
                                   scf::ParallelOp>();
        scfToCfTarget.markUnknownOpDynamicallyLegal(
            [](Operation *) { return true; });

        for (auto func : mop.getOps<func::FuncOp>()) {
          if (!needsWholeFunctionSCFToCF(func))
            continue;
          if (failed(applyPartialConversion(func, scfToCfTarget,
                                            frozenSCFToCF))) {
            func.emitError()
                << "failed to lower nested SCF to ControlFlow (SCFToCF)";
            return signalPassFailure();
          }
        }
      }

      RewritePatternSet scfLoweringPatterns(ctx);
      scfLoweringPatterns.add<SCFExecuteRegionInline, SCFExecuteRegionToCF,
                              SCFIndexSwitchToCF,
                              SCFWhileToCF, CFSwitchToCondBr>(ctx);
      (void)applyPatternsAndFoldGreedily(mop, std::move(scfLoweringPatterns));

      bool hasUnsupportedSCF = false;
      mop.walk([&](Operation *op) {
        if (isa<scf::ExecuteRegionOp, scf::IndexSwitchOp, scf::WhileOp>(op)) {
          hasUnsupportedSCF = true;
          op->emitError() << "Unsupported SCF op remained after pre-lowering";
          return WalkResult::interrupt();
        }
        if (isa<cf::SwitchOp>(op)) {
          hasUnsupportedSCF = true;
          op->emitError()
              << "Unsupported CF op remained after pre-lowering: cf.switch";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasUnsupportedSCF)
        return signalPassFailure();
    }

    PTOToEmitCTypeConverter typeConverter(ctx, targetArch);

    // 2. Pre-convert SCF structural op types (e.g. scf.if/scf.for results)
    // using the same type converter. This avoids creating emitc.variable with
    // unsupported types such as memref.
    {
      RewritePatternSet scfTypePatterns(ctx);
      ConversionTarget scfTypeTarget(*ctx);
      scf::populateSCFStructuralTypeConversionsAndLegality(
          typeConverter, scfTypePatterns, scfTypeTarget);
      scfTypeTarget.markUnknownOpDynamicallyLegal(
          [](Operation *) { return true; });

      if (failed(applyPartialConversion(mop, scfTypeTarget,
                                        std::move(scfTypePatterns)))) {
        mop.emitError("failed to reconcile SCF structural types");
        return signalPassFailure();
      }
    }

    // 3. 配置转换目标
    ConversionTarget target(*ctx);

    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<pto::PTODialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<mlir::scf::SCFDialect>(); 
    
    // If we introduced CFG branches (e.g. from scf.while), make sure they are
    // updated to use legalized operand types.
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&](Operation *op) {
          return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                  typeConverter);
        });

    // [关键] 允许 Cast 存在，最后统一清理
    target.addLegalOp<UnrealizedConversionCastOp>(); 

    target.addIllegalOp<func::ReturnOp>();
    target.addIllegalOp<func::FuncOp>(); 
    target.addIllegalOp<func::CallOp>();

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<ModuleOp>();

    auto solver = std::make_unique<DataFlowSolver>();
    solver->load<dataflow::DeadCodeAnalysis>();
    solver->load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    RewritePatternSet patterns(ctx);
    populatePTOToEmitCPatterns(patterns, typeConverter, ctx, *solver, targetArch);

    // 4. 执行转换
    if (failed(applyPartialConversion(mop, target, std::move(patterns)))) {
      llvm::errs() << "Conversion FAILED! Rolling back executed.\n";
      return signalPassFailure();
    }

    // =========================================================================
    // 5. [终极清理] 
    // 顺序至关重要：
    // Step A: 先移除所有 Cast，让 Loop 的 Operand 类型变成底层类型 (如 int32)
    // Step B: 再根据新的 Operand 类型，修复 Loop IV 的类型
    // =========================================================================
    
    // --- Step A: 清理 UnrealizedConversionCastOp ---
    // Prefer dropping redundant/unused casts; otherwise lower to emitc.cast
    // so the C++ emitter can print it.
    auto isEmitCPointerLikeType = [](Type ty) {
      if (isa<emitc::PointerType>(ty))
        return true;
      if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty))
        return opaqueTy.getValue().ends_with("*");
      return false;
    };

    llvm::SmallVector<UnrealizedConversionCastOp> castsToErase;
    bool castCleanupFailed = false;
    mop.walk([&](UnrealizedConversionCastOp cast) {
      if (castCleanupFailed)
        return;

      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
        cast.emitError() << "unsupported unrealized_conversion_cast shape";
        castCleanupFailed = true;
        return;
      }

      Value input = cast.getOperand(0);
      Value output = cast.getResult(0);
      Type inTy = input.getType();
      Type outTy = output.getType();

      if (output.use_empty()) {
        castsToErase.push_back(cast);
        return;
      }

      if (inTy == outTy) {
        output.replaceAllUsesWith(input);
        castsToErase.push_back(cast);
        return;
      }

      // SCF/CFG type conversion can transiently materialize pointer->memref
      // bridge casts. At this stage, the producing value is already in the
      // lowered EmitC pointer form; keep it and drop the bridge cast.
      if (isEmitCPointerLikeType(inTy) && isa<BaseMemRefType>(outTy)) {
        output.replaceAllUsesWith(input);
        castsToErase.push_back(cast);
        return;
      }

      if (emitc::isSupportedEmitCType(inTy) && emitc::isSupportedEmitCType(outTy)) {
        OpBuilder builder(cast);
        auto c = builder.create<emitc::CastOp>(cast.getLoc(), outTy, input);
        output.replaceAllUsesWith(c.getResult());
        castsToErase.push_back(cast);
        return;
      }

      cast.emitError() << "cannot lower unrealized_conversion_cast(" << inTy
                       << " -> " << outTy << ") to emitc.cast";
      castCleanupFailed = true;
    });

    for (auto cast : castsToErase)
      cast.erase();

    if (castCleanupFailed)
      return signalPassFailure();

    // --- Step A2: Sink casts of emitc.variable "reads" to their use sites ---
    //
    // SCFToEmitC lowers scf.if/scf.for results via mutable `emitc.variable` and
    // `emitc.assign`. During type conversion, casts from the variable handle to
    // the converted type may be materialized right after the variable
    // declaration, effectively snapshotting the value *before* assignments. That
    // produces wrong C++ (use-before-init / stale reads).
    //
    // Fix by re-materializing the cast at each use site so it reads the variable
    // at the point of use.
    {
      SmallVector<emitc::CastOp> castOpsToSink;
      mop.walk([&](emitc::CastOp castOp) {
        if (castOp.getSource().getDefiningOp<emitc::VariableOp>())
          castOpsToSink.push_back(castOp);
      });

      for (emitc::CastOp castOp : castOpsToSink) {
        Value src = castOp.getSource();
        Type dstTy = castOp.getResult().getType();
        Value oldRes = castOp.getResult();

        // Replace each use with a freshly inserted cast right before the user.
        for (OpOperand &use : llvm::make_early_inc_range(oldRes.getUses())) {
          Operation *user = use.getOwner();
          OpBuilder b(user);
          b.setInsertionPoint(user);
          auto newCast = b.create<emitc::CastOp>(castOp.getLoc(), dstTy, src);
          use.set(newCast.getResult());
        }

        castOp.erase();
      }
    }

    // --- Step B: 修复 Loop 归纳变量 (IV) ---
    // 此时 emitc.for 的 operand 已经是 int32 了，我们检查 IV 是否匹配，不匹配则修正
    mop.walk([&](emitc::ForOp forOp) {
       Type boundTy = forOp.getLowerBound().getType(); 
       BlockArgument iv = forOp.getBody()->getArgument(0); 
       
       if (iv.getType() != boundTy) {
         iv.setType(boundTy); // 强制将 IV 类型 (index) 修改为与边界一致 (int32)
       }
    });
    
    // --- Step C: 消除冗余 Tile 变量 (Dead Code Elimination) [新增] ---
    // 逻辑：如果一个 emitc.variable 没有被读取（use_empty），
    // 那么它自己，以及给它赋值的 TASSIGN 都可以删除。
    // 注意：TASSIGN(v15, v9) 会把 v15 作为 Operand 0 使用，所以 v15 不是严格的 use_empty。
    // 我们需要检查：v15 是否除了 TASSIGN 之外没有其他 User。

    llvm::SmallVector<emitc::VariableOp> deadVars;
    mop.walk([&](emitc::VariableOp varOp) {
        // 检查该变量的所有 User
        bool isRead = false;
        for (Operation* user : varOp.getResult().getUsers()) {
            // 如果 User 是 TASSIGN 且变量是第0个参数(dst)，不算"读取"
            if (auto call = dyn_cast<emitc::CallOpaqueOp>(user)) {
                if (call.getCallee() == "TASSIGN" && call.getOperand(0) == varOp.getResult()) {
                    continue; // 这是一个赋值操作，不算有效使用
                }
            }
            // 如果还有其他用途（如 TLOAD, TMOV, TMATMUL），则该变量有用
            isRead = true;
            break;
        }

        if (!isRead) {
            deadVars.push_back(varOp);
        }
    });

    for (auto varOp : deadVars) {
        // 1. 先删除所有使用该变量的 TASSIGN
        llvm::SmallVector<Operation*> usersToErase;
        for (Operation* user : varOp.getResult().getUsers()) {
             // 我们上面已经确认过，剩下的 user 只能是 TASSIGN
             usersToErase.push_back(user);
        }
        for (auto u : usersToErase) u->erase();

        // 2. 删除变量定义本身
        varOp.erase();
    }

    llvm::SmallVector<emitc::ConstantOp> deadConsts;
    mop.walk([&](emitc::ConstantOp constOp) {
      if (constOp.getResult().use_empty())
        deadConsts.push_back(constOp);
    });
    for (auto constOp : deadConsts)
      constOp.erase();

    // =========================================================================
  }
  };
} // namespace

std::unique_ptr<Pass> mlir::pto::createEmitPTOManualPass() {
  return std::make_unique<EmitPTOManualPass>();
}

std::unique_ptr<Pass> mlir::pto::createEmitPTOManualPass(PTOArch arch) {
  return std::make_unique<EmitPTOManualPass>(arch);
}
