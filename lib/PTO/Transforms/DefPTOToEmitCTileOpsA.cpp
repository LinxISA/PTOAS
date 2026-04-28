// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

struct PTOLoadScalarToEmitC : public OpConversionPattern<pto::LoadScalarOp> {
  using OpConversionPattern<pto::LoadScalarOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = peelUnrealized(adaptor.getPtr());
    Value offset = peelUnrealized(adaptor.getOffset());

    Type dstTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!dstTy)
      return failure();

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{dstTy}, "PTOAS__PTR_LOAD",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct PTOStoreScalarToEmitC : public OpConversionPattern<pto::StoreScalarOp> {
  using OpConversionPattern<pto::StoreScalarOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = peelUnrealized(adaptor.getPtr());
    Value offset = peelUnrealized(adaptor.getOffset());
    Value val = peelUnrealized(adaptor.getValue());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__PTR_STORE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset, val});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tabs lowering -> TABS(dst, src)
//===----------------------------------------------------------------------===//

struct PTOTAbsToTABS : public OpConversionPattern<pto::TAbsOp> {
  using OpConversionPattern<pto::TAbsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAbsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TABS(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TABS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tadd lowering -> TADD(dst, src0, src1)
//===----------------------------------------------------------------------===//

struct PTOTAddToTADD : public OpConversionPattern<pto::TAddOp> {
  using OpConversionPattern<pto::TAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOInitializeL2G2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp> {
  PTOInitializeL2G2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2G2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    Value gmAddr = peelUnrealized(adaptor.getGmAddr());
    Value localAddr = peelUnrealized(adaptor.getLocalAddr());
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == 1)
      c2vBuf = localAddr;
    else if (op.getDirMask() == 2)
      v2cBuf = localAddr;
    else if (op.getDirMask() == 3) {
      c2vBuf = localAddr;
      v2cBuf = peelUnrealized(adaptor.getPeerLocalAddr());
    } else
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{gmAddr, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOInitializeL2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2LPipeOp> {
  PTOInitializeL2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                              PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    auto gmPtrTy = emitc::OpaqueType::get(ctx, "__gm__ void *");
    Value nullGm =
        makeEmitCOpaqueConstant(rewriter, op.getLoc(), gmPtrTy, "nullptr");
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);
    Value localAddr = peelUnrealized(adaptor.getLocalAddr());

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == 1)
      c2vBuf = localAddr;
    else if (op.getDirMask() == 2)
      v2cBuf = localAddr;
    else if (op.getDirMask() == 3) {
      c2vBuf = localAddr;
      v2cBuf = peelUnrealized(adaptor.getPeerLocalAddr());
    } else
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{nullGm, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOBuildAsyncSessionToEmitC
    : public OpConversionPattern<mlir::pto::BuildAsyncSessionOp> {
  PTOBuildAsyncSessionToEmitC(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<mlir::pto::BuildAsyncSessionOp>(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(mlir::pto::BuildAsyncSessionOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Location loc = op.getLoc();

    auto sessionTy =
        dyn_cast<emitc::OpaqueType>(getTypeConverter()->convertType(op.getSession().getType()));
    if (!sessionTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async session type");

    FailureOr<Value> scratchTile =
        buildAsyncScratchTileValue(rewriter, loc, op.getScratch(),
                                   adaptor.getScratch());
    if (failed(scratchTile))
      return rewriter.notifyMatchFailure(op, "failed to materialize async scratch tile");

    Value workspace =
        castToGMBytePointer(rewriter, loc, peelUnrealized(adaptor.getWorkspace()));

    Value session = rewriter
                        .create<emitc::VariableOp>(
                            loc, sessionTy, emitc::OpaqueAttr::get(ctx, ""))
                        .getResult();

    auto u32Ty = emitc::OpaqueType::get(ctx, "uint32_t");
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");

    auto makeU32Const = [&](uint64_t value) -> Value {
      return makeEmitCOpaqueConstant(rewriter, loc, u32Ty,
                                     std::to_string(value) + "u");
    };
    uint64_t syncId = op.getSyncIdAttr() ? op.getSyncIdAttr().getInt() : 0;
    uint64_t blockBytes =
        op.getBlockBytesAttr() ? op.getBlockBytesAttr().getInt() : 32 * 1024;
    uint64_t commBlockOffset =
        op.getCommBlockOffsetAttr() ? op.getCommBlockOffsetAttr().getInt() : 0;
    uint64_t queueNum = op.getQueueNumAttr() ? op.getQueueNumAttr().getInt() : 1;
    uint64_t channelGroupIdx = op.getChannelGroupIdxAttr()
                                   ? op.getChannelGroupIdxAttr().getInt()
                                   : UINT32_MAX;

    Value syncIdVal = makeU32Const(syncId);
    Value channelGroupIdxVal =
        channelGroupIdx == UINT32_MAX
            ? makeEmitCOpaqueConstant(rewriter, loc, u32Ty, "UINT32_MAX")
            : makeU32Const(channelGroupIdx);

    auto baseConfigTy =
        emitc::OpaqueType::get(ctx, "pto::comm::sdma::SdmaBaseConfig");
    Value baseConfig =
        rewriter
            .create<emitc::VariableOp>(
                loc, baseConfigTy,
                emitc::OpaqueAttr::get(
                    ctx, "{" + std::to_string(blockBytes) + "ULL, " +
                             std::to_string(commBlockOffset) + "ULL, " +
                             std::to_string(queueNum) + "u}"))
            .getResult();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{*scratchTile, workspace, session, syncIdVal, baseConfig,
                   channelGroupIdxVal});

    rewriter.replaceOp(op, session);
    return success();
  }
};

template <typename AsyncOp>
struct PTOAsyncTransferToEmitC : public OpConversionPattern<AsyncOp> {
  using OpConversionPattern<AsyncOp>::OpConversionPattern;

  explicit PTOAsyncTransferToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                   StringRef callee)
      : OpConversionPattern<AsyncOp>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(AsyncOp op, typename AsyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value dstGT = dst;
    Value srcGT = src;
    if (!isEmitCGlobalTensorLikeType(dstGT.getType())) {
      auto dstMrTy = dyn_cast<MemRefType>(op.getDst().getType());
      if (!dstMrTy)
        return rewriter.notifyMatchFailure(op, "expected dst to lower to GlobalTensor or memref");
      dstGT = buildGlobalTensorFromMemref(rewriter, op.getLoc(), dst, dstMrTy,
                                          op.getDst().getDefiningOp()
                                              ? op.getDst().getDefiningOp()
                                              : op.getOperation());
    }
    if (!isEmitCGlobalTensorLikeType(srcGT.getType())) {
      auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType());
      if (!srcMrTy)
        return rewriter.notifyMatchFailure(op, "expected src to lower to GlobalTensor or memref");
      srcGT = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                          op.getSrc().getDefiningOp()
                                              ? op.getSrc().getDefiningOp()
                                              : op.getOperation());
    }
    if (!dstGT || !srcGT)
      return rewriter.notifyMatchFailure(op, "failed to build GlobalTensor operands");

    Type eventTy = this->getTypeConverter()->convertType(op.getEvent().getType());
    if (!eventTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async event type");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{eventTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{dstGT, srcGT, peelUnrealized(adaptor.getSession())});
    return success();
  }

  std::string callee;
};

template <typename AsyncEventOp>
struct PTOAsyncEventToEmitC : public OpConversionPattern<AsyncEventOp> {
  explicit PTOAsyncEventToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                StringRef callee)
      : OpConversionPattern<AsyncEventOp>(typeConverter, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(AsyncEventOp op,
                                typename AsyncEventOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        this->getTypeConverter()->convertType(op.getCompleted().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async event result type");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{resultTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getEvent()),
                   peelUnrealized(adaptor.getSession())});
    return success();
  }

  std::string callee;
};

struct PTODeclareTileMemRefToEmitC
    : public OpConversionPattern<mlir::pto::DeclareTileMemRefOp> {
  using OpConversionPattern<
      mlir::pto::DeclareTileMemRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareTileMemRefOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type convertedType = getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert declare_tile_memref result type");
    rewriter.replaceOp(op, makeEmitCOpaqueConstant(rewriter, op.getLoc(),
                                                   convertedType, "nullptr"));
    return success();
  }
};

struct PTODeclareEventIdArrayToEmitC
    : public OpConversionPattern<mlir::pto::DeclareEventIdArrayOp> {
  using OpConversionPattern<
      mlir::pto::DeclareEventIdArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareEventIdArrayOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type arrayTy = getTypeConverter()->convertType(op.getArray().getType());
    if (!arrayTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map declared eventid_array type");

    auto array = rewriter
                     .create<emitc::VariableOp>(
                         op.getLoc(), arrayTy,
                         emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                     .getResult();
    rewriter.replaceOp(op, array);
    return success();
  }
};

struct PTOEventIdArrayGetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArrayGetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArrayGetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = peelUnrealized(adaptor.getArray());
    Value index = peelUnrealized(adaptor.getIndex());

    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map eventid_array get result type");

    auto load =
        rewriter.create<emitc::SubscriptOp>(op.getLoc(), resultTy, array, index);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct PTOEventIdArraySetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArraySetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArraySetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = peelUnrealized(adaptor.getArray());
    Value index = peelUnrealized(adaptor.getIndex());
    Value value = peelUnrealized(adaptor.getValue());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__EVENTID_ARRAY_STORE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{array, index, value});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTPushToEmitC : public OpConversionPattern<mlir::pto::TPushOp> {
  PTOTPushToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                  PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TPushOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TPushOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    // Read the tile type token from the already-converted OpaqueType, which
    // preserves the exact layout produced by BindTileOp / PointerCastOp EmitC.
    Value convertedTile = peelUnrealized(adaptor.getTile());
    auto opaqueT = dyn_cast<emitc::OpaqueType>(convertedTile.getType());
    if (!opaqueT || !opaqueT.getValue().contains("Tile<"))
      return rewriter.notifyMatchFailure(op, "failed to resolve tile token");
    std::string tileTok = opaqueT.getValue().str();
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    std::string callee =
        "TPUSH<" + *pipeTok + ", " + tileTok + ", " + *splitTok + ">";
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getPipeHandle()), convertedTile});
    return success();
  }

  PTOArch targetArch;
};

struct PTOTPopToEmitC : public OpConversionPattern<mlir::pto::TPopOp> {
  PTOTPopToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                 PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TPopOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TPopOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    Value convertedTile = peelUnrealized(adaptor.getTile());
    auto opaqueT = dyn_cast<emitc::OpaqueType>(convertedTile.getType());
    if (!opaqueT || !opaqueT.getValue().contains("Tile<"))
      return rewriter.notifyMatchFailure(op, "failed to resolve tile token");
    std::string tileTok = opaqueT.getValue().str();
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    std::string callee =
        "TPOP<" + *pipeTok + ", " + tileTok + ", " + *splitTok + ">";
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getPipeHandle()), convertedTile});
    return success();
  }

  PTOArch targetArch;
};

struct PTOTFreeToEmitC : public OpConversionPattern<mlir::pto::TFreeOp> {
  PTOTFreeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                  PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TFreeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TFreeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    std::string callee =
        "TFREE<" + *pipeTok + ", " + *splitTok + ">";
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getPipeHandle())});
    return success();
  }

  PTOArch targetArch;
};

//===----------------------------------------------------------------------===//
// populate patterns
//===----------------------------------------------------------------------===
struct ReinterpretCastToEmitC : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto resMrTy = dyn_cast<MemRefType>(op.getType());
    if (!resMrTy)
      return failure();

    auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(resMrTy.getMemorySpace());
    const bool isGm = (!asAttr || asAttr.getAddressSpace() == pto::AddressSpace::GM);

    bool emitAddPtrTrace = op->hasAttr("pto.addptr_trace");
    Value source = peelUnrealized(adaptor.getSource());
    auto offsets = adaptor.getOffsets();
    Value offsetVal = offsets.empty() ? Value() : offsets[0];

    // GM: keep pointer arithmetic.
    if (isGm) {
      if (!offsetVal) {
        rewriter.replaceOp(op, source);
        return success();
      }

      Type resultType = getTypeConverter()->convertType(op.getType());
      if (!resultType)
        return failure();

      auto addOp = rewriter.create<emitc::AddOp>(loc, resultType, source, offsetVal);
      if (emitAddPtrTrace) {
        rewriter.setInsertionPointAfter(addOp);
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "PTOAS__ADDPTR_TRACE",
            ArrayAttr{}, ArrayAttr{},
            ValueRange{addOp.getResult(), source, offsetVal});
      }
      rewriter.replaceOp(op, addOp.getResult());
      return success();
    }

    // UB/L1/L0 tiles: materialize a new Tile view by assigning an adjusted
    // underlying pointer (in elements).
    pto::AddressSpace as = asAttr.getAddressSpace();

    // Element type token.
    Type elemTy = resMrTy.getElementType();
    std::string elemTok = getEmitCScalarTypeToken(elemTy);
    int64_t elemBytes = getEmitCScalarByteWidth(elemTy);

    // Tile role.
    const char *roleTok = "TileType::Vec";
    switch (as) {
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
    case pto::AddressSpace::GM:
      roleTok = "TileType::Vec";
      break;
    }

    // Shape (fallback to 32x32).
    int64_t rows = 32, cols = 32;
    if (resMrTy.getRank() >= 2 && resMrTy.hasStaticShape()) {
      rows = resMrTy.getDimSize(0);
      cols = resMrTy.getDimSize(1);
    }
    int64_t templateRows =
        renderTileTemplateDim(rows, elemTy, pto::BLayout::RowMajor, 0);
    int64_t templateCols =
        renderTileTemplateDim(cols, elemTy, pto::BLayout::RowMajor, 1);

    // Keep a conservative default config for now.
    std::string tileTypeStr =
        std::string("Tile<") + roleTok + ", " + elemTok + ", " +
        std::to_string(templateRows) + ", " + std::to_string(templateCols) +
        ", BLayout::RowMajor, " + std::to_string(templateRows) + ", " +
        std::to_string(templateCols) +
        ", SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>";

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value tile = rewriter
                     .create<emitc::VariableOp>(loc, tileType,
                                                emitc::OpaqueAttr::get(ctx, ""))
                     .getResult();

    // Compute an integer address and assign it to the new tile.
    // NOTE: pto-isa TASSIGN requires an integral address (not a pointer).
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    auto rcU64 = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});

    // Non-GM reinterpret_cast operands come from UB/L1/L0 tiles.
    // We need the underlying address, but `__cce_get_tile_ptr()` is only valid
    // inside `__tf__` functions. Use `tile.data()` (via a post-processed marker)
    // and compute the adjusted address in bytes.
    Value rawPtr = source;
    if (auto ot = dyn_cast<emitc::OpaqueType>(source.getType())) {
      // Only Tiles have a `.data()` member. For plain address-space pointers
      // (e.g. `__ubuf__ float*`), use the pointer value directly.
      if (ot.getValue().starts_with("Tile<")) {
        rawPtr = materializeTileDataValue(rewriter, loc, source, as, elemTok);
      }
    }

    Value baseAddr = rawPtr;
    if (isSetFFTsPointerLikeType(rawPtr.getType())) {
      baseAddr = rewriter
                     .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                                  /*args=*/ArrayAttr{},
                                                  /*templateArgs=*/rcU64,
                                                  /*operands=*/ValueRange{rawPtr})
                     .getResult(0);
    } else if (rawPtr.getType() != u64Ty) {
      baseAddr = rewriter.create<emitc::CastOp>(loc, u64Ty, rawPtr).getResult();
    }

    Value addr = baseAddr;
    if (offsetVal) {
      Value offU64 = offsetVal;
      if (offU64.getType() != u64Ty)
        offU64 = rewriter.create<emitc::CastOp>(loc, u64Ty, offU64).getResult();

      auto bytesAttr = emitc::OpaqueAttr::get(ctx, std::to_string(elemBytes));
      Value bytesVal = rewriter.create<emitc::ConstantOp>(loc, u64Ty, bytesAttr);
      Value byteOff = rewriter.create<emitc::MulOp>(loc, u64Ty, offU64, bytesVal);
      addr = rewriter.create<emitc::AddOp>(loc, u64Ty, baseAddr, byteOff);
    }

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         /*args=*/ArrayAttr{},
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/ValueRange{tile, addr});

    rewriter.replaceOp(op, tile);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.taddc lowering -> TADDC(dst, src0, src1, src2)
//===----------------------------------------------------------------------===//

struct PTOTAddCToTADDC : public OpConversionPattern<pto::TAddCOp> {
  using OpConversionPattern<pto::TAddCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src2 = peelUnrealized(adaptor.getSrc2());
    Value dst  = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TADDC yet.
    // Decompose: dst = src0 + src1 + src2
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, dst, src2});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tadds lowering -> TADDS(dst, src, scalar)
//===----------------------------------------------------------------------===//

struct PTOAddSToTADDS : public OpConversionPattern<pto::TAddSOp> {
  using OpConversionPattern<pto::TAddSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value dst    = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TADDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.taddsc lowering -> TADDSC(dst, src0, scalar, src1)
//===----------------------------------------------------------------------===//

struct PTOAddSCToTADDSC : public OpConversionPattern<pto::TAddSCOp> {
  using OpConversionPattern<pto::TAddSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0    = peelUnrealized(adaptor.getSrc0());
    Value scalar  = peelUnrealized(adaptor.getScalar());
    Value src1    = peelUnrealized(adaptor.getSrc1());
    Value dst     = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TADDSC yet.
    // Decompose: dst = src0 + scalar + src1
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, scalar});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, dst, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOTAndToEmitC : public OpConversionPattern<pto::TAndOp> {
  using OpConversionPattern<pto::TAndOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAndOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a   = peelUnrealized(adaptor.getSrc0());
    Value b   = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TAND",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, a, b});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOConcatToEmitC : public OpConversionPattern<pto::TConcatOp> {
  using OpConversionPattern<pto::TConcatOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TConcatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TCONCAT",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOConcatidxToEmitC : public OpConversionPattern<pto::TConcatidxOp> {
  using OpConversionPattern<pto::TConcatidxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TConcatidxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src0Idx = peelUnrealized(adaptor.getSrc0Idx());
    Value src1Idx = peelUnrealized(adaptor.getSrc1Idx());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TCONCAT",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOAndSToEmitC : public OpConversionPattern<pto::TAndSOp> {
  using OpConversionPattern<pto::TAndSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAndSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TANDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOTCIToEmitC : public OpConversionPattern<pto::TCIOp> {
  using OpConversionPattern<pto::TCIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value S = peelUnrealized(adaptor.getOperands()[0]);

    // The TCI scalar template parameter should follow the original PTO IR
    // scalar type, not the converted EmitC value type.
    std::string scalarTok = "int32_t";
    if (auto it = dyn_cast<IntegerType>(op->getOperand(0).getType())) {
      bool isUnsigned = it.isUnsigned();
      if (it.getWidth() == 16)
        scalarTok = isUnsigned ? "uint16_t" : "int16_t";
      else
        scalarTok = isUnsigned ? "uint32_t" : "int32_t";
    }

    // descending -> "0"/"1"
    std::string descTok = op.getDescending() ? "1" : "0";

    ArrayAttr targs;
    if (auto ot = dst.getType().dyn_cast<emitc::OpaqueType>()) {
      std::string tileTok = ot.getValue().str(); // "Tile<...>"
      targs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, tileTok),
          emitc::OpaqueAttr::get(ctx, scalarTok),
          emitc::OpaqueAttr::get(ctx, descTok),
      });
    } else {
      targs = rewriter.getArrayAttr({});
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCI",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, S});

    rewriter.eraseOp(op);
    return success();
  }
};
static std::string cmpModeTok(pto::CmpModeAttr a) {
  // 生成 "CmpMode::GT" 这种 token
  auto m = a.getValue(); // 取 enum
  switch (m) {
    case pto::CmpMode::EQ: return "CmpMode::EQ";
    case pto::CmpMode::NE: return "CmpMode::NE";
    case pto::CmpMode::LT: return "CmpMode::LT";
    case pto::CmpMode::LE: return "CmpMode::LE";
    case pto::CmpMode::GT: return "CmpMode::GT";
    case pto::CmpMode::GE: return "CmpMode::GE";
  }
  return "CmpMode::EQ";
}
struct PTOColExpandToEmitC : public OpConversionPattern<pto::TColExpandOp> {
  using OpConversionPattern<pto::TColExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPAND",
        /*args=*/ArrayAttr(),           
        /*templateArgs=*/ArrayAttr(),
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandMulToEmitC : public OpConversionPattern<pto::TColExpandMulOp> {
  using OpConversionPattern<pto::TColExpandMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDMUL",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandAddToEmitC : public OpConversionPattern<pto::TColExpandAddOp> {
  using OpConversionPattern<pto::TColExpandAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDADD",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandDivToEmitC : public OpConversionPattern<pto::TColExpandDivOp> {
  using OpConversionPattern<pto::TColExpandDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDDIV",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandExpdifToEmitC
    : public OpConversionPattern<pto::TColExpandExpdifOp> {
  using OpConversionPattern<pto::TColExpandExpdifOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandExpdifOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDEXPDIF",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandSubToEmitC : public OpConversionPattern<pto::TColExpandSubOp> {
  using OpConversionPattern<pto::TColExpandSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDSUB",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandMaxToEmitC : public OpConversionPattern<pto::TColExpandMaxOp> {
  using OpConversionPattern<pto::TColExpandMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDMAX",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandMinToEmitC : public OpConversionPattern<pto::TColExpandMinOp> {
  using OpConversionPattern<pto::TColExpandMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDMIN",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTTriToEmitC : public OpConversionPattern<pto::TTriOp> {
  using OpConversionPattern<pto::TTriOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TTriOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value diagonal = peelUnrealized(adaptor.getDiagonal());

    ArrayAttr templateArgs;
    if (auto dstOT = dst.getType().dyn_cast<emitc::OpaqueType>()) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, std::to_string(op.getUpperOrLower())),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value, 2> operands{dst, diagonal};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTRI",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOCmpToEmitC : public OpConversionPattern<pto::TCmpOp> {
  using OpConversionPattern<pto::TCmpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
	
    Value dst  = peelUnrealized(adaptor.getDst());
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());

    std::string tok = "CmpMode::EQ";
    if (auto a = op.getCmpModeAttr())
      tok = cmpModeTok(a);

    auto modeTy = emitc::OpaqueType::get(ctx, "CmpMode");
    Value modeVal = rewriter.create<emitc::ConstantOp>(
        loc, modeTy, emitc::OpaqueAttr::get(ctx, tok));

    auto argsAttr = rewriter.getArrayAttr({});

    rewriter.create<emitc::CallOpaqueOp>(
        loc,
        TypeRange{},
        "TCMP",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, modeVal});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOCmpSToEmitC : public OpConversionPattern<pto::TCmpSOp> {
  using OpConversionPattern<pto::TCmpSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCmpSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());

    // cmpMode -> token
    auto cmpAttr = op.getCmpModeAttr();          // PTO_CmpModeAttr
    std::string tok = cmpModeTok(cmpAttr);

    auto modeTy = emitc::OpaqueType::get(ctx, "CmpMode");
    Value modeVal = rewriter.create<emitc::ConstantOp>(
        loc, modeTy, emitc::OpaqueAttr::get(ctx, tok));

    rewriter.create<emitc::CallOpaqueOp>(
        loc,
        TypeRange{},
        "TCMPS",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, scalar, modeVal});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOColMaxToEmitC : public OpConversionPattern<pto::TColMaxOp> {
  using OpConversionPattern<pto::TColMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TCOLMAX(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLMAX",
        /*args=*/ArrayAttr{},          // default: print all operands
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMaxToEmitC : public OpConversionPattern<pto::TColArgMaxOp> {
  using OpConversionPattern<pto::TColArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLARGMAX",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColMinToEmitC : public OpConversionPattern<pto::TColMinOp> {
  using OpConversionPattern<pto::TColMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TCOLMIN(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLMIN",
        /*args=*/ArrayAttr{},          // default: print all operands
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMinToEmitC : public OpConversionPattern<pto::TColArgMinOp> {
  using OpConversionPattern<pto::TColArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLARGMIN",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColSumToEmitC : public OpConversionPattern<pto::TColSumOp> {
  using OpConversionPattern<pto::TColSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // Check if tmp exists before accessing it
    if (op.getTmp()) {
      // Format 2: with tmp and isBinary
      Value tmp = peelUnrealized(adaptor.getTmp());
      bool isBinary = false;
      if (auto a = op.getIsBinaryAttr())
        isBinary = a.getValue();

      auto boolTy = emitc::OpaqueType::get(ctx, "bool");
      auto tok = isBinary ? "true" : "false";
      Value isBinaryVal = rewriter.create<emitc::ConstantOp>(
          loc, boolTy, emitc::OpaqueAttr::get(ctx, tok));

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TCOLSUM",
          /*args=*/ArrayAttr(),             
          /*templateArgs=*/ArrayAttr(),
          /*operands=*/ValueRange{dst, src, tmp, isBinaryVal});
    } else {
      // Format 1: without tmp and isBinary
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TCOLSUM",
          /*args=*/ArrayAttr(),             
          /*templateArgs=*/ArrayAttr(),
          /*operands=*/ValueRange{dst, src});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColProdToEmitC : public OpConversionPattern<pto::TColProdOp> {
  using OpConversionPattern<pto::TColProdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColProdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLPROD",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
static std::string roundModeTok(mlir::pto::RoundModeAttr attr) {
  using RM = mlir::pto::RoundMode;
  switch (attr.getValue()) {
  case RM::NONE:      return "RoundMode::CAST_NONE";
  case RM::RINT:      return "RoundMode::CAST_RINT";
  case RM::ROUND:     return "RoundMode::CAST_ROUND";
  case RM::FLOOR:     return "RoundMode::CAST_FLOOR";
  case RM::CEIL:      return "RoundMode::CAST_CEIL";
  case RM::TRUNC:     return "RoundMode::CAST_TRUNC";
  case RM::ODD:       return "RoundMode::CAST_ODD";
  case RM::CAST_RINT: return "RoundMode::CAST_RINT";
  }
  return "RoundMode::CAST_RINT";
}
static std::string saturationModeTok(mlir::pto::SaturationModeAttr attr) {
  using SM = mlir::pto::SaturationMode;
  switch (attr.getValue()) {
  case SM::ON:  return "SaturationMode::ON";
  case SM::OFF: return "SaturationMode::OFF";
  }
  return "SaturationMode::ON";
}
struct PTOCvtToEmitC : public OpConversionPattern<pto::TCvtOp> {
  using OpConversionPattern<pto::TCvtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCvtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value tmp = adaptor.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value{};

    pto::RoundModeAttr rmAttr = op.getRmodeAttr();
    std::string rmTok = rmAttr ? roundModeTok(rmAttr)
                               : std::string("RoundMode::CAST_RINT");
    auto rmodeTy = emitc::OpaqueType::get(ctx, "RoundMode");
    Value rmodeVal = rewriter.create<emitc::ConstantOp>(
        loc, rmodeTy, emitc::OpaqueAttr::get(ctx, rmTok));

    SmallVector<Value, 5> operands{dst, src};
    if (tmp)
      operands.push_back(tmp);
    operands.push_back(rmodeVal);

    if (auto satAttr = op.getSatModeAttr()) {
      auto satModeTy = emitc::OpaqueType::get(ctx, "SaturationMode");
      Value satModeVal = rewriter.create<emitc::ConstantOp>(
          loc, satModeTy, emitc::OpaqueAttr::get(ctx, saturationModeTok(satAttr)));
      operands.push_back(satModeVal);
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCVT",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTORandomToEmitC : public OpConversionPattern<pto::TRandomOp> {
  using OpConversionPattern<pto::TRandomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRandomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    SmallVector<Value, 7> operands{
        dst,
        peelUnrealized(adaptor.getKey0()),
        peelUnrealized(adaptor.getKey1()),
        peelUnrealized(adaptor.getCounter0()),
        peelUnrealized(adaptor.getCounter1()),
        peelUnrealized(adaptor.getCounter2()),
        peelUnrealized(adaptor.getCounter3()),
    };
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, std::to_string(op.getRounds()))});

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "PTOAS__TRANDOM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tdiv lowering -> TDIV(dst, src0, src1)
//===----------------------------------------------------------------------===//

