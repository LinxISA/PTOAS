// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

struct PTODivToTDIV : public OpConversionPattern<pto::TDivOp> {
  using OpConversionPattern<pto::TDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TDIV",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tdivs lowering -> TDIVS(dst, src, scalar)  or  TDIVS(dst, scalar, src)
// Order is determined by operand types: if src is tile_buf, order is (tile, scalar)
// Otherwise, order is (scalar, tile)
//===----------------------------------------------------------------------===//

struct PTODivSToEmitC : public OpConversionPattern<pto::TDivSOp> {
  using OpConversionPattern<pto::TDivSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());
    // Preserve source order from textual parse:
    // ins(tile, scalar)   -> TDIVS(dst, tile, scalar)
    // ins(scalar, tile)   -> TDIVS(dst, scalar, tile)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TDIVS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tdivs (TDivSOp) lowering -> TDIVS(dst, src, scalar)  or  TDIVS(dst, scalar, src)
// Order is determined by operand types: if src is tile_buf, order is (tile, scalar)
// Otherwise, order is (scalar, tile)
//===----------------------------------------------------------------------===//

struct PTOTDivSToEmitC : public OpConversionPattern<pto::TDivSOp> {
  using OpConversionPattern<pto::TDivSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TDIVS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.texp lowering -> TEXP(dst, src)
//===----------------------------------------------------------------------===//

struct PTOExpToEmitC : public OpConversionPattern<pto::TExpOp> {
  using OpConversionPattern<pto::TExpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXP",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.texpands lowering -> TEXPANDS(dst, scalar)
//===----------------------------------------------------------------------===//

struct PTOExpandsToEmitC : public OpConversionPattern<pto::TExpandsOp> {
  using OpConversionPattern<pto::TExpandsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExpandsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXPANDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.textract lowering -> TEXTRACT(dst, src, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOExtractToEmitC : public OpConversionPattern<pto::TExtractOp> {
  using OpConversionPattern<pto::TExtractOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0  = peelUnrealized(adaptor.getIndexRow());
    Value c0  = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXTRACT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.textract_fp lowering -> TEXTRACT_FP(dst, src, fp, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOExtractFPToEmitC : public OpConversionPattern<pto::TExtractFPOp> {
  using OpConversionPattern<pto::TExtractFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExtractFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0 = peelUnrealized(adaptor.getIndexRow());
    Value c0 = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXTRACT_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, fp, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tinsert lowering -> TINSERT(dst, src, indexRow, indexCol)
// Keep lowering arch-agnostic and let PTO-ISA infer proper A5 path.
//===----------------------------------------------------------------------===//

struct PTOInsertToEmitC : public OpConversionPattern<pto::TInsertOp> {
  using OpConversionPattern<pto::TInsertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TInsertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0  = peelUnrealized(adaptor.getIndexRow());
    Value c0  = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TINSERT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tinsert_fp lowering -> TINSERT_FP(dst, src, fp, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOInsertFPToEmitC : public OpConversionPattern<pto::TInsertFPOp> {
  using OpConversionPattern<pto::TInsertFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TInsertFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0 = peelUnrealized(adaptor.getIndexRow());
    Value c0 = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TINSERT_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, fp, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad lowering -> TFILLPAD(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadToEmitC : public OpConversionPattern<pto::TFillPadOp> {
  using OpConversionPattern<pto::TFillPadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad_inplace lowering -> TFILLPAD_INPLACE(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadInplaceToEmitC
    : public OpConversionPattern<pto::TFillPadInplaceOp> {
  using OpConversionPattern<pto::TFillPadInplaceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadInplaceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD_INPLACE",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad_expand lowering -> TFILLPAD_EXPAND(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadExpandToEmitC
    : public OpConversionPattern<pto::TFillPadExpandOp> {
  using OpConversionPattern<pto::TFillPadExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD_EXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tgather lowering
// - Index form  : TGATHER(dst, src0, indices, tmp)
// - Compare form: TGATHER<DstT, SrcT, CDstT, TmpT, CmpMode::GT, 7>(dst, src0, kValue, cdst, tmp)
// - Mask form : TGATHER<dstTileTok, srcTileTok, pto::MaskPattern::Pxxxx>(dst, src0)
//===----------------------------------------------------------------------===//

static std::string maskPatternTok(mlir::pto::MaskPatternAttr a) {

  auto v = a.getValue(); // enum
  return (std::string("pto::MaskPattern::") + mlir::pto::stringifyMaskPattern(v).str());
}

struct PTOGatherToEmitC : public OpConversionPattern<pto::TGatherOp> {
  using OpConversionPattern<pto::TGatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst  = peelUnrealized(adaptor.getDst());
    Value src0 = peelUnrealized(adaptor.getSrc());

    auto getOpaqueTok = [&](Value v, StringRef name) -> FailureOr<std::string> {
      if (auto ot = v.getType().dyn_cast<emitc::OpaqueType>())
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(op, (name + " must be emitc::OpaqueType (tile)").str());
    };

    // Case 1: index-based TGATHER(dst, src0, indices, tmp)
    if (Value idx = adaptor.getIndices()) {
      idx = peelUnrealized(idx);
      Value tmp = peelUnrealized(adaptor.getTmp());

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TGATHER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
          /*operands=*/ValueRange{dst, src0, idx, tmp});

      rewriter.eraseOp(op);
      return success();
    }

    // Case 2: compare-based TGATHER<DstT, SrcT, CDstT, TmpT, CmpMode::GT, offset>(...)
    if (Value cdst = adaptor.getCdst()) {
      cdst = peelUnrealized(cdst);
      Value tmp = peelUnrealized(adaptor.getTmp());
      Value kValue = peelUnrealized(adaptor.getKValue());

      auto dstTokOr = getOpaqueTok(dst, "dst");
      auto srcTokOr = getOpaqueTok(src0, "src0");
      auto cdstTokOr = getOpaqueTok(cdst, "cdst");
      auto tmpTokOr = getOpaqueTok(tmp, "tmp");
      if (failed(dstTokOr) || failed(srcTokOr) || failed(cdstTokOr) || failed(tmpTokOr))
        return failure();

      auto cmpAttr = op.getCmpModeAttr();
      std::string cmpTok = cmpAttr ? cmpModeTok(cmpAttr) : "CmpMode::EQ";
      int64_t offset = 0;
      if (auto offsetAttr = op.getOffsetAttr())
        offset = offsetAttr.getInt();

      auto targs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, *dstTokOr),
          emitc::OpaqueAttr::get(ctx, *srcTokOr),
          emitc::OpaqueAttr::get(ctx, *cdstTokOr),
          emitc::OpaqueAttr::get(ctx, *tmpTokOr),
          emitc::OpaqueAttr::get(ctx, cmpTok),
          emitc::OpaqueAttr::get(ctx, std::to_string(offset)),
      });

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TGATHER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
          /*operands=*/ValueRange{dst, src0, kValue, cdst, tmp});

      rewriter.eraseOp(op);
      return success();
    }

    // Case 3: mask-pattern TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src0)
    auto mp = op.getMaskPatternAttr();
    if (!mp)
      return rewriter.notifyMatchFailure(op, "expected maskPattern, indices, or cdst on tgather");

    auto dstTokOr = getOpaqueTok(dst, "dst");
    auto srcTokOr = getOpaqueTok(src0, "src0");
    if (failed(dstTokOr) || failed(srcTokOr))
      return failure();

    // mp is an EnumAttr; stringify name is "P0101" etc.
    // We emit MaskPattern::P0101 (because generated C++ has `using namespace pto;`)
    std::string mpTok = std::string("MaskPattern::") +
                        mlir::pto::stringifyMaskPattern(mp.getValue()).str();

    auto targs = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, *dstTokOr),
        emitc::OpaqueAttr::get(ctx, *srcTokOr),
        emitc::OpaqueAttr::get(ctx, mpTok),
    });

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHER",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, src0});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOGatherbToEmitC : public OpConversionPattern<pto::TGatherBOp> {
  using OpConversionPattern<pto::TGatherBOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGatherBOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src     = peelUnrealized(adaptor.getSrc());
    Value offsets = peelUnrealized(adaptor.getOffsets());
    Value dst     = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHERB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, offsets});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TLOG lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOLogToEmitC : public OpConversionPattern<pto::TLogOp> {
  using OpConversionPattern<pto::TLogOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLogOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TLOG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};



//===----------------------------------------------------------------------===//
// TLRELU lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

	struct PTOLReluToEmitC : public OpConversionPattern<pto::TLReluOp> {
	  using OpConversionPattern<pto::TLReluOp>::OpConversionPattern;
	
	  LogicalResult matchAndRewrite(pto::TLReluOp op, OpAdaptor adaptor,
	                                ConversionPatternRewriter &rewriter) const override {
	    auto loc = op.getLoc();
	
	    Value src = peelUnrealized(adaptor.getSrc());
	    Value slope = peelUnrealized(adaptor.getSlope());
	    Value dst = peelUnrealized(adaptor.getDst());

            SmallVector<Value, 3> operands{dst, src, slope};

	    rewriter.create<emitc::CallOpaqueOp>(
	        loc, TypeRange{}, "TLRELU",
	        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
	        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMAX lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMaxToEmitC : public OpConversionPattern<pto::TMaxOp> {
  using OpConversionPattern<pto::TMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMAXS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

	struct PTOMaxSToEmitC : public OpConversionPattern<pto::TMaxSOp> {
	  using OpConversionPattern<pto::TMaxSOp>::OpConversionPattern;
	
	  LogicalResult matchAndRewrite(pto::TMaxSOp op, OpAdaptor adaptor,
	                                ConversionPatternRewriter &rewriter) const override {
	    auto loc = op.getLoc();
	
	    Value src0 = peelUnrealized(adaptor.getSrc());
	    Value scalar = peelUnrealized(adaptor.getScalar());
	    Value dst  = peelUnrealized(adaptor.getDst());

	    SmallVector<Value, 3> operands{dst, src0, scalar};
	    rewriter.create<emitc::CallOpaqueOp>(
	        loc, TypeRange{}, "TMAXS",
	        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// TMIN lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMinToEmitC : public OpConversionPattern<pto::TMinOp> {
  using OpConversionPattern<pto::TMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (fix APFloat -> FloatAttr)  (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMinsToEmitC : public OpConversionPattern<pto::TMinSOp> {
  using OpConversionPattern<pto::TMinSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMinSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMINS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TMOV op -> EmitC)
//===----------------------------------------------------------------------===//

struct PTOMovToEmitC : public OpConversionPattern<pto::TMovOp> {
  using OpConversionPattern<pto::TMovOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMovOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value fp;
    if (op.getFp())
      fp = peelUnrealized(adaptor.getFp());
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = peelUnrealized(adaptor.getPreQuantScalar());

    auto dstOT = dst.getType().dyn_cast<emitc::OpaqueType>();
    auto srcOT = src.getType().dyn_cast<emitc::OpaqueType>();
    if (!dstOT || !srcOT)
      return rewriter.notifyMatchFailure(
          op, "tmov lowering expects opaque dst/src types");

    auto modeTok = [&](pto::AccToVecMode mode) -> StringRef {
      switch (mode) {
      case pto::AccToVecMode::SingleModeVec0:
        return "pto::AccToVecMode::SingleModeVec0";
      case pto::AccToVecMode::SingleModeVec1:
        return "pto::AccToVecMode::SingleModeVec1";
      case pto::AccToVecMode::DualModeSplitM:
        return "pto::AccToVecMode::DualModeSplitM";
      case pto::AccToVecMode::DualModeSplitN:
        return "pto::AccToVecMode::DualModeSplitN";
      }
      llvm_unreachable("unknown AccToVecMode");
    };

    auto modeAttr = op.getAccToVecModeAttr();
    auto reluTok = [&](pto::ReluPreMode mode) -> StringRef {
      switch (mode) {
      case pto::ReluPreMode::NoRelu:
        return "ReluPreMode::NoRelu";
      case pto::ReluPreMode::NormalRelu:
        return "ReluPreMode::NormalRelu";
      }
      llvm_unreachable("unknown ReluPreMode");
    };

    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;

    SmallVector<Value, 4> operands{dst, src};
    SmallVector<Attribute, 5> templateArgVec{
        emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
        emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
    };
    StringRef callee = "TMOV";

    if (hasFp) {
      auto fpOT = fp.getType().dyn_cast<emitc::OpaqueType>();
      if (!fpOT)
        return rewriter.notifyMatchFailure(
            op, "tmov fp lowering expects opaque fp type");
      operands.push_back(fp);
      templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()));
      if (hasMode)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, modeTok(modeAttr.getValue())));
      if (hasMode || reluNonDefault)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
      callee = hasMode ? "TMOV" : "TMOV_FP";
    } else if (hasPreQuantScalar) {
      operands.push_back(preQuantScalar);
      if (hasMode)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, modeTok(modeAttr.getValue())));
      if (hasMode || reluNonDefault)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
    } else if (hasMode) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, modeTok(modeAttr.getValue())));
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
    } else if (reluNonDefault) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
    }

    ArrayAttr templateArgs =
        templateArgVec.size() == 2 && !hasFp && !hasPreQuantScalar &&
                !hasMode && !reluNonDefault
            ? ArrayAttr{}
            : rewriter.getArrayAttr(templateArgVec);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, callee,
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMOV_FP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMovFPToEmitC : public OpConversionPattern<pto::TMovFPOp> {
  using OpConversionPattern<pto::TMovFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMovFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value fp  = peelUnrealized(adaptor.getFp());

    // TMOV_FP<DstTileData, AccTile, FbTile>(dstTileData, cTile, fbTile)
    ArrayAttr templateArgs;
    auto dstOT = dst.getType().dyn_cast<emitc::OpaqueType>();
    auto srcOT = src.getType().dyn_cast<emitc::OpaqueType>();
    auto fpOT  = fp.getType().dyn_cast<emitc::OpaqueType>();
    if (dstOT && srcOT && fpOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value, 3> operands{dst, src, fp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMOV_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOQuantToEmitC : public OpConversionPattern<pto::TQuantOp> {
  using OpConversionPattern<pto::TQuantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TQuantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value fp  = peelUnrealized(adaptor.getFp());

    // Optional offset (INT8_ASYM only): passed as pointer (&offset)
    Value offsetPtr;
    if (op.getOffset()) {
      Value offset = peelUnrealized(adaptor.getOffset());
      auto offsetOT = offset.getType().dyn_cast<emitc::OpaqueType>();
      if (offsetOT) {
        offsetPtr = rewriter
                        .create<emitc::ApplyOp>(
                            loc, emitc::PointerType::get(offsetOT), "&", offset)
                        .getResult();
      }
    }

    // TQUANT<QuantType, DstTile, SrcTile, FpTile>(dst, src, fp[, &offset])
    std::string quantTypeStr =
        op.getQuantType() == pto::QuantType::INT8_SYM
            ? "pto::QuantType::INT8_SYM"
            : "pto::QuantType::INT8_ASYM";
    ArrayAttr templateArgs;
    auto dstOT = dst.getType().dyn_cast<emitc::OpaqueType>();
    auto srcOT = src.getType().dyn_cast<emitc::OpaqueType>();
    auto fpOT  = fp.getType().dyn_cast<emitc::OpaqueType>();
    if (dstOT && srcOT && fpOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, quantTypeStr),
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value> operands{dst, src, fp};
    if (offsetPtr)
      operands.push_back(offsetPtr);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TQUANT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTODequantToEmitC : public OpConversionPattern<pto::TDequantOp> {
  using OpConversionPattern<pto::TDequantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDequantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scale  = peelUnrealized(adaptor.getScale());
    Value offset = peelUnrealized(adaptor.getOffset());

    // TDEQUANT<DstTile, SrcTile, ParaTile>(dst, src, scale, offset)
    ArrayAttr templateArgs;
    auto dstOT   = dst.getType().dyn_cast<emitc::OpaqueType>();
    auto srcOT   = src.getType().dyn_cast<emitc::OpaqueType>();
    auto scaleOT = scale.getType().dyn_cast<emitc::OpaqueType>();
    if (dstOT && srcOT && scaleOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, scaleOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TDEQUANT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/SmallVector<Value>{dst, src, scale, offset});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMRGSORT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMrgSortToEmitC : public OpConversionPattern<pto::TMrgSortOp> {
  using OpConversionPattern<pto::TMrgSortOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMrgSortOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    if (op.isFormat1()) {
      Value src = peelUnrealized(adaptor.getSrcs().front());
      Value dst = peelUnrealized(adaptor.getDsts().front());
      Value blockLen = peelUnrealized(adaptor.getBlockLen());

      SmallVector<Value, 3> operands{dst, src, blockLen};
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TMRGSORT",
          ArrayAttr{}, ArrayAttr{}, operands);
    } else if (op.isFormat2()) {
      // pto-isa API:
      //   TMRGSORT<DstTile, TmpTile, Src0, Src1[, Src2[, Src3]], exhausted>(
      //       dst, executedNumList, tmp, src0, src1[, src2[, src3]]);
      auto *ctx = rewriter.getContext();

      Value dst = peelUnrealized(adaptor.getDsts()[0]);
      Value tmp = peelUnrealized(adaptor.getTmp());
      Value excuted = peelUnrealized(adaptor.getExcuted());

      SmallVector<Value, 4> srcs;
      srcs.reserve(adaptor.getSrcs().size());
      for (Value v : adaptor.getSrcs())
        srcs.push_back(peelUnrealized(v));

      auto dstOT = dst.getType().dyn_cast<emitc::OpaqueType>();
      auto tmpOT = tmp.getType().dyn_cast<emitc::OpaqueType>();
      if (!dstOT || !tmpOT || srcs.size() < 2 || srcs.size() > 4)
        return op.emitOpError("format2 expects dst/tmp tilebufs and 2 to 4 srcs");

      SmallVector<Attribute, 8> targs;
      targs.reserve(2 + srcs.size() + 1);
      targs.push_back(emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()));
      targs.push_back(emitc::OpaqueAttr::get(ctx, tmpOT.getValue().str()));
      for (Value v : srcs) {
        auto ot = v.getType().dyn_cast<emitc::OpaqueType>();
        if (!ot)
          return op.emitOpError("format2 expects tilebuf srcs");
        targs.push_back(emitc::OpaqueAttr::get(ctx, ot.getValue().str()));
      }
      targs.push_back(emitc::OpaqueAttr::get(ctx, op.getExhausted() ? "true" : "false"));
      ArrayAttr templateArgs = rewriter.getArrayAttr(targs);

      SmallVector<Value, 7> operands{dst, excuted, tmp};
      operands.append(srcs.begin(), srcs.end());

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TMRGSORT",
          /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
    } else {
      return op.emitOpError("unsupported mrgsort_dps format");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMulToEmitC : public OpConversionPattern<pto::TMulOp> {
  using OpConversionPattern<pto::TMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMULS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMulsToEmitC : public OpConversionPattern<pto::TMulSOp> {
  using OpConversionPattern<pto::TMulSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMulSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc0());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMULS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNEG DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTONegToEmitC : public OpConversionPattern<pto::TNegOp> {
  using OpConversionPattern<pto::TNegOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TNegOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TNEG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNOT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTONotToEmitC : public OpConversionPattern<pto::TNotOp> {
  using OpConversionPattern<pto::TNotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TNotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TNOT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrToEmitC : public OpConversionPattern<pto::TOrOp> {
  using OpConversionPattern<pto::TOrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TOrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrsToEmitC : public OpConversionPattern<pto::TOrSOp> {
  using OpConversionPattern<pto::TOrSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TOrSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc());
    Value dst  = peelUnrealized(adaptor.getDst());
    // NOTE: The conversion type system may materialize integers as emitc.opaque
    // (e.g. "int32_t"). For EmitC call emission we can pass the scalar through
    // directly without arith casts here.
    Value s = adaptor.getScalar();

    SmallVector<Value, 3> operands{dst, src0, s};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTADD DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartAddToEmitC : public OpConversionPattern<pto::TPartAddOp> {
  using OpConversionPattern<pto::TPartAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMAX DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMaxToEmitC : public OpConversionPattern<pto::TPartMaxOp> {
  using OpConversionPattern<pto::TPartMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMIN DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMinToEmitC : public OpConversionPattern<pto::TPartMinOp> {
  using OpConversionPattern<pto::TPartMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMulToEmitC : public OpConversionPattern<pto::TPartMulOp> {
  using OpConversionPattern<pto::TPartMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPreluToEmitC : public OpConversionPattern<pto::TPReluOp> {
  using OpConversionPattern<pto::TPReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());

    // C++ interface: TPRELU(dst, src0, src1, tmp) — last parameter is tmp.
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRECIP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORecipToEmitC : public OpConversionPattern<pto::TRecipOp> {
  using OpConversionPattern<pto::TRecipOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRecipOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRECIP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOReluToEmitC : public OpConversionPattern<pto::TReluOp> {
  using OpConversionPattern<pto::TReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREM DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemToEmitC : public OpConversionPattern<pto::TRemOp> {
  using OpConversionPattern<pto::TRemOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModToEmitC : public OpConversionPattern<pto::TFModOp> {
  using OpConversionPattern<pto::TFModOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMOD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREMS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemSToEmitC : public OpConversionPattern<pto::TRemSOp> {
  using OpConversionPattern<pto::TRemSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVector<Value, 4> operands{dst, src, scalar, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREMS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModSToEmitC : public OpConversionPattern<pto::TFModSOp> {
  using OpConversionPattern<pto::TFModSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMODS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPAND DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandToEmitC : public OpConversionPattern<pto::TRowExpandOp> {
  using OpConversionPattern<pto::TRowExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowExpandAddToEmitC : public OpConversionPattern<pto::TRowExpandAddOp> {
  using OpConversionPattern<pto::TRowExpandAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowExpandExpdifToEmitC
    : public OpConversionPattern<pto::TRowExpandExpdifOp> {
  using OpConversionPattern<pto::TRowExpandExpdifOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandExpdifOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDEXPDIF",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDDIV DPS/memref op)
//===----------------------------------------------------------------------===//
// Helper: replace or erase based on whether op has results.
static void replaceOrEraseWithOpaqueCall(Operation *op,
                                        StringRef callee,
                                        ArrayRef<Value> args,
                                        ConversionPatternRewriter &rewriter) {
  TypeRange resultTypes = op->getResultTypes();
  auto call = rewriter.create<emitc::CallOpaqueOp>(
      op->getLoc(), resultTypes, callee, ArrayAttr{}, ArrayAttr{}, ValueRange(args));
  if (resultTypes.empty())
    rewriter.eraseOp(op);
  else
    rewriter.replaceOp(op, call.getResults());
}

static void replaceOrEraseWithOpaqueCallAndReturnDst(Operation *op, Value dst,
                                                     StringRef callee,
                                                     ArrayRef<Value> args,
                                                     ConversionPatternRewriter &rewriter) {
  rewriter.create<emitc::CallOpaqueOp>(
      op->getLoc(), TypeRange{}, callee, ArrayAttr{}, ArrayAttr{}, ValueRange(args));
  if (op->getNumResults() == 1)
    rewriter.replaceOp(op, dst);
  else
    rewriter.eraseOp(op);
}

// ---------- TOp ----------
struct PTOTGemvBiasToTGEMV_BIAS
    : public OpConversionPattern<pto::TGemvBiasOp> {
  using OpConversionPattern<pto::TGemvBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TGEMV_BIAS",
                                {dst, a, b, bias}, rewriter);
    return success();
  }
};

struct PTOTGemvMXToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxOp> {
  using OpConversionPattern<pto::TGemvMxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTGemvMXAccToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxAccOp> {
  using OpConversionPattern<pto::TGemvMxAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = peelUnrealized(adaptor.getCIn());
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, cIn, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTGemvMXBiasToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxBiasOp> {
  using OpConversionPattern<pto::TGemvMxBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value bias    = peelUnrealized(adaptor.getBias());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, a, aScale, b, bScale, bias}, rewriter);
    return success();
  }
};

struct PTOTMatmulBiasToTMATMUL_BIAS
    : public OpConversionPattern<pto::TMatmulBiasOp> {
  using OpConversionPattern<pto::TMatmulBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_BIAS",
                                {dst, a, b, bias}, rewriter);
    return success();
  }
};

struct PTOTMatmulMXToTMATMUL_MX
    : public OpConversionPattern<pto::TMatmulMxOp> {
  using OpConversionPattern<pto::TMatmulMxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX",
                                {dst, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTMatmulMXAccToTMATMUL_MX_ACC
    : public OpConversionPattern<pto::TMatmulMxAccOp> {
  using OpConversionPattern<pto::TMatmulMxAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = peelUnrealized(adaptor.getCIn());
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX",
                                {dst, cIn, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTMatmulMXBiasToTMATMUL_MX_BIAS
    : public OpConversionPattern<pto::TMatmulMxBiasOp> {
  using OpConversionPattern<pto::TMatmulMxBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value bias    = peelUnrealized(adaptor.getBias());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX",
                                {dst, a, aScale, b, bScale, bias}, rewriter);
    return success();
  }
};

struct PTORowExpandDivToEmitC : public OpConversionPattern<pto::TRowExpandDivOp> {
  using OpConversionPattern<pto::TRowExpandDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDDIV",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandMulToEmitC : public OpConversionPattern<pto::TRowExpandMulOp> {
  using OpConversionPattern<pto::TRowExpandMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDSUB DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandSubToEmitC : public OpConversionPattern<pto::TRowExpandSubOp> {
  using OpConversionPattern<pto::TRowExpandSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDSUB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowExpandMaxToEmitC : public OpConversionPattern<pto::TRowExpandMaxOp> {
  using OpConversionPattern<pto::TRowExpandMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowExpandMinToEmitC : public OpConversionPattern<pto::TRowExpandMinOp> {
  using OpConversionPattern<pto::TRowExpandMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMAX DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowMaxToEmitC : public OpConversionPattern<pto::TRowMaxOp> {
  using OpConversionPattern<pto::TRowMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowArgMaxToEmitC
    : public OpConversionPattern<pto::TRowArgMaxOp> {
  using OpConversionPattern<pto::TRowArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMIN DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowMinToEmitC : public OpConversionPattern<pto::TRowMinOp> {
  using OpConversionPattern<pto::TRowMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowArgMinToEmitC
    : public OpConversionPattern<pto::TRowArgMinOp> {
  using OpConversionPattern<pto::TRowArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWSUM DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowSumToEmitC : public OpConversionPattern<pto::TRowSumOp> {
  using OpConversionPattern<pto::TRowSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWSUM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowProdToEmitC : public OpConversionPattern<pto::TRowProdOp> {
  using OpConversionPattern<pto::TRowProdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowProdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWPROD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRSQRT DPS/memref op)
// - no-tmp form : TRSQRT(dst, src)
// - tmp form    : TRSQRT(dst, src, tmp)
//===----------------------------------------------------------------------===//

struct PTORsqrtToEmitC : public OpConversionPattern<pto::TRsqrtOp> {
  using OpConversionPattern<pto::TRsqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRsqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    SmallVector<Value, 3> operands{dst, src};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSCATTER DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOScatterToEmitC : public OpConversionPattern<pto::TScatterOp> {
  using OpConversionPattern<pto::TScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value idx = peelUnrealized(adaptor.getIndexes());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, idx};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSCATTER",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSEL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSelToEmitC : public OpConversionPattern<pto::TSelOp> {
  using OpConversionPattern<pto::TSelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value mask = peelUnrealized(adaptor.getMask());
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 5> operands{dst, mask, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSEL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSELS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSelSToEmitC : public OpConversionPattern<pto::TSelSOp> {
  using OpConversionPattern<pto::TSelSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSelSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value mask = peelUnrealized(adaptor.getMask());
    Value src  = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 5> operands{dst, mask, src, tmp, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSELS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShlSToEmitC : public OpConversionPattern<pto::TShlOp> {
  using OpConversionPattern<pto::TShlOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShlOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShrSToEmitC : public OpConversionPattern<pto::TShrOp> {
  using OpConversionPattern<pto::TShrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TSHLS/TSHRS DPS: shift by scalar)
//===----------------------------------------------------------------------===//

struct PTOShlSConstToEmitC : public OpConversionPattern<pto::TShlSOp> {
  using OpConversionPattern<pto::TShlSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShlSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHLS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOShrSConstToEmitC : public OpConversionPattern<pto::TShrSOp> {
  using OpConversionPattern<pto::TShrSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShrSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHRS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (TSORT32 DPS/memref op: ins(src, idx[, tmp]) outs(dst))
//===----------------------------------------------------------------------===//

struct PTOSORT32SToEmitC : public OpConversionPattern<pto::TSort32Op> {
  using OpConversionPattern<pto::TSort32Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSort32Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value tmp = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src, idx, tmp});
    else
      operands.assign({dst, src, idx});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSORT32",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSQRT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSqrtSToEmitC : public OpConversionPattern<pto::TSqrtOp> {
  using OpConversionPattern<pto::TSqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSTORE_FP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOStoreFPSToEmitC : public OpConversionPattern<pto::TStoreFPOp> {
  using OpConversionPattern<pto::TStoreFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TStoreFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, fp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSTORE_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUB DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSToEmitC : public OpConversionPattern<pto::TSubOp> {
  using OpConversionPattern<pto::TSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubCSToEmitC : public OpConversionPattern<pto::TSubCOp> {
  using OpConversionPattern<pto::TSubCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src2 = peelUnrealized(adaptor.getSrc2());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TSUBC yet.
    // Decompose: dst = src0 - src1 + src2
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, dst, src2});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSSToEmitC : public OpConversionPattern<pto::TSubSOp> {
  using OpConversionPattern<pto::TSubSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUBS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBSC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSCToEmitC : public OpConversionPattern<pto::TSubSCOp> {
  using OpConversionPattern<pto::TSubSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TSUBSC yet.
    // Decompose: dst = src0 - scalar + src1
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUBS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, scalar});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, dst, src1});

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORToEmitC : public OpConversionPattern<pto::TXorOp> {
  using OpConversionPattern<pto::TXorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TXorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());
    Value tmp = peelUnrealized(adaptor.getTmp());
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOTTransToEmitC : public OpConversionPattern<pto::TTransOp> {
  using OpConversionPattern<pto::TTransOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TTransOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTRANS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORSToEmitC : public OpConversionPattern<pto::TXorSOp> {
  using OpConversionPattern<pto::TXorSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TXorSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, scalar, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
  struct PTOPrintToTPRINT : public OpConversionPattern<pto::TPrintOp> {
  using OpConversionPattern<pto::TPrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());

    SmallVector<Value, 4> operands{src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRINT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

// pto.print "format", %scalar -> PRINTF("format", scalar)
struct PTOPrintOpToEmitC : public OpConversionPattern<pto::PrintOp> {
  using OpConversionPattern<pto::PrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    std::string fmt = op.getFormat().str();
    if (fmt.empty())
      fmt = "%f";
    std::string quoted = "\"";
    for (char c : fmt) {
      if (c == '"' || c == '\\')
        quoted += '\\';
      else if (c == '\n')
        quoted += "\\n";
      else if (c == '\t')
        quoted += "\\t";
      else
        quoted += c;
    }
    quoted += "\"";

    Value scalar = peelUnrealized(adaptor.getScalar());
    auto argsAttr = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, quoted),
         IntegerAttr::get(IndexType::get(ctx), 0)});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "cce::printf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

// pto.trap -> TRAP()
struct PTOTrapOpToEmitC : public OpConversionPattern<pto::TrapOp> {
  using OpConversionPattern<pto::TrapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TrapOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "trap",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});

    rewriter.eraseOp(op);
    return success();
  }
};

// =============================================================================
// 2. BindTileOp Lowering (FIX: Trace back to physical address)
// =============================================================================
