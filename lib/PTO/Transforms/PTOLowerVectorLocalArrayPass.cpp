// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOLowerVectorLocalArrayPass.cpp ----------------------------------===//

#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWERVECTORLOCALARRAY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static bool isSupportedLocalArrayElementType(Type type) {
  return isa<IntegerType, FloatType>(type);
}

static bool isSupportedVectorArrayType(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 && !vectorType.isScalable() &&
         vectorType.getDimSize(0) > 0 &&
         isSupportedLocalArrayElementType(vectorType.getElementType());
}

static pto::LocalArrayType getLocalArrayType(VectorType vectorType) {
  return pto::LocalArrayType::get(vectorType.getContext(),
                                  vectorType.getDimSize(0),
                                  vectorType.getElementType());
}

static FailureOr<Value>
materializeSingleVectorIndex(Location loc, ArrayRef<int64_t> staticPosition,
                             ValueRange dynamicPosition,
                             PatternRewriter &rewriter) {
  if (staticPosition.size() != 1)
    return failure();

  int64_t staticIndex = staticPosition.front();
  if (staticIndex == ShapedType::kDynamic) {
    if (dynamicPosition.size() != 1)
      return failure();
    return dynamicPosition.front();
  }

  if (!dynamicPosition.empty() || staticIndex < 0)
    return failure();
  return rewriter.create<arith::ConstantIndexOp>(loc, staticIndex).getResult();
}

struct ArithVectorConstantLowering
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto vectorType = dyn_cast<VectorType>(op.getType());
    if (!vectorType || !isSupportedVectorArrayType(vectorType))
      return failure();

    auto elementsAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!elementsAttr)
      return rewriter.notifyMatchFailure(
          op, "expected dense vector constant for local_array lowering");

    Type elemType = vectorType.getElementType();
    SmallVector<Value> elements;
    elements.reserve(vectorType.getDimSize(0));
    for (Attribute attr : elementsAttr.getValues<Attribute>()) {
      auto typedAttr = dyn_cast<TypedAttr>(attr);
      if (!typedAttr)
        return rewriter.notifyMatchFailure(
            op, "expected typed scalar attributes in dense vector constant");
      elements.push_back(
          rewriter.create<arith::ConstantOp>(op.getLoc(), elemType, typedAttr));
    }

    rewriter.replaceOpWithNewOp<pto::LocalArrayFromElementsOp>(
        op, getLocalArrayType(vectorType), elements);
    return success();
  }
};

struct VectorFromElementsLowering
    : public OpConversionPattern<vector::FromElementsOp> {
  using OpConversionPattern<vector::FromElementsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::FromElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vectorType = dyn_cast<VectorType>(op.getType());
    if (!vectorType || !isSupportedVectorArrayType(vectorType))
      return failure();

    rewriter.replaceOpWithNewOp<pto::LocalArrayFromElementsOp>(
        op, getLocalArrayType(vectorType), adaptor.getElements());
    return success();
  }
};

struct VectorInsertLowering : public OpConversionPattern<vector::InsertOp> {
  using OpConversionPattern<vector::InsertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vectorType = dyn_cast<VectorType>(op.getDest().getType());
    if (!vectorType || !isSupportedVectorArrayType(vectorType))
      return failure();
    if (op.getSource().getType() != vectorType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "only scalar element inserts are supported");

    FailureOr<Value> index = materializeSingleVectorIndex(
        op.getLoc(), op.getStaticPosition(), op.getDynamicPosition(),
        rewriter);
    if (failed(index))
      return rewriter.notifyMatchFailure(
          op, "only one-dimensional vector.insert is supported");

    rewriter.replaceOpWithNewOp<pto::LocalArrayInsertOp>(
        op, getLocalArrayType(vectorType), adaptor.getSource(),
        adaptor.getDest(), *index);
    return success();
  }
};

struct VectorExtractLowering : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern<vector::ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vectorType = dyn_cast<VectorType>(op.getVector().getType());
    if (!vectorType || !isSupportedVectorArrayType(vectorType))
      return failure();
    if (op.getResult().getType() != vectorType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "only scalar element extracts are supported");

    FailureOr<Value> index = materializeSingleVectorIndex(
        op.getLoc(), op.getStaticPosition(), op.getDynamicPosition(),
        rewriter);
    if (failed(index))
      return rewriter.notifyMatchFailure(
          op, "only one-dimensional vector.extract is supported");

    rewriter.replaceOpWithNewOp<pto::LocalArrayExtractOp>(
        op, vectorType.getElementType(), adaptor.getVector(), *index);
    return success();
  }
};

struct PTOLowerVectorLocalArrayPass
    : public pto::impl::PTOLowerVectorLocalArrayBase<
          PTOLowerVectorLocalArrayPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](VectorType vectorType) -> Type {
      if (!isSupportedVectorArrayType(vectorType))
        return Type{};
      return getLocalArrayType(vectorType);
    });

    RewritePatternSet patterns(ctx);
    patterns.add<ArithVectorConstantLowering, VectorFromElementsLowering,
                 VectorInsertLowering, VectorExtractLowering>(typeConverter,
                                                              ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, pto::PTODialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalDialect<vector::VectorDialect>();
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      return !isSupportedVectorArrayType(op.getType());
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerVectorLocalArrayPass() {
  return std::make_unique<PTOLowerVectorLocalArrayPass>();
}
