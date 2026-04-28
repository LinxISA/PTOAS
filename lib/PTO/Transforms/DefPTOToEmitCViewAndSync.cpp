// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

struct PTOMGatherToMGATHER : public OpConversionPattern<pto::MGatherOp> {
  using OpConversionPattern<pto::MGatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Value mem = peelUnrealized(adaptor.getMem());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value dst = peelUnrealized(adaptor.getDst());

    Value memArg = maybeWrapGlobalMemrefAsGlobalTensor(
        rewriter, op.getLoc(), mem, op.getMem().getType(), op.getOperation());

    ArrayAttr templateArgs;
    if (op.getGatherOob() != pto::GatherOOB::Undefined) {
      auto gatherOobTok = [&](pto::GatherOOB mode) -> StringRef {
        switch (mode) {
        case pto::GatherOOB::Undefined:
          return "pto::GatherOOB::Undefined";
        case pto::GatherOOB::Clamp:
          return "pto::GatherOOB::Clamp";
        case pto::GatherOOB::Wrap:
          return "pto::GatherOOB::Wrap";
        case pto::GatherOOB::Zero:
          return "pto::GatherOOB::Zero";
        }
        llvm_unreachable("unknown GatherOOB");
      };
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, gatherOobTok(op.getGatherOob()))});
    }

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MGATHER",
        ArrayAttr{}, templateArgs,
        ValueRange{dst, memArg, idx});

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, dst);
    }
    return success();
  }
};

struct AffineApplyMulConstToEmitC
    : public OpConversionPattern<affine::AffineApplyOp> {
  using OpConversionPattern<affine::AffineApplyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto map = op.getAffineMap();

    if (map.getNumDims() != 0 || map.getNumSymbols() != 1)
      return failure();

    auto expr = map.getResult(0);
    auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!bin || bin.getKind() != AffineExprKind::Mul)
      return failure();

    auto lhs = bin.getLHS();
    auto rhs = bin.getRHS();

    auto symExpr = dyn_cast<AffineSymbolExpr>(lhs);
    auto constExpr = dyn_cast<AffineConstantExpr>(rhs);
    if (!symExpr || !constExpr)
      return failure();

    Value inputVal = adaptor.getMapOperands()[0];

    std::string valStr = std::to_string(constExpr.getValue());
    auto cstAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
    auto cstOp = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), inputVal.getType(), cstAttr);

    rewriter.replaceOpWithNewOp<emitc::MulOp>(
        op, inputVal.getType(), inputVal, cstOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Kernel inference helpers
//===----------------------------------------------------------------------===//

enum class KernelKind { VecAdd, Matmul, Unknown };

static KernelKind inferKernelKind(func::FuncOp f) {
  bool hasAdd = false;
  bool hasMM  = false;
  f.walk([&](Operation *op) {
    if (isa<mlir::pto::TAddOp>(op)) hasAdd = true;
    if (isa<mlir::pto::TMatmulOp>(op)) hasMM = true;
    if (isa<mlir::pto::TMatmulAccOp>(op)) hasMM = true;
  });
  if (hasMM)  return KernelKind::Matmul;
  if (hasAdd) return KernelKind::VecAdd;
  return KernelKind::Unknown;
}

static void inferTileMNK(func::FuncOp f, int &M, int &N, int &K) {
  M = 32; N = 32; K = 32;
  SmallVector<memref::SubViewOp, 4> subs;
  f.walk([&](memref::SubViewOp sv) { subs.push_back(sv); });

  auto readShape2D = [&](memref::SubViewOp sv, int &d0, int &d1) {
    auto resTy = mlir::cast<MemRefType>(sv.getResult().getType());
    if (resTy.getRank() == 2 && resTy.hasStaticShape()) {
      d0 = (int)resTy.getDimSize(0);
      d1 = (int)resTy.getDimSize(1);
    }
  };

  if (subs.empty()) return;

  int a0=32, a1=32;
  readShape2D(subs[0], a0, a1);
  M = a0; N = a1;

  if (subs.size() >= 2) {
    int b0=32, b1=32;
    readShape2D(subs[0], a0, a1);
    readShape2D(subs[1], b0, b1);
    M = a0; K = a1; N = b1;
  }
}

static std::optional<StringRef> getKernelKindMacro(func::FuncOp funcOp) {
  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;

  switch (kernelKindAttr.getKernelKind()) {
  case FunctionKernelKind::Cube:
    return StringRef("__DAV_CUBE__");
  case FunctionKernelKind::Vector:
    return StringRef("__DAV_VEC__");
  }

  llvm_unreachable("unexpected kernel kind");
}

struct FuncToEmitC : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Convert the function signature with the type converter.
    Type convertedTy = getTypeConverter()->convertType(op.getFunctionType());
    auto funcType = dyn_cast_or_null<FunctionType>(convertedTy);
    if (!funcType)
      return rewriter.notifyMatchFailure(op, "failed to convert function type");
    if (funcType.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot return multiple values");

    // Create the EmitC function with the converted signature.
    auto emitcFunc =
        rewriter.create<emitc::FuncOp>(op.getLoc(), op.getName(), funcType);

    for (const auto &namedAttr : op->getAttrs()) {
      StringRef name = namedAttr.getName().strref();
      if (name == op.getFunctionTypeAttrName() ||
          name == SymbolTable::getSymbolAttrName() ||
          name == pto::kPTOEntryAttrName ||
          name == pto::kLegacyHACCEntryAttrName ||
          name == "pto.internal.entry")
        continue;
      emitcFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    if (op.isDeclaration()) {
      emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"extern"}));
      rewriter.eraseOp(op);
      return success();
    }

    if (pto::isPTOEntryFunction(op)) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"__global__ AICORE"}));
    } else if (op.isPrivate()) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"static", "AICORE"}));
    } else {
      emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"AICORE"}));
    }

    std::optional<StringRef> kernelKindMacro = getKernelKindMacro(op);
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    // Inline the original body, then convert region/block argument types to
    // match the converted signature (also covers CFG blocks introduced by
    // pre-lowering, e.g. scf.while -> cf.br/cf.cond_br).
    rewriter.inlineRegionBefore(op.getBody(), emitcFunc.getBody(),
                                emitcFunc.end());

    TypeConverter::SignatureConversion entryConv(op.getNumArguments());
    for (unsigned i = 0; i < op.getNumArguments(); ++i)
      entryConv.addInputs(i, funcType.getInput(i));

    if (failed(rewriter.convertRegionTypes(&emitcFunc.getBody(),
                                           *getTypeConverter(), &entryConv)))
      return failure();

    // Preserve the existing function prologue shape. `kernel_kind` functions are
    // emitted with the same macro guard/reset sequence that used to come from
    // early pto.section wrapping, but only after SCF pre-lowering has finished.
    {
      Block &entryBlock = emitcFunc.getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), "using T = float;");
      if (kernelKindMacro) {
        std::string startMacro = "\n#if defined(" + kernelKindMacro->str() + ")";
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), startMacro);
        if (*kernelKindMacro == "__DAV_VEC__") {
          rewriter.create<emitc::VerbatimOp>(op.getLoc(), "set_mask_norm();");
          rewriter.create<emitc::VerbatimOp>(op.getLoc(),
                                             "set_vector_mask(-1, -1);");
          if (needsNoSplitGuard)
            rewriter.create<emitc::VerbatimOp>(
                op.getLoc(), "if (get_subblockid() == 0) {");
        }
      }
    }

    if (kernelKindMacro) {
      Block &lastBlock = emitcFunc.getBody().back();
      rewriter.setInsertionPoint(lastBlock.getTerminator());
      if (*kernelKindMacro == "__DAV_VEC__" && needsNoSplitGuard)
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), "}");
      std::string endMacro = "#endif // " + kernelKindMacro->str() + "\n";
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), endMacro);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SubView lowering to GlobalTensor (keep your existing code)
//===----------------------------------------------------------------------===

enum class Role { A, B, C, Unknown };

template <typename MatmulLikeOp>
static std::optional<Role> inferMatmulLikeSubviewRole(MatmulLikeOp op,
                                                      Value buffer) {
  if (op.getLhs() == buffer)
    return Role::A;
  if (op.getRhs() == buffer)
    return Role::B;
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromLoadUser(mlir::pto::TLoadOp load) {
  Value buffer = load.getDst();
  if (!buffer)
    return std::nullopt;
  for (Operation *user : buffer.getUsers()) {
    if (auto matmul = dyn_cast<mlir::pto::TMatmulOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmul, buffer))
        return role;
      continue;
    }
    if (auto matmulAcc = dyn_cast<mlir::pto::TMatmulAccOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmulAcc, buffer))
        return role;
    }
  }
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromUser(Operation *user, Value result) {
  if (auto load = dyn_cast<mlir::pto::TLoadOp>(user))
    return inferSubviewRoleFromLoadUser(load);
  if (auto store = dyn_cast<mlir::pto::TStoreOp>(user)) {
    if (store.getDst() == result)
      return Role::C;
  }
  return std::nullopt;
}

static Role inferSubviewRole(memref::SubViewOp sv) {
  Value result = sv.getResult();
  for (Operation *user : result.getUsers()) {
    if (auto role = inferSubviewRoleFromUser(user, result))
      return *role;
  }
  return Role::Unknown;
}

// =============================================================================
// 4. MemRef SubView -> Explicit Shape/Stride Construction (Full Implementation)
// =============================================================================
struct SubviewToEmitCPattern : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  // 辅助函数：尝试从 OpFoldResult 中提取静态整数值
  std::optional<int64_t> extractStaticInt(OpFoldResult ofr) const {
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        return intAttr.getInt();
    } else {
      Value v = ofr.get<Value>();
      if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto iAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
          return iAttr.getInt();
      } else if (auto idxOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
        return idxOp.value();
      }
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    
    // 获取源 MemRef 类型信息
    auto srcType = mlir::cast<MemRefType>(op.getSource().getType());
    int64_t rank = srcType.getRank();

	    auto elemTypeToString = [&](Type elemTy) -> std::string {
	      if (elemTy.isF16())
	        return "half";
	      if (elemTy.isBF16())
	        return "bfloat16_t";
	      if (elemTy.isF32())
	        return "float";
	      if (elemTy.isF64())
	        return "double";
      if (elemTy.isInteger(8)) {
        if (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
          return "int8_t";
        return "uint8_t";
      }
      if (elemTy.isInteger(16)) {
        if (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
          return "int16_t";
        return "uint16_t";
      }
      if (elemTy.isInteger(32)) {
        if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
          return "int32_t";
        return "uint32_t";
      }
      if (elemTy.isInteger(64)) {
        return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
      }
      return "float";
    };

    // -------------------------------------------------------------------------
    // Part 1: 指针偏移计算 (Runtime Pointer Arithmetic)
    // -------------------------------------------------------------------------
    
    // 准备类型: unsigned
    Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
    
    // Helper: 创建 unsigned 常量
    auto mkU32 = [&](int64_t v) -> Value {
      return rewriter.create<emitc::ConstantOp>(
          loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(v)));
    };

    // Helper: 将 OpFoldResult 转为 EmitC Value (用于计算)
    auto ofrToEmitCValue = [&](OpFoldResult ofr) -> Value {
      if (auto v = ofr.dyn_cast<Value>()) {
        Value rv = rewriter.getRemappedValue(v);
        // 如果类型不匹配，插入 Cast
        if (rv.getType() != u32Ty)
             return rewriter.create<emitc::CastOp>(loc, u32Ty, rv).getResult();
        return rv;
      }
      if (auto attr = ofr.dyn_cast<Attribute>()) {
         if (auto ia = dyn_cast<IntegerAttr>(attr))
             return mkU32(ia.getValue().getSExtValue());
      }
      return mkU32(0);
    };

    // 1. 获取 Source 的 Strides (支持动态 Stride 收集)
    SmallVector<OpFoldResult> sourceStrides;

    if (auto rc = op.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
        sourceStrides = rc.getMixedStrides();
    } else {
        SmallVector<int64_t> strideInts;
        int64_t offset = ShapedType::kDynamic;
        bool useTypeStrides = succeeded(getStridesAndOffset(srcType, strideInts, offset));
        (void)offset;
        if (useTypeStrides) {
          for (int64_t s : strideInts) {
            if (s == ShapedType::kDynamic)
              useTypeStrides = false;
          }
        }
        if (useTypeStrides) {
            for (int64_t s : strideInts) {
                sourceStrides.push_back(rewriter.getIndexAttr(s));
            }
        } else {
            // Fallback: Compact Layout
            auto shape = srcType.getShape();
            int64_t current = 1;
            sourceStrides.resize(rank);
            for (int i = rank - 1; i >= 0; --i) {
                sourceStrides[i] = rewriter.getIndexAttr(current);
                if (shape[i] != ShapedType::kDynamic) current *= shape[i];
            }
        }
    }

    // 2. 计算运行时 Offset
    auto staticOffsets = op.getStaticOffsets();
    auto dynamicOffsets = adaptor.getOffsets();
    int dynOffIdx = 0;
    Value totalOffset = mkU32(0);

    for (int i = 0; i < rank; ++i) {
        // A. 获取 Offset
        Value offVal;
        if (staticOffsets[i] == ShapedType::kDynamic) {
            Value rawDyn = dynamicOffsets[dynOffIdx++];
            offVal = rewriter.create<emitc::CastOp>(loc, u32Ty, rawDyn);
        } else {
            offVal = mkU32(staticOffsets[i]);
        }

        // B. 获取 Stride (用于指针计算)
        Value strideVal = mkU32(1);
        if (i < (int)sourceStrides.size()) {
            strideVal = ofrToEmitCValue(sourceStrides[i]);
        }

        // C. 累加
        Value term = rewriter.create<emitc::MulOp>(loc, u32Ty, offVal, strideVal);
        totalOffset = rewriter.create<emitc::AddOp>(loc, u32Ty, totalOffset, term);
    }

    // 3. 生成新指针
    //
    // NOTE: Some toolchains may materialize kernel pointer params as `void*` even
    // when the underlying element type is i16. Pointer arithmetic on `void*`
    // is ill-formed in C++, so we explicitly cast to a typed pointer for i16.
    Value sourcePtr = adaptor.getSource();
    Value tileCandidate = sourcePtr;
    if (auto castOp = sourcePtr.getDefiningOp<emitc::CastOp>()) {
      tileCandidate = castOp.getOperand();
    } else if (auto uc =
                   sourcePtr.getDefiningOp<UnrealizedConversionCastOp>()) {
      tileCandidate = uc.getOperand(0);
    }
    if (auto ot = dyn_cast<emitc::OpaqueType>(tileCandidate.getType())) {
      auto tyStr = ot.getValue();
      if (tyStr.find("Tile<") != std::string::npos ||
          tyStr.find("ConvTile<") != std::string::npos) {
        std::string elemTok = elemTypeToString(srcType.getElementType());
        pto::AddressSpace as = pto::AddressSpace::GM;
        if (auto asAttr =
                dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace()))
          as = asAttr.getAddressSpace();
        sourcePtr =
            materializeTileDataValue(rewriter, loc, tileCandidate, as, elemTok);
        if (tileDataReturnsIntegralAddress(as))
          sourcePtr =
              materializeAddressAsPointer(rewriter, loc, sourcePtr, as, elemTok);
      }
    }
    Value newPtr;
    {
      auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
      Type elemTy = resTy.getElementType();
      if (elemTy.isInteger(16)) {
        std::string castElemTypeStr = "int16_t";
        if (cast<IntegerType>(elemTy).isUnsigned())
          castElemTypeStr = "uint16_t";

        std::string qualifier = "__gm__";
        if (Attribute ms = srcType.getMemorySpace()) {
          if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(ms)) {
            qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
          }
        }

        auto typedPtrTy = emitc::OpaqueType::get(ctx, qualifier + " " + castElemTypeStr + "*");
        Value typedSourcePtr = rewriter.create<emitc::CastOp>(loc, typedPtrTy, sourcePtr);
        newPtr = rewriter.create<emitc::AddOp>(loc, typedPtrTy, typedSourcePtr, totalOffset);
      } else {
        newPtr = rewriter.create<emitc::AddOp>(loc, sourcePtr.getType(), sourcePtr, totalOffset);
      }
    }


    // -------------------------------------------------------------------------
    // Part 2: For non-GM memrefs, keep pointer (no GlobalTensor).
    // -------------------------------------------------------------------------
    bool isGlobal = true;
    if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace())) {
      auto as = asAttr.getAddressSpace();
      isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
    }
    if (!isGlobal) {
      Type dstTy = getTypeConverter()->convertType(op.getType());
      if (!dstTy)
        return failure();
      if (newPtr.getType() != dstTy)
        newPtr = rewriter.create<emitc::CastOp>(loc, dstTy, newPtr);
      rewriter.replaceOp(op, newPtr);
      return success();
    }

    // -------------------------------------------------------------------------
    // Part 3: 生成 GlobalTensor 类型 (Shape/Stride Template Generation)
    // -------------------------------------------------------------------------
    
    // When emitting C++ with `declareVariablesAtTop`, value declarations are
    // hoisted before body statements. Avoid introducing local `using` aliases
    // for templated types (Shape/Stride/GlobalTensor) because those aliases
    // would appear after the hoisted declarations and break compilation
    // (`unknown type name`).
    //
    // Instead, use the fully spelled template types as EmitC opaque types.

    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    
    // 1. 解析具体元素类型
    std::string elemTypeStr = getElemTypeStringForGT(resTy.getElementType());

    // 2. 生成 Shape 模板参数，之后会右对齐有效维度并补齐到 5 维（高维填 1）
    SmallVector<int64_t> shapeParamsVec;
    SmallVector<Value> sizeValues; // 每个维度对应的运行时 size（统一为 unsigned）
    auto resShape = resTy.getShape();
    auto mixedSizes = op.getMixedSizes();
    sizeValues.reserve(rank);
    for (int i = 0; i < resTy.getRank(); ++i) {
      if (resShape[i] == ShapedType::kDynamic) {
        shapeParamsVec.push_back(-1);
      } else {
        shapeParamsVec.push_back(resShape[i]);
      }
      // size 值：优先从 op.getMixedSizes() 取（可动态/静态），否则退化为类型里的静态 shape。
      if (i < (int)mixedSizes.size())
        sizeValues.push_back(ofrToEmitCValue(mixedSizes[i]));
      else
        sizeValues.push_back(
            mkU32(resShape[i] == ShapedType::kDynamic ? 1 : resShape[i]));
    }

    // 3. 生成 Stride 模板参数 + 运行时 stride 值（考虑 subview step）
    SmallVector<int64_t> strideTemplateVec;
    SmallVector<Value> strideValues; // 每个维度对应的运行时 stride（统一为 unsigned）
    strideTemplateVec.reserve(rank);
    strideValues.reserve(rank);
    auto subViewSteps = op.getMixedStrides();
    for (int i = 0; i < rank; ++i) {
      OpFoldResult srcStrideOfr =
          (i < (int)sourceStrides.size()) ? sourceStrides[i]
                                          : rewriter.getIndexAttr(1);
      OpFoldResult stepOfr = (i < (int)subViewSteps.size())
                                 ? subViewSteps[i]
                                 : rewriter.getIndexAttr(1);

      auto srcStatic = extractStaticInt(srcStrideOfr);
      auto stepStatic = extractStaticInt(stepOfr);
      if (srcStatic && stepStatic) {
        int64_t finalStride = (*srcStatic) * (*stepStatic);
        strideTemplateVec.push_back(finalStride);
        strideValues.push_back(mkU32(finalStride));
        continue;
      }

      strideTemplateVec.push_back(-1);
      Value srcV = ofrToEmitCValue(srcStrideOfr);
      Value stepV = ofrToEmitCValue(stepOfr);
      // 尽量避免乘以 1 生成冗余指令
      if (stepStatic && *stepStatic == 1)
        strideValues.push_back(srcV);
      else if (srcStatic && *srcStatic == 1)
        strideValues.push_back(stepV);
      else
        strideValues.push_back(
            rewriter.create<emitc::MulOp>(loc, u32Ty, srcV, stepV));
    }

    // 3.1 右对齐到 5 维：shape 补 1；已有维度继承原 stride；
    //      被补出来的高维按“紧密升维”规则连续推导：stride[i] = shape[i+1] * stride[i+1]
    SmallVector<int64_t, 5> finalShape;
    SmallVector<int64_t, 5> finalStride;
    buildGlobalTensorShapeAndStride(shapeParamsVec, strideTemplateVec,
                                    finalShape, finalStride);
    Value oneU32 = mkU32(1);
    SmallVector<Value, 5> finalShapeValues(5, oneU32);
    SmallVector<Value, 5> finalStrideValues(5, oneU32);
    int shift = 5 - rank;

    // 先放入原始 shape/stride（保持用户提供的值）
    for (int i = 0; i < rank && i < 5; ++i) {
      finalShapeValues[shift + i] = sizeValues[i];
      finalStrideValues[shift + i] = strideValues[i];
    }

    // 从低维到高维倒推补齐 stride（仅对补出来的前置维度生效）
    for (int i = 3; i >= 0; --i) {
      // 如果该维已由原始 rank 覆盖，则保持原值
      if (i >= shift)
        continue;
      if (finalStride[i] != -1) {
        finalStrideValues[i] = mkU32(finalStride[i]);
        continue;
      }
      // 动态推导：stride[i] = shape[i+1] * stride[i+1]
      if (finalShape[i + 1] == 1) {
        finalStrideValues[i] = finalStrideValues[i + 1];
      } else {
        finalStrideValues[i] = rewriter.create<emitc::MulOp>(
            loc, u32Ty, finalShapeValues[i + 1], finalStrideValues[i + 1]);
      }
    }

    std::string shapeParams = joinIntTemplateParams(finalShape);
    std::string strideParams = joinIntTemplateParams(finalStride);

    // Spelled-out C++ types.
    std::string shapeCppType = "pto::Shape<" + shapeParams + ">";
    std::string strideCppType = "pto::Stride<" + strideParams + ">";

    // 3.0 Layout: prefer the attribute from InferPTOLayout; only fall back to
    // local inference when the pass is disabled.
    std::string layoutEnum = "pto::Layout::ND";
    if (auto layout = resolveLayoutForGlobalTensor(op, op.getSource())) {
      layoutEnum = layoutToEmitCString(*layout);
    } else {
      bool allStatic =
          llvm::all_of(finalShape, [](int64_t value) { return value != -1; }) &&
          llvm::all_of(finalStride, [](int64_t value) { return value != -1; });

      int layoutTag = 0; // ND
      auto elemBytes = 4; // default float
      if (elemTypeStr.find("half") != std::string::npos ||
          elemTypeStr.find("f16") != std::string::npos ||
          elemTypeStr.find("bf16") != std::string::npos)
        elemBytes = 2;
      else if (elemTypeStr.find("double") != std::string::npos ||
               elemTypeStr.find("f64") != std::string::npos)
        elemBytes = 8;

      if (allStatic) {
        if (finalShape[2] == 16 &&
            finalShape[2] * finalShape[3] * elemBytes == 512 &&
            finalStride[4] == 1 && finalStride[3] == finalShape[4]) {
          layoutTag = 2; // NZ
        } else {
          bool isRow = finalStride[4] == 1;
          for (int i = 3; i >= 0; --i)
            isRow &= (finalStride[i] ==
                      multiplyOrDynamic(finalStride[i + 1], finalShape[i + 1]));
          bool isCol = finalStride[0] == 1;
          for (int i = 0; i < 4; ++i)
            isCol &= (finalStride[i + 1] ==
                      multiplyOrDynamic(finalStride[i], finalShape[i]));
          if (isCol)
            layoutTag = 1; // DN
          else
            layoutTag = isRow ? 0 : 0; // fallback ND
        }
      }

      if (layoutTag == 1)
        layoutEnum = "pto::Layout::DN";
      else if (layoutTag == 2)
        layoutEnum = "pto::Layout::NZ";
    }
    // GlobalTensor takes a Layout non-type template parameter; directly use the
    // enum constant.


    // -------------------------------------------------------------------------
    // Part 3: 显式对象实例化 (Explicit Object Instantiation)
    // -------------------------------------------------------------------------

    // A. Instantiate Shape object.
    auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, shapeCppType);
    SmallVector<Value> shapeArgs;
    // 从 adaptor.getSizes() 获取 subview 的所有 dynamic sizes
    for (Value dynSize : adaptor.getSizes()) {
        shapeArgs.push_back(dynSize);
    }
    
    auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, 
        shapeTypeOpaque, // 返回类型
        shapeCppType,    // 调用的“函数名”即类名构造函数
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{}, 
        /*operands=*/ValueRange(shapeArgs)
    );
    
    // B. Instantiate Stride object.
    auto strideTypeOpaque = emitc::OpaqueType::get(ctx, strideCppType);
    // 仅传入动态 stride 维度对应的值，匹配 pto::Stride 的 N-parameter ctor（并满足其 static_assert）。
    SmallVector<Value> strideCtorArgs;
    strideCtorArgs.reserve(5);
    for (int i = 0; i < 5; ++i) {
      if (finalStride[i] == -1)
        strideCtorArgs.push_back(finalStrideValues[i]);
    }
    auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, strideTypeOpaque, strideCppType,
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(strideCtorArgs));

    // C. Instantiate GlobalTensor object (ptr + shape + stride).
    std::string gtCppType = "GlobalTensor<" + elemTypeStr + ", " + shapeCppType +
                            ", " + strideCppType + ", " + layoutEnum + ">";
    auto gtType = emitc::OpaqueType::get(ctx, gtCppType);

    // 准备构造参数: [ptr, shape_instance, stride_instance]
    SmallVector<Value> gtConstructorArgs;
    gtConstructorArgs.push_back(newPtr);
    gtConstructorArgs.push_back(shapeInstOp.getResult(0)); // 拿到 shape_inst 的 SSA Value
    gtConstructorArgs.push_back(strideInstOp.getResult(0)); // 拿到 stride_inst 的 SSA Value

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, 
        gtType, 
        gtCppType,
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(gtConstructorArgs)
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper: build GlobalTensor from a static MemRef (for TLOAD/TSTORE)
//===----------------------------------------------------------------------===//

static std::string getElemTypeStringForGT(Type elemTy) {
  if (elemTy.isF16()) return "half";
  if (elemTy.isBF16()) return "bfloat16_t";
  if (elemTy.isF32()) return "float";
  if (elemTy.isF64()) return "double";
  if (elemTy.isInteger(8)) {
    if (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
      return "int8_t";
    return "uint8_t";
  }
  if (elemTy.isInteger(16)) {
    if (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
      return "int16_t";
    return "uint16_t";
  }
  if (elemTy.isInteger(32)) {
    if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
      return "int32_t";
    return "uint32_t";
  }
  if (elemTy.isInteger(64)) {
    return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
  }
  return "float";
}

static bool hasStaticShape(MemRefType mrTy) {
  return llvm::none_of(mrTy.getShape(), [](int64_t dim) {
    return dim == ShapedType::kDynamic;
  });
}

static bool getStaticMemrefLayout(MemRefType mrTy, SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset) {
  if (failed(getStridesAndOffset(mrTy, strides, offset))) {
    strides.clear();
    int64_t stride = 1;
    ArrayRef<int64_t> shape = mrTy.getShape();
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      strides.push_back(stride);
      stride *= shape[i];
    }
    std::reverse(strides.begin(), strides.end());
    offset = 0;
  }
  return offset != ShapedType::kDynamic &&
         llvm::none_of(strides, [](int64_t strideValue) {
           return strideValue == ShapedType::kDynamic;
         });
}

static Value applyStaticMemrefOffset(ConversionPatternRewriter &rewriter,
                                     Location loc, Value basePtr,
                                     int64_t offset) {
  if (offset == 0)
    return basePtr;
  auto *ctx = rewriter.getContext();
  Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
  auto offVal = rewriter.create<emitc::ConstantOp>(
      loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(offset)));
  return rewriter.create<emitc::AddOp>(loc, basePtr.getType(), basePtr, offVal);
}

static int getGlobalTensorElementBytes(StringRef elemTypeStr) {
  if (elemTypeStr.contains("half") || elemTypeStr.contains("bf16"))
    return 2;
  if (elemTypeStr.contains("double"))
    return 8;
  return 4;
}

static int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return -1;
  return lhs * rhs;
}

static void buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> strides,
                                            SmallVectorImpl<int64_t> &shape5D,
                                            SmallVectorImpl<int64_t> &stride5D) {
  shape5D.assign(5, 1);
  stride5D.assign(5, 1);
  int rank = static_cast<int>(shape.size());
  int shift = 5 - rank;
  for (int i = 0; i < rank && i < 5; ++i) {
    shape5D[shift + i] = shape[i];
    stride5D[shift + i] = strides[i];
  }
  for (int i = 3; i >= 0; --i) {
    if (i >= shift)
      continue;
    stride5D[i] = multiplyOrDynamic(shape5D[i + 1], stride5D[i + 1]);
  }
}

static std::string joinIntTemplateParams(ArrayRef<int64_t> values) {
  std::string result;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      result += ", ";
    result += std::to_string(values[i]);
  }
  return result;
}

static std::string inferFallbackGlobalTensorLayout(ArrayRef<int64_t> shape5D,
                                                   ArrayRef<int64_t> stride5D,
                                                   StringRef elemTypeStr) {
  int elemBytes = getGlobalTensorElementBytes(elemTypeStr);
  if (shape5D[2] == 16 && multiplyOrDynamic(shape5D[2], shape5D[3]) * elemBytes == 512 &&
      stride5D[4] == 1 && stride5D[3] == shape5D[4]) {
    return "pto::Layout::NZ";
  }

  bool isRowMajor = stride5D[4] == 1;
  for (int i = 3; i >= 0 && isRowMajor; --i)
    isRowMajor = stride5D[i] == multiplyOrDynamic(stride5D[i + 1], shape5D[i + 1]);

  bool isColMajor = stride5D[0] == 1;
  for (int i = 0; i < 4 && isColMajor; ++i)
    isColMajor = stride5D[i + 1] == multiplyOrDynamic(stride5D[i], shape5D[i]);

  if (isColMajor)
    return "pto::Layout::DN";
  return isRowMajor ? "pto::Layout::ND" : "pto::Layout::ND";
}

static std::string resolveGlobalTensorLayout(Operation *anchor, Value basePtr,
                                             ArrayRef<int64_t> shape5D,
                                             ArrayRef<int64_t> stride5D,
                                             StringRef elemTypeStr) {
  if (auto layout = resolveLayoutForGlobalTensor(anchor, basePtr))
    return layoutToEmitCString(*layout);
  return inferFallbackGlobalTensorLayout(shape5D, stride5D, elemTypeStr);
}

struct GlobalTensorTypeNames {
  std::string shapeTypeName;
  std::string strideTypeName;
  std::string tensorTypeName;
  std::string layoutConstName;
};

static GlobalTensorTypeNames getGlobalTensorTypeNames(Operation *anchor) {
  std::string suffix = "_" + std::to_string(reinterpret_cast<uintptr_t>(anchor));
  return {
      "GTShape" + suffix,
      "GTStride" + suffix,
      "GT" + suffix,
      "GT" + suffix + "_layout",
  };
}
static Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                         Location loc, Value basePtr,
                                         MemRefType mrTy,
                                         Operation *anchor) {
  auto *ctx = rewriter.getContext();

  ArrayRef<int64_t> shape = mrTy.getShape();
  if (!hasStaticShape(mrTy))
    return Value();

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (!getStaticMemrefLayout(mrTy, strides, offset))
    return Value();

  Value ptr = applyStaticMemrefOffset(rewriter, loc, basePtr, offset);
  GlobalTensorTypeNames names = getGlobalTensorTypeNames(anchor);
  std::string elemTypeStr = getElemTypeStringForGT(mrTy.getElementType());
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, strides, shape5D, stride5D);

  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.shapeTypeName + " = pto::Shape<" +
               joinIntTemplateParams(shape5D) + ">;");
  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.strideTypeName + " = pto::Stride<" +
               joinIntTemplateParams(stride5D) + ">;");

  std::string layoutEnum = resolveGlobalTensorLayout(anchor, basePtr, shape5D,
                                                     stride5D, elemTypeStr);
  rewriter.create<emitc::VerbatimOp>(loc, "constexpr pto::Layout " +
                                              names.layoutConstName + " = " +
                                              layoutEnum + ";");

  auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, names.shapeTypeName);
  auto strideTypeOpaque = emitc::OpaqueType::get(ctx, names.strideTypeName);
  auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, shapeTypeOpaque, names.shapeTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});
  auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, strideTypeOpaque, names.strideTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});

  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.tensorTypeName + " = GlobalTensor<" + elemTypeStr +
               ", " + names.shapeTypeName + ", " + names.strideTypeName +
               ", " + names.layoutConstName + ">;");
  auto gtType = emitc::OpaqueType::get(ctx, names.tensorTypeName);

  SmallVector<Value> gtArgs;
  gtArgs.push_back(ptr);
  gtArgs.push_back(shapeInstOp.getResult(0));
  gtArgs.push_back(strideInstOp.getResult(0));

  auto gtInst = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, names.tensorTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange(gtArgs));

  return gtInst.getResult(0);
}

static Value maybeWrapGlobalMemrefAsGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value loweredValue,
    Type originalType, Operation *anchor) {
  auto mrTy = dyn_cast<MemRefType>(originalType);
  if (!mrTy)
    return loweredValue;

  bool isGlobal = true;
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace())) {
    auto as = asAttr.getAddressSpace();
    isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
  }
  if (!isGlobal)
    return loweredValue;

  if (Value gt =
          buildGlobalTensorFromMemref(rewriter, loc, loweredValue, mrTy, anchor))
    return gt;
  return loweredValue;
}

static Value castToGMBytePointer(ConversionPatternRewriter &rewriter,
                                 Location loc, Value value) {
  auto *ctx = rewriter.getContext();
  auto targetTy = emitc::OpaqueType::get(ctx, "__gm__ uint8_t*");
  if (value.getType() == targetTy)
    return value;

  auto castTyAttr =
      rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "__gm__ uint8_t*")});
  if (isSetFFTsPointerLikeType(value.getType())) {
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, targetTy, "reinterpret_cast",
                                     ArrayAttr{}, castTyAttr,
                                     ValueRange{value})
        .getResult(0);
  }
  return rewriter.create<emitc::CastOp>(loc, targetTy, value).getResult();
}

static std::string tileBufBLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string blTok = "BLayout::RowMajor";
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout())) {
    if (static_cast<int32_t>(blAttr.getValue()) == 1)
      blTok = "BLayout::ColMajor";
  }
  return blTok;
}

static std::string tileBufSLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string slTok = "SLayout::NoneBox";
  if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout())) {
    int32_t slVal = static_cast<int32_t>(slAttr.getValue());
    slTok = (slVal == 1) ? "SLayout::RowMajor"
                         : (slVal == 2) ? "SLayout::ColMajor"
                                        : "SLayout::NoneBox";
  }
  return slTok;
}

static std::string tileBufPadToken(pto::TileBufConfigAttr configAttr) {
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
  return padTok;
}

static pto::BLayout getTileBufBLayoutValue(pto::TileBufConfigAttr configAttr) {
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
    return blAttr.getValue();
  return pto::BLayout::RowMajor;
}

static int64_t renderTileTemplateDim(int64_t rawDim, Type elemTy,
                                     pto::BLayout blayout, int dimIdx) {
  assert(dimIdx >= 0 && dimIdx < 2 &&
         "renderTileTemplateDim expects a rank-2 rows/cols dimension index");
  if (rawDim == ShapedType::kDynamic)
    return rawDim;
  if (!pto::isPTOFloat4PackedType(elemTy))
    return rawDim;
  int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

static FailureOr<Value> buildAsyncScratchTileValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalScratch,
    Value emittedScratch) {
  Value scratch = peelUnrealized(emittedScratch);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(scratch.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return scratch;
  }

  auto memTy = dyn_cast<MemRefType>(originalScratch.getType());
  if (!memTy)
    return failure();

  ArrayRef<int64_t> shape = memTy.getShape();
  if (!memTy.hasStaticShape() || shape.empty() || shape.size() > 2)
    return failure();

  int64_t rows = shape.size() == 1 ? 1 : shape[0];
  int64_t cols = shape.size() == 1 ? shape[0] : shape[1];

  auto *ctx = rewriter.getContext();
  pto::TileBufConfigAttr configAttr = pto::TileBufConfigAttr::getDefault(ctx);
  if (auto bind = originalScratch.getDefiningOp<pto::BindTileOp>()) {
    configAttr = bind.getConfig();
  } else if (auto cast = originalScratch.getDefiningOp<pto::PointerCastOp>()) {
    if (auto config = cast.getConfig())
      configAttr = *config;
  }

  int32_t fractal = 512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = frAttr.getInt();

  Type elemTy = memTy.getElementType();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  int64_t templateRows = renderTileTemplateDim(rows, elemTy, blayout, 0);
  int64_t templateCols = renderTileTemplateDim(cols, elemTy, blayout, 1);
  std::string elemTypeStr = getEmitCScalarTypeToken(elemTy);
  std::string tileTypeStr =
      "Tile<TileType::Vec, " + elemTypeStr + ", " +
      std::to_string(templateRows) + ", " + std::to_string(templateCols) +
      ", " + tileBufBLayoutToken(configAttr) + ", " +
      std::to_string(templateRows) + ", " + std::to_string(templateCols) +
      ", " + tileBufSLayoutToken(configAttr) + ", " +
      std::to_string(fractal) + ", " + tileBufPadToken(configAttr) + ">";

  Value tile = rewriter
                   .create<emitc::VariableOp>(
                       loc, emitc::OpaqueType::get(ctx, tileTypeStr),
                       emitc::OpaqueAttr::get(ctx, ""))
                   .getResult();
  auto addr = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
  Value scratchAddr =
      rewriter
          .create<emitc::CallOpaqueOp>(loc, emitc::OpaqueType::get(ctx, "uint64_t"),
                                       "reinterpret_cast", ArrayAttr{}, addr,
                                       ValueRange{scratch})
          .getResult(0);
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange{tile, scratchAddr});
  return tile;
}

//===----------------------------------------------------------------------===//
// pto.pointer_cast lowering
//===----------------------------------------------------------------------===
struct PointerCastConversion : public OpConversionPattern<pto::PointerCastOp> {
  static bool getIndexConst(Value v, int64_t &out) {
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        out = ia.getValue().getSExtValue();
        return true;
      }
    }
    return false;
  }

  using OpConversionPattern<pto::PointerCastOp>::OpConversionPattern;

  enum class TileRole { Vec, Mat, Left, Right, Acc, Bias, Scaling };

  static void collectUserOpsThroughCasts(Value v, SmallVectorImpl<Operation *> &out) {
    for (Operation *u : v.getUsers()) {
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(u)) {
        for (Value r : castOp.getResults())
          collectUserOpsThroughCasts(r, out);
        continue;
      }
      out.push_back(u);
    }
  }

  static Value peelUnrealized(Value v) {
    while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      v = castOp.getOperand(0);
    }
    return v;
  }

  static TileRole inferRole(pto::PointerCastOp op) {
    // 1. 优先检查 AddressSpace
    if (auto memRefTy = dyn_cast<MemRefType>(op.getType())) {
      Attribute memorySpace = memRefTy.getMemorySpace();
      if (auto ptoAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
        switch (ptoAttr.getAddressSpace()) {
          case pto::AddressSpace::LEFT:  return TileRole::Left;
          case pto::AddressSpace::RIGHT: return TileRole::Right;
          case pto::AddressSpace::ACC:   return TileRole::Acc;
          case pto::AddressSpace::BIAS:  return TileRole::Bias; 
          case pto::AddressSpace::MAT:   return TileRole::Mat;
          case pto::AddressSpace::SCALING: return TileRole::Scaling;
          default: break; 
        }
      }
    }

    // 2. 通过 Usage 推导 (Fallback)
    SmallVector<Operation *, 8> users;
    collectUserOpsThroughCasts(op.getResult(), users);

    for (Operation *user : users) {
      if (auto mm = dyn_cast<pto::TMatmulOp>(user)) {
        if (mm.getDst() && peelUnrealized(mm.getDst()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mm.getLhs()) == op.getResult()) return TileRole::Left;
        if (peelUnrealized(mm.getRhs()) == op.getResult()) return TileRole::Right;
      }
      if (auto mmacc = dyn_cast<pto::TMatmulAccOp>(user)) {
        if (mmacc.getDst() && peelUnrealized(mmacc.getDst()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mmacc.getAccIn()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mmacc.getLhs()) == op.getResult()) return TileRole::Left;
        if (peelUnrealized(mmacc.getRhs()) == op.getResult()) return TileRole::Right;
      }
    }

    return TileRole::Vec;
  }

  // [新增] 辅助函数：判断 Value 是否源自 arith.constant
  static bool isConstant(Value v, int64_t &outVal) {
    if (!v) return false;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
       if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) {
           outVal = attr.getInt();
           return true;
       }
    }
    return false;
  }

  LogicalResult matchAndRewrite(pto::PointerCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto selfType = mlir::cast<MemRefType>(op.getType());
    ArrayRef<int64_t> shape = selfType.getShape();
    Type elemType = selfType.getElementType();
    
    // 1. 推导 Tile Role
    TileRole role = inferRole(op);

    // 2. 类型字符串生成 (elemTypeStr, dimStr)
    std::string elemTypeStr = getEmitCScalarTypeToken(elemType);

    std::string dimStr;
    pto::BLayout blayout = pto::BLayout::RowMajor;
    auto dimToString = [&](int64_t dim, const char *symbol,
                           int dimIdx) -> std::string {
        if (dim == ShapedType::kDynamic)
          return std::string(symbol);
        return std::to_string(renderTileTemplateDim(dim, elemType, blayout,
                                                    dimIdx));
    };

    // 3. Role Token
    const char *roleTok = "TileType::Vec";
    switch (role) {
      case TileRole::Left:  roleTok = "TileType::Left"; break;
      case TileRole::Right: roleTok = "TileType::Right"; break;
      case TileRole::Acc:   roleTok = "TileType::Acc"; break;
      case TileRole::Bias:  roleTok = "TileType::Bias"; break;
      case TileRole::Mat:   roleTok = "TileType::Mat"; break;
      case TileRole::Vec:   roleTok = "TileType::Vec"; break;
      case TileRole::Scaling: roleTok = "TileType::Scaling"; break;
    }

    // 4. Config & Layout (support BLayoutAttr/SLayoutAttr/PadValueAttr after namespace change)
    std::string layoutParams = "BLayout::RowMajor";
    std::string extraParams = "";
    if (auto configOpt = op.getConfig()) {
        auto config = *configOpt;
        int32_t blVal = 0;
        if (auto attr = dyn_cast<BLayoutAttr>(config.getBLayout()))
            blVal = static_cast<int32_t>(attr.getValue());
 
        if (blVal == 1) layoutParams = "BLayout::ColMajor";
        blayout = blVal == 1 ? pto::BLayout::ColMajor : pto::BLayout::RowMajor;

        int32_t slVal = 0;
        if (auto attr = dyn_cast<SLayoutAttr>(config.getSLayout()))
            slVal = static_cast<int32_t>(attr.getValue());

        std::string slStr = (slVal == 1) ? "SLayout::RowMajor" : (slVal == 2) ? "SLayout::ColMajor" : "SLayout::NoneBox";

        int32_t frVal = 0;
        if (auto attr = dyn_cast<IntegerAttr>(config.getSFractalSize())) frVal = attr.getInt();

        int32_t padVal = 0;
        if (auto attr = dyn_cast<PadValueAttr>(config.getPad()))
            padVal = static_cast<int32_t>(attr.getValue());

        std::string padStr = "PadValue::Null";
        switch (padVal) {
            case 1: padStr = "PadValue::Zero"; break;
            case 2: padStr = "PadValue::Max";  break;
            case 3: padStr = "PadValue::Min";  break;
        }

        int32_t compactVal = 0;
        if (auto attr = dyn_cast<CompactModeAttr>(config.getCompactMode()))
            compactVal = static_cast<int32_t>(attr.getValue());

        std::string compactStr = "CompactMode::Null";
        switch (compactVal) {
            case 1: compactStr = "CompactMode::Normal"; break;
            case 2: compactStr = "CompactMode::RowPlusOne"; break;
        }

        if (!slStr.empty()) {
            extraParams += ", " + slStr + ", " + std::to_string(frVal) + ", " +
                           padStr + ", " + compactStr;
        }
    } else {
        extraParams = ", SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null";
    }

    if (role == TileRole::Left)
      dimStr = dimToString(shape[0], "M", 0) + ", " +
               dimToString(shape[1], "K", 1);
    else if (role == TileRole::Right)
      dimStr = dimToString(shape[0], "K", 0) + ", " +
               dimToString(shape[1], "N", 1);
    else if (role == TileRole::Bias)
      dimStr = "1, " + dimToString(shape[1], "N", 1);
    else
      dimStr = dimToString(shape[0], "M", 0) + ", " +
               dimToString(shape[1], "N", 1);

    // [核心修改] Valid Dims 处理逻辑 (支持混合静态/动态)
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
    bool rowIsConst = vRow && isConstant(vRow, cRow);
    bool colIsConst = vCol && isConstant(vCol, cCol);

    auto makeCtorDimValue = [&](Value emitted, int64_t fallback) -> Value {
      if (emitted)
        return emitted;
      return makeEmitCIntConstant(
          rewriter, loc, emitc::OpaqueType::get(ctx, "int32_t"), fallback);
    };
    auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
      if (!emitted || !pto::isPTOFloat4PackedType(elemType))
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
                           renderTileTemplateDim(rowIsConst ? cRow : shape[0],
                                                 elemType, blayout, 0)));
      constructorArgs.push_back(
          makeCtorDimValue(maybeScaleDynamicValid(vColEmitC, 1),
                           renderTileTemplateDim(colIsConst ? cCol : shape[1],
                                                 elemType, blayout, 1)));
    } else {
      if (rowIsConst) {
        vrowTok = std::to_string(
            renderTileTemplateDim(cRow, elemType, blayout, 0));
      } else if (vRow) {
        vrowTok = "-1";
        rowIsDynamic = true;
        useConstructor = true;
      } else {
        vrowTok = std::to_string(
            renderTileTemplateDim(shape[0], elemType, blayout, 0));
      }

      if (colIsConst) {
        vcolTok = std::to_string(
            renderTileTemplateDim(cCol, elemType, blayout, 1));
      } else if (vCol) {
        vcolTok = "-1";
        colIsDynamic = true;
        useConstructor = true;
      } else {
        vcolTok = std::to_string(
            renderTileTemplateDim(shape[1], elemType, blayout, 1));
      }

      if (useConstructor) {
        if (rowIsDynamic && vRowEmitC)
          constructorArgs.push_back(maybeScaleDynamicValid(vRowEmitC, 0));
        if (colIsDynamic && vColEmitC)
          constructorArgs.push_back(maybeScaleDynamicValid(vColEmitC, 1));
      }
    }

    // 5. 生成 Tile 类型字符串
    std::string tileTypeStr =
      std::string("Tile<") + roleTok + ", " + elemTypeStr + ", " + dimStr + ", " +
      layoutParams + ", " + vrowTok + ", " + vcolTok + extraParams + ">";

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value resultValue;

    if (useConstructor) {
        // 使用 CallOpaqueOp 生成构造函数调用 (Tile v = Tile(...))
        auto ctorOp = rewriter.create<emitc::CallOpaqueOp>(
            loc, 
            tileType,        // Result Type
            tileTypeStr,     // Callee Name (类名)
            ArrayAttr{},     // args
            ArrayAttr{},     // template_args
            ValueRange(constructorArgs) // operands
        );
        resultValue = ctorOp.getResult(0);
    } else {
        // 静态情况 (Tile v;)
        auto varOp = rewriter.create<emitc::VariableOp>(
            loc, 
            tileType, 
            emitc::OpaqueAttr::get(ctx, "")
        );
        resultValue = varOp.getResult();
    }

    // TASSIGN: pto-isa expects an integral address.
    Value addr = adaptor.getAddrs()[0];
    if (isa<emitc::PointerType>(addr.getType()) ||
        (isa<emitc::OpaqueType>(addr.getType()) &&
         cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"))) {
      auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
      auto rcU64 = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      addr = rewriter.create<emitc::CallOpaqueOp>(
                 loc, u64Ty, "reinterpret_cast",
                 /*args=*/ArrayAttr{}, /*templateArgs=*/rcU64,
                 /*operands=*/ValueRange{addr})
                 .getResult(0);
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TASSIGN",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{resultValue, addr});

    rewriter.replaceOp(op, resultValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.load_dps / pto.store_dps lowering (FIX: keep optional result)
//===----------------------------------------------------------------------===

struct PTOTLoadToTLOAD : public OpConversionPattern<pto::TLoadOp> {
  using OpConversionPattern<pto::TLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tload");

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value srcArg = src;
    if (auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType())) {
      bool isGlobal = true;
      if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(srcMrTy.getMemorySpace())) {
        auto as = asAttr.getAddressSpace();
        isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
      }
      if (isGlobal) {
        if (Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                                  op.getOperation()))
          srcArg = gt;
      }
    }

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TLOAD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, srcArg});

    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct PTOTPrefetchToTPREFETCH : public OpConversionPattern<pto::TPrefetchOp> {
  using OpConversionPattern<pto::TPrefetchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrefetchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tprefetch");

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value srcArg = src;
    if (auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType())) {
      bool isGlobal = true;
      if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(srcMrTy.getMemorySpace())) {
        auto as = asAttr.getAddressSpace();
        isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
      }
      if (isGlobal) {
        if (Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                                  op.getOperation()))
          srcArg = gt;
      }
    }

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPREFETCH",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dst, srcArg});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTPackToTPACK : public OpConversionPattern<pto::TPackOp> {
  using OpConversionPattern<pto::TPackOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPackOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tpack");

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPACK",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dst, src});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTStoreToTSTORE : public OpConversionPattern<pto::TStoreOp> {
  using OpConversionPattern<pto::TStoreOp>::OpConversionPattern;

  static std::string stPhaseTok(pto::STPhase phase) {
    switch (phase) {
      case pto::STPhase::Unspecified: return "STPhase::Unspecified";
      case pto::STPhase::Partial: return "STPhase::Partial";
      case pto::STPhase::Final: return "STPhase::Final";
    }
    return "STPhase::Unspecified";
  }

  static std::string atomicTypeTok(pto::AtomicType atomicType) {
    switch (atomicType) {
      case pto::AtomicType::AtomicNone: return "AtomicType::AtomicNone";
      case pto::AtomicType::AtomicAdd: return "AtomicType::AtomicAdd";
    }
    return "AtomicType::AtomicNone";
  }

  static std::string reluPreModeTok(pto::ReluPreMode reluPreMode) {
    switch (reluPreMode) {
      case pto::ReluPreMode::NoRelu: return "ReluPreMode::NoRelu";
      case pto::ReluPreMode::NormalRelu: return "ReluPreMode::NormalRelu";
    }
    return "ReluPreMode::NoRelu";
  }

  LogicalResult matchAndRewrite(pto::TStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tstore");

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = peelUnrealized(adaptor.getPreQuantScalar());
    Value dstArg = dst;
    if (auto dstMrTy = dyn_cast<MemRefType>(op.getDst().getType())) {
      bool isGlobal = true;
      if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(dstMrTy.getMemorySpace())) {
        auto as = asAttr.getAddressSpace();
        isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
      }
      if (isGlobal) {
        if (Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), dst, dstMrTy,
                                                  op.getOperation()))
          dstArg = gt;
      }
    }

    const auto phase = op.getStPhase();
    const auto atomicType = op.getAtomicType();
    const auto reluPreMode = op.getReluPreMode();
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool phaseNonDefault = phase != pto::STPhase::Unspecified;
    const bool atomicNonDefault = atomicType != pto::AtomicType::AtomicNone;
    const bool reluNonDefault = reluPreMode != pto::ReluPreMode::NoRelu;

    auto getOpaqueTok = [&](Value v, StringRef name) -> FailureOr<std::string> {
      if (auto ot = v.getType().dyn_cast<emitc::OpaqueType>())
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(op, (name + " must be emitc::OpaqueType").str());
    };

    ArrayAttr targs;
    // Map op attributes/operands to the exact TSTORE overload family:
    //  1) TSTORE(dst, src)
    //  2) TSTORE<Phase>(dst, src)
    //  3) TSTORE<TileData, GlobalData, AtomicType>(dst, src)
    //  4) TSTORE<Phase, TileData, GlobalData, AtomicType>(dst, src)
    //  5) TSTORE<TileData, GlobalData, AtomicType, ReluPreMode>(dst, src)
    //  6) TSTORE<Phase, TileData, GlobalData, AtomicType, ReluPreMode>(dst, src)
    //  7) TSTORE<TileData, GlobalData, AtomicType, ReluPreMode>(dst, src, preQuant)
    //  8) TSTORE<Phase, TileData, GlobalData, AtomicType, ReluPreMode>(dst, src, preQuant)
    if (!hasPreQuantScalar && !reluNonDefault && !atomicNonDefault) {
      if (phaseNonDefault) {
        targs = rewriter.getArrayAttr({
            emitc::OpaqueAttr::get(ctx, stPhaseTok(phase)),
        });
      } else {
        targs = ArrayAttr{};
      }
    } else {
      auto srcTokOr = getOpaqueTok(src, "src");
      auto dstTokOr = getOpaqueTok(dstArg, "dst");
      if (failed(srcTokOr) || failed(dstTokOr))
        return failure();

      // If there is no preQuant and relu stays default, emit the atomic-only
      // overloads (#3/#4) without ReluPreMode template argument.
      if (!hasPreQuantScalar && !reluNonDefault) {
        if (phaseNonDefault) {
          targs = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, stPhaseTok(phase)),
              emitc::OpaqueAttr::get(ctx, *srcTokOr),
              emitc::OpaqueAttr::get(ctx, *dstTokOr),
              emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)),
          });
        } else {
          targs = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, *srcTokOr),
              emitc::OpaqueAttr::get(ctx, *dstTokOr),
              emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)),
          });
        }
      } else {
        // Relu/preQuant families (#5/#6/#7/#8): keep AtomicType + ReluPreMode.
        if (phaseNonDefault) {
          targs = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, stPhaseTok(phase)),
              emitc::OpaqueAttr::get(ctx, *srcTokOr),
              emitc::OpaqueAttr::get(ctx, *dstTokOr),
              emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)),
              emitc::OpaqueAttr::get(ctx, reluPreModeTok(reluPreMode)),
          });
        } else {
          targs = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, *srcTokOr),
              emitc::OpaqueAttr::get(ctx, *dstTokOr),
              emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)),
              emitc::OpaqueAttr::get(ctx, reluPreModeTok(reluPreMode)),
          });
        }
      }
    }

    SmallVector<Value, 3> operands{dstArg, src};
    if (hasPreQuantScalar)
      operands.push_back(preQuantScalar);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSTORE",
        /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
        /*operands=*/operands);

    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.matmul_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
struct PTOTMatmulToTMATMUL : public OpConversionPattern<pto::TMatmulOp> {
  using OpConversionPattern<pto::TMatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取操作数 (剥离 Cast)
    Value lhs = peelUnrealized(adaptor.getLhs()); // A (Left)
    Value rhs = peelUnrealized(adaptor.getRhs()); // B (Right)
    Value dst = peelUnrealized(adaptor.getDst()); // C (Acc)

    // 2. 直接生成函数调用 TMATMUL(dst, lhs, rhs)
    // 假设输入已经在对应的 L0 Buffer 中
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TMATMUL",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tgemv lowering
//===----------------------------------------------------------------------===//
struct PTOTGemvToTGEMV : public OpConversionPattern<pto::TGemvOp> {
  using OpConversionPattern<pto::TGemvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取操作数 (剥离 Cast)
    Value lhs = peelUnrealized(adaptor.getLhs()); // A (Matrix)
    Value rhs = peelUnrealized(adaptor.getRhs()); // B (Vector)
    Value dst = peelUnrealized(adaptor.getDst()); // C (Result)

    // 2. 直接生成函数调用 TGEMV(dst, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TGEMV",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tgemv.acc lowering
//===----------------------------------------------------------------------===//
struct PTOTGemvAccToTGEMVACC : public OpConversionPattern<pto::TGemvAccOp> {
  using OpConversionPattern<pto::TGemvAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.tgemv.acc");

    // 1. 获取操作数
    Value accIn = peelUnrealized(adaptor.getAccIn()); // AccOld
    Value lhs   = peelUnrealized(adaptor.getLhs());   // A (Matrix)
    Value rhs   = peelUnrealized(adaptor.getRhs());   // B (Vector)
    Value dst   = peelUnrealized(adaptor.getDst());   // AccNew

    // 2. 直接生成函数调用 TGEMV_ACC(dst, accIn, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TGEMV_ACC",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, accIn, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.matmul_acc_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
struct PTOTMatmulAccToTMATMULACC : public OpConversionPattern<pto::TMatmulAccOp> {
  using OpConversionPattern<pto::TMatmulAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.tmatmul.acc");

    // 1. 获取操作数
    Value accIn = peelUnrealized(adaptor.getAccIn()); // AccOld
    Value lhs   = peelUnrealized(adaptor.getLhs());   // A (Left)
    Value rhs   = peelUnrealized(adaptor.getRhs());   // B (Right)
    Value dst   = peelUnrealized(adaptor.getDst());   // AccNew

    // 2. 直接生成函数调用 TMATMUL_ACC(dst, accIn, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TMATMUL_ACC",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, accIn, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Return lowering
//===----------------------------------------------------------------------===

static constexpr llvm::StringLiteral kAutoSyncTailPendingModeAttr =
    "__pto.auto_sync_tail_mode";

struct ReturnToEmitC : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (auto emitcFunc = op->getParentOfType<emitc::FuncOp>()) {
      if (auto modeAttr =
              emitcFunc->getAttrOfType<StringAttr>(kAutoSyncTailPendingModeAttr)) {
        auto *ctx = rewriter.getContext();
        rewriter.setInsertionPoint(op);
        auto args = rewriter.getArrayAttr(
            {emitc::OpaqueAttr::get(ctx, modeAttr.getValue())});
        rewriter.create<emitc::CallOpaqueOp>(
            op.getLoc(), TypeRange{}, "ptoas_auto_sync_tail",
            args, ArrayAttr{}, ValueRange{});
      }
    }

    auto vals = adaptor.getOperands();
    if (vals.empty()) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, Value{});
      return success();
    }
    if (vals.size() == 1) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, vals[0]);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "EmitC cannot return multiple values");
  }
};

struct CallToEmitC : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot lower calls with multiple results");

    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert call result types");

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, op.getCalleeAttr(),
                                               resultTypes,
                                               adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sync lowering
//===----------------------------------------------------------------------===

static constexpr llvm::StringLiteral kAutoSyncTailBarrierAttr =
    "pto.auto_sync_tail_barrier";
static constexpr llvm::StringLiteral kAutoSyncTailHintAttr =
    "pto.auto_sync_tail_hint";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyBarrierAll =
    "barrier_all";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyMte3ToSEvent0 =
    "setwait_mte3_to_s_event0";
static constexpr llvm::StringLiteral kAutoSyncTailModeBarrierAllToken =
    "PTOAutoSyncTailMode::kBarrierAll";
static constexpr llvm::StringLiteral kAutoSyncTailModeMte3ToSEvent0Token =
    "PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0";

static std::string getAutoSyncTailModeToken(Operation *op) {
  if (op) {
    if (auto hintAttr = op->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr)) {
      if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
        return kAutoSyncTailModeBarrierAllToken.str();
      if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
        return kAutoSyncTailModeMte3ToSEvent0Token.str();
    }
  }

  auto func = op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
  if (!func)
    return kAutoSyncTailModeBarrierAllToken.str();

  auto hintAttr = func->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr);
  if (!hintAttr)
    return kAutoSyncTailModeBarrierAllToken.str();

  if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
    return kAutoSyncTailModeBarrierAllToken.str();
  if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
    return kAutoSyncTailModeMte3ToSEvent0Token.str();

  // Fallback to the conservative behavior when seeing unknown policies.
  return kAutoSyncTailModeBarrierAllToken.str();
}

static std::string getPipeName(pto::PIPE pipe) {
  switch (pipe) {
    case pto::PIPE::PIPE_S: return "PIPE_S";
    case pto::PIPE::PIPE_V: return "PIPE_V";
    case pto::PIPE::PIPE_M: return "PIPE_M";
    case pto::PIPE::PIPE_MTE1: return "PIPE_MTE1";
    case pto::PIPE::PIPE_MTE2: return "PIPE_MTE2";
    case pto::PIPE::PIPE_MTE3: return "PIPE_MTE3";
    case pto::PIPE::PIPE_ALL: return "PIPE_ALL";
    case pto::PIPE::PIPE_MTE4: return "PIPE_MTE4";
    case pto::PIPE::PIPE_MTE5: return "PIPE_MTE5";
    case pto::PIPE::PIPE_V2: return "PIPE_V2";
    case pto::PIPE::PIPE_FIX: return "PIPE_FIX";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1A: return "VIRTUAL_PIPE_MTE2_L1A";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1B: return "VIRTUAL_PIPE_MTE2_L1B";
    // 默认回退
    default: return "PIPE_ALL"; 
  }
}

//===----------------------------------------------------------------------===//
// pto.barrier lowering -> pipe_barrier(...)
//===----------------------------------------------------------------------===//
struct PTOBarrierToEmitC : public OpConversionPattern<pto::BarrierOp> {
  using OpConversionPattern<pto::BarrierOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::BarrierOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->hasAttr(kAutoSyncTailBarrierAttr)) {
      auto modeAttr = rewriter.getStringAttr(getAutoSyncTailModeToken(op));
      if (auto emitcFunc = op->getParentOfType<emitc::FuncOp>()) {
        emitcFunc->setAttr(kAutoSyncTailPendingModeAttr, modeAttr);
      } else if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
        funcOp->setAttr(kAutoSyncTailPendingModeAttr, modeAttr);
      }
      rewriter.eraseOp(op);
      return success();
    }

    // [FIX] op.getPipe() returns PipeAttr. 
    // We must call .getPipe() on the attribute to get the actual Enum value.
    pto::PIPE pipeEnum = op.getPipe().getPipe();

    // Convert Enum to String (e.g., PIPE_ALL -> "PIPE_ALL")
    std::string pipeStr = pto::stringifyPIPE(pipeEnum).str();
    auto *ctx = rewriter.getContext();

    auto args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeStr)
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, 
        TypeRange{},        // void return
        "pipe_barrier",     // function name
        args,               // arguments
        ArrayAttr{},        // template args
        ValueRange{}        // operands
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sync lowering (robust for bracket form pto.set_flag[...] / pto.wait_flag[...])
// Replace your PTOSyncToRuntimeCall with the code below.
//===----------------------------------------------------------------------===//

static bool tryConvertPipeAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto pipe = dyn_cast<mlir::pto::PipeAttr>(attr)) {
    token = mlir::pto::stringifyPIPE(pipe.getPipe()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

static bool tryConvertEventAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto event = dyn_cast<mlir::pto::EventAttr>(attr)) {
    token = mlir::pto::stringifyEVENT(event.getEvent()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

static bool tryAssignSyncTokens(Attribute srcAttr, Attribute dstAttr,
                                Attribute evtAttr, std::string &srcTok,
                                std::string &dstTok, std::string &evtTok) {
  std::string localSrc;
  std::string localDst;
  std::string localEvt;
  if (!tryConvertPipeAttrToToken(srcAttr, localSrc) ||
      !tryConvertPipeAttrToToken(dstAttr, localDst) ||
      !tryConvertEventAttrToToken(evtAttr, localEvt)) {
    return false;
  }
  srcTok = std::move(localSrc);
  dstTok = std::move(localDst);
  evtTok = std::move(localEvt);
  return true;
}

static bool tryExtractSyncTokensFromNamedAttrs(Operation *op,
                                               StringRef srcName,
                                               StringRef dstName,
                                               StringRef evtName,
                                               std::string &srcTok,
                                               std::string &dstTok,
                                               std::string &evtTok) {
  return tryAssignSyncTokens(op->getAttr(srcName), op->getAttr(dstName),
                             op->getAttr(evtName), srcTok, dstTok, evtTok);
}

static bool tryExtractSyncTokensFromArrayAttr(Operation *op, StringRef attrName,
                                              std::string &srcTok,
                                              std::string &dstTok,
                                              std::string &evtTok) {
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!arrayAttr || arrayAttr.size() < 3)
    return false;
  return tryAssignSyncTokens(arrayAttr[0], arrayAttr[1], arrayAttr[2], srcTok,
                             dstTok, evtTok);
}

static bool tryExtractFallbackSyncTokens(Operation *op, std::string &srcTok,
                                         std::string &dstTok,
                                         std::string &evtTok) {
  SmallVector<std::string, 2> pipes;
  std::string event;
  for (NamedAttribute namedAttr : op->getAttrs()) {
    std::string token;
    if (tryConvertPipeAttrToToken(namedAttr.getValue(), token)) {
      pipes.push_back(std::move(token));
      continue;
    }
    if (event.empty() &&
        tryConvertEventAttrToToken(namedAttr.getValue(), token)) {
      event = std::move(token);
    }
  }
  if (pipes.size() < 2 || event.empty())
    return false;
  srcTok = pipes[0];
  dstTok = pipes[1];
  evtTok = event;
  return true;
}

static LogicalResult extractSyncTripletTokens(Operation *op,
                                             std::string &srcTok,
                                             std::string &dstTok,
                                             std::string &evtTok,
                                             ConversionPatternRewriter &rewriter) {
  if (tryExtractSyncTokensFromNamedAttrs(op, "src_pipe", "dst_pipe", "event_id",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "srcPipe", "dstPipe", "eventId",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "src", "dst", "event", srcTok,
                                         dstTok, evtTok)) {
    return success();
  }

  for (StringRef attrName : {"args", "pipes", "sync", "triplet", "attrs"}) {
    if (tryExtractSyncTokensFromArrayAttr(op, attrName, srcTok, dstTok,
                                          evtTok)) {
      return success();
    }
  }

  if (tryExtractFallbackSyncTokens(op, srcTok, dstTok, evtTok))
    return success();
  return rewriter.notifyMatchFailure(
      op, "cannot extract PIPE/PIPE/EVENT tokens from pto.{set,wait}_flag");
}
static inline std::string pipeTokFromPipeEnum(mlir::pto::PIPE p) {
  return mlir::pto::stringifyPIPE(p).str();
}
static inline std::string evtTokFromEventEnum(mlir::pto::EVENT e) {
  return mlir::pto::stringifyEVENT(e).str();
}
static inline std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a) {
  return mlir::pto::stringifyPIPE(a.getPipe()).str();
}
static inline std::string evtTokFromEventAttr(mlir::pto::EventAttr a) {
  return mlir::pto::stringifyEVENT(a.getEvent()).str();
}

template <typename T, typename = void>
struct HasGetSrcPipe : std::false_type {};
template <typename T>
struct HasGetSrcPipe<T, std::void_t<decltype(std::declval<T>().getSrcPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipe : std::false_type {};
template <typename T>
struct HasGetDstPipe<T, std::void_t<decltype(std::declval<T>().getDstPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventId : std::false_type {};
template <typename T>
struct HasGetEventId<T, std::void_t<decltype(std::declval<T>().getEventId())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetSrcPipeAttr : std::false_type {};
template <typename T>
struct HasGetSrcPipeAttr<T, std::void_t<decltype(std::declval<T>().getSrcPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipeAttr : std::false_type {};
template <typename T>
struct HasGetDstPipeAttr<T, std::void_t<decltype(std::declval<T>().getDstPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventIdAttr : std::false_type {};
template <typename T>
struct HasGetEventIdAttr<T, std::void_t<decltype(std::declval<T>().getEventIdAttr())>> : std::true_type {};

template <typename SyncOpT>
static LogicalResult extractSyncTokens(SyncOpT op,
                                      std::string &srcTok,
                                      std::string &dstTok,
                                      std::string &evtTok,
                                      ConversionPatternRewriter &rewriter) {
  if constexpr (HasGetSrcPipe<SyncOpT>::value &&
                HasGetDstPipe<SyncOpT>::value &&
                HasGetEventId<SyncOpT>::value) {
    auto s = op.getSrcPipe();
    auto d = op.getDstPipe();
    auto e = op.getEventId();

    if constexpr (std::is_same<decltype(s), mlir::pto::PIPE>::value) srcTok = pipeTokFromPipeEnum(s);
    else srcTok = pipeTokFromPipeAttr(s);

    if constexpr (std::is_same<decltype(d), mlir::pto::PIPE>::value) dstTok = pipeTokFromPipeEnum(d);
    else dstTok = pipeTokFromPipeAttr(d);

    if constexpr (std::is_same<decltype(e), mlir::pto::EVENT>::value) evtTok = evtTokFromEventEnum(e);
    else evtTok = evtTokFromEventAttr(e);

    return success();
  }

  if constexpr (HasGetSrcPipeAttr<SyncOpT>::value &&
                HasGetDstPipeAttr<SyncOpT>::value &&
                HasGetEventIdAttr<SyncOpT>::value) {
    auto s = op.getSrcPipeAttr();
    auto d = op.getDstPipeAttr();
    auto e = op.getEventIdAttr();
    srcTok = pipeTokFromPipeAttr(s);
    dstTok = pipeTokFromPipeAttr(d);
    evtTok = evtTokFromEventAttr(e);
    return success();
  }

  return extractSyncTripletTokens(op.getOperation(), srcTok, dstTok, evtTok, rewriter);
}
struct PTOSetFlagToEmitC : public OpConversionPattern<mlir::pto::SetFlagOp> {
  using OpConversionPattern<mlir::pto::SetFlagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::SetFlagOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    std::string srcTok, dstTok, evtTok;
    if (failed(extractSyncTokens(op, srcTok, dstTok, evtTok, rewriter)))
      return failure();

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        emitc::OpaqueAttr::get(ctx, evtTok),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "set_flag",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOWaitFlagToEmitC : public OpConversionPattern<mlir::pto::WaitFlagOp> {
  using OpConversionPattern<mlir::pto::WaitFlagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::WaitFlagOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    std::string srcTok, dstTok, evtTok;
    if (failed(extractSyncTokens(op, srcTok, dstTok, evtTok, rewriter)))
      return failure();

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        emitc::OpaqueAttr::get(ctx, evtTok),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "wait_flag",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOSyncToEmitC : public OpConversionPattern<mlir::pto::TSyncOp> {
  using OpConversionPattern<mlir::pto::TSyncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::TSyncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> operands;
    operands.reserve(adaptor.getEvents().size());
    for (Value event : adaptor.getEvents())
      operands.push_back(peelUnrealized(event));

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TSYNC",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(operands));
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSyncFlagDynToEmitC : public ConversionPattern {
  PTOSyncFlagDynToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                        StringRef opName, StringRef callee)
      : ConversionPattern(typeConverter, opName, /*benefit=*/1, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected exactly one dynamic event-id operand");

    auto srcAttr = op->getAttrOfType<mlir::pto::PipeAttr>("src_pipe");
    auto dstAttr = op->getAttrOfType<mlir::pto::PipeAttr>("dst_pipe");
    if (!srcAttr || !dstAttr)
      return rewriter.notifyMatchFailure(op, "missing PipeAttr src_pipe/dst_pipe attrs");

    auto *ctx = rewriter.getContext();
    std::string srcTok = pipeTokFromPipeAttr(srcAttr);
    std::string dstTok = pipeTokFromPipeAttr(dstAttr);

    Value eventVal = operands.front();
    eventVal =
        emitCCast(rewriter, op->getLoc(), emitc::OpaqueType::get(ctx, "event_t"), eventVal);

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee,
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{eventVal});
    return success();
  }

private:
  std::string callee;
};

struct PTOGetBufToEmitC : public OpConversionPattern<mlir::pto::GetBufOp> {
  using OpConversionPattern<mlir::pto::GetBufOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::GetBufOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "get_buf expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "get_buf op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        op.getBufIdAttr(),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "get_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTORlsBufToEmitC : public OpConversionPattern<mlir::pto::RlsBufOp> {
  using OpConversionPattern<mlir::pto::RlsBufOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::RlsBufOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "rls_buf expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "rls_buf op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        op.getBufIdAttr(),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "rls_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOSetFFTsToEmitC : public OpConversionPattern<mlir::pto::SetFFTsOp> {
  using OpConversionPattern<mlir::pto::SetFFTsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::SetFFTsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    Value fftsAddr = peelUnrealized(adaptor.getFfts());
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");

    if (isSetFFTsPointerLikeType(fftsAddr.getType())) {
      auto castTyAttr =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      fftsAddr =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                           /*args=*/ArrayAttr{},
                                           /*templateArgs=*/castTyAttr,
                                           /*operands=*/ValueRange{fftsAddr})
              .getResult(0);
    } else if (fftsAddr.getType() != u64Ty) {
      fftsAddr =
          rewriter.create<emitc::CastOp>(loc, u64Ty, fftsAddr).getResult();
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "set_ffts_base_addr",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{fftsAddr});
    return success();
  }
};

struct PTOSyncSetToEmitC : public OpConversionPattern<mlir::pto::SyncSetOp> {
  PTOSyncSetToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                    PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SyncSetOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult
  matchAndRewrite(mlir::pto::SyncSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    IntegerAttr eventIdAttr = op.getEventIdAttr();
    Value eventIdDyn = adaptor.getEventIdDyn();
    int64_t fftsMode = 2;
    if (IntegerAttr fftsModeAttr = op.getFftsModeAttr())
      fftsMode = fftsModeAttr.getInt();

    if ((eventIdAttr != nullptr) == static_cast<bool>(eventIdDyn))
      return rewriter.notifyMatchFailure(
          op, "expects exactly one of static event_id attr or dynamic event_id operand");

    // A5 inter-core sync mirrors +16 only for cube-side producer (PIPE_FIX).
    // Vec-side producer (PIPE_MTE3) emits a single set; hardware handles the
    // subblock mapping in PTO-ISA custom flow.
    if (targetArch == PTOArch::A5) {
      pto::PIPE pipe = op.getPipe().getPipe();
      bool needsMirrorPlus16 = (pipe == pto::PIPE::PIPE_FIX);
      std::string pipeTok = pipeTokFromPipeAttr(op.getPipe());
      auto emitSet = [&](Value eventOperand, IntegerAttr eventLiteral,
                         bool isDynamic) {
        if (isDynamic) {
          auto args = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, pipeTok),
              IntegerAttr::get(IndexType::get(ctx), 0),
          });
          rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "set_intra_block",
                                               /*args=*/args,
                                               /*templateArgs=*/ArrayAttr{},
                                               /*operands=*/ValueRange{eventOperand});
          return;
        }
        auto args = rewriter.getArrayAttr({
            emitc::OpaqueAttr::get(ctx, pipeTok),
            eventLiteral,
        });
        rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "set_intra_block",
                                             /*args=*/args,
                                             /*templateArgs=*/ArrayAttr{},
                                             /*operands=*/ValueRange{});
      };

      if (eventIdAttr) {
        emitSet(Value{}, eventIdAttr, /*isDynamic=*/false);
        if (needsMirrorPlus16) {
          auto plus16 = IntegerAttr::get(eventIdAttr.getType(),
                                         eventIdAttr.getInt() + 16);
          emitSet(Value{}, plus16, /*isDynamic=*/false);
        }
      } else {
        Value eventI32 = castInterCoreEventIdToI32(rewriter, loc, eventIdDyn);
        emitSet(eventI32, IntegerAttr{}, /*isDynamic=*/true);
        if (needsMirrorPlus16) {
          auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
          Value c16 = makeEmitCIntConstant(rewriter, loc, i32Ty, 16);
          Value eventI32Plus16 =
              rewriter.create<emitc::AddOp>(loc, i32Ty, eventI32, c16).getResult();
          emitSet(eventI32Plus16, IntegerAttr{}, /*isDynamic=*/true);
        }
      }

      rewriter.eraseOp(op);
      return success();
    }

    InterCoreSyncCallDesc desc;
    if (eventIdAttr) {
      desc = buildInterCoreSyncSetCall(rewriter, loc, targetArch, op.getPipe(),
                                       eventIdAttr, fftsMode);
    } else {
      desc = buildInterCoreSyncSetCallDyn(rewriter, loc, targetArch, op.getPipe(),
                                          eventIdDyn, fftsMode);
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, desc.callee,
                                         /*args=*/desc.args,
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/desc.operands);

    rewriter.eraseOp(op);
    return success();
  }

  PTOArch targetArch;
};

struct PTOSyncWaitToEmitC : public OpConversionPattern<mlir::pto::SyncWaitOp> {
  PTOSyncWaitToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                     PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SyncWaitOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult
  matchAndRewrite(mlir::pto::SyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    IntegerAttr eventIdAttr = op.getEventIdAttr();
    Value eventIdDyn = adaptor.getEventIdDyn();

    if ((eventIdAttr != nullptr) == static_cast<bool>(eventIdDyn))
      return rewriter.notifyMatchFailure(
          op, "expects exactly one of static event_id attr or dynamic event_id operand");

    InterCoreSyncCallDesc desc;
    if (eventIdAttr) {
      desc = buildInterCoreSyncWaitCall(rewriter, targetArch, op.getPipe(),
                                        eventIdAttr);
    } else {
      desc = buildInterCoreSyncWaitCallDyn(rewriter, loc, targetArch, op.getPipe(),
                                           eventIdDyn);
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, desc.callee,
                                         desc.args, ArrayAttr{}, desc.operands);

    rewriter.eraseOp(op);
    return success();
  }

  PTOArch targetArch;
};

// GetBlockIdxOp Lowering (pto.get_block_idx -> get_block_idx())
struct PTOGetBlockIdxToEmitC
    : public OpConversionPattern<mlir::pto::GetBlockIdxOp> {
  using OpConversionPattern<mlir::pto::GetBlockIdxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetBlockIdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_block_idx", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetBlockNumOp Lowering (pto.get_block_num -> get_block_num())
struct PTOGetBlockNumToEmitC
    : public OpConversionPattern<mlir::pto::GetBlockNumOp> {
  using OpConversionPattern<mlir::pto::GetBlockNumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetBlockNumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_block_num", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetSubBlockIdxOp Lowering (pto.get_block_idx -> get_subblockid())
struct PTOGetSubBlockIdxToEmitC
    : public OpConversionPattern<mlir::pto::GetSubBlockIdxOp> {
  using OpConversionPattern<mlir::pto::GetSubBlockIdxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetSubBlockIdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_subblockid", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetSubBlockNumOp Lowering.
struct PTOGetSubBlockNumToEmitC
    : public OpConversionPattern<mlir::pto::GetSubBlockNumOp> {
  using OpConversionPattern<mlir::pto::GetSubBlockNumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetSubBlockNumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_subblockdim", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};


struct PTOMScatterToMSCATTER : public OpConversionPattern<pto::MScatterOp> {
  using OpConversionPattern<pto::MScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Value src = peelUnrealized(adaptor.getSrc());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value mem = peelUnrealized(adaptor.getMem());

    Value memArg = maybeWrapGlobalMemrefAsGlobalTensor(
        rewriter, op.getLoc(), mem, op.getMem().getType(), op.getOperation());

    auto scatterAtomicTok = [&](pto::ScatterAtomicOp atomic) -> StringRef {
      switch (atomic) {
      case pto::ScatterAtomicOp::None:
        return "pto::ScatterAtomicOp::None";
      case pto::ScatterAtomicOp::Add:
        return "pto::ScatterAtomicOp::Add";
      case pto::ScatterAtomicOp::Max:
        return "pto::ScatterAtomicOp::Max";
      case pto::ScatterAtomicOp::Min:
        return "pto::ScatterAtomicOp::Min";
      }
      llvm_unreachable("unknown ScatterAtomicOp");
    };
    auto scatterOobTok = [&](pto::ScatterOOB mode) -> StringRef {
      switch (mode) {
      case pto::ScatterOOB::Undefined:
        return "pto::ScatterOOB::Undefined";
      case pto::ScatterOOB::Skip:
        return "pto::ScatterOOB::Skip";
      case pto::ScatterOOB::Clamp:
        return "pto::ScatterOOB::Clamp";
      case pto::ScatterOOB::Wrap:
        return "pto::ScatterOOB::Wrap";
      }
      llvm_unreachable("unknown ScatterOOB");
    };

    SmallVector<Attribute, 2> templateArgVec;
    if (op.getScatterAtomicOp() != pto::ScatterAtomicOp::None ||
        op.getScatterOob() != pto::ScatterOOB::Undefined) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, scatterAtomicTok(op.getScatterAtomicOp())));
      if (op.getScatterOob() != pto::ScatterOOB::Undefined)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, scatterOobTok(op.getScatterOob())));
    }
    ArrayAttr templateArgs =
        templateArgVec.empty() ? ArrayAttr{} : rewriter.getArrayAttr(templateArgVec);

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MSCATTER",
        ArrayAttr{}, templateArgs,
        ValueRange{memArg, src, idx});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOSetValToSETVAL : public OpConversionPattern<pto::TSetValOp> {
  using OpConversionPattern<pto::TSetValOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSetValOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value dst = peelUnrealized(adaptor.getDst());
    Value val = peelUnrealized(adaptor.getVal());

    // ---- offset: SSA index operand ----
    Value offset = peelUnrealized(adaptor.getOffset());

    // Emit a marker call and let the ptoas post-processing step lower it to
    // the corresponding tile setter.
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__TILE_SET_VALUE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dst, offset, val});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOGetValToGETVAL : public OpConversionPattern<pto::TGetValOp> {
  using OpConversionPattern<pto::TGetValOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGetValOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());

    // ---- offset: SSA index operand ----
    Value offset = peelUnrealized(adaptor.getOffset());

    // Emit a marker call and let the ptoas post-processing step lower it to
    // the corresponding tile getter.
    Type dstTy = getTypeConverter()->convertType(op.getDst().getType());
    if (!dstTy)
      return failure();
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(),
        TypeRange{dstTy},
        "PTOAS__TILE_GET_VALUE",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{src, offset});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct PTOTAxpyToEmitC : public OpConversionPattern<pto::TAxpyOp> {
  using OpConversionPattern<pto::TAxpyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAxpyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TAXPY",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOHistogramToEmitC : public OpConversionPattern<pto::THistogramOp> {
  using OpConversionPattern<pto::THistogramOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::THistogramOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value dst = peelUnrealized(adaptor.getDst());

    auto templateArgs = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, op.getIsMSB() ? "true" : "false")});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "THISTOGRAM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/ValueRange{dst, src, idx});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOGetScaleAddrToEmitC
    : public OpConversionPattern<pto::TGetScaleAddrOp> {
  using OpConversionPattern<pto::TGetScaleAddrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGetScaleAddrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGET_SCALE_ADDR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSetValidShapeToEmitC : public OpConversionPattern<pto::SetValidShapeOp> {
  using OpConversionPattern<pto::SetValidShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetValidShapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
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

    Value src = peelAllCasts(peelUnrealized(adaptor.getSource()));
    Value row = peelUnrealized(adaptor.getValidRow());
    Value col = peelUnrealized(adaptor.getValidCol());

    if (!isTileLike(src))
      return rewriter.notifyMatchFailure(
          op, "set_validshape source must lower to a tile-like value");

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__TILE_SET_VALIDSHAPE", ArrayAttr{},
        ArrayAttr{}, ValueRange{src, row, col});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTAssignToEmitC : public OpConversionPattern<pto::TAssignOp> {
  using OpConversionPattern<pto::TAssignOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAssignOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
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

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value tile = peelAllCasts(peelUnrealized(adaptor.getTile()));
    if (!isTileLike(tile))
      return rewriter.notifyMatchFailure(
          op, "tassign tile must lower to a tile-like value");

    Value addr = peelUnrealized(adaptor.getAddr());
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    if (isa<emitc::PointerType>(addr.getType()) ||
        (isa<emitc::OpaqueType>(addr.getType()) &&
         cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"))) {
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      addr = rewriter
                 .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                              ArrayAttr{}, rcU64,
                                              ValueRange{addr})
                 .getResult(0);
    } else if (addr.getType() != u64Ty) {
      addr = rewriter.create<emitc::CastOp>(loc, u64Ty, addr).getResult();
    }

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{tile, addr});
    rewriter.replaceOp(op, tile);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.load_scalar / pto.store_scalar lowering -> ptr[offset]
//===----------------------------------------------------------------------===//

