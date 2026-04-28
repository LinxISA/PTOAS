// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitC.cpp - PTO to EmitC conversion pass ----------------------===//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <climits>

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"                   
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
#define GEN_PASS_DEF_EMITPTOMANUAL
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

static std::string getElemTypeStringForGT(Type elemTy);
static bool getStaticMemrefLayout(MemRefType mrTy,
                                  SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset);
static int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs);
static void buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> strides,
                                            SmallVectorImpl<int64_t> &shape5D,
                                            SmallVectorImpl<int64_t> &stride5D);
static std::string joinIntTemplateParams(ArrayRef<int64_t> values);

static const char *addrSpaceQualifier(pto::AddressSpace as) {
  switch (as) {
  case pto::AddressSpace::Zero:
    return "__gm__";
  case pto::AddressSpace::VEC:
    return "__ubuf__";
  case pto::AddressSpace::GM:
    return "__gm__";
  case pto::AddressSpace::MAT:
    return "__cbuf__";
  case pto::AddressSpace::LEFT:
    return "__ca__";
  case pto::AddressSpace::RIGHT:
    return "__cb__";
  case pto::AddressSpace::ACC:
    return "__cc__";
  case pto::AddressSpace::BIAS:
    // Bias tiles are special in pto-isa; keep a safe fallback qualifier.
    return "__gm__";
  case pto::AddressSpace::SCALING:
    // pto-isa TileType::Scaling maps to __fbuf__ (see pto/common/memory.hpp).
    return "__fbuf__";
  }
  return "__gm__";
}

static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";
static constexpr llvm::StringLiteral kLoweredSetValidShapeConfigAttrName =
    "__pto.lowered_set_validshape_config";
static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";

static Value peelUnrealized(Value v) {
  if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    return castOp.getOperand(0);
  return v;
}

static Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                         Location loc, Value basePtr,
                                         MemRefType mrTy, Operation *anchor);

static Value maybeWrapGlobalMemrefAsGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value loweredValue,
    Type originalType, Operation *anchor);

static std::optional<mlir::pto::Layout> getLayoutAttrFromOp(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<mlir::pto::LayoutAttr>("layout"))
    return attr.getLayout();
  return std::nullopt;
}

static std::optional<mlir::pto::Layout> resolveLayoutFromValueChain(Value v) {
  v = peelUnrealized(v);
  while (Operation *def = v.getDefiningOp()) {
    if (auto layout = getLayoutAttrFromOp(def))
      return layout;
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      v = peelUnrealized(subview.getSource());
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      v = peelUnrealized(reinterpret.getSource());
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      v = peelUnrealized(cast.getSource());
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (unrealized->getNumOperands() == 0)
        break;
      v = peelUnrealized(unrealized.getOperand(0));
      continue;
    }
    break;
  }
  return std::nullopt;
}

static std::optional<mlir::pto::Layout>
resolveLayoutForGlobalTensor(Operation *anchor, Value basePtr) {
  if (auto layout = getLayoutAttrFromOp(anchor))
    return layout;
  return resolveLayoutFromValueChain(basePtr);
}

static std::string layoutToEmitCString(mlir::pto::Layout layout) {
  switch (layout) {
  case mlir::pto::Layout::ND:
    return "pto::Layout::ND";
  case mlir::pto::Layout::DN:
    return "pto::Layout::DN";
  case mlir::pto::Layout::NZ:
    return "pto::Layout::NZ";
  }
  return "pto::Layout::ND";
}

static bool isEmitCGlobalTensorLikeType(Type ty) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
  return opaqueTy && opaqueTy.getValue().contains("GlobalTensor<");
}

static std::string getEmitCScalarTypeToken(Type elemTy) {
  if (pto::isPTOFloat8Type(elemTy) &&
      (elemTy.isFloat8E4M3() || elemTy.isFloat8E4M3FN() ||
       elemTy.isFloat8E4M3FNUZ() || elemTy.isFloat8E4M3B11FNUZ()))
    return "float8_e4m3_t";
  if (pto::isPTOFloat8Type(elemTy) &&
      (elemTy.isFloat8E5M2() || elemTy.isFloat8E5M2FNUZ()))
    return "float8_e5m2_t";
  if (isa<pto::HiF8Type>(elemTy))
    return "hifloat8_t";
  if (isa<pto::F4E1M2x2Type>(elemTy))
    return "float4_e1m2x2_t";
  if (isa<pto::F4E2M1x2Type>(elemTy))
    return "float4_e2m1x2_t";
  if (elemTy.isF16())
    return "half";
  if (elemTy.isBF16())
    return "bfloat16_t";
  if (elemTy.isF32())
    return "float";
  if (elemTy.isF64())
    return "double";
  if (elemTy.isInteger(8))
    return (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8)) ? "int8_t"
                                                                       : "uint8_t";
  if (elemTy.isInteger(16))
    return (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
               ? "int16_t"
               : "uint16_t";
  if (elemTy.isInteger(32))
    return (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
               ? "int32_t"
               : "uint32_t";
  if (elemTy.isInteger(64))
    return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
  return "float";
}

static int64_t getEmitCScalarByteWidth(Type elemTy) {
  if (pto::getPTOStorageElemByteSize(elemTy) == 1)
    return 1;
  if (elemTy.isF16() || elemTy.isBF16() || elemTy.isInteger(16))
    return 2;
  if (elemTy.isF32() || elemTy.isInteger(32))
    return 4;
  if (elemTy.isF64() || elemTy.isInteger(64))
    return 8;
  return 4;
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PTOToEmitCTypeConverter : public TypeConverter {
public:
  PTOToEmitCTypeConverter(MLIRContext *Ctx, PTOArch targetArch) {
    // ---------------------------------------------------------
    // 1. 基本类型 (f32, i32, index)
    // ---------------------------------------------------------
    addConversion([Ctx](FloatType type) -> Type {
      if (type.isFloat8E4M3() || type.isFloat8E4M3FN() ||
          type.isFloat8E4M3FNUZ() || type.isFloat8E4M3B11FNUZ())
        return emitc::OpaqueType::get(Ctx, "float8_e4m3_t");
      if (type.isFloat8E5M2() || type.isFloat8E5M2FNUZ())
        return emitc::OpaqueType::get(Ctx, "float8_e5m2_t");
      if (type.isF32()) return emitc::OpaqueType::get(Ctx, "float");
      if (type.isF16()) return emitc::OpaqueType::get(Ctx, "half");
      if (type.isBF16()) return emitc::OpaqueType::get(Ctx, "bfloat16_t");
      if (type.isF64()) return emitc::OpaqueType::get(Ctx, "double");
      llvm::errs() << "[Debug] Unsupported FloatType: " << type << "\n";
      return Type{};
    });

    addConversion([Ctx](pto::HiF8Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "hifloat8_t");
    });
    addConversion([Ctx](pto::F4E1M2x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e1m2x2_t");
    });
    addConversion([Ctx](pto::F4E2M1x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e2m1x2_t");
    });

    addConversion([Ctx](IntegerType type) -> Type {
      if (type.getWidth() == 1)
        return type;

      // Prefer fixed-width C types. Preserve signedness if the MLIR integer is
      // explicitly signed/unsigned; treat signless as signed by default.
      const bool isUnsigned = type.isUnsignedInteger();
      switch (type.getWidth()) {
      case 8:
        return emitc::OpaqueType::get(Ctx, isUnsigned ? "uint8_t" : "int8_t");
      case 16:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint16_t" : "int16_t");
      case 32:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint32_t" : "int32_t");
      case 64:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint64_t" : "int64_t");
      default:
        llvm::errs() << "[Debug] Unsupported IntegerType width: "
                     << type.getWidth() << "\n";
        return emitc::OpaqueType::get(Ctx, "int32_t"); // Fallback
      }
    });

    addConversion([Ctx](IndexType type) -> Type {
      return emitc::OpaqueType::get(Ctx, "int32_t");
    });

    // vector<4xi16> (e.g. TMRGSORT executedNumList) -> pto::MrgSortExecutedNumList
    addConversion([Ctx](VectorType type) -> Type {
      if (type.getRank() == 1 && type.getNumElements() == 4 &&
          type.getElementType().isInteger(16))
        return emitc::OpaqueType::get(Ctx, "pto::MrgSortExecutedNumList");
      return Type{};
    });

    // ---------------------------------------------------------
    // 2. PTO 特殊类型 (透传或转换)
    // ---------------------------------------------------------
    addConversion([Ctx](emitc::OpaqueType type) { return type; });
    addConversion([Ctx](emitc::PointerType type) { return type; });

    // ---------------------------------------------------------
    // 2.5 PtrType 转换 (指针类型)
    // ---------------------------------------------------------
    addConversion([this, Ctx](pto::PtrType type) -> std::optional<Type> {
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType);
      if (!newElemType)
        return std::nullopt;

      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
        llvm::errs() << "  [Error] PtrType elem type is not OpaqueType: "
                     << newElemType << "\n";
        return std::nullopt;
      }

      std::string qualifier = "__gm__";

      std::string finalTypeStr = qualifier + " " + elemTypeStr;
      return emitc::PointerType::get(
          emitc::OpaqueType::get(Ctx, finalTypeStr));
    });

    addConversion([Ctx](pto::PipeType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "auto");
    });

    addConversion([Ctx](pto::EventIdArrayType type) -> Type {
      std::string tok = "PTOAS_EventIdArray<" + std::to_string(type.getSize()) + ">";
      return emitc::OpaqueType::get(Ctx, tok);
    });

    addConversion([Ctx](pto::AsyncSessionType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncSession");
    });

    addConversion([Ctx](pto::AsyncEventType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncEvent");
    });

    // ---------------------------------------------------------
    // 3. MemRef 转换 (Debug 重点)
    // ---------------------------------------------------------
    addConversion([this, Ctx](MemRefType type) -> std::optional<Type> {
      LLVM_DEBUG(llvm::dbgs() << "Converting MemRef: " << type << "\n");

      // A. 转换元素类型
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType); 
      if (!newElemType) {
        llvm::errs() << "  [Error] Failed to convert element type: " << elemType << "\n";
        return std::nullopt;
      }
      
      // 获取元素类型的字符串
      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
         llvm::errs() << "  [Error] Converted element type is not OpaqueType: " << newElemType << "\n";
         return std::nullopt;
      }

      // B. 处理 Memory Space
      std::string qualifier = "";
      Attribute memorySpace = type.getMemorySpace();
      
      if (!memorySpace) {
         qualifier = "__gm__";
      } else if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
         qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
      } else {
         llvm::errs() << "  [Warning] Unknown MemorySpace Attribute type: " << memorySpace << "\n";
         qualifier = "__gm__"; // Fallback
      }

      std::string finalTypeStr = qualifier + " " + elemTypeStr;
      LLVM_DEBUG(llvm::dbgs() << "  [Success] -> " << finalTypeStr << "*\n");
      
      return emitc::PointerType::get(emitc::OpaqueType::get(Ctx, finalTypeStr));
    });

    // ---------------------------------------------------------
    // 4. Function & Materialization
    // ---------------------------------------------------------
    addConversion([this](FunctionType type) -> Type {
      SmallVector<Type> inputs;
      if (failed(convertTypes(type.getInputs(), inputs))) return Type{};
      SmallVector<Type> results;
      if (failed(convertTypes(type.getResults(), results))) return Type{};
      return FunctionType::get(type.getContext(), inputs, results);
    });

    auto materializeCast = [](OpBuilder &Builder, Type ResultType,
                              ValueRange Inputs, Location Loc) -> Value {
      if (Inputs.size() != 1) return Value();
      return Builder.create<UnrealizedConversionCastOp>(Loc, ResultType, Inputs[0]).getResult(0);
    };

    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
    // Needed for region/block signature conversions (e.g. CFG block args).
    addArgumentMaterialization(materializeCast);
  }
};

static constexpr unsigned kPTOIndexBitWidth =
    32; // keep consistent with IndexType conversion

// Forward declarations (definitions below).
static inline std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a);
static emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth);
static emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth);
static emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth);
static emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth);
static Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal);
static Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value);
static Value emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src);
static FailureOr<std::string> buildEmitCOpaqueConstantLiteral(Type targetType,
                                                              Attribute valueAttr);
static Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth);
static bool needsA5NoSplitVectorGuard(Operation *op);

static FailureOr<std::string> getTileSplitToken(int64_t split) {
  switch (split) {
  case 0:
    return std::string("TileSplitAxis::TILE_NO_SPLIT");
  case 1:
    return std::string("TileSplitAxis::TILE_UP_DOWN");
  case 2:
    return std::string("TileSplitAxis::TILE_LEFT_RIGHT");
  default:
    return failure();
  }
}

static FailureOr<std::string>
getTPipeDirectionToken(bool isL2G2L, int8_t dirMask, PTOArch targetArch) {
  if (dirMask == 1) {
    if (isL2G2L && targetArch == PTOArch::A5)
      return std::string("Direction::DIR_C2V_GM");
    return std::string("Direction::DIR_C2V");
  }
  if (dirMask == 2) {
    if (isL2G2L && targetArch == PTOArch::A5)
      return std::string("Direction::DIR_V2C_GM");
    return std::string("Direction::DIR_V2C");
  }
  if (dirMask == 3)
    return std::string("Direction::DIR_BOTH");
  return failure();
}

static std::string buildTPipeToken(int32_t flagBase, llvm::StringRef dirTok,
                                   int32_t slotSize, int32_t slotNum,
                                   int32_t localSlotNum, bool nosplit) {
  std::string token = "TPipe<" + std::to_string(flagBase) + ", " + dirTok.str() +
                      ", " + std::to_string(slotSize) + ", " +
                      std::to_string(slotNum);
  token += ", " + std::to_string(localSlotNum);
  token += nosplit ? ", true" : ", false";
  token += ">";
  return token;
}

static FailureOr<std::string> buildTPipeTokenFromInitOp(Operation *op,
                                                        PTOArch targetArch) {
  if (auto initOp = dyn_cast<pto::InitializeL2G2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr())
      return failure();
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/true, initOp.getDirMask(), targetArch);
    if (failed(dirTok))
      return failure();
    int32_t localSlotNum = initOp.getLocalSlotNumAttr()
                               ? initOp.getLocalSlotNumAttr().getInt()
                               : initOp.getSlotNum();
    return buildTPipeToken(initOp.getFlagBaseAttr().getInt(), *dirTok,
                           initOp.getSlotSize(), initOp.getSlotNum(),
                           localSlotNum,
                           initOp.getNosplitAttr() &&
                               initOp.getNosplitAttr().getValue());
  }

  if (auto initOp = dyn_cast<pto::InitializeL2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr())
      return failure();
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/false, initOp.getDirMask(), targetArch);
    if (failed(dirTok))
      return failure();
    return buildTPipeToken(initOp.getFlagBaseAttr().getInt(), *dirTok,
                           initOp.getSlotSize(), initOp.getSlotNum(), 2,
                           initOp.getNosplitAttr() &&
                               initOp.getNosplitAttr().getValue());
  }

  return failure();
}

static FailureOr<std::string> getTPipeTokenFromValue(Value pipeHandle,
                                                     PTOArch targetArch) {
  pipeHandle = peelUnrealized(pipeHandle);
  Operation *def = pipeHandle.getDefiningOp();
  if (!def)
    return failure();
  return buildTPipeTokenFromInitOp(def, targetArch);
}

static bool isSetFFTsPointerLikeType(Type ty) {
  if (isa<emitc::PointerType>(ty))
    return true;
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty))
    return opaqueTy.getValue().ends_with("*");
  return false;
}

static bool tileDataReturnsIntegralAddress(pto::AddressSpace as) {
  return as == pto::AddressSpace::BIAS;
}

static emitc::OpaqueType getTileDataResultType(MLIRContext *ctx,
                                               pto::AddressSpace as,
                                               StringRef elemTok) {
  if (tileDataReturnsIntegralAddress(as))
    return emitc::OpaqueType::get(ctx, "uint64_t");
  return emitc::OpaqueType::get(
      ctx, std::string(addrSpaceQualifier(as)) + " " + elemTok.str() + "*");
}

static Value materializeTileDataValue(ConversionPatternRewriter &rewriter,
                                      Location loc, Value tile,
                                      pto::AddressSpace as,
                                      StringRef elemTok) {
  auto rawTy = getTileDataResultType(rewriter.getContext(), as, elemTok);
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, rawTy, "PTOAS__TILE_DATA",
                                   ArrayAttr{}, ArrayAttr{},
                                   ValueRange{tile})
      .getResult(0);
}

static Value materializeAddressAsPointer(ConversionPatternRewriter &rewriter,
                                         Location loc, Value addr,
                                         pto::AddressSpace as,
                                         StringRef elemTok) {
  auto *ctx = rewriter.getContext();
  std::string ptrTyStr =
      std::string(addrSpaceQualifier(as)) + " " + elemTok.str() + "*";
  auto ptrTy = emitc::OpaqueType::get(ctx, ptrTyStr);
  if (isSetFFTsPointerLikeType(addr.getType())) {
    if (addr.getType() == ptrTy)
      return addr;
    return rewriter.create<emitc::CastOp>(loc, ptrTy, addr).getResult();
  }
  auto castTyAttr =
      rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, ptrTyStr)});
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, ptrTy, "reinterpret_cast",
                                   ArrayAttr{}, castTyAttr,
                                   ValueRange{addr})
      .getResult(0);
}

struct InterCoreSyncCallDesc {
  const char *callee = nullptr;
  ArrayAttr args;
  SmallVector<Value, 2> operands;
};

static Value castInterCoreEventIdToI32(ConversionPatternRewriter &rewriter,
                                       Location loc, Value eventId) {
  auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
  if (eventId.getType() == i32Ty)
    return eventId;
  return emitCCast(rewriter, loc, i32Ty, eventId);
}

static Attribute getFFTSModeCodegenArg(ConversionPatternRewriter &rewriter,
                                       int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  if (fftsMode == 2)
    return emitc::OpaqueAttr::get(ctx, "FFTS_MODE_VAL");
  return emitc::OpaqueAttr::get(ctx, std::to_string(fftsMode));
}

static Value createFFTSMsg(ConversionPatternRewriter &rewriter, Location loc,
                           Value eventI32, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  auto msgTy = emitc::OpaqueType::get(ctx, "uint16_t");
  auto msgArgs = rewriter.getArrayAttr({
      getFFTSModeCodegenArg(rewriter, fftsMode),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, msgTy, "getFFTSMsg",
                                   /*args=*/msgArgs,
                                   /*templateArgs=*/ArrayAttr{},
                                   /*operands=*/ValueRange{eventI32})
      .getResult(0);
}

static InterCoreSyncCallDesc buildInterCoreSyncSetCall(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  if (targetArch == PTOArch::A3) {
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value eventVal =
        makeEmitCIntConstant(rewriter, loc, i32Ty, eventIdAttr.getInt());
    Value msgVal = createFFTSMsg(rewriter, loc, eventVal, fftsMode);

    InterCoreSyncCallDesc desc;
    desc.callee = "ffts_cross_core_sync";
    desc.args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
    });
    desc.operands.push_back(msgVal);
    return desc;
  }

  InterCoreSyncCallDesc desc;
  desc.callee = "set_intra_block";
  desc.args = rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(ctx, pipeTok), eventIdAttr});
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncSetCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  Value eventI32 = castInterCoreEventIdToI32(rewriter, loc, eventIdVal);

  if (targetArch == PTOArch::A3) {
    Value msgVal = createFFTSMsg(rewriter, loc, eventI32, fftsMode);

    InterCoreSyncCallDesc desc;
    desc.callee = "ffts_cross_core_sync";
    desc.args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
    });
    desc.operands.push_back(msgVal);
    return desc;
  }

  InterCoreSyncCallDesc desc;
  desc.callee = "set_intra_block";
  desc.args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, pipeTok),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  desc.operands.push_back(eventI32);
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncWaitCall(
    ConversionPatternRewriter &rewriter, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  InterCoreSyncCallDesc desc;
  if (targetArch == PTOArch::A3) {
    desc.callee = "wait_flag_dev";
    desc.args = rewriter.getArrayAttr({eventIdAttr});
    return desc;
  }

  desc.callee = "wait_intra_block";
  desc.args = rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(ctx, pipeTok), eventIdAttr});
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncWaitCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  Value eventI32 = castInterCoreEventIdToI32(rewriter, loc, eventIdVal);

  InterCoreSyncCallDesc desc;
  if (targetArch == PTOArch::A3) {
    desc.callee = "wait_flag_dev";
    desc.args = rewriter.getArrayAttr({IntegerAttr::get(IndexType::get(ctx), 0)});
    desc.operands.push_back(eventI32);
    return desc;
  }

  desc.callee = "wait_intra_block";
  desc.args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, pipeTok),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  desc.operands.push_back(eventI32);
  return desc;
}

static bool hasInterCoreSyncOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SyncSetOp, pto::SyncWaitOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool hasSetFFTsOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SetFFTsOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

//===----------------------------------------------------------------------===//
// Arith -> EmitC (full dialect coverage for scalar ops)
//===----------------------------------------------------------------------===//

template <typename ArithOp, typename EmitCOp>
struct ArithSimpleBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<EmitCOp>(op, dstTy, adaptor.getOperands());
    return success();
  }
};

// Integer bitwise ops (andi/ori/xori) on signless integers: perform in unsigned
// to avoid signedness pitfalls, then cast back.
template <typename ArithOp, typename EmitCOp>
struct ArithUnsignedBitwiseBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = this->getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<EmitCOp>(op, dstTy, adaptor.getLhs(),
                                           adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value resU = rewriter.create<EmitCOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, resU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithDivUIToEmitC : public OpConversionPattern<arith::DivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value divU = rewriter.create<emitc::DivOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, divU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithRemUIToEmitC : public OpConversionPattern<arith::RemUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value remU = rewriter.create<emitc::RemOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, remU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCeilDivUIToEmitC : public OpConversionPattern<arith::CeilDivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value one = makeEmitCIntConstant(rewriter, loc, uTy, 1);
    Value rhsMinusOne = rewriter.create<emitc::SubOp>(loc, uTy, rhsU, one);
    Value num = rewriter.create<emitc::AddOp>(loc, uTy, lhsU, rhsMinusOne);
    Value divU = rewriter.create<emitc::DivOp>(loc, uTy, num, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, divU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCeilDivSIToEmitC : public OpConversionPattern<arith::CeilDivSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
    Value one = makeEmitCIntConstant(rewriter, loc, dstTy, 1);

    Value q0 = rewriter.create<emitc::DivOp>(loc, dstTy, adaptor.getLhs(),
                                             adaptor.getRhs());
    Value r = rewriter.create<emitc::RemOp>(loc, dstTy, adaptor.getLhs(),
                                            adaptor.getRhs());

    Value rNeZero = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                                  emitc::CmpPredicate::ne, r,
                                                  zero);
    Value lhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getLhs(),
                                      zero);
    Value rhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getRhs(),
                                      zero);
    Value signsSame =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::eq, lhsLt0, rhsLt0);
    Value adjust =
        rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                             rNeZero, signsSame);

    Value qPlusOne = rewriter.create<emitc::AddOp>(loc, dstTy, q0, one);
    Value result = rewriter.create<emitc::ConditionalOp>(loc, dstTy, adjust,
                                                         qPlusOne, q0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithFloorDivSIToEmitC : public OpConversionPattern<arith::FloorDivSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::FloorDivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
    Value one = makeEmitCIntConstant(rewriter, loc, dstTy, 1);

    Value q0 = rewriter.create<emitc::DivOp>(loc, dstTy, adaptor.getLhs(),
                                             adaptor.getRhs());
    Value r = rewriter.create<emitc::RemOp>(loc, dstTy, adaptor.getLhs(),
                                            adaptor.getRhs());

    Value rNeZero = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                                  emitc::CmpPredicate::ne, r,
                                                  zero);
    Value lhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getLhs(),
                                      zero);
    Value rhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getRhs(),
                                      zero);
    Value signsDifferent =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::ne, lhsLt0, rhsLt0);
    Value adjust =
        rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                             rNeZero, signsDifferent);

    Value qMinusOne = rewriter.create<emitc::SubOp>(loc, dstTy, q0, one);
    Value result = rewriter.create<emitc::ConditionalOp>(loc, dstTy, adjust,
                                                         qMinusOne, q0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftLeftToEmitC : public OpConversionPattern<arith::ShLIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShLIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      // Compute on u8 and truncate to i1.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseLeftShiftOp>(loc, u8Ty, lhsU8,
                                                            rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value shU =
        rewriter.create<emitc::BitwiseLeftShiftOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, shU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftRightUIToEmitC : public OpConversionPattern<arith::ShRUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      // (x >> y) on i1 is either x (y==0) or 0 (y!=0); approximate in u8.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseRightShiftOp>(loc, u8Ty, lhsU8,
                                                             rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value shU =
        rewriter.create<emitc::BitwiseRightShiftOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, shU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftRightSIToEmitC : public OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      // (x >> y) on i1 is either x (y==0) or 0 (y!=0); approximate in u8.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseRightShiftOp>(loc, u8Ty, lhsU8,
                                                             rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    // Signed arithmetic shift; cast RHS to unsigned to interpret shift amount.
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value sh =
        rewriter.create<emitc::BitwiseRightShiftOp>(loc, dstTy, adaptor.getLhs(),
                                                    rhsU);
    rewriter.replaceOp(op, sh);
    return success();
  }
};

struct ArithNegFToEmitC : public OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::UnaryMinusOp>(op, dstTy, adaptor.getOperand());
    return success();
  }
};

struct ArithRemFToEmitC : public OpConversionPattern<arith::RemFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::RemFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // Use builtin `fmod` when possible. For f16, compute in float and cast back.
    Type callTy = dstTy;
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF16()) {
        auto f32Ty = emitc::OpaqueType::get(rewriter.getContext(), "float");
        lhs = emitCCast(rewriter, loc, f32Ty, lhs);
        rhs = emitCCast(rewriter, loc, f32Ty, rhs);
        callTy = f32Ty;
      }
    }

    // Prefer `__builtin_fmod*` to avoid relying on extra headers.
    llvm::StringRef callee = "__builtin_fmod";
    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF32() || opFloatTy.isF16())
        callee = "__builtin_fmodf";
      else if (opFloatTy.isF64())
        callee = "__builtin_fmod";
    }

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{callTy}, callee, ValueRange{lhs, rhs},
        /*args=*/ArrayAttr{}, /*template_args=*/ArrayAttr{});
    Value result = call.getResult(0);
    if (callTy != dstTy)
      result = emitCCast(rewriter, loc, dstTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithSelectToEmitC : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getCondition().getType().isInteger(1))
      return rewriter.notifyMatchFailure(
          op, "only scalar i1 conditions supported for arith.select");

    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto cond =
        rewriter.create<emitc::ConditionalOp>(op.getLoc(), dstTy,
                                              adaptor.getCondition(),
                                              adaptor.getTrueValue(),
                                              adaptor.getFalseValue());
    rewriter.replaceOp(op, cond.getResult());
    return success();
  }
};

struct ArithExtUIToEmitC : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // i1 -> iN: bool to integer already behaves as 0/1.
    if (srcIntTy.getWidth() == 1) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto uSrcTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), srcIntTy.getWidth());
    auto uDstTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
    Value srcU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                           srcIntTy.getWidth());
    Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
    Value result = emitCCast(rewriter, loc, dstTy, extU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithExtSIToEmitC : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // i1 sign-extension: 0 -> 0, 1 -> -1.
    if (srcIntTy.getWidth() == 1) {
      Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
      Value asInt = emitCCast(rewriter, loc, dstTy, adaptor.getIn());
      Value neg = rewriter.create<emitc::SubOp>(loc, dstTy, zero, asInt).getResult();
      rewriter.replaceOp(op, neg);
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

template <typename CastOp>
struct ArithCastToEmitC : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

struct ArithIndexCastUIToEmitC : public OpConversionPattern<arith::IndexCastUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::IndexCastUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // MemRef casts are handled elsewhere; for safety, fall back to emitc.cast.
    if (isa<MemRefType>(op.getIn().getType()) || isa<MemRefType>(op.getType())) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto getBW = [](Type t) -> std::optional<unsigned> {
      if (auto i = dyn_cast<IntegerType>(t))
        return i.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };

    auto srcBW = getBW(op.getIn().getType());
    auto dstBW = getBW(op.getType());
    if (!srcBW || !dstBW)
      return rewriter.notifyMatchFailure(op, "unsupported index_castui types");

    if (*dstBW <= *srcBW) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto uSrcTy = getUnsignedIntOpaqueType(rewriter.getContext(), *srcBW);
    auto uDstTy = getUnsignedIntOpaqueType(rewriter.getContext(), *dstBW);
    Value srcU = emitCCast(rewriter, loc, uSrcTy, adaptor.getIn());
    Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
    Value result = emitCCast(rewriter, loc, dstTy, extU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithUIToFPToEmitC : public OpConversionPattern<arith::UIToFPOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer input");

    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // Convert via an unsigned integer type of the same width.
    if (srcIntTy.getWidth() == 1) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }
    auto uSrcTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), srcIntTy.getWidth());
    Value srcU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                           srcIntTy.getWidth());
    Value fp = rewriter.create<emitc::CastOp>(loc, dstTy, srcU).getResult();
    rewriter.replaceOp(op, fp);
    return success();
  }
};

struct ArithFPToUIToEmitC : public OpConversionPattern<arith::FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::FPToUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    if (!dstIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer result");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    auto uDstTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
    Value asU = rewriter.create<emitc::CastOp>(loc, uDstTy, adaptor.getIn()).getResult();
    Value result = emitCCast(rewriter, loc, dstTy, asU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithBitcastToEmitC : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // For pointer-like types, a regular cast is fine.
    if (isa<emitc::PointerType>(dstTy)) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    // Only support scalar int/float/index bitcasts here.
    auto srcTy = op.getIn().getType();
    auto dstOrigTy = op.getType();

    auto getBitWidth = [](Type t) -> std::optional<unsigned> {
      if (auto it = dyn_cast<IntegerType>(t))
        return it.getWidth();
      if (auto ft = dyn_cast<FloatType>(t))
        return ft.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };
    auto srcBW = getBitWidth(srcTy);
    auto dstBW = getBitWidth(dstOrigTy);
    if (!srcBW || !dstBW || *srcBW != *dstBW)
      return rewriter.notifyMatchFailure(op, "bitcast requires equal bitwidth");

    // Determine the template argument from the destination type string.
    auto dstOpaque = dyn_cast<emitc::OpaqueType>(dstTy);
    if (!dstOpaque)
      return rewriter.notifyMatchFailure(op, "expected emitc opaque dest type");

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      dstOpaque.getValue())});
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{dstTy}, "ptoas_bitcast", /*operands=*/ValueRange{adaptor.getIn()},
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

// arith.cmpf lowering with ordered/unordered semantics.
struct ArithCmpFToEmitC : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  struct CmpFConfig {
    bool unordered = false;
    emitc::CmpPredicate predicate = emitc::CmpPredicate::eq;
  };

  static Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
                     Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::ne,
                              v, v)
        .getResult();
  }

  static Value isNotNaN(ConversionPatternRewriter &rewriter, Location loc,
                        Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::eq,
                              v, v)
        .getResult();
  }

  static std::optional<Value> buildSpecialCmpFResult(
      arith::CmpFPredicate predicate, ConversionPatternRewriter &rewriter,
      Location loc, Type i1Ty, Value lhs, Value rhs) {
    switch (predicate) {
    case arith::CmpFPredicate::AlwaysFalse:
      return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "false");
    case arith::CmpFPredicate::AlwaysTrue:
      return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "true");
    case arith::CmpFPredicate::ORD:
      return rewriter.create<emitc::LogicalAndOp>(
                 loc, i1Ty, isNotNaN(rewriter, loc, lhs),
                 isNotNaN(rewriter, loc, rhs))
          .getResult();
    case arith::CmpFPredicate::UNO:
      return rewriter.create<emitc::LogicalOrOp>(
                 loc, i1Ty, isNaN(rewriter, loc, lhs),
                 isNaN(rewriter, loc, rhs))
          .getResult();
    default:
      return std::nullopt;
    }
  }

  static std::optional<CmpFConfig>
  getCmpFConfig(arith::CmpFPredicate predicate) {
    switch (predicate) {
    case arith::CmpFPredicate::OEQ:
      return CmpFConfig{false, emitc::CmpPredicate::eq};
    case arith::CmpFPredicate::OGT:
      return CmpFConfig{false, emitc::CmpPredicate::gt};
    case arith::CmpFPredicate::OGE:
      return CmpFConfig{false, emitc::CmpPredicate::ge};
    case arith::CmpFPredicate::OLT:
      return CmpFConfig{false, emitc::CmpPredicate::lt};
    case arith::CmpFPredicate::OLE:
      return CmpFConfig{false, emitc::CmpPredicate::le};
    case arith::CmpFPredicate::ONE:
      return CmpFConfig{false, emitc::CmpPredicate::ne};
    case arith::CmpFPredicate::UEQ:
      return CmpFConfig{true, emitc::CmpPredicate::eq};
    case arith::CmpFPredicate::UGT:
      return CmpFConfig{true, emitc::CmpPredicate::gt};
    case arith::CmpFPredicate::UGE:
      return CmpFConfig{true, emitc::CmpPredicate::ge};
    case arith::CmpFPredicate::ULT:
      return CmpFConfig{true, emitc::CmpPredicate::lt};
    case arith::CmpFPredicate::ULE:
      return CmpFConfig{true, emitc::CmpPredicate::le};
    case arith::CmpFPredicate::UNE:
      return CmpFConfig{true, emitc::CmpPredicate::ne};
    default:
      return std::nullopt;
    }
  }

  static Value buildCmpFResult(const CmpFConfig &config,
                               ConversionPatternRewriter &rewriter,
                               Location loc, Type i1Ty, Value lhs, Value rhs) {
    Value cmp = rewriter
                    .create<emitc::CmpOp>(loc, i1Ty, config.predicate, lhs, rhs)
                    .getResult();
    Value unord = rewriter.create<emitc::LogicalOrOp>(
        loc, i1Ty, isNaN(rewriter, loc, lhs), isNaN(rewriter, loc, rhs));
    if (config.unordered)
      return rewriter
          .create<emitc::LogicalOrOp>(loc, i1Ty, unord, cmp)
          .getResult();
    Value ord = rewriter.create<emitc::LogicalAndOp>(
        loc, i1Ty, isNotNaN(rewriter, loc, lhs), isNotNaN(rewriter, loc, rhs));
    return rewriter
        .create<emitc::LogicalAndOp>(loc, i1Ty, ord, cmp)
        .getResult();
  }

  LogicalResult matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getLhs().getType()))
      return rewriter.notifyMatchFailure(op, "cmpf only supported on scalar floats");

    auto loc = op.getLoc();
    auto i1Ty = rewriter.getI1Type();
    if (auto special = buildSpecialCmpFResult(op.getPredicate(), rewriter, loc,
                                              i1Ty, adaptor.getLhs(),
                                              adaptor.getRhs())) {
      rewriter.replaceOp(op, *special);
      return success();
    }

    auto config = getCmpFConfig(op.getPredicate());
    if (!config)
      return rewriter.notifyMatchFailure(op, "unsupported cmpf predicate");
    rewriter.replaceOp(op, buildCmpFResult(*config, rewriter, loc, i1Ty,
                                           adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct ArithAddUIExtendedToEmitC
    : public OpConversionPattern<arith::AddUIExtendedOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddUIExtendedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getSum().getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();
    if (newResultTypes.size() != 2)
      return failure();

    Type sumDstTy = newResultTypes[0];
    Type overflowDstTy = newResultTypes[1];

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    auto wideTy = getWiderUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);

    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
    Value rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    Value sumWide =
        rewriter.create<emitc::AddOp>(loc, wideTy, lhsWide, rhsWide).getResult();

    Value sumN = emitCCast(rewriter, loc, uTy, sumWide);
    Value sum = emitCCast(rewriter, loc, sumDstTy, sumN);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value high = rewriter
                     .create<emitc::BitwiseRightShiftOp>(loc, wideTy, sumWide,
                                                         shiftAmt)
                     .getResult();
    Value zeroWide = makeEmitCIntConstant(rewriter, loc, wideTy, 0);
    Value overflow =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne, high, zeroWide)
            .getResult();
    overflow = emitCCast(rewriter, loc, overflowDstTy, overflow);

    rewriter.replaceOp(op, {sum, overflow});
    return success();
  }
};

template <typename ArithOp, bool isUnsigned>
struct ArithMulExtendedToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getResult(0).getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      newResultTypes)))
      return failure();
    if (newResultTypes.size() != 2)
      return failure();

    Type lowDstTy = newResultTypes[0];
    Type highDstTy = newResultTypes[1];

    Type wideTy = isUnsigned ? (Type)getWiderUnsignedIntOpaqueType(rewriter.getContext(),
                                                                   bitWidth)
                             : (Type)getWiderSignedIntOpaqueType(rewriter.getContext(),
                                                                 bitWidth);

    Value lhsWide;
    Value rhsWide;
    if constexpr (isUnsigned) {
      Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                      bitWidth);
      Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                      bitWidth);
      lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
      rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    } else {
      lhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getLhs());
      rhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getRhs());
    }

    Value prodWide =
        rewriter.create<emitc::MulOp>(loc, wideTy, lhsWide, rhsWide).getResult();
    Value low = emitCCast(rewriter, loc, lowDstTy, prodWide);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value highWide = rewriter
                         .create<emitc::BitwiseRightShiftOp>(loc, wideTy, prodWide,
                                                             shiftAmt)
                         .getResult();
    Value high = emitCCast(rewriter, loc, highDstTy, highWide);

    rewriter.replaceOp(op, {low, high});
    return success();
  }
};

using ArithMulSIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulSIExtendedOp, /*isUnsigned=*/false>;
using ArithMulUIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulUIExtendedOp, /*isUnsigned=*/true>;

struct ArithMinMaxIToEmitCBase {
  static Value makeSelect(ConversionPatternRewriter &rewriter, Location loc,
                          Type dstTy, Value cond, Value trueV, Value falseV) {
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, cond, trueV, falseV)
        .getResult();
  }
};

struct ArithMaxSIToEmitC : public OpConversionPattern<arith::MaxSIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt,
                                           adaptor.getLhs(), adaptor.getRhs())
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getRhs(),
                           adaptor.getLhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMinSIToEmitC : public OpConversionPattern<arith::MinSIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt,
                                           adaptor.getLhs(), adaptor.getRhs())
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getLhs(),
                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMaxUIToEmitC : public OpConversionPattern<arith::MaxUIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value lhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                           bitWidth);
    Value rhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                           bitWidth);
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt, lhsU, rhsU)
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getRhs(),
                           adaptor.getLhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMinUIToEmitC : public OpConversionPattern<arith::MinUIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value lhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                           bitWidth);
    Value rhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                           bitWidth);
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt, lhsU, rhsU)
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getLhs(),
                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

// Floating-point max/min variants.
struct ArithFloatMinMaxToEmitCBase {
  static Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
                     Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::ne,
                              v, v)
        .getResult();
  }

  static Value makeFZero(ConversionPatternRewriter &rewriter, Location loc,
                         Type ty) {
    return makeEmitCOpaqueConstant(rewriter, loc, ty, "0.0f");
  }
};

struct ArithMaxNumFToEmitC : public OpConversionPattern<arith::MaxNumFOp>,
                             ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxNumFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    Value lhsNaN = isNaN(rewriter, loc, adaptor.getLhs());
    Value rhsNaN = isNaN(rewriter, loc, adaptor.getRhs());

    Value cmpLt = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::lt,
                                            adaptor.getLhs(), adaptor.getRhs())
                      .getResult();
    Value maxNoNaN =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, cmpLt, adaptor.getRhs(),
                                          adaptor.getLhs())
            .getResult();

    Value rhsOrMax =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, adaptor.getLhs(),
                                          maxNoNaN)
            .getResult();
    Value res =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, adaptor.getRhs(),
                                          rhsOrMax)
            .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMinNumFToEmitC : public OpConversionPattern<arith::MinNumFOp>,
                             ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinNumFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    Value lhsNaN = isNaN(rewriter, loc, adaptor.getLhs());
    Value rhsNaN = isNaN(rewriter, loc, adaptor.getRhs());

    Value cmpLt = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::lt,
                                            adaptor.getLhs(), adaptor.getRhs())
                      .getResult();
    Value minNoNaN =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, cmpLt, adaptor.getLhs(),
                                          adaptor.getRhs())
            .getResult();

    Value rhsOrMin =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, adaptor.getLhs(),
                                          minNoNaN)
            .getResult();
    Value res =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, adaptor.getRhs(),
                                          rhsOrMin)
            .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

template <typename ArithOp, bool isMaximum>
struct ArithMinMaxFPropagateNaNToEmitC : public OpConversionPattern<ArithOp>,
                                        ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  static Value buildPrimaryCandidate(ConversionPatternRewriter &rewriter,
                                     Location loc, Type dstTy, Value lhs,
                                     Value rhs) {
    Value cmpLt =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::lt, lhs, rhs)
            .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(
            loc, dstTy, cmpLt, isMaximum ? rhs : lhs, isMaximum ? lhs : rhs)
        .getResult();
  }

  static Value buildSignBitValue(ConversionPatternRewriter &rewriter,
                                 Location loc, Value lhs, FloatType floatTy) {
    auto bitsTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), floatTy.getWidth());
    auto templateArgs = rewriter.getArrayAttr({emitc::OpaqueAttr::get(
        rewriter.getContext(), cast<emitc::OpaqueType>(bitsTy).getValue())});
    Value lhsBits =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, TypeRange{bitsTy}, "ptoas_bitcast",
                                         ValueRange{lhs}, ArrayAttr{},
                                         templateArgs)
            .getResult(0);
    Value oneBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 1);
    Value shiftAmount =
        makeEmitCIntConstant(rewriter, loc, bitsTy, floatTy.getWidth() - 1);
    Value signMask = rewriter
                         .create<emitc::BitwiseLeftShiftOp>(loc, bitsTy, oneBits,
                                                            shiftAmount)
                         .getResult();
    return rewriter
        .create<emitc::BitwiseAndOp>(loc, bitsTy, lhsBits, signMask)
        .getResult();
  }

  static Value buildSignedZeroCandidate(ConversionPatternRewriter &rewriter,
                                        Location loc, Type dstTy, Value lhs,
                                        Value rhs, FloatType floatTy) {
    Value zero = makeFZero(rewriter, loc, dstTy);
    Value equal = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::eq, lhs, rhs)
                      .getResult();
    Value lhsZero = rewriter
                        .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                              emitc::CmpPredicate::eq, lhs,
                                              zero)
                        .getResult();
    Value bothZero = rewriter
                         .create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                                      equal, lhsZero)
                         .getResult();
    auto bitsTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), floatTy.getWidth());
    Value zeroBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 0);
    Value lhsIsNegZero =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne,
                                  buildSignBitValue(rewriter, loc, lhs, floatTy),
                                  zeroBits)
            .getResult();
    Value tie = rewriter
                    .create<emitc::ConditionalOp>(
                        loc, dstTy, lhsIsNegZero, isMaximum ? rhs : lhs,
                        isMaximum ? lhs : rhs)
                    .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, bothZero, tie,
                                      buildPrimaryCandidate(rewriter, loc, dstTy,
                                                            lhs, rhs))
        .getResult();
  }

  static Value buildNaNPropagatingResult(ConversionPatternRewriter &rewriter,
                                         Location loc, Type dstTy, Value lhs,
                                         Value rhs, FloatType floatTy) {
    Value lhsNaN = isNaN(rewriter, loc, lhs);
    Value rhsNaN = isNaN(rewriter, loc, rhs);
    Value noNaN =
        buildSignedZeroCandidate(rewriter, loc, dstTy, lhs, rhs, floatTy);
    Value rhsOrNoNaN = rewriter
                           .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, rhs,
                                                         noNaN)
                           .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, lhs, rhsOrNoNaN)
        .getResult();
  }

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getType()))
      return rewriter.notifyMatchFailure(op, "expected scalar float type");

    auto loc = op.getLoc();
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto floatTy = cast<FloatType>(op.getType());
    rewriter.replaceOp(op, buildNaNPropagatingResult(
                               rewriter, loc, dstTy, adaptor.getLhs(),
                               adaptor.getRhs(), floatTy));
    return success();
  }
};

using ArithMaximumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MaximumFOp, /*isMaximum=*/true>;
using ArithMinimumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MinimumFOp, /*isMaximum=*/false>;

//===----------------------------------------------------------------------===//
// Arith -> EmitC helpers
//===----------------------------------------------------------------------===//

static emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "int16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "int32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "int64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "__int128");
  default:
    llvm::errs() << "[Debug] Unsupported signed integer bitwidth: " << bitWidth
                 << "\n";
    return emitc::OpaqueType::get(ctx, "int64_t");
  }
}

static emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "uint16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "uint32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "uint64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "unsigned __int128");
  default:
    llvm::errs() << "[Debug] Unsupported unsigned integer bitwidth: "
                 << bitWidth << "\n";
    return emitc::OpaqueType::get(ctx, "uint64_t");
  }
}

static emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getSignedIntOpaqueType(ctx, 16);
  case 16:
    return getSignedIntOpaqueType(ctx, 32);
  case 32:
    return getSignedIntOpaqueType(ctx, 64);
  case 64:
    return getSignedIntOpaqueType(ctx, 128);
  default:
    return getSignedIntOpaqueType(ctx, 128);
  }
}

static emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getUnsignedIntOpaqueType(ctx, 16);
  case 16:
    return getUnsignedIntOpaqueType(ctx, 32);
  case 32:
    return getUnsignedIntOpaqueType(ctx, 64);
  case 64:
    return getUnsignedIntOpaqueType(ctx, 128);
  default:
    return getUnsignedIntOpaqueType(ctx, 128);
  }
}

static Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal) {
  auto attr = emitc::OpaqueAttr::get(rewriter.getContext(), literal);
  return rewriter.create<emitc::ConstantOp>(loc, type, attr);
}

static Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value) {
  return makeEmitCOpaqueConstant(rewriter, loc, type, std::to_string(value));
}

static FailureOr<std::string> buildEmitCOpaqueConstantLiteral(Type targetType,
                                                              Attribute valueAttr) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(targetType);
  if (!opaqueTy)
    return failure();

  if (opaqueTy.getValue() == "pto::MrgSortExecutedNumList") {
    auto dense = dyn_cast_or_null<DenseIntElementsAttr>(valueAttr);
    if (!dense)
      return failure();

    auto vecTy = dyn_cast<VectorType>(dense.getType());
    if (!vecTy || vecTy.getRank() != 1 || vecTy.getNumElements() != 4 ||
        !vecTy.getElementType().isInteger(16))
      return failure();

    std::string literal;
    llvm::raw_string_ostream os(literal);
    os << "pto::MrgSortExecutedNumList{";
    bool first = true;
    for (APInt elem : dense.getValues<APInt>()) {
      if (!first)
        os << ", ";
      first = false;
      os << elem.getZExtValue();
    }
    os << "}";
    os.flush();
    return literal;
  }

  return failure();
}

static Value emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src) {
  if (src.getType() == dstType)
    return src;
  return rewriter.createOrFold<emitc::CastOp>(loc, dstType, src);
}

// For signless iN integers lowered to signed C++ types, this creates a value
// representing the same N-bit pattern in an unsigned C++ type of the same
// width. This avoids incorrect sign-extension when later widening to a larger
// unsigned type.
static Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth) {
  auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
  return emitCCast(rewriter, loc, uTy, v);
}

struct ArithMulIToEmitC : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern<arith::MulIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    // i1 mul is equivalent to bitwise AND (mod 2 arithmetic).
    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<emitc::BitwiseAndOp>(op, opTy, adaptor.getLhs(),
                                                      adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value mulU = rewriter.create<emitc::MulOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, mulU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithAddIToEmitC : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    // i1 add is equivalent to XOR (mod 2 arithmetic).
    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<emitc::BitwiseXorOp>(op, opTy, adaptor.getLhs(),
                                                      adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value addU = rewriter.create<emitc::AddOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, addU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCastOPToEmitC : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    if (adaptor.getIn().getType() == newTy) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, newTy, adaptor.getIn());
    return success();
  }
};

struct ArithSubIToEmitC : public OpConversionPattern<arith::SubIOp> {
  using OpConversionPattern<arith::SubIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::SubIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    // i1 sub is equivalent to XOR (mod 2 arithmetic).
    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<emitc::BitwiseXorOp>(op, opTy, adaptor.getLhs(),
                                                      adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value subU = rewriter.create<emitc::SubOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, subU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithDivSIToEmitC : public OpConversionPattern<arith::DivSIOp> {
  using OpConversionPattern<arith::DivSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::DivOp>(op, newTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithRemSIToEmitC : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern<arith::RemSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::RemOp>(op, newTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithTruncIToEmitC : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // to-i1 conversions: Arith wants truncation to the low bit, while C/C++
    // casts to bool are equivalent to `v != 0`. Implement as `(bool)(v & 1)`.
    if (dstIntTy.getWidth() == 1) {
      if (srcIntTy.getWidth() == 1) {
        rewriter.replaceOp(op, adaptor.getIn());
        return success();
      }

      auto uSrcTy =
          getUnsignedIntOpaqueType(rewriter.getContext(), srcIntTy.getWidth());
      Value inU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                                     srcIntTy.getWidth());
      Value one = makeEmitCIntConstant(rewriter, loc, uSrcTy, 1);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, uSrcTy, inU, one);
      Value asBool = emitCCast(rewriter, loc, dstTy, masked);
      rewriter.replaceOp(op, asBool);
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

struct ArithConstantToEmitC : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    if (!newType)
      return failure();

    // `adaptor.getValue()` may be null if attribute conversion isn't defined.
    // Use the original attribute as fallback and always cast null-safely.
    Attribute valueAttr = adaptor.getValue();
    if (!valueAttr)
      valueAttr = op.getValue();

    if (auto opaqueLiteral = buildEmitCOpaqueConstantLiteral(newType, valueAttr);
        succeeded(opaqueLiteral)) {
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), *opaqueLiteral);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    if (auto floatAttr = dyn_cast_or_null<FloatAttr>(valueAttr)) {
      SmallString<32> valStr;
      floatAttr.getValue().toString(valStr);
      llvm::StringRef s(valStr);
      // Ensure the literal parses as a floating-point constant in C/C++.
      // `APFloat::toString` may emit "1" for integral values; make it "1.0".
      const bool hasFloatMarker =
          s.contains('.') || s.contains('e') || s.contains('E') ||
          s.contains('p') || s.contains('P') || s.starts_with("0x") ||
          s.starts_with("0X") || s.starts_with("nan") ||
          s.starts_with("-nan") || s.starts_with("inf") ||
          s.starts_with("-inf");
      if (!hasFloatMarker)
        valStr.append(".0");
      // Suffix: keep `f` for f16/f32; omit for f64.
      if (!floatAttr.getType().isF64())
        valStr.append("f");
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(valueAttr)) {
      std::string valStr = std::to_string(intAttr.getValue().getSExtValue());
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    return failure();
  }
};
//===----------------------------------------------------------------------===//
// pto.mgather lowering -> MGATHER(dst, src, indexes)  (pto-isa)
//===----------------------------------------------------------------------===//

