// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/InsertSyncDebug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
// [P0 新增] 引入副作用接口和 PTO 接口
#include "mlir/Interfaces/SideEffectInterfaces.h"
 
#define DEBUG_TYPE "pto-ir-translator"
 
using namespace mlir;
using namespace mlir::pto;

static bool isInsertSyncTraceEnabled() {
  return isInsertSyncDebugEnabled(InsertSyncDebugLevel::Trace);
}

static void dumpInt64List(llvm::raw_ostream &os, ArrayRef<int64_t> values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    os << values[i];
    if (i + 1 != values.size())
      os << ", ";
  }
  os << "]";
}

static void dumpValueSummary(llvm::raw_ostream &os, Value value) {
  if (!value) {
    os << "<null>";
    return;
  }
  if (Operation *op = value.getDefiningOp())
    os << op->getName();
  else
    os << "block_arg";
  os << ":" << value.getType();
}

struct StaticAliasInfo {
  bool hasStaticOffset{true};
  bool preserveParentRegion{true};
  int64_t offsetBytes{0};
  int64_t sizeBytes{-1};
  std::optional<StaticMemRegion> relativeRegion;
};

static int64_t getElementSizeBytes(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    type = shapedType.getElementType();
  if (auto intType = dyn_cast<IntegerType>(type)) {
    int64_t bitWidth = intType.getWidth();
    return bitWidth > 0 ? std::max<int64_t>(1, bitWidth / 8) : 1;
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    int64_t bitWidth = floatType.getWidth();
    return bitWidth > 0 ? std::max<int64_t>(1, bitWidth / 8) : 1;
  }
  return 1;
}

static bool hasOnlyStaticValues(ArrayRef<int64_t> values) {
  for (int64_t value : values)
    if (value == ShapedType::kDynamic)
      return false;
  return true;
}

static std::optional<int64_t> getStaticBoundingSpan(int64_t size,
                                                    int64_t stride) {
  if (size == ShapedType::kDynamic || stride == ShapedType::kDynamic ||
      size < 0 || stride < 0)
    return std::nullopt;
  if (size == 0)
    return 0;
  return (size - 1) * stride + 1;
}

static bool shouldPreserveParentRegion(Operation *op) {
  if (!op)
    return true;
  return isa<pto::BindTileOp, memref::CastOp>(op);
}

static std::optional<StaticMemRegion>
makeRootRegionFromMemRefType(MemRefType type) {
  if (!type || !type.hasStaticShape())
    return std::nullopt;

  int64_t baseOffset;
  SmallVector<int64_t, 4> strides;
  if (failed(mlir::getStridesAndOffset(type, strides, baseOffset)))
    return std::nullopt;
  if (strides.size() != static_cast<size_t>(type.getRank()) ||
      !hasOnlyStaticValues(strides))
    return std::nullopt;

  StaticMemRegion region;
  region.elemSizeBytes = getElementSizeBytes(type);
  region.baseOffsetBytes = 0;
  region.offsets.assign(type.getRank(), 0);
  region.strides.assign(strides.begin(), strides.end());
  region.sizes.reserve(type.getRank());
  ArrayRef<int64_t> shape = type.getShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    std::optional<int64_t> span = getStaticBoundingSpan(shape[i], strides[i]);
    if (!span)
      return std::nullopt;
    region.sizes.push_back(*span);
  }
  return region;
}

static std::optional<StaticMemRegion>
composeSubviewRegion(const StaticMemRegion &relativeRegion,
                     const std::optional<StaticMemRegion> &parentRegion) {
  if (!relativeRegion.isPrecise() || !parentRegion ||
      !parentRegion->isPrecise())
    return std::nullopt;
  if (parentRegion->offsets.size() != relativeRegion.offsets.size())
    return std::nullopt;
  if (parentRegion->elemSizeBytes != relativeRegion.elemSizeBytes)
    return std::nullopt;

  StaticMemRegion region;
  region.elemSizeBytes = relativeRegion.elemSizeBytes;
  region.baseOffsetBytes =
      parentRegion->baseOffsetBytes + relativeRegion.baseOffsetBytes;
  region.offsets.reserve(relativeRegion.offsets.size());
  region.sizes.reserve(relativeRegion.sizes.size());
  region.strides.reserve(relativeRegion.strides.size());

  for (size_t i = 0; i < relativeRegion.offsets.size(); ++i) {
    int64_t parentStride = parentRegion->strides[i];
    int64_t childStride = relativeRegion.strides[i];
    if (parentStride == ShapedType::kDynamic ||
        childStride == ShapedType::kDynamic || parentStride < 0 ||
        childStride < 0)
      return std::nullopt;

    int64_t rootStride = parentStride * childStride;
    std::optional<int64_t> span =
        getStaticBoundingSpan(relativeRegion.sizes[i], rootStride);
    if (!span)
      return std::nullopt;

    region.offsets.push_back(parentRegion->offsets[i] +
                             relativeRegion.offsets[i] * parentStride);
    region.sizes.push_back(*span);
    region.strides.push_back(rootStride);
  }
  return region;
}
 
// [辅助函数] 尝试从 Operation 中计算相对于 Source 的字节偏移量和新大小
// 返回值: pair<offsetInBytes, sizeInBytes>
// 如果无法计算静态值，返回 {-1, -1} 表示这是动态的
static StaticAliasInfo getStaticAliasInfo(Operation *op, Value src) {
  StaticAliasInfo aliasInfo;
  aliasInfo.preserveParentRegion = shouldPreserveParentRegion(op);
  auto srcType = dyn_cast<MemRefType>(src.getType());
  if (!srcType)
    return aliasInfo;
  
  int64_t elemSize = getElementSizeBytes(srcType);
 
  // === Case 1: memref.subview ===
  if (auto subView = dyn_cast<memref::SubViewOp>(op)) {
    aliasInfo.preserveParentRegion = false;
    int64_t baseOffset;
    SmallVector<int64_t, 4> strides;
    if (failed(mlir::getStridesAndOffset(srcType, strides, baseOffset))) {
        aliasInfo.hasStaticOffset = false;
        return aliasInfo;
    }
 
    auto staticSizes = subView.getStaticSizes();
    auto staticOffsets = subView.getStaticOffsets();
    auto staticSubStrides = subView.getStaticStrides();

    if (staticOffsets.empty() || staticOffsets.size() > strides.size() ||
        staticSizes.size() != staticOffsets.size() ||
        staticSubStrides.size() != staticOffsets.size()) {
      aliasInfo.hasStaticOffset = false;
      return aliasInfo;
    }

    int64_t flatSpan = 0;
    bool hasZeroSize = false;
    for (size_t i = 0; i < staticSizes.size(); ++i) {
      int64_t s = staticSizes[i];
      if (s == ShapedType::kDynamic) {
        aliasInfo.hasStaticOffset = false;
        return aliasInfo;
      }
      if (s == 0)
        hasZeroSize = true;
    }
 
    int64_t totalOffset = 0;
    for (size_t i = 0; i < staticOffsets.size(); ++i) {
      int64_t off = staticOffsets[i];
      int64_t subStride = staticSubStrides[i];
      if (off == ShapedType::kDynamic ||
          subStride == ShapedType::kDynamic || subStride < 0) {
        aliasInfo.hasStaticOffset = false;
        return aliasInfo;
      }
      
      int64_t stride = 1; 
      if (i < strides.size() && strides[i] != ShapedType::kDynamic) {
          stride = strides[i];
      } else {
          aliasInfo.hasStaticOffset = false;
          return aliasInfo;
      }
      
      totalOffset += off * stride;
      if (!hasZeroSize)
        flatSpan += (staticSizes[i] - 1) * stride * subStride;
    }
 
    int64_t byteOffset = totalOffset * elemSize;
    aliasInfo.offsetBytes = byteOffset;
    aliasInfo.sizeBytes = hasZeroSize ? 0 : (flatSpan + 1) * elemSize;
    StaticMemRegion region;
    region.elemSizeBytes = elemSize;
    region.baseOffsetBytes = byteOffset;
    region.offsets.assign(staticOffsets.begin(), staticOffsets.end());
    region.sizes.assign(staticSizes.begin(), staticSizes.end());
    region.strides.assign(staticSubStrides.begin(), staticSubStrides.end());
    aliasInfo.relativeRegion = region;
    if (isInsertSyncTraceEnabled()) {
      llvm::errs() << "  [AliasRange] memref.subview flat range\n";
      llvm::errs() << "    src=";
      dumpValueSummary(llvm::errs(), src);
      llvm::errs() << "\n";
      llvm::errs() << "    srcType=" << srcType << " elemBytes=" << elemSize
                   << " baseOffset=" << baseOffset << "\n";
      llvm::errs() << "    staticOffsets=";
      dumpInt64List(llvm::errs(), staticOffsets);
      llvm::errs() << " staticSizes=";
      dumpInt64List(llvm::errs(), staticSizes);
      llvm::errs() << " staticSubStrides=";
      dumpInt64List(llvm::errs(), staticSubStrides);
      llvm::errs() << " sourceStrides=";
      dumpInt64List(llvm::errs(), strides);
      llvm::errs() << "\n";
      llvm::errs() << "    computedFlatOffsetBytes=" << byteOffset
                   << " computedFlatSizeBytes=" << aliasInfo.sizeBytes
                   << " (size = strided bounding span * elemBytes)\n";
    }
    return aliasInfo;
  }
 
  // === Case 2: memref.reinterpret_cast ===
  if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    aliasInfo.preserveParentRegion = false;
    auto staticOffsets = castOp.getStaticOffsets();
    if (staticOffsets.empty() || staticOffsets[0] == ShapedType::kDynamic) {
        return aliasInfo;
    }
    int64_t byteOffset = staticOffsets[0] * elemSize;
    aliasInfo.offsetBytes = byteOffset;
    if (isInsertSyncTraceEnabled()) {
      llvm::errs() << "  [AliasRange] memref.reinterpret_cast offset\n";
      llvm::errs() << "    src=";
      dumpValueSummary(llvm::errs(), src);
      llvm::errs() << " staticOffsets=";
      dumpInt64List(llvm::errs(), staticOffsets);
      llvm::errs() << " computedOffsetBytes=" << byteOffset << "\n";
    }
    return aliasInfo;
  }
 
  return aliasInfo;
}
 
// ============================================================================
// 1. 构建入口
// ============================================================================
void PTOIRTranslator::Build() {
  Region &funcRegion = func_.getBody();
  UpdateKernelArgMemInfo();
  RecursionIR(&funcRegion);
}
 
// ============================================================================
// 2. 更新 Kernel 参数内存信息 (GM Global Memory)
// ============================================================================
void PTOIRTranslator::UpdateKernelArgMemInfo() {
  auto funcParamSize = func_.getNumArguments();
  for (size_t i = 0; i < funcParamSize; i++) {
    Value funcArg = func_.getArgument(i);
    Type argType = funcArg.getType();
 
    if (!isa<pto::PtrType>(argType) && !isa<MemRefType>(argType)) {
      continue;
    }
 
    std::unique_ptr<BaseMemInfo> newMemInfo = std::make_unique<BaseMemInfo>(
        funcArg,                  // baseBuffer
        funcArg,                  // rootBuffer
        pto::AddressSpace::GM,    // Scope
        SmallVector<uint64_t>{0}, // Base Addresses
        0                         // Allocate Size
    );
 
    buffer2MemInfoMap_[funcArg].emplace_back(newMemInfo->clone());
  }
}
 
// ============================================================================
// 3. 递归遍历 IR (核心分发逻辑)
// ============================================================================
void PTOIRTranslator::RecursionIR(Region *region) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {

    // --- Case A: 内存分配 (AllocTile) ---
    if (auto allocOp = dyn_cast<pto::AllocTileOp>(op)) {
      if (failed(UpdateAllocTileOpMemInfo(allocOp))) {
        return WalkResult::interrupt();
      }
    }
    // 支持标准 memref.alloc
    else if (auto memAllocOp = dyn_cast<memref::AllocOp>(op)) {
       if (failed(UpdateMemrefAllocOpMemInfo(memAllocOp))) {
          return WalkResult::interrupt();
       }
    }
    else if (auto declareOp = dyn_cast<pto::DeclareTileMemRefOp>(op)) {
      if (failed(UpdateDeclareTileMemRefOpMemInfo(declareOp))) {
        return WalkResult::interrupt();
      }
    }
    else if (auto castOp = dyn_cast<pto::PointerCastOp>(op)) {
      if (failed(UpdatePointerCastOpMemInfo(castOp))) return WalkResult::interrupt();
    }
    
    // --- Case B: 别名/视图操作 ---
    else if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
      UpdateAliasBufferInfo(makeViewOp.getResult(), makeViewOp.getPtr());
    } 
    else if (auto bindTileOp = dyn_cast<pto::BindTileOp>(op)) {
      UpdateAliasBufferInfo(bindTileOp.getResult(), bindTileOp.getSource());
    }
    else if (auto subViewOp = dyn_cast<pto::PartitionViewOp>(op)) {
      UpdateAliasBufferInfo(subViewOp.getResult(), subViewOp.getSource());
    } 
    else if (auto memrefSubView = dyn_cast<memref::SubViewOp>(op)) {
      UpdateAliasBufferInfo(memrefSubView.getResult(), memrefSubView.getSource());
    }
    else if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
      UpdateAliasBufferInfo(castOp.getResult(), castOp.getSource());
    }
    // [Fix] 添加 CollapseShape 和 ExpandShape 的支持
    else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(op)) {
      UpdateAliasBufferInfo(collapseOp.getResult(), collapseOp.getSrc());
    }
    else if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(op)) {
      UpdateAliasBufferInfo(expandOp.getResult(), expandOp.getSrc());
    }
 
    // --- Case C: 控制流 (SCF) ---
    else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      UpdateForOpInfo(forOp);
      return WalkResult::skip();
    } 
    else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      UpdateWhileOpInfo(whileOp);
      return WalkResult::skip();
    } 
    else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      UpdateIfOpInfo(ifOp);
      return WalkResult::skip();
    } 
    else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      UpdateYieldOpInfo(yieldOp);
    }
    // --- Case D: 带有 OpPipeInterface 的计算/搬运指令 ---
    else if (isa<pto::OpPipeInterface>(op)) {
      UpdatePTOOpInfo(op);
    }
    
    return WalkResult::advance();
  });
 
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("PTO InjectSync Traverse IR Failed!");
  }
}
 
// ============================================================================
// 4. 处理 AllocTile / PointerCast
// ============================================================================
LogicalResult PTOIRTranslator::UpdateAllocTileOpMemInfo(pto::AllocTileOp op) {
  Value res = op.getResult();
  
  auto tileType = dyn_cast<pto::TileBufType>(res.getType());
  uint64_t sizeInBytes = 0;
  uint64_t baseAddr = 0;

  // If alloc_tile carries an explicit address, record it when it's a constant.
  if (Value addr = op.getAddr()) {
    llvm::APInt apIntValue;
    if (matchPattern(addr, m_ConstantInt(&apIntValue))) {
        // 将 APInt 转换为 int64_t，再转为 uint64_t
        int64_t c = apIntValue.getSExtValue();  // 有符号扩展转换
        // 如果确定是无符号值，也可以用：apIntValue.getZExtValue()
        baseAddr = static_cast<uint64_t>(c);
    }
  }

  // 1. 计算大小
  if (tileType) {
    ArrayRef<int64_t> shape = tileType.getShape();
    bool isStatic = true;
    for (int64_t dim : shape) {
      if (dim == ShapedType::kDynamic) {
        isStatic = false;
        break;
      }
    }

    if (isStatic) {
      int64_t elemSize = tileType.getElementType().getIntOrFloatBitWidth() / 8;
      int64_t numElements = 1;
      for (auto dim : shape) numElements *= dim;
      sizeInBytes = numElements * elemSize;
    }
  }

  // 2. 解析地址空间
  // 默认设为 MAT (Matrix Buffer)，但优先读取 Type 中的属性
  pto::AddressSpace space = pto::AddressSpace::MAT; 
  
  if (tileType) {
      if (auto attr = tileType.getMemorySpace()) {
          // 尝试转换为 PTO 的 AddressSpaceAttr
          if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
              space = ptoAttr.getAddressSpace();
          }
      }
  }

  // 3. 注册 Buffer 信息
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,                  
      res,                  
      space, // 使用解析出的 space                 
      SmallVector<uint64_t>{baseAddr},
      sizeInBytes             
  );

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}
 
LogicalResult PTOIRTranslator::UpdatePointerCastOpMemInfo(pto::PointerCastOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType) return failure();
 
  if (op.getAddrs().empty()) {
    return op.emitError("PointerCast must have at least one address operand");
  }
  Value rootSrc = op.getAddrs().front(); 
 
  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t numElements = 1;
    for (auto dim : memRefType.getShape()) numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }
 
  pto::AddressSpace space = pto::AddressSpace::GM; 
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
      space = ptoAttr.getAddressSpace();
    }
  }
 
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,          
      rootSrc,      
      space,
      SmallVector<uint64_t>{0}, 
      sizeInBytes
  );
  newMemInfo->preciseRegion = makeRootRegionFromMemRefType(memRefType);
 
  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateDeclareTileMemRefOpMemInfo(pto::DeclareTileMemRefOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType)
    return failure();

  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    if (elemSize == 0)
      elemSize = 1;

    int64_t numElements = 1;
    for (auto dim : memRefType.getShape())
      numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }

  pto::AddressSpace space = pto::AddressSpace::MAT;
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr))
      space = ptoAttr.getAddressSpace();
  }

  // declare_tile_memref is only a symbolic placeholder. Use its SSA result as
  // both base/root so later bind_tile aliases and tpop consumers can be
  // connected by InsertSync without inventing a fake allocation.
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,
      res,
      space,
      SmallVector<uint64_t>{0},
      sizeInBytes);
  newMemInfo->preciseRegion = makeRootRegionFromMemRefType(memRefType);

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}
 
// ============================================================================
// 5. [P0 修改] 更新 PTO Op 信息 (通用接口版)
// ============================================================================
void PTOIRTranslator::UpdatePTOOpInfo(Operation *op) {
  // 1. 获取流水线类型 (现在通过 Interface)
  pto::PipelineType pipe = getOpPipeline(op);
  
  // 如果 Op 不属于任何关心的流水线，直接跳过，不建立 Sync 节点
  if (pipe == pto::PipelineType::PIPE_UNASSIGNED) return;
 
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
 
  // 2. [关键] 使用 MemoryEffects 接口自动获取读写依赖
  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
     SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
     memEffect.getEffects(effects);
     
     for (auto &effect : effects) {
       Value val = effect.getValue();
       if (!val) continue;
 
       // 只有当 Value 在我们的 BufferMap 中有记录时，才视为有效依赖
       // (过滤掉比如 Loop Iterator 或其他标量)
       if (isa<MemoryEffects::Read>(effect.getEffect())) {
          UpdateDefUseVec({val}, useVec);
       } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
          UpdateDefUseVec({val}, defVec);
       }
     }
  } else {
    // 如果算子有 Pipe 属性但没实现 MemoryEffects，这是一个定义错误
    // 我们可以打印个 Warning 或者保持为空 (认为无副作用)
    LLVM_DEBUG(llvm::dbgs() << "Warning: Op " << op->getName() 
                            << " has Pipe but no MemoryEffects interface.\n");
  }
 
  // 3. 构建 Compound Node
  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, op->getName());
  compoundElement->elementOp = op;
 
  // 4. 设置 Core Type (用于区分 Cube/Vector 资源)
  // Matmul (M) 和 L1->L0 搬运 (MTE1) 通常涉及 Cube 资源
  if (pipe == pto::PipelineType::PIPE_M || pipe == pto::PipelineType::PIPE_MTE1) {
    compoundElement->compoundCoreType = pto::TCoreType::CUBE; 
  } else {
    // MTE2, MTE3, Vector 归类为 Vector Core (或者对应 MTE 资源)
    compoundElement->compoundCoreType = pto::TCoreType::VECTOR;
  }
 
  syncIR_.emplace_back(std::move(compoundElement));
  index++;
}
 
// ============================================================================
// 6. [P0 修改] 获取 Op 的 Pipeline 类型
// ============================================================================
pto::PipelineType PTOIRTranslator::getOpPipeline(Operation *op) {
  // 1. 优先尝试通过接口获取
  if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op)) {
    // 注意：假设 pto::Pipe (ODS Enum) 和 pto::PipelineType (C++ Enum) 的数值定义是一致的
    // 或者在这里做一个 switch-case 映射
    // 目前假设直接 cast 是安全的 (0=S, 1=V, 2=M ...)
    return static_cast<pto::PipelineType>(pipeOp.getPipe());
  }
 
  // 2. 如果没实现接口，返回 Unassigned
  return pto::PipelineType::PIPE_UNASSIGNED;
}
 
// ============================================================================
// 7. 控制流处理 (SCF Support)
// ============================================================================
 
void PTOIRTranslator::UpdateForOpInfo(scf::ForOp forOp) {
  auto forBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  forBeginElement->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forBeginElement));
  
  std::unique_ptr<InstanceElement> &forElement = syncIR_[index];
  index++;
  
  auto *forBeginPtr = dyn_cast<LoopInstanceElement>(forElement.get());
  assert(forBeginPtr != nullptr && "Sync IR Construction failed.");
  
  if (!forOp.getInitArgs().empty()) {
    assert(forOp.getInitArgs().size() == forOp.getRegionIterArgs().size());
    for (auto [i, arg] : llvm::enumerate(forOp.getInitArgs())) {
      UpdateAliasBufferInfo(forOp.getRegionIterArgs()[i], arg);
    }
  }
 
  RecursionIR(&forOp.getRegion());
 
  forBeginPtr->endId = index;
  auto forEnd = forBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}
 
void PTOIRTranslator::UpdateWhileOpInfo(scf::WhileOp whileOp) {
  auto loopBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  loopBeginElement->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(loopBeginElement));
  
  auto *loopBeginPtr = dyn_cast<LoopInstanceElement>(syncIR_.back().get());
  index++;
 
  if (!whileOp.getInits().empty()) {
    for (auto [initArg, blockArg] : llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments())) {
      UpdateAliasBufferInfo(blockArg, initArg);
    }
    auto conditionOp = whileOp.getConditionOp();
    for (auto [yieldedArg, blockArg] : llvm::zip(conditionOp.getArgs(), whileOp.getAfterArguments())) {
      UpdateAliasBufferInfo(blockArg, yieldedArg);
    }
  }
 
  RecursionIR(&whileOp.getBefore());
  RecursionIR(&whileOp.getAfter());
 
  loopBeginPtr->endId = index;
  auto forEnd = loopBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}
 
void PTOIRTranslator::UpdateIfOpInfo(scf::IfOp ifOp) {
  auto ifBeginElement = std::make_unique<BranchInstanceElement>(index, index, KindOfBranch::IF_BEGIN);
  ifBeginElement->elementOp = ifOp.getOperation();
  auto *ifPtr = ifBeginElement.get();
  
  syncIR_.emplace_back(std::move(ifBeginElement));
  index++;
 
  // 1. 处理 Then 区域
  RecursionIR(&ifOp.getThenRegion());
  
  // Then 的结束占位符
  auto placeHolder = std::make_unique<PlaceHolderInstanceElement>(index, ifPtr->GetIndex());

  // 直接指向Then Block的yieldop
  placeHolder->elementOp = ifOp.getThenRegion().front().getTerminator();

  syncIR_.emplace_back(std::move(placeHolder));
  index++;
  
  ifPtr->branchId = index;
 
  // 2. 处理 Else 区域 (总是创建 SyncIR 节点，即使 IR 中没有 Else)
  auto ifElseElement = ifPtr->CloneBranch(KindOfBranch::ELSE_BEGIN);
  ifElseElement->elementOp = ifOp.getOperation();
  auto *elsePtr = ifElseElement.get();
 
  syncIR_.emplace_back(std::move(ifElseElement));
  index++;
 
  if (ifOp.elseBlock()) {
    RecursionIR(&ifOp.getElseRegion());
  }
  
  // Else 的结束占位符
  auto elsePlaceHolder = std::make_unique<PlaceHolderInstanceElement>(index, elsePtr->GetIndex());
  
  if (ifOp.elseBlock()) {
      // 如果有真实的 Else Block，映射到 ifOp (CodeGen 需定位到 Else Yield 前)
      elsePlaceHolder->elementOp = ifOp.getElseRegion().front().getTerminator();
      elsePlaceHolder->isVirtualElse = false;
  } else {
      // 如果没有 Else Block，标记为虚拟，映射到 ifOp
      elsePlaceHolder->elementOp = ifOp.getOperation();
      elsePlaceHolder->isVirtualElse = true;
      elsePlaceHolder->parentIfOp = ifOp.getOperation();
  }
  
  syncIR_.emplace_back(std::move(elsePlaceHolder));
  index++;
  
  elsePtr->endId = index;
  ifPtr->endId = index;
 
  // 3. If End
  auto ifEndElement = ifPtr->CloneBranch(KindOfBranch::IF_END);
  ifEndElement->elementOp = ifOp.getOperation();
  syncIR_.emplace_back(std::move(ifEndElement));
  index++;
}
 
void PTOIRTranslator::UpdateYieldOpInfo(scf::YieldOp yieldOp) {
  auto *parentOp = yieldOp->getParentOp();
  if (!parentOp || isa<scf::WhileOp>(parentOp)) return;
 
  assert(parentOp->getResults().size() == yieldOp->getOpOperands().size());
  for (auto [yieldVal, resultVal] : llvm::zip(yieldOp->getOpOperands(), parentOp->getResults())) {
    UpdateAliasBufferInfo(resultVal, yieldVal.get());
  }
}
 
// ============================================================================
// 8. 辅助函数
// ============================================================================
void PTOIRTranslator::UpdateAliasBufferInfo(Value result, Value source) {
  if (!result || !source) return;
  if (!buffer2MemInfoMap_.contains(source)) return;
 
  StaticAliasInfo aliasInfo;
 
  if (auto op = result.getDefiningOp()) {
    aliasInfo = getStaticAliasInfo(op, source);
  }

  int64_t deltaOffset = aliasInfo.hasStaticOffset ? aliasInfo.offsetBytes : 0;
  int64_t newSize = aliasInfo.sizeBytes;
 
  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto newInfo = parentInfo->clone(result);
    SmallVector<uint64_t> parentBases = parentInfo->baseAddresses;
 
    if (!newInfo->baseAddresses.empty()) {
        newInfo->baseAddresses[0] += deltaOffset;
    } else {
        newInfo->baseAddresses.push_back(deltaOffset);
    }
 
    if (newSize > 0) {
        newInfo->allocateSize = newSize;
    }

    if (aliasInfo.relativeRegion) {
      newInfo->preciseRegion =
          composeSubviewRegion(*aliasInfo.relativeRegion,
                               parentInfo->preciseRegion);
    } else if (!aliasInfo.preserveParentRegion) {
      newInfo->preciseRegion.reset();
    }

    if (isInsertSyncTraceEnabled()) {
      llvm::errs() << "  [AliasInfo] ";
      if (Operation *op = result.getDefiningOp())
        llvm::errs() << "op=" << op->getName() << " ";
      llvm::errs() << "source=";
      dumpValueSummary(llvm::errs(), source);
      llvm::errs() << " result=";
      dumpValueSummary(llvm::errs(), result);
      llvm::errs() << "\n";
      llvm::errs() << "    root=";
      dumpValueSummary(llvm::errs(), newInfo->rootBuffer);
      llvm::errs() << " scope=" << static_cast<int>(newInfo->scope)
                   << " deltaOffsetBytes=" << deltaOffset
                   << " newSizeBytes=" << newSize << "\n";
      llvm::errs() << "    parentBases=[";
      for (size_t i = 0; i < parentBases.size(); ++i) {
        llvm::errs() << parentBases[i];
        if (i + 1 != parentBases.size())
          llvm::errs() << ", ";
      }
      llvm::errs() << "] parentSizeBytes=" << parentInfo->allocateSize
                   << " resultBases=[";
      for (size_t i = 0; i < newInfo->baseAddresses.size(); ++i) {
        llvm::errs() << newInfo->baseAddresses[i];
        if (i + 1 != newInfo->baseAddresses.size())
          llvm::errs() << ", ";
      }
      llvm::errs() << "] resultSizeBytes=" << newInfo->allocateSize << "\n";
      if (newInfo->preciseRegion && newInfo->preciseRegion->isPrecise()) {
        llvm::errs() << "    preciseRegion offsets=";
        dumpInt64List(llvm::errs(), newInfo->preciseRegion->offsets);
        llvm::errs() << " sizes=";
        dumpInt64List(llvm::errs(), newInfo->preciseRegion->sizes);
        llvm::errs() << " strides=";
        dumpInt64List(llvm::errs(), newInfo->preciseRegion->strides);
        llvm::errs() << " elemBytes="
                     << newInfo->preciseRegion->elemSizeBytes << "\n";
      } else {
        llvm::errs() << "    preciseRegion=<none>\n";
      }
    }
 
    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
}
 
// ============================================================================
// 实现 UpdateMemrefAllocOpMemInfo
// ============================================================================
LogicalResult PTOIRTranslator::UpdateMemrefAllocOpMemInfo(memref::AllocOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType) return failure();
 
  // 1. 计算大小 (Bytes)
  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    if (elemSize == 0) elemSize = 1; // bool case
    
    int64_t numElements = 1;
    for (auto dim : memRefType.getShape()) numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }
 
  // 2. 解析地址空间 (Scope)
  // 默认视为 MAT/UB (Local Memory)，这是 alloc 的常见用途
  // 如果有显式属性，则覆盖
  pto::AddressSpace space = pto::AddressSpace::MAT; 
  
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
      space = ptoAttr.getAddressSpace();
    }
  }
 
  // 3. 注册 Buffer 信息
  // 对于 alloc，它自己就是 Root
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,                      // baseBuffer
      res,                      // rootBuffer (Self is root)
      space,
      SmallVector<uint64_t>{0}, // Base Addresses (Offset 0)
      sizeInBytes
  );
  newMemInfo->preciseRegion = makeRootRegionFromMemRefType(memRefType);
 
  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}
 
void PTOIRTranslator::UpdateDefUseVec(ValueRange values, SmallVector<const BaseMemInfo *> &vec) {
  for (Value v : values) {
    if (buffer2MemInfoMap_.contains(v)) {
      for (auto &memInfo : buffer2MemInfoMap_[v]) {
        vec.push_back(memInfo.get());
      }
    }
  }
}
 
// ============================================================================
// 9. 调试与打印支持
// ============================================================================
 
std::string PTOIRTranslator::getPipelineName(pto::PipelineType pipe) {
  switch (pipe) {
  case pto::PipelineType::PIPE_MTE1: return "MTE1";
  case pto::PipelineType::PIPE_MTE2: return "MTE2";
  case pto::PipelineType::PIPE_MTE3: return "MTE3";
  case pto::PipelineType::PIPE_M:    return "CUBE";
  case pto::PipelineType::PIPE_V:    return "VECTOR";
  case pto::PipelineType::PIPE_S:    return "SCALAR";
  case pto::PipelineType::PIPE_ALL:  return "BARRIER";
  default: return "UNKNOWN";
  }
}
 
void PTOIRTranslator::printMemInfoList(llvm::raw_ostream &os, 
                                       const SmallVector<const BaseMemInfo *> &list, 
                                       AsmState &state) {
  os << "[";
  bool first = true;
  for (const auto *info : list) {
    if (!first) os << ", ";
    info->rootBuffer.printAsOperand(os, state);
    // [Fix] 打印 MAT 或 VEC 或 GM
    if (info->scope == pto::AddressSpace::GM) os << "(GM)";
    else if (info->scope == pto::AddressSpace::MAT) os << "(MAT)";
    else if (info->scope == pto::AddressSpace::VEC) os << "(VEC)";
    else os << "(Other)"; // 处理 LEFT/RIGHT/ACC 等其他情况
    first = false;
  }
  os << "]";
}
 
void PTOIRTranslator::print() {
  llvm::errs() << "\n=== PTO IR Translator Dump ===\n";
  
  AsmState state(func_); 
 
  llvm::errs() << "--- Buffer Analysis (Value -> Root) ---\n";
  for (auto &it : buffer2MemInfoMap_) {
    Value v = it.first;
    auto &infoList = it.second;
    
    llvm::errs() << "  ";
    v.printAsOperand(llvm::errs(), state);
    llvm::errs() << " -> ";
    
    for (auto &mem : infoList) {
        mem->rootBuffer.printAsOperand(llvm::errs(), state);
        llvm::errs() << " ";
    }
    llvm::errs() << "\n";
  }
 
  llvm::errs() << "\n--- SyncIR Structure ---\n";
  for (const auto &element : syncIR_) {
    unsigned id = element->GetIndex();
    llvm::errs() << llvm::formatv("{0,4}: ", id); 
 
    switch (element->GetKind()) {
    case InstanceElement::KindTy::COMPOUND: {
      auto *comp = dyn_cast<CompoundInstanceElement>(element.get());
      llvm::errs() << "COMPOUND [" << getPipelineName(comp->kPipeValue) << "] ";
      llvm::errs() << comp->opName.getStringRef() << "\n";
      
      llvm::errs() << "      DEF: ";
      printMemInfoList(llvm::errs(), comp->defVec, state);
      llvm::errs() << "\n      USE: ";
      printMemInfoList(llvm::errs(), comp->useVec, state);
      llvm::errs() << "\n";
      break;
    }
    case InstanceElement::KindTy::LOOP: 
        llvm::errs() << "LOOP\n"; break;
    case InstanceElement::KindTy::BRANCH: 
        llvm::errs() << "BRANCH\n"; break;
    case InstanceElement::KindTy::PLACE_HOLDER: 
        llvm::errs() << "PLACE_HOLDER\n"; break;
    }
  }
  llvm::errs() << "==============================\n\n";
}
