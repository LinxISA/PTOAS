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

#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/InsertSyncDebug.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;

static bool isTraceEnabled() {
  return isInsertSyncDebugEnabled(InsertSyncDebugLevel::Trace);
}
 
// [Debug] 打印 Value 详细信息
static void printValueDebug(const char* tag, Value v) {
  if (!isTraceEnabled())
    return;
  llvm::errs() << tag << ": ";
  if (!v) {
    llvm::errs() << "NULL\n";
    return;
  }
  
  if (auto *op = v.getDefiningOp()) {
    llvm::errs() << "OpResult defined by [" << op->getName() << "]";
  } else {
    llvm::errs() << "BlockArgument";
  }
  llvm::errs() << " | Type: " << v.getType() << "\n";
}

static const char *getScopeName(pto::AddressSpace scope) {
  switch (scope) {
  case pto::AddressSpace::Zero:
    return "Zero";
  case pto::AddressSpace::GM:
    return "GM";
  case pto::AddressSpace::VEC:
    return "VEC";
  case pto::AddressSpace::MAT:
    return "MAT";
  case pto::AddressSpace::ACC:
    return "ACC";
  case pto::AddressSpace::LEFT:
    return "LEFT";
  case pto::AddressSpace::RIGHT:
    return "RIGHT";
  case pto::AddressSpace::BIAS:
    return "BIAS";
  case pto::AddressSpace::SCALING:
    return "SCALING";
  }
  return "UNKNOWN";
}

static void printBaseList(ArrayRef<uint64_t> bases) {
  llvm::errs() << "[";
  for (size_t i = 0; i < bases.size(); ++i) {
    llvm::errs() << bases[i];
    if (i + 1 != bases.size())
      llvm::errs() << ", ";
  }
  llvm::errs() << "]";
}

static void printInt64List(ArrayRef<int64_t> values) {
  llvm::errs() << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    llvm::errs() << values[i];
    if (i + 1 != values.size())
      llvm::errs() << ", ";
  }
  llvm::errs() << "]";
}

static void printRegionDebug(const char *tag,
                             const std::optional<StaticMemRegion> &region) {
  if (!isTraceEnabled())
    return;
  llvm::errs() << tag << ": ";
  if (!region || !region->isPrecise()) {
    llvm::errs() << "<none>\n";
    return;
  }
  llvm::errs() << "offsets=";
  printInt64List(region->offsets);
  llvm::errs() << " sizes=";
  printInt64List(region->sizes);
  llvm::errs() << " strides=";
  printInt64List(region->strides);
  llvm::errs() << " elemBytes=" << region->elemSizeBytes
               << " baseOffsetBytes=" << region->baseOffsetBytes << "\n";
}

static bool hasComparablePreciseRegions(const BaseMemInfo *a,
                                        const BaseMemInfo *b) {
  if (!a || !b || !a->preciseRegion || !b->preciseRegion)
    return false;
  const StaticMemRegion &ar = *a->preciseRegion;
  const StaticMemRegion &br = *b->preciseRegion;
  if (!ar.isPrecise() || !br.isPrecise())
    return false;
  if (ar.offsets.size() != br.offsets.size())
    return false;
  if (ar.elemSizeBytes != br.elemSizeBytes)
    return false;
  return true;
}

static bool arePreciseRegionsProvenDisjoint(const BaseMemInfo *a,
                                            const BaseMemInfo *b) {
  if (!hasComparablePreciseRegions(a, b))
    return false;

  const StaticMemRegion &ar = *a->preciseRegion;
  const StaticMemRegion &br = *b->preciseRegion;
  for (size_t dim = 0; dim < ar.offsets.size(); ++dim) {
    if (ar.sizes[dim] < 0 || br.sizes[dim] < 0)
      return false;
    int64_t aStart = ar.offsets[dim];
    int64_t bStart = br.offsets[dim];
    int64_t aEnd = aStart + ar.sizes[dim];
    int64_t bEnd = bStart + br.sizes[dim];
    if (aEnd <= bStart || bEnd <= aStart) {
      if (isTraceEnabled()) {
        llvm::errs() << "    [RegionOverlap] precise regions are disjoint "
                     << "by dim=" << dim << " A=[" << aStart << ", " << aEnd
                     << ") B=[" << bStart << ", " << bEnd << ")\n";
      }
      return true;
    }
  }

  if (isTraceEnabled())
    llvm::errs() << "    [RegionOverlap] precise regions not proven "
                    "disjoint; falling back to flat range\n";
  return false;
}

static void printMemInfoDebug(const char *tag, const BaseMemInfo *info) {
  if (!isTraceEnabled())
    return;
  llvm::errs() << tag << ": ";
  if (!info) {
    llvm::errs() << "<null>\n";
    return;
  }
  llvm::errs() << "scope=" << getScopeName(info->scope)
               << " bases=";
  printBaseList(info->baseAddresses);
  llvm::errs() << " sizeBytes=" << info->allocateSize << "\n";
  printValueDebug("      base", info->baseBuffer);
  printValueDebug("      root", info->rootBuffer);
  printRegionDebug("      region", info->preciseRegion);
}
 
// [Fix & Debug] 增强版 GetRealRoot
static Value GetRealRoot(Value v) {
  const bool trace = isTraceEnabled();
  if (trace) {
    llvm::errs() << "  [Trace] GetRealRoot Start:\n";
    printValueDebug("    Current", v);
  }
  
  int depth = 0;
  const int maxDepth = 20;
 
  while (v && depth++ < maxDepth) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
        if (trace)
          llvm::errs() << "    -> Reached BlockArgument. Stop.\n";
        break; 
    }
 
    if (auto op = dyn_cast<memref::CollapseShapeOp>(defOp)) {
        if (trace)
          llvm::errs() << "    -> Hit CollapseShapeOp. Peel off.\n";
        v = op.getSrc();
        continue;
    }
    if (auto op = dyn_cast<memref::ExpandShapeOp>(defOp)) {
        if (trace)
          llvm::errs() << "    -> Hit ExpandShapeOp. Peel off.\n";
        v = op.getSrc();
        continue;
    }
    if (auto op = dyn_cast<memref::ViewOp>(defOp)) {
        if (trace)
          llvm::errs() << "    -> Hit ViewOp. Peel off.\n";
        v = op.getSource();
        continue;
    }
    if (auto view = dyn_cast<ViewLikeOpInterface>(defOp)) {
        if (trace)
          llvm::errs() << "    -> Hit ViewLikeInterface. Peel off.\n";
        v = view.getViewSource();
        continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
        v = cast.getSource();
        continue;
    }
    if (auto reCast = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
        v = reCast.getSource();
        continue;
    }
 
    if (trace) {
      llvm::errs() << "    -> Hit Alloc/Other [" << defOp->getName()
                   << "]. Stop.\n";
    }
    break;
  }
  return v;
}
 
bool MemoryDependentAnalyzer::DepBetween(
    const SmallVector<const BaseMemInfo *> &a,
    const SmallVector<const BaseMemInfo *> &b,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  
  // [Debug Log] 关键入口信息
  if (isTraceEnabled()) {
    llvm::errs() << "\n[DepBetween] Checking dependency...\n";
    llvm::errs() << "  Vec A Size: " << a.size() << "\n";
    llvm::errs() << "  Vec B Size: " << b.size() << "\n";
  }
 
  bool hasAlias = false;
  for (auto &i : a) {
    for (auto &j : b) {
      if (MemAlias(i, j)) {
        depBaseMemInfosVec.push_back(std::make_pair(i, j));
        hasAlias = true;
      }
    }
  }
  return hasAlias;
}
 
bool MemoryDependentAnalyzer::MemAlias(const BaseMemInfo *a,
                                       const BaseMemInfo *b) {
  pto::AddressSpace as = a->scope;
  pto::AddressSpace bs = b->scope;
 
  // [Debug Log] 打印比较对象
  if (isTraceEnabled()) {
    llvm::errs() << "  [MemAlias Check]\n";
    printValueDebug("    Root A", a->rootBuffer);
    printValueDebug("    Root B", b->rootBuffer);
    llvm::errs() << "    Scope A: " << (int)as << ", Scope B: " << (int)bs
                 << "\n";
    printMemInfoDebug("    MemInfo A", a);
    printMemInfoDebug("    MemInfo B", b);
  }
 
  if (as != bs) {
    if (isTraceEnabled())
      llvm::errs() << "    -> Scope Mismatch. False.\n";
    return false;
  }
 
  // 1. GM 内存
  if (as == pto::AddressSpace::GM) {
    return isGMBufferOverlap(a, b);
  }
 
  // 2. Local Memory (UB/L1)
  
  if (a->rootBuffer == b->rootBuffer) {
    if (arePreciseRegionsProvenDisjoint(a, b)) {
      if (isTraceEnabled())
        llvm::errs() << "    -> Same root precise regions disjoint. "
                        "Alias=false.\n";
      return false;
    }
    if (a->baseAddresses.empty() || b->baseAddresses.empty()) {
      if (isTraceEnabled())
        llvm::errs() << "    -> Same root but unknown base list. Alias=true.\n";
      return true;
    }
    bool overlap = isBufferAddressRangeOverlap(a, b);
    if (isTraceEnabled())
      llvm::errs() << "    -> Same root address range overlap="
                   << (overlap ? "true" : "false") << "\n";
    return overlap;
  }
 
  // 2.2 深层比较：穿透 View
  Value realRootA = GetRealRoot(a->rootBuffer);
  Value realRootB = GetRealRoot(b->rootBuffer);
 
  if (isTraceEnabled()) {
    llvm::errs() << "    [Deep Check] Surface Roots differ. Digging deeper...\n";
    printValueDebug("      Real Root A", realRootA);
    printValueDebug("      Real Root B", realRootB);
  }
 
  if (realRootA == realRootB && realRootA != nullptr) {
      if (arePreciseRegionsProvenDisjoint(a, b)) {
        if (isTraceEnabled())
          llvm::errs() << "      -> MATCH, but precise regions are disjoint. "
                          "Alias=false.\n";
        return false;
      }
      if (isTraceEnabled())
        llvm::errs() << "      -> MATCH! Real roots are the same. Alias=true "
                        "without refined range check.\n";
      return true;
  } else {
      if (isTraceEnabled())
        llvm::errs() << "      -> Mismatch. Real roots differ.\n";
  }
 
  return false;
}
 
bool MemoryDependentAnalyzer::isGMBufferOverlap(const BaseMemInfo *a,
                                                const BaseMemInfo *b) {
  if (a->rootBuffer != b->rootBuffer) {
    Value realRootA = GetRealRoot(a->rootBuffer);
    Value realRootB = GetRealRoot(b->rootBuffer);
    
    if (realRootA != realRootB) {
        if (isTraceEnabled())
          llvm::errs() << "    -> GM real roots differ. Alias=false.\n";
        return false;
    }
    if (arePreciseRegionsProvenDisjoint(a, b)) {
      if (isTraceEnabled())
        llvm::errs() << "    -> GM real roots match, but precise regions are "
                        "disjoint. Alias=false.\n";
      return false;
    }
    if (isTraceEnabled())
      llvm::errs() << "    -> GM real roots match. Alias=true without "
                      "refined range check.\n";
    return true; 
  }
 
  if (arePreciseRegionsProvenDisjoint(a, b)) {
    if (isTraceEnabled())
      llvm::errs() << "    -> GM precise regions disjoint. Alias=false.\n";
    return false;
  }

  if (a->baseAddresses.empty() || b->baseAddresses.empty()) {
    if (isTraceEnabled())
      llvm::errs() << "    -> GM unknown base list. Alias=true.\n";
    return true;
  }
  if (a->allocateSize == 0 || b->allocateSize == 0) {
    if (isTraceEnabled())
      llvm::errs() << "    -> GM unknown allocation size. Alias=true.\n";
    return true;
  }
 
  return isBufferAddressRangeOverlap(a, b);
}
 
bool MemoryDependentAnalyzer::isBufferAddressRangeOverlap(
    const BaseMemInfo *a, const BaseMemInfo *b) {
  int aBaseAddressesSize = static_cast<int>(a->baseAddresses.size());
  int bBaseAddressesSize = static_cast<int>(b->baseAddresses.size());
  
  for (int i = 0; i < aBaseAddressesSize; i++) {
    for (int j = 0; j < bBaseAddressesSize; j++) {
      if (isBufferOverlap(a, b, i, j)) {
        return true;
      }
    }
  }
  return false;
}
 
bool MemoryDependentAnalyzer::isBufferOverlap(const BaseMemInfo *a,
                                              const BaseMemInfo *b, int aIndex,
                                              int bIndex) {
  uint64_t aStart = a->baseAddresses[aIndex];
  uint64_t bStart = b->baseAddresses[bIndex];
  uint64_t aEnd = aStart + a->allocateSize;
  uint64_t bEnd = bStart + b->allocateSize;
 
  uint64_t maxStart = std::max(aStart, bStart);
  uint64_t minEnd = std::min(aEnd, bEnd);
 
  bool overlap = maxStart < minEnd;
  if (isTraceEnabled()) {
    llvm::errs() << "    [RangeOverlap] A[" << aIndex << "]=[" << aStart
                 << ", " << aEnd << ") size=" << a->allocateSize
                 << " B[" << bIndex << "]=[" << bStart << ", " << bEnd
                 << ") size=" << b->allocateSize
                 << " maxStart=" << maxStart << " minEnd=" << minEnd
                 << " => " << (overlap ? "overlap" : "disjoint");
    if (overlap)
      llvm::errs() << " overlapBytes=" << (minEnd - maxStart);
    llvm::errs() << "\n";
  }
  return overlap;
}
