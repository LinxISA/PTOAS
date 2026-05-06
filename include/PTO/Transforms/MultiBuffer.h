// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_MULTIBUFFER_H
#define PTO_TRANSFORMS_MULTIBUFFER_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {

/// Attribute name for multi-buffer depth on `memref.alloc` (integer slot count N>=2).
inline constexpr llvm::StringLiteral kPtoMultiBufferAttrName = "pto.multi_buffer";

/// Upper bound for N; must stay consistent with `MAX_MULTI_BUFFER_NUM` in
/// insert-sync's SyncCommon.h. The static_assert that pins these two values
/// together lives in PTOPlanMemory.cpp (which already includes both headers)
/// so this header stays cheap to include from CV/multi-buffer paths.
inline constexpr unsigned kPtoMultiBufferMaxNum = 16;

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_MULTIBUFFER_H
