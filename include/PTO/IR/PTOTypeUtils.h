// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_IR_PTOTYPEUTILS_H
#define PTO_IR_PTOTYPEUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::pto {

bool isPTOFloat8Type(Type t);
bool isPTOFloat8E4M3FamilyType(Type t);
bool isPTOFloat8E5M2FamilyType(Type t);
bool isPTOHiFloat8Type(Type t);
bool isPTOFloat4PackedType(Type t);
bool isPTOLowPrecisionType(Type t);

unsigned getPTOStorageElemByteSize(Type t);

} // namespace mlir::pto

#endif // PTO_IR_PTOTYPEUTILS_H
