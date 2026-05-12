# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint, MLIRError
from mlir.dialects import func, pto
from mlir.ir import IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            i32 = IntegerType.get_signless(32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            full_ty = pto.TileBufType.get([32, 32], i32, vec, [32, 32], cfg, ctx)
            scalar_ty = pto.TileBufType.get([1, 32], i32, vec, [1, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tcolexpandexpdif_dtype_invalid", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src0 = pto.AllocTileOp(full_ty).result
                src1 = pto.AllocTileOp(scalar_ty).result
                dst = pto.AllocTileOp(full_ty).result
                pto.TColExpandExpdifOp(src0, src1, dst)
                func.ReturnOp([])

            ok = m.operation.verify()
            if ok:
                return m
            raise SystemExit(1)


if __name__ == "__main__":
    try:
        print(build())
    except MLIRError as err:
        print(f"EXPECTED_VERIFIER_FAILURE: {err}")
