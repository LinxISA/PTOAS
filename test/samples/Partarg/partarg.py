# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.dialects import arith, func, pto
from mlir.ir import Context, F32Type, IndexType, InsertionPoint, IntegerType, Location, Module


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ui32 = IntegerType.get_unsigned(32, ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_ui32 = pto.PtrType.get(ui32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tv2_ui32 = pto.TensorViewType.get(2, ui32, ctx)
            tile_view_f32 = pto.PartitionTensorViewType.get([16, 32], f32, ctx)
            tile_view_ui32 = pto.PartitionTensorViewType.get([16, 32], ui32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(
                bl, sl, pto.TileConfig.fractalABSize, pd, ctx
            )
            tile_f32 = pto.TileBufType.get([16, 32], f32, vec, [16, 32], cfg, ctx)
            tile_ui32 = pto.TileBufType.get([16, 32], ui32, vec, [16, 32], cfg, ctx)

            fn_ty = func.FunctionType.get(
                [ptr_f32, ptr_f32, ptr_ui32, ptr_ui32, ptr_f32, ptr_ui32],
                [],
            )
            with InsertionPoint(m.body):
                fn = func.FuncOp("partarg_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c16 = arith.ConstantOp(IndexType.get(ctx), 16).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                src0_ptr, src1_ptr, src0_idx_ptr, src1_idx_ptr, dst_ptr, dst_idx_ptr = (
                    entry.arguments
                )

                src0_tv = pto.MakeTensorViewOp(tv2_f32, src0_ptr, [c16, c32], [c32, c1]).result
                src1_tv = pto.MakeTensorViewOp(tv2_f32, src1_ptr, [c16, c32], [c32, c1]).result
                src0_idx_tv = pto.MakeTensorViewOp(tv2_ui32, src0_idx_ptr, [c16, c32], [c32, c1]).result
                src1_idx_tv = pto.MakeTensorViewOp(tv2_ui32, src1_idx_ptr, [c16, c32], [c32, c1]).result
                dst_tv = pto.MakeTensorViewOp(tv2_f32, dst_ptr, [c16, c32], [c32, c1]).result
                dst_idx_tv = pto.MakeTensorViewOp(tv2_ui32, dst_idx_ptr, [c16, c32], [c32, c1]).result

                src0 = pto.PartitionViewOp(tile_view_f32, src0_tv, offsets=[c0, c0], sizes=[c16, c32]).result
                src1 = pto.PartitionViewOp(tile_view_f32, src1_tv, offsets=[c0, c0], sizes=[c16, c32]).result
                src0_idx = pto.PartitionViewOp(tile_view_ui32, src0_idx_tv, offsets=[c0, c0], sizes=[c16, c32]).result
                src1_idx = pto.PartitionViewOp(tile_view_ui32, src1_idx_tv, offsets=[c0, c0], sizes=[c16, c32]).result
                dst = pto.PartitionViewOp(tile_view_f32, dst_tv, offsets=[c0, c0], sizes=[c16, c32]).result
                dst_idx = pto.PartitionViewOp(tile_view_ui32, dst_idx_tv, offsets=[c0, c0], sizes=[c16, c32]).result

                src0_tile = pto.AllocTileOp(tile_f32).result
                src1_tile = pto.AllocTileOp(tile_f32).result
                src0_idx_tile = pto.AllocTileOp(tile_ui32).result
                src1_idx_tile = pto.AllocTileOp(tile_ui32).result
                dst_tile = pto.AllocTileOp(tile_f32).result
                dst_idx_tile = pto.AllocTileOp(tile_ui32).result

                pto.TLoadOp(None, src0, src0_tile)
                pto.TLoadOp(None, src1, src1_tile)
                pto.TLoadOp(None, src0_idx, src0_idx_tile)
                pto.TLoadOp(None, src1_idx, src1_idx_tile)

                pto.TPartArgMaxOp(
                    src0_tile,
                    src1_tile,
                    src0_idx_tile,
                    src1_idx_tile,
                    dst_tile,
                    dst_idx_tile,
                )
                pto.TPartArgMinOp(
                    src0_tile,
                    src1_tile,
                    src0_idx_tile,
                    src1_idx_tile,
                    dst_tile,
                    dst_idx_tile,
                )

                pto.TStoreOp(None, dst_tile, dst)
                pto.TStoreOp(None, dst_idx_tile, dst_idx)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
