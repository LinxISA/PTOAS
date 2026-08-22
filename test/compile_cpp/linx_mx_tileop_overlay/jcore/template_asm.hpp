// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once

#include <common/pto_tile.hpp>
#include <jcore/type.hpp>
#include <type_traits>

using half = __half;
using bfloat16_t = __bf16;

namespace pto {

template <typename T>
inline constexpr bool is_e8m0_scale_v =
    std::is_same_v<typename T::DType, __fp8_e8m0>;

template <typename TileT>
void TASSIGN(TileT &, uint64_t) {}

#define PTO_MX_ZERO_SCALE(NAME)                                                   \
  template <typename D, typename A, typename B>                                  \
  void NAME(D &, A &, B &) {                                                     \
    static_assert(!is_e8m0_scale_v<A> && !is_e8m0_scale_v<B>);                   \
  }

#define PTO_MX_ONE_SCALE(NAME)                                                    \
  template <typename D, typename A, typename X, typename B>                      \
  void NAME(D &, A &, X &, B &) {                                                \
    static_assert(is_e8m0_scale_v<X> != is_e8m0_scale_v<B>);                     \
  }

#define PTO_MX_TWO_SCALE(NAME)                                                    \
  template <typename D, typename A, typename SA, typename B, typename SB>         \
  void NAME(D &, A &, SA &, B &, SB &) {                                         \
    static_assert(is_e8m0_scale_v<SA> && is_e8m0_scale_v<SB>);                   \
  }

#define PTO_MX_ACC_ZERO_SCALE(NAME)                                               \
  template <typename D, typename C, typename A, typename B>                      \
  void NAME(D &, C &, A &, B &) {                                                \
    static_assert(!is_e8m0_scale_v<A> && !is_e8m0_scale_v<B>);                   \
  }

#define PTO_MX_ACC_ONE_SCALE(NAME)                                                \
  template <typename D, typename C, typename A, typename X, typename B>           \
  void NAME(D &, C &, A &, X &, B &) {                                           \
    static_assert(is_e8m0_scale_v<X> != is_e8m0_scale_v<B>);                     \
  }

#define PTO_MX_ACC_TWO_SCALE(NAME)                                                \
  template <typename D, typename C, typename A, typename SA, typename B,          \
            typename SB>                                                         \
  void NAME(D &, C &, A &, SA &, B &, SB &) {                                    \
    static_assert(is_e8m0_scale_v<SA> && is_e8m0_scale_v<SB>);                   \
  }

#define PTO_MX_BIAS_ZERO_SCALE(NAME)                                              \
  template <typename D, typename A, typename B, typename Bias>                   \
  void NAME(D &, A &, B &, Bias &) {                                             \
    static_assert(!is_e8m0_scale_v<A> && !is_e8m0_scale_v<B>);                   \
  }

#define PTO_MX_BIAS_ONE_SCALE(NAME)                                               \
  template <typename D, typename A, typename X, typename B, typename Bias>        \
  void NAME(D &, A &, X &, B &, Bias &) {                                        \
    static_assert(is_e8m0_scale_v<X> != is_e8m0_scale_v<B>);                     \
  }

#define PTO_MX_BIAS_TWO_SCALE(NAME)                                               \
  template <typename D, typename A, typename SA, typename B, typename SB,         \
            typename Bias>                                                       \
  void NAME(D &, A &, SA &, B &, SB &, Bias &) {                                 \
    static_assert(is_e8m0_scale_v<SA> && is_e8m0_scale_v<SB>);                   \
  }

PTO_MX_ZERO_SCALE(TMATMUL_MX)
PTO_MX_ONE_SCALE(TMATMUL_MX)
PTO_MX_TWO_SCALE(TMATMUL_MX)
PTO_MX_ACC_ZERO_SCALE(TMATMUL_MX_ACC)
PTO_MX_ACC_ONE_SCALE(TMATMUL_MX_ACC)
PTO_MX_ACC_TWO_SCALE(TMATMUL_MX_ACC)
PTO_MX_BIAS_ZERO_SCALE(TMATMUL_MX_BIAS)
PTO_MX_BIAS_ONE_SCALE(TMATMUL_MX_BIAS)
PTO_MX_BIAS_TWO_SCALE(TMATMUL_MX_BIAS)

PTO_MX_ZERO_SCALE(TGEMV_MX)
PTO_MX_ONE_SCALE(TGEMV_MX)
PTO_MX_TWO_SCALE(TGEMV_MX)
PTO_MX_ACC_ZERO_SCALE(TGEMV_MX_ACC)
PTO_MX_ACC_ONE_SCALE(TGEMV_MX_ACC)
PTO_MX_ACC_TWO_SCALE(TGEMV_MX_ACC)
PTO_MX_BIAS_ZERO_SCALE(TGEMV_MX_BIAS)
PTO_MX_BIAS_ONE_SCALE(TGEMV_MX_BIAS)
PTO_MX_BIAS_TWO_SCALE(TGEMV_MX_BIAS)

} // namespace pto
