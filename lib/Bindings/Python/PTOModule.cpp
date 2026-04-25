// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- DialectPTO.cpp -----------------------------------------------------===//
//
// Python bindings for the PTO dialect types (pybind11 version).
//
// This file is intended to be built via declare_mlir_python_extension(...)
// with PYTHON_BINDINGS_LIBRARY pybind11, and linked with MLIRCAPIPTO.
//
//===----------------------------------------------------------------------===//

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "pto-c/Dialect/PTO.h"
#include "mlir-c/IR.h"
#include "PTO/IR/PTO.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Support.h"
#include "mlir/IR/BuiltinTypes.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

static std::vector<int64_t> toInt64Vector(const py::sequence &seq) {
  std::vector<int64_t> out;
  out.reserve(seq.size());
  for (py::handle h : seq)
    out.push_back(py::cast<int64_t>(h));
  return out;
}

static py::list shapeToPyList(const int64_t *data, intptr_t n) {
  py::list lst;
  for (intptr_t i = 0; i < n; ++i)
    lst.append(py::int_(data[i]));
  return lst;
}

void populatePTODialectSubmodule(pybind11::module &m);
void populatePTODialectSubmodule(pybind11::module &m) {
  (void)m;
}

static py::object wrapAttr(py::object cls, MlirAttribute attr) {
  if (mlirAttributeIsNull(attr))
    return py::none();
  return cls.attr("__call__")(attr);
}

static py::object wrapType(py::object cls, MlirType type) {
  if (mlirTypeIsNull(type))
    return py::none();
  return cls.attr("__call__")(type);
}

static int32_t getIntOrEnumValue(py::object value, const char *typeName) {
  if (py::isinstance<py::int_>(value))
    return value.cast<int32_t>();
  if (py::hasattr(value, "value"))
    return value.attr("value").cast<int32_t>();
  throw std::runtime_error(std::string(typeName) + " expects int or enum value");
}

template <typename IsaFn, typename BuildFn, typename ValueFn>
static void bindIntBackedAttr(py::module_ &m, const char *name, IsaFn isaFn,
                              BuildFn buildFn, ValueFn valueFn,
                              const char *typeName) {
  mlir_attribute_subclass(m, name, isaFn)
      .def_classmethod(
          "get",
          [buildFn, typeName](py::object cls, py::object value,
                              MlirContext ctx) -> py::object {
            return wrapAttr(cls, buildFn(ctx, getIntOrEnumValue(value, typeName)));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly("value", [valueFn](MlirAttribute self) -> int32_t {
        return valueFn(self);
      });
}

template <typename IsaFn, typename BuildFn>
static void bindNullaryType(py::module_ &m, const char *name, IsaFn isaFn,
                            BuildFn buildFn) {
  mlir_type_subclass(m, name, isaFn)
      .def_classmethod(
          "get",
          [buildFn](py::object cls, MlirContext context) {
            return wrapType(cls, buildFn(context));
          },
          py::arg("cls"), py::arg("context") = py::none());
}

static void bindRegisterDialect(py::module_ &m) {
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__pto__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      py::arg("context"), py::arg("load") = true);
}

static void bindBaseEnums(py::module_ &m) {
  py::enum_<mlir::pto::AddressSpace>(m, "AddressSpace")
      .value("Zero", mlir::pto::AddressSpace::Zero)
      .value("GM", mlir::pto::AddressSpace::GM)
      .value("MAT", mlir::pto::AddressSpace::MAT)
      .value("LEFT", mlir::pto::AddressSpace::LEFT)
      .value("RIGHT", mlir::pto::AddressSpace::RIGHT)
      .value("ACC", mlir::pto::AddressSpace::ACC)
      .value("VEC", mlir::pto::AddressSpace::VEC)
      .value("BIAS", mlir::pto::AddressSpace::BIAS)
      .value("SCALING", mlir::pto::AddressSpace::SCALING)
      .export_values();
  py::enum_<mlir::pto::BLayout>(m, "BLayout")
      .value("RowMajor", mlir::pto::BLayout::RowMajor)
      .value("ColMajor", mlir::pto::BLayout::ColMajor);
  py::enum_<mlir::pto::SLayout>(m, "SLayout")
      .value("NoneBox", mlir::pto::SLayout::NoneBox)
      .value("RowMajor", mlir::pto::SLayout::RowMajor)
      .value("ColMajor", mlir::pto::SLayout::ColMajor);
  py::enum_<mlir::pto::PadValue>(m, "PadValue")
      .value("Null", mlir::pto::PadValue::Null)
      .value("Zero", mlir::pto::PadValue::Zero)
      .value("Max", mlir::pto::PadValue::Max)
      .value("Min", mlir::pto::PadValue::Min);
  py::enum_<mlir::pto::CompactMode>(m, "CompactMode")
      .value("Null", mlir::pto::CompactMode::Null)
      .value("Normal", mlir::pto::CompactMode::Normal)
      .value("RowPlusOne", mlir::pto::CompactMode::RowPlusOne);
}

static void bindComputeEnums(py::module_ &m) {
  py::enum_<mlir::pto::RoundMode>(m, "RoundMode")
      .value("NONE", mlir::pto::RoundMode::NONE)
      .value("RINT", mlir::pto::RoundMode::RINT)
      .value("ROUND", mlir::pto::RoundMode::ROUND)
      .value("FLOOR", mlir::pto::RoundMode::FLOOR)
      .value("CEIL", mlir::pto::RoundMode::CEIL)
      .value("TRUNC", mlir::pto::RoundMode::TRUNC)
      .value("ODD", mlir::pto::RoundMode::ODD)
      .value("CAST_RINT", mlir::pto::RoundMode::CAST_RINT);
  py::enum_<mlir::pto::SaturationMode>(m, "SaturationMode")
      .value("ON", mlir::pto::SaturationMode::ON)
      .value("OFF", mlir::pto::SaturationMode::OFF);
  py::enum_<MlirPTOCmpMode>(m, "CmpMode")
      .value("EQ", MlirPTOCmpMode_EQ)
      .value("NE", MlirPTOCmpMode_NE)
      .value("LT", MlirPTOCmpMode_LT)
      .value("LE", MlirPTOCmpMode_LE)
      .value("GT", MlirPTOCmpMode_GT)
      .value("GE", MlirPTOCmpMode_GE)
      .export_values();
  py::enum_<mlir::pto::Layout>(m, "Layout")
      .value("ND", mlir::pto::Layout::ND)
      .value("DN", mlir::pto::Layout::DN)
      .value("NZ", mlir::pto::Layout::NZ);
  py::enum_<mlir::pto::AccToVecMode>(m, "AccToVecMode")
      .value("SingleModeVec0", mlir::pto::AccToVecMode::SingleModeVec0)
      .value("SingleModeVec1", mlir::pto::AccToVecMode::SingleModeVec1)
      .value("DualModeSplitM", mlir::pto::AccToVecMode::DualModeSplitM)
      .value("DualModeSplitN", mlir::pto::AccToVecMode::DualModeSplitN)
      .export_values();
  py::enum_<mlir::pto::ReluPreMode>(m, "ReluPreMode")
      .value("NoRelu", mlir::pto::ReluPreMode::NoRelu)
      .value("NormalRelu", mlir::pto::ReluPreMode::NormalRelu)
      .export_values();
}

static void bindPipeEnum(py::module_ &m) {
  py::enum_<mlir::pto::PIPE>(m, "PIPE")
      .value("PIPE_S", mlir::pto::PIPE::PIPE_S)
      .value("PIPE_V", mlir::pto::PIPE::PIPE_V)
      .value("PIPE_M", mlir::pto::PIPE::PIPE_M)
      .value("PIPE_MTE1", mlir::pto::PIPE::PIPE_MTE1)
      .value("PIPE_MTE2", mlir::pto::PIPE::PIPE_MTE2)
      .value("PIPE_MTE3", mlir::pto::PIPE::PIPE_MTE3)
      .value("PIPE_ALL", mlir::pto::PIPE::PIPE_ALL)
      .value("PIPE_MTE4", mlir::pto::PIPE::PIPE_MTE4)
      .value("PIPE_MTE5", mlir::pto::PIPE::PIPE_MTE5)
      .value("PIPE_V2", mlir::pto::PIPE::PIPE_V2)
      .value("PIPE_FIX", mlir::pto::PIPE::PIPE_FIX)
      .value("VIRTUAL_PIPE_MTE2_L1A", mlir::pto::PIPE::VIRTUAL_PIPE_MTE2_L1A)
      .value("VIRTUAL_PIPE_MTE2_L1B", mlir::pto::PIPE::VIRTUAL_PIPE_MTE2_L1B)
      .value("PIPE_NUM", mlir::pto::PIPE::PIPE_NUM)
      .value("PIPE_UNASSIGNED", mlir::pto::PIPE::PIPE_UNASSIGNED);
}

static void bindSyncOpAndEventEnums(py::module_ &m) {
  py::enum_<mlir::pto::SyncOpType>(m, "SyncOpType")
      .value("TLOAD", mlir::pto::SyncOpType::TLOAD)
      .value("TSTORE_ACC", mlir::pto::SyncOpType::TSTORE_ACC)
      .value("TSTORE_VEC", mlir::pto::SyncOpType::TSTORE_VEC)
      .value("TMOV_M2L", mlir::pto::SyncOpType::TMOV_M2L)
      .value("TMOV_M2S", mlir::pto::SyncOpType::TMOV_M2S)
      .value("TMOV_M2B", mlir::pto::SyncOpType::TMOV_M2B)
      .value("TMOV_M2V", mlir::pto::SyncOpType::TMOV_M2V)
      .value("TMOV_V2M", mlir::pto::SyncOpType::TMOV_V2M)
      .value("TMATMUL", mlir::pto::SyncOpType::TMATMUL)
      .value("TVEC", mlir::pto::SyncOpType::TVEC)
      .value("TVECWAIT_EVENT", mlir::pto::SyncOpType::TVECWAIT_EVENT)
      .export_values();
  py::enum_<mlir::pto::EVENT>(m, "EVENT")
      .value("EVENT_ID0", mlir::pto::EVENT::EVENT_ID0)
      .value("EVENT_ID1", mlir::pto::EVENT::EVENT_ID1)
      .value("EVENT_ID2", mlir::pto::EVENT::EVENT_ID2)
      .value("EVENT_ID3", mlir::pto::EVENT::EVENT_ID3)
      .value("EVENT_ID4", mlir::pto::EVENT::EVENT_ID4)
      .value("EVENT_ID5", mlir::pto::EVENT::EVENT_ID5)
      .value("EVENT_ID6", mlir::pto::EVENT::EVENT_ID6)
      .value("EVENT_ID7", mlir::pto::EVENT::EVENT_ID7)
      .export_values();
}

static py::object bindMaskAndQuantEnums(py::module_ &m) {
  py::enum_<mlir::pto::MaskPattern>(m, "MaskPattern")
      .value("P0101", mlir::pto::MaskPattern::P0101)
      .value("P1010", mlir::pto::MaskPattern::P1010)
      .value("P0001", mlir::pto::MaskPattern::P0001)
      .value("P0010", mlir::pto::MaskPattern::P0010)
      .value("P0100", mlir::pto::MaskPattern::P0100)
      .value("P1000", mlir::pto::MaskPattern::P1000)
      .value("P1111", mlir::pto::MaskPattern::P1111)
      .export_values();
  py::enum_<mlir::pto::QuantType>(m, "QuantType")
      .value("INT8_SYM", mlir::pto::QuantType::INT8_SYM)
      .value("INT8_ASYM", mlir::pto::QuantType::INT8_ASYM)
      .export_values();
  return m.attr("MaskPattern");
}

static py::object bindSyncEnums(py::module_ &m) {
  bindPipeEnum(m);
  bindSyncOpAndEventEnums(m);
  return bindMaskAndQuantEnums(m);
}

static void bindSimpleAttrs(py::module_ &m) {
  bindIntBackedAttr(m, "BLayoutAttr", mlirPTOAttrIsABLayoutAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOBLayoutAttrGet(ctx, value);
                    },
                    mlirPTOBLayoutAttrGetValue, "BLayoutAttr.get");
  bindIntBackedAttr(m, "SLayoutAttr", mlirPTOAttrIsASLayoutAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOSLayoutAttrGet(ctx, value);
                    },
                    mlirPTOSLayoutAttrGetValue, "SLayoutAttr.get");
  bindIntBackedAttr(m, "PadValueAttr", mlirPTOAttrIsAPadValueAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOPadValueAttrGet(ctx, value);
                    },
                    mlirPTOPadValueAttrGetValue, "PadValueAttr.get");
  bindIntBackedAttr(m, "CompactModeAttr", mlirPTOAttrIsACompactModeAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOCompactModeAttrGet(ctx, value);
                    },
                    mlirPTOCompactModeAttrGetValue, "CompactModeAttr.get");
  bindIntBackedAttr(m, "AccToVecModeAttr", mlirPTOAttrIsAAccToVecModeAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOAccToVecModeAttrGet(ctx, value);
                    },
                    mlirPTOAccToVecModeAttrGetValue,
                    "AccToVecModeAttr.get");
  bindIntBackedAttr(m, "ReluPreModeAttr", mlirPTOAttrIsAReluPreModeAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOReluPreModeAttrGet(ctx, value);
                    },
                    mlirPTOReluPreModeAttrGetValue, "ReluPreModeAttr.get");
}

static void bindExtendedAttrs(py::module_ &m) {
  bindIntBackedAttr(m, "AddressSpaceAttr", mlirPTOAttrIsAAddressSpaceAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOAddressSpaceAttrGet(ctx, value);
                    },
                    mlirPTOAddressSpaceAttrGetValue,
                    "AddressSpaceAttr.get");
  bindIntBackedAttr(m, "RoundModeAttr", mlirPTOAttrIsARoundModeAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTORoundModeAttrGet(ctx, value);
                    },
                    mlirPTORoundModeAttrGetValue, "RoundModeAttr.get");
  bindIntBackedAttr(m, "SaturationModeAttr",
                    mlirPTOAttrIsASaturationModeAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOSaturationModeAttrGet(ctx, value);
                    },
                    mlirPTOSaturationModeAttrGetValue,
                    "SaturationModeAttr.get");
  bindIntBackedAttr(m, "PipeAttr", mlirPTOAttrIsAPipeAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOPipeAttrGet(ctx, value);
                    },
                    mlirPTOPipeAttrGetValue, "PipeAttr.get");
  bindIntBackedAttr(m, "LayoutAttr", mlirPTOAttrIsALayoutAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOLayoutAttrGet(ctx, value);
                    },
                    mlirPTOLayoutAttrGetValue, "LayoutAttr.get");
  bindIntBackedAttr(m, "SyncOpTypeAttr", mlirPTOAttrIsASyncOpTypeAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOSyncOpTypeAttrGet(ctx, value);
                    },
                    mlirPTOSyncOpTypeAttrGetValue, "SyncOpTypeAttr.get");
}

static void bindCmpModeAttr(py::module_ &m) {
  mlir_attribute_subclass(m, "CmpModeAttr", mlirAttributeIsAPTOCmpModeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, MlirPTOCmpMode value) {
            return wrapAttr(cls, mlirPTOCmpModeAttrGet(ctx, value));
          },
          "cls"_a, "context"_a, "value"_a)
      .def_property_readonly("value",
                             [](MlirAttribute self) {
                               return mlirPTOCmpModeAttrGetValue(self);
                             });
}

static void bindEventAndQuantAttrs(py::module_ &m) {
  bindIntBackedAttr(m, "EventAttr", mlirPTOAttrIsAEventAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOEventAttrGet(ctx, value);
                    },
                    mlirPTOEventAttrGetValue, "EventAttr.get");
  bindIntBackedAttr(m, "QuantTypeAttr", mlirPTOAttrIsAQuantTypeAttr,
                    [](MlirContext ctx, int32_t value) {
                      return mlirPTOQuantTypeAttrGet(ctx, value);
                    },
                    mlirPTOQuantTypeAttrGetValue, "QuantTypeAttr.get");
}

static void bindMaskPatternAttr(py::module_ &m, py::object maskPatternEnumType) {
  mlir_attribute_subclass(m, "MaskPatternAttr", [](MlirAttribute attr) {
    return mlirPTOAttrIsAMaskPatternAttr(attr);
  })
      .def_classmethod(
          "get",
          [maskPatternEnumType](py::object cls, py::object value,
                                MlirContext ctx) -> py::object {
            if (py::isinstance(value, maskPatternEnumType)) {
              auto enumValue =
                  static_cast<MlirPTOMaskPattern>(value.attr("value").cast<int32_t>());
              return wrapAttr(cls, mlirPTOMaskPatternAttrGetEnum(ctx, enumValue));
            }
            MlirAttribute attr =
                mlirPTOMaskPatternAttrGet(ctx, getIntOrEnumValue(value, "MaskPatternAttr.get"));
            if (mlirAttributeIsNull(attr))
              throw std::runtime_error(
                  "MaskPatternAttr.get(int, ...) only accepts unambiguous values {0,3,6,7}; "
                  "use MaskPattern enum for ISA values and get_legacy_raw(...) for historical raw encodings");
            return wrapAttr(cls, attr);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_classmethod(
          "get_legacy_raw",
          [](py::object cls, int32_t value, MlirContext ctx) -> py::object {
            MlirAttribute attr = mlirPTOMaskPatternAttrGetLegacyRaw(ctx, value);
            if (mlirAttributeIsNull(attr))
              throw std::runtime_error(
                  "MaskPatternAttr.get_legacy_raw(...) only accepts historical raw values {0,3,4,5}");
            return wrapAttr(cls, attr);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly("value", [](MlirAttribute self) {
        return static_cast<mlir::pto::MaskPattern>(
            mlirPTOMaskPatternAttrGetEnumValue(self));
      })
      .def_property_readonly("int_value", [](MlirAttribute self) -> int32_t {
        return mlirPTOMaskPatternAttrGetValue(self);
      });
}

static void bindTypedAttrs(py::module_ &m, py::object maskPatternEnumType) {
  bindCmpModeAttr(m);
  bindEventAndQuantAttrs(m);
  bindMaskPatternAttr(m, maskPatternEnumType);
}

static void bindScalarTypes(py::module_ &m) {
  mlir_type_subclass(m, "PtrType", [](MlirType type) -> bool {
    return mlirPTOTypeIsAPtrType(type);
  })
      .def_classmethod(
          "get",
          [](py::object cls, MlirType elementType, MlirContext context) {
            MlirContext ctx = context.ptr ? context : mlirTypeGetContext(elementType);
            return wrapType(cls, mlirPTOPtrTypeGet(ctx, elementType));
          },
          py::arg("cls"), py::arg("element_type"),
          py::arg("context") = py::none())
      .def_property_readonly("element_type", [](MlirType self) {
        return mlirPTOPtrTypeGetElementType(self);
      });
  bindNullaryType(m, "AsyncSessionType", mlirPTOTypeIsAAsyncSessionType,
                  mlirPTOAsyncSessionTypeGet);
  bindNullaryType(m, "AsyncEventType", mlirPTOTypeIsAAsyncEventType,
                  mlirPTOAsyncEventTypeGet);
  bindNullaryType(m, "HiF8Type", mlirPTOTypeIsAHiF8Type, mlirPTOHiF8TypeGet);
  bindNullaryType(m, "F4E1M2x2Type", mlirPTOTypeIsAF4E1M2x2Type,
                  mlirPTOF4E1M2x2TypeGet);
  bindNullaryType(m, "F4E2M1x2Type", mlirPTOTypeIsAF4E2M1x2Type,
                  mlirPTOF4E2M1x2TypeGet);
}

static void bindTensorViewType(py::module_ &m) {
  mlir_type_subclass(m, "TensorViewType", [](MlirType type) -> bool {
    return mlirPTOTypeIsATensorViewType(type);
  })
      .def_classmethod(
          "get",
          [](py::object cls, py::object shapeOrRank, MlirType elementType,
             MlirContext context) -> py::object {
            std::vector<int64_t> shape;
            if (py::isinstance<py::int_>(shapeOrRank)) {
              shape.assign(shapeOrRank.cast<int64_t>(),
                           mlir::ShapedType::kDynamic);
            } else {
              shape = toInt64Vector(shapeOrRank.cast<py::sequence>());
            }
            return wrapType(
                cls, mlirPTOTensorViewTypeGet(context, shape.size(), shape.data(),
                                              elementType));
          },
          py::arg("cls"), py::arg("shape_or_rank"), py::arg("element_type"),
          py::arg("context") = py::none())
      .def_property_readonly("rank", [](MlirType self) {
        return mlirPTOTensorViewTypeGetRank(self);
      })
      .def_property_readonly("element_type", [](MlirType self) {
        return mlirPTOTensorViewTypeGetElementType(self);
      })
      .def_property_readonly("shape", [](MlirType self) {
        intptr_t count = 0;
        return shapeToPyList(mlirPTOTensorViewTypeGetShape(self, &count), count);
      });
}

static void bindPartitionTensorViewType(py::module_ &m) {
  mlir_type_subclass(m, "PartitionTensorViewType", [](MlirType type) -> bool {
    return mlirPTOTypeIsAPartitionTensorViewType(type);
  })
      .def_classmethod(
          "get",
          [](py::object cls, py::sequence shape, MlirType elementType,
             MlirContext context) {
            auto dims = toInt64Vector(shape);
            return wrapType(cls, mlirPTOPartitionTensorViewTypeGet(
                                     context, dims.size(), dims.data(), elementType));
          },
          py::arg("cls"), py::arg("shape"), py::arg("element_type"),
          py::arg("context") = py::none())
      .def_property_readonly("rank", [](MlirType self) {
        return mlirPTOPartitionTensorViewTypeGetRank(self);
      })
      .def_property_readonly("element_type", [](MlirType self) {
        return mlirPTOPartitionTensorViewTypeGetElementType(self);
      })
      .def_property_readonly("shape", [](MlirType self) {
        intptr_t count = 0;
        return shapeToPyList(
            mlirPTOPartitionTensorViewTypeGetShape(self, &count), count);
      });
}

static void bindViewTypes(py::module_ &m) {
  bindTensorViewType(m);
  bindPartitionTensorViewType(m);
}

static void bindTileTypes(py::module_ &m) {
  mlir_type_subclass(m, "TileType", [](MlirType type) -> bool {
    return mlirPTOTypeIsATileType(type);
  })
      .def_classmethod(
          "get",
          [](py::object cls, py::sequence shape, MlirType elementType,
             MlirContext context) {
            auto dims = toInt64Vector(shape);
            return wrapType(
                cls, mlirPTOTileTypeGet(context, dims.size(), dims.data(), elementType));
          },
          py::arg("cls"), py::arg("shape"), py::arg("element_type"),
          py::arg("context") = py::none())
      .def_property_readonly("rank", [](MlirType self) {
        return mlirPTOTileTypeGetRank(self);
      })
      .def_property_readonly("element_type", [](MlirType self) {
        return mlirPTOTileTypeGetElementType(self);
      })
      .def_property_readonly("shape", [](MlirType self) {
        intptr_t count = 0;
        return shapeToPyList(mlirPTOTileTypeGetShape(self, &count), count);
      });
}

static MlirAttribute resolveCompactMode(py::object value, MlirContext ctx) {
  if (value.is_none())
    return mlirPTOCompactModeAttrGet(
        ctx, static_cast<int32_t>(mlir::pto::CompactMode::Null));
  if (py::isinstance<py::int_>(value) || py::hasattr(value, "value"))
    return mlirPTOCompactModeAttrGet(ctx, getIntOrEnumValue(value, "compact_mode"));
  return value.cast<MlirAttribute>();
}

static std::vector<int64_t> resolveValidShape(const std::vector<int64_t> &shape,
                                              py::object validShapeObj) {
  std::vector<int64_t> validShape = shape;
  if (validShapeObj.is_none())
    return validShape;
  py::list values = validShapeObj.cast<py::list>();
  if (static_cast<size_t>(values.size()) != shape.size())
    throw std::runtime_error("valid_shape rank must match shape rank");
  validShape.resize(values.size());
  for (ssize_t index = 0; index < values.size(); ++index)
    validShape[index] = values[index].is_none() ? -1 : values[index].cast<int64_t>();
  return validShape;
}

static void bindTileBufConfigAttr(py::module_ &m) {
  mlir_attribute_subclass(m, "TileBufConfigAttr", [](MlirAttribute attr) {
    return mlirPTOAttrIsATileBufConfigAttr(attr);
  })
      .def_classmethod(
          "get_default",
          [](py::object cls, MlirContext ctx) {
            return wrapAttr(cls, mlirPTOTileBufConfigAttrGetDefault(ctx));
          },
          py::arg("cls"), py::arg("context") = py::none())
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute blayout, MlirAttribute slayout,
             int32_t sFractalSize, MlirAttribute pad, MlirContext ctx,
             py::object compactModeObj) {
            MlirAttribute sizeAttr =
                mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 32), sFractalSize);
            MlirAttribute compactMode = resolveCompactMode(compactModeObj, ctx);
            return wrapAttr(cls, mlirPTOTileBufConfigAttrGetWithCompactMode(
                                     ctx, blayout, slayout, sizeAttr, pad,
                                     compactMode));
          },
          py::arg("cls"), py::arg("blayout"), py::arg("slayout"),
          py::arg("s_fractal_size"), py::arg("pad"),
          py::arg("context") = py::none(),
          py::arg("compact_mode") = py::none());
}

static void bindTileBufType(py::module_ &m) {
  mlir_type_subclass(m, "TileBufType", [](MlirType type) -> bool {
    return mlirPTOTypeIsATileBufType(type);
  })
      .def_classmethod(
          "get",
          [](py::object cls, std::vector<int64_t> shape, MlirType elementType,
             MlirAttribute memorySpace, py::object validShapeObj,
             py::object configObj, MlirContext ctx) {
            auto validShape = resolveValidShape(shape, validShapeObj);
            MlirType type = configObj.is_none()
                                ? mlirPTOTileBufTypeGetWithValidShape(
                                      ctx, shape.size(), shape.data(), elementType,
                                      memorySpace, validShape.size(),
                                      validShape.data())
                                : mlirPTOTileBufTypeGetWithValidShapeAndConfig(
                                      ctx, shape.size(), shape.data(), elementType,
                                      memorySpace, validShape.size(),
                                      validShape.data(),
                                      configObj.cast<MlirAttribute>());
            return wrapType(cls, type);
          },
          py::arg("cls"), py::arg("shape"), py::arg("element_type"),
          py::arg("memory_space"), py::arg("valid_shape") = py::none(),
          py::arg("config") = py::none(),
          py::arg("context") = py::none())
      .def_classmethod(
          "upcast_type",
          [](py::object cls, MlirType type) {
            return mlirPTOTypeIsATileBufType(type) ? cls(type) : py::none();
          },
          py::arg("cls"), py::arg("type"));
}

static void bindTileBufTypes(py::module_ &m) {
  bindTileBufConfigAttr(m);
  bindTileBufType(m);
}

static void bindPTOModule(pybind11::module &m) {
  m.doc() = "PTO dialect Python bindings (pybind11).";
  bindRegisterDialect(m);
  bindBaseEnums(m);
  bindComputeEnums(m);
  py::object maskPatternEnumType = bindSyncEnums(m);
  bindSimpleAttrs(m);
  bindExtendedAttrs(m);
  bindTypedAttrs(m, maskPatternEnumType);
  bindScalarTypes(m);
  bindViewTypes(m);
  bindTileTypes(m);
  bindTileBufTypes(m);
  populatePTODialectSubmodule(m);
}

PYBIND11_MODULE(_pto, m) {
  bindPTOModule(m);
}
