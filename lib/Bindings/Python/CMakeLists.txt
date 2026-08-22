# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# lib/Bindings/Python/CMakeLists.txt

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

include(TableGen)
include(AddMLIR)

# ---- 1) Python native extension: mlir._mlir_libs._pto ----
nanobind_add_module(_pto
  NB_DOMAIN mlir
  FREE_THREADED
  NB_SHARED
  PTOModule.cpp
)

# CPython extension runtimes intentionally resolve Python ABI symbols from the
# interpreter at module-load time. The project hardening cache enables
# ``-z defs`` globally, so restore the standard extension-module behavior only
# for nanobind's shared runtime on ELF platforms. RELRO/NOW and the remaining
# hardening flags stay enabled.
if(UNIX AND NOT APPLE)
  # nanobind 2.x names the shared domain runtime nanobind-mlir. nanobind 3.x
  # names it from the selected ABI/features (normally nanobind, with optional
  # -abi3/-ft suffixes). Resolve the target created by nanobind_add_module
  # rather than assuming one nanobind release's private target spelling.
  set(PTOAS_NANOBIND_RUNTIME_TARGET "")
  foreach(_candidate
      nanobind-mlir nanobind-abi3-ft nanobind-ft nanobind-abi3 nanobind)
    if(TARGET ${_candidate})
      set(PTOAS_NANOBIND_RUNTIME_TARGET ${_candidate})
      break()
    endif()
  endforeach()
  if(NOT PTOAS_NANOBIND_RUNTIME_TARGET)
    message(FATAL_ERROR "nanobind_add_module did not create a shared runtime target")
  endif()

  target_link_options(${PTOAS_NANOBIND_RUNTIME_TARGET} PRIVATE
    "LINKER:-z,undefs")
  # NB_SHARED creates the runtime outside LLVM's library directory.
  # Install it into the directory exported to auditwheel so every architecture
  # can resolve and vendor the runtime into the repaired wheel.
  install(TARGETS ${PTOAS_NANOBIND_RUNTIME_TARGET}
    LIBRARY DESTINATION lib
    COMPONENT PTOASPythonRuntime
  )
endif()

target_compile_definitions(_pto PRIVATE
  MLIR_BINDINGS_PYTHON_DOMAIN=mlir
)

target_include_directories(_pto PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_link_options(_pto PRIVATE)

target_link_libraries(_pto PRIVATE

  PTOCAPI
  PTOIR
  MLIRIR
  MLIRCAPIIR
  MLIRSupport
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRDestinationStyleOpInterface
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRLoopLikeInterface
  MLIRViewLikeInterface
  MLIRFunctionInterfaces
)

# 关键：放到 mlir/_mlir_libs 下（匹配 MLIR dialect python 的 import 习惯）
set_target_properties(_pto PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  BUILD_RPATH "${LLVM_BUILD_LIBRARY_DIR}"
  INSTALL_RPATH "${LLVM_BUILD_LIBRARY_DIR}"
)
# macOS: 避免 ld 警告 "-undefined suppress is deprecated"，仅保留 -undefined dynamic_lookup
if(APPLE)
  target_link_options(_pto PRIVATE "LINKER:-undefined,dynamic_lookup")
endif()

install(TARGETS _pto
  LIBRARY DESTINATION "${MLIR_PYTHON_PACKAGE_DIR}/mlir/_mlir_libs"
)

# ---- 2) Generate ODS Python op bindings: _pto_ops_gen.py ----
set(PTO_TD_DIR "${CMAKE_SOURCE_DIR}/include/PTO/IR")
set(LLVM_TARGET_DEFINITIONS ${PTO_TD_DIR}/PTOOps.td)

mlir_tablegen(_pto_ops_gen.py
  -gen-python-op-bindings
  --bind-dialect=pto
)

add_public_tablegen_target(PTOPythonGen)
add_dependencies(_pto PTOPythonGen)

# ---- 3) Copy generated python + handwritten pto.py into build/python ----
set(PTO_PY_SRC "${CMAKE_SOURCE_DIR}/python/pto/dialects/pto.py")

add_custom_command(TARGET _pto POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python/mlir/dialects"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${PTO_PY_SRC}"
          "${CMAKE_BINARY_DIR}/python/mlir/dialects/pto.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_BINARY_DIR}/_pto_ops_gen.py"
          "${CMAKE_BINARY_DIR}/python/mlir/dialects/_pto_ops_gen.py"
  VERBATIM
)

install(FILES
  "${PTO_PY_SRC}"
  "${CMAKE_CURRENT_BINARY_DIR}/_pto_ops_gen.py"
  DESTINATION mlir/dialects
)

add_custom_command(TARGET _pto POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  VERBATIM
)
