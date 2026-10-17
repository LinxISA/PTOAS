# PTOAS (PTO Assembler & Optimizer)

## Unreleased: PTO ISA 0.58.7 development audit

- Treats the unpublished PTO-SPEC development manifest as the architectural
  golden without presenting it as a published release.
- Records the exact PTOAS capability boundary: 94 golden Tile operation names
  are present but semantic/schema parity is unproven, 25 remain unavailable,
  and 15 retired public operations remain dialect-only compatibility forms.
- Rejects every retired public operation on the Linx target and verifies that
  missing golden operations have neither an ODS declaration nor a PTO-BC
  opcode assignment.
- Records that PTOAS does not assemble or encode scalar machine instructions;
  the 0.58.7 scalar atomic encoding change is consumed by scalar toolchains,
  while PTOAS audits its catalog identity only.

## Unreleased: PTO ISA 0.58.3 alignment

- Locks the exact PTO-SPEC v0.58.3 release identity and all four catalogs.
- Adds Linx CUBE CELL layouts (`cube_m16`, `cube_m32`, `cube_n8`) and
  fail-closed CUBE operand verification.
- Updates TGEMV/TGEMVMX operand order and distinct ACC/BIAS TileOP calls.
- Records TLOAD/TSTORE byte row-stride roles and verifies the 0.58.3
  B.IOT/B.IOS/B.FPATR/DTYPE_NONE hard-break fields.

## 版本
- 版本号：v0.1.0
- 发布日期：2026-02-14

## 变更摘要
- PTOAS 首次发布

## 概述
PTOAS（PTO Assembler & Optimizer）是面向 PTO Bytecode 的编译器工具链，基于 LLVM/MLIR release/19.x 构建。它提供 PTO Dialect 的定义、解析、验证、优化与代码生成能力，并输出可调用 `pto-isa` 的 C++ 代码。

PTOAS很快将集成到以下框架中，敬请期待
- PyPTO
- TileLang

## 本仓库的目标用户
PTOAS 主要面向：
- 编译器与框架后端开发者
- 高性能算子/内核开发者
- 需要进行 PTO Bytecode 生成、调试与落地的工程团队

## 主要能力
- PTO Dialect 全流程（定义、解析、验证、打印）
- 与 Tile 抽象/地址空间/同步模型配套的 IR 支撑
- PTO Bytecode → C++ 生成
- Python 端的 Dialect 构建与测试样例

## 平台与依赖最低配置
- **操作系统**：macOS (Darwin) 或 Linux (Ubuntu 20.04+)
- **编译器**：Clang >= 12 或 GCC >= 9（支持 C++17）
- **构建工具**：CMake >= 3.20，Ninja
- **Python**：Python 3.8+

## 如何使用PTOAS以及PTO IR的详细描述
- 构建与环境配置：`README.md`
- PTO Bytecode 定义：`docs/PTO_IR_manual.md`
