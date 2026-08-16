# ptobc maintenance notes (PTOAS)

This tool encodes/decodes PTO-BC v0.

## When you change the PTO dialect / IR
If you change any of the following:
- `include/PTO/IR/PTOOps.td` (add/remove ops, rename mnemonics)
- operand counts / region structure / immediates semantics

…then you **must** update the PTO-BC v0 opcode/schema tables (regenerate `tools/ptobc/generated/ptobc_opcodes_v0.h`) and ensure tests pass.

For the PTO ISA 0.58.1 hard break, the opcode assignments remain unchanged and
the audited schema transformation changes only TIMG2COL from 2 to 4 operands
and TPREFETCH from 2 to 5 operands. Apply or verify that transformation with:

```bash
python3 tools/ptobc/update_v0581_schema.py \
  tools/ptobc/generated/ptobc_opcodes_v0.h --write
python3 tools/ptobc/update_v0581_schema.py \
  tools/ptobc/generated/ptobc_opcodes_v0.h
```

The second command is the non-mutating CI audit. Historical 0.58.0 opcode
assignments and their round-trip test remain release evidence.

## Required gates
Run (or rely on CI):
- `ctest -R ptobc_stage9_e2e`
- `ctest -R ptobc_to_ptoas_smoke`
- `ctest -R ptobc_opcode_coverage_check`
- `ctest -R ptobc_v0581_contract_encode`

## Notes
- `ptobc_opcode_coverage_check` is a heuristic based on `mnemonic = "..."` occurrences.
  If PTOOps.td patterns change, update `tools/ptobc/tests/opcode_coverage_check.py`.
