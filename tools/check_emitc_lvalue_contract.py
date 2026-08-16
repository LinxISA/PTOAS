#!/usr/bin/env python3
"""Guard the fail-closed boundary for void mutating EmitC calls."""

from pathlib import Path


def main() -> int:
    source = (Path(__file__).resolve().parents[1] / "lib/PTO/Transforms/PTOToEmitC.cpp").read_text()
    start = source.index("if (call->getNumResults() == 0)")
    end = source.index("// Result-producing call_opaque operations", start)
    void_path = source[start:end]
    required = (
        "cannot preserve mutable lvalue operands for this void call",
        "signalPassFailure();",
        "return;",
    )
    missing = [token for token in required if token not in void_path]
    if missing:
        raise SystemExit(f"void mutating lvalue path is not fail closed: missing {missing}")
    if "create<emitc::LoadOp>" in void_path:
        raise SystemExit("void mutating lvalue path must not materialize input-only loads")
    print("EmitC void-lvalue contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
