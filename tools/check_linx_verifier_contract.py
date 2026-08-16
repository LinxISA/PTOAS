#!/usr/bin/env python3
"""Check that decoded representations cannot bypass Linx verification."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = (root / "lib/PTO/IR/PTO.cpp").read_text()
    helper_start = source.index("static bool shouldBypassDecodedMemrefVerifier")
    helper_end = source.index("\n}\n", helper_start)
    helper = source[helper_start:helper_end]
    required = "if (isVerifierTargetLinx(op))\n    return false;"
    if required not in helper:
        raise SystemExit(
            "decoded memref/bind_tile verifier bypass must be disabled for Linx"
        )
    print("Linx decoded-representation verifier contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
