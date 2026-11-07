"""Check generated dispatch entry points without loading a CUDA driver."""

import argparse
from pathlib import Path
import re


REQUIRED = {
    "cast": ("cast_bf16_f32", "cast_f32_bf16"),
    "unary": ("ucopy_f32", "ucopy_bf16"),
    "binary": ("badd_f32",),
}


def check(target: Path) -> None:
    outputs = sorted(target.glob("**/build/candle-kernels-*/out/ptx.rs"))
    if not outputs:
        raise RuntimeError(f"No compiled candle-kernels PTX found under {target}")
    for bindings in outputs:
        for module, symbols in REQUIRED.items():
            path = bindings.parent / f"{module}.ptx"
            ptx = path.read_text()
            # Match declarations, not references or comments containing a name.
            ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
            for symbol in symbols:
                if not re.search(r"\.entry\s+" + re.escape(symbol) + r"\s*\(", ptx):
                    raise RuntimeError(f"Missing dispatch CUDA entry {symbol} in {path}")
        print(f"DISPATCH_S0_PTX_SYMBOLS=PASS: {bindings.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", type=Path, default=Path("target"))
    check(parser.parse_args().target)
