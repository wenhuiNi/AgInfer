from __future__ import annotations

import pathlib
import sys

from aginfer.providers import LayerNormPayload, LayerNormProblem
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_layer_norm_payload_fixture.py OUTPUT")
    payload = LayerNormPayload(
        LayerNormProblem(CudaArch.SM120), 64_000, "6" * 64
    ).to_bytes()
    pathlib.Path(sys.argv[1]).write_bytes(payload)


if __name__ == "__main__":
    main()
