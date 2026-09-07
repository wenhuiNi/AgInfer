from __future__ import annotations

import pathlib
import sys

from aginfer.providers import (
    CublasLtAlgorithm,
    CublasLtDType,
    CublasLtLinearPayload,
    CublasLtLinearProblem,
    PatchProjectionPayload,
    PatchProjectionProblem,
)
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_patch_projection_payload_fixture.py OUTPUT")
    linear = CublasLtLinearPayload(
        CublasLtLinearProblem(CudaArch.SM120, CublasLtDType.F32, 256, 1152, 588),
        120803,
        CublasLtAlgorithm(10, 11, 0, 1, 0, 0, 0, 0, 0),
        2304,
        256,
        256,
        256,
        256,
        256,
    )
    payload = PatchProjectionPayload(
        PatchProjectionProblem(CudaArch.SM120), 160_000, "e" * 64, linear
    )
    pathlib.Path(sys.argv[1]).write_bytes(payload.to_bytes())


if __name__ == "__main__":
    main()
