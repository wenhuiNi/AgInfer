from __future__ import annotations

import pathlib
import sys

from aginfer.providers import (
    AotActivationProblem,
    CudaKernelDType,
    CudaKernelId,
    CudaKernelPayload,
)
from aginfer.schema import CudaArch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_activation_payload_fixture.py OUTPUT")
    problems = (
        AotActivationProblem(
            CudaArch.SM120,
            CudaKernelId.GELU_BF16,
            CudaKernelDType.BF16,
            204_800,
        ),
        AotActivationProblem(
            CudaArch.SM120,
            CudaKernelId.GELU_F32,
            CudaKernelDType.F32,
            1_101_824,
        ),
        AotActivationProblem(
            CudaArch.SM120,
            CudaKernelId.GELU_BF16,
            CudaKernelDType.BF16,
            15_859_712,
        ),
        AotActivationProblem(
            CudaArch.SM120,
            CudaKernelId.SILU_F32,
            CudaKernelDType.F32,
            1_024,
            "silu",
        ),
    )
    payloads = tuple(
        CudaKernelPayload.for_problem(
            problem, module_bytes=100_000, module_sha256="9" * 64
        ).to_bytes()
        for problem in problems
    )
    pathlib.Path(sys.argv[1]).write_bytes(b"".join(payloads))


if __name__ == "__main__":
    main()
