from __future__ import annotations

import sys
from pathlib import Path

from aginfer.providers import (
    AotCastProblem,
    CudaKernelDType,
    CudaKernelId,
    CudaKernelPayload,
)
from aginfer.schema import CudaArch


problem = AotCastProblem(
    CudaArch.SM120,
    CudaKernelId.CAST_BF16_TO_F32,
    CudaKernelDType.BF16,
    CudaKernelDType.F32,
    51_200,
)
payload = CudaKernelPayload.for_problem(
    problem,
    module_bytes=12_296,
    module_sha256="1" * 64,
)
Path(sys.argv[1]).write_bytes(payload.to_bytes())
