from __future__ import annotations

import sys
from pathlib import Path

from aginfer.providers import (
    CublasLtAlgorithm,
    CublasLtDType,
    CublasLtLinearPayload,
    CublasLtLinearProblem,
)
from aginfer.schema import CudaArch


def main() -> None:
    payload = CublasLtLinearPayload(
        problem=CublasLtLinearProblem(
            target_arch=CudaArch.SM120,
            dtype=CublasLtDType.BF16,
            m=50,
            n=256,
            k=1024,
        ),
        cublaslt_version=120805,
        algorithm=CublasLtAlgorithm(
            algorithm_id=23,
            tile_id=20,
            split_k=1,
            reduction_scheme=0,
            cta_swizzling=0,
            custom_option=0,
            stages_id=14,
            inner_shape_id=0,
            cluster_shape_id=0,
        ),
        workspace_bytes=4 * 1024 * 1024,
        x_alignment=256,
        weight_alignment=256,
        bias_alignment=256,
        output_alignment=256,
        workspace_alignment=256,
    )
    Path(sys.argv[1]).write_bytes(payload.to_bytes())


if __name__ == "__main__":
    main()
