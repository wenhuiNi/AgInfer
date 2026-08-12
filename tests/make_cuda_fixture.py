from __future__ import annotations

import sys
from pathlib import Path

from aginfer.aim import AimWriter, VariantPayload
from aginfer.plan import compile_execution_plan
from aginfer.schema import CudaArch, Platform


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: make_cuda_fixture.py <sm89.cubin> <output.aim>", file=sys.stderr)
        return 2
    cubin_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    plan = compile_execution_plan(
        {
            "cuda_arch": "sm89",
            "arena_bytes": 0,
            "workspace_bytes": 0,
            "shape_dispatch": [
                {
                    "profile": "vector16",
                    "inputs": [
                        {"name": "left", "dtype": "F32", "shape": [16]},
                        {"name": "right", "dtype": "F32", "shape": [16]},
                    ],
                    "outputs": [{"name": "output", "dtype": "F32", "shape": [16]}],
                    "launches": [
                        {
                            "kernel": "vector_add_f32",
                            "grid": [1, 1, 1],
                            "block": [16, 1, 1],
                            "arguments": [
                                {"tensor": "left"},
                                {"tensor": "right"},
                                {"tensor": "output"},
                                {"scalar_u32": 16},
                            ],
                        }
                    ],
                }
            ],
            "cuda_graph_templates": [],
        },
        CudaArch.SM89,
        weight_size=1,
    )
    manifest = {
        "model_family": "runtime_validation",
        "runtime_abi": 1,
        "toolchain": {
            "cuda_driver_min": 12000,
            "cuda_runtime_min": 12000,
            "cuda_runtime_max": 12999,
            "cublaslt_abi": 12,
            "cudnn_abi": 9,
        },
    }
    AimWriter.write(
        output_path,
        platform=Platform.LINUX_X86_64_GNU,
        manifest=manifest,
        graph={"opset": 1, "subgraphs": ["vector_add_f32"]},
        tensors={"count": 0, "items": []},
        variants=[VariantPayload(CudaArch.SM89, cubin_path.read_bytes(), b"\0", plan.data)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

