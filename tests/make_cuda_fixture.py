from __future__ import annotations

import sys
from pathlib import Path

from aginfer.aim import AimWriter, Compatibility, VariantPayload
from aginfer.plan import compile_execution_plan
from aginfer.schema import CudaArch, Platform


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: make_cuda_fixture.py <arch> <kernels.cubin> <output.aim>", file=sys.stderr)
        return 2
    arch = CudaArch.parse(sys.argv[1])
    cubin_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    plan = compile_execution_plan(
        {
            "cuda_arch": arch.name_string,
            "arena_bytes": 0,
            "workspace_bytes": 0,
            "shape_dispatch": [
                {
                    "profile": "vector16",
                    "inputs": [
                        {"id": 0, "name": "left", "dtype": "F32", "shape": [16]},
                        {"id": 1, "name": "right", "dtype": "F32", "shape": [16]},
                    ],
                    "outputs": [{"id": 2, "name": "output", "dtype": "F32", "shape": [16]}],
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
        arch,
        weight_size=1,
    )
    manifest = {
        "model_family": "runtime_validation",
        "runtime_abi": 1,
    }
    AimWriter.write(
        output_path,
        platform=Platform.LINUX_X86_64_GNU,
        manifest=manifest,
        graph={"opset": 1, "subgraphs": ["vector_add_f32"]},
        tensors={"count": 0, "items": []},
        compatibility=Compatibility(cuda_driver_min=12000),
        variants=[VariantPayload(arch, cubin_path.read_bytes(), b"\0", plan.data)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
