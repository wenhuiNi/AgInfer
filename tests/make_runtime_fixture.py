from __future__ import annotations

import json
import sys
import struct
from pathlib import Path

from aginfer.aim import AimWriter, VariantPayload
from aginfer.plan import compile_execution_plan
from aginfer.schema import CudaArch, Platform


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    def plan(arch: CudaArch) -> bytes:
        return compile_execution_plan(
            {
                "cuda_arch": arch.name_string,
                "arena_bytes": 4096,
                "workspace_bytes": 2048,
                "shape_dispatch": [
                    {
                        "profile": "default",
                        "inputs": [
                            {"name": "input_ids", "dtype": "I32", "shape": [1]},
                            {"name": "pixel_values", "dtype": "F16", "shape": [1]},
                            {"name": "state", "dtype": "F16", "shape": [1]},
                        ],
                        "outputs": [{"name": "actions", "dtype": "F16", "shape": [1]}],
                        "launches": [
                            {"kernel": "noop", "grid": [1, 1, 1], "block": [1, 1, 1], "arguments": []}
                        ],
                    }
                ],
                "cuda_graph_templates": [{"profile": "default"}],
            },
            arch,
            weight_size=7,
        ).data
    manifest = {
        "runtime_abi": 1,
        "toolchain": {
            "cuda_driver_min": 12000,
            "cuda_runtime_min": 12000,
            "cuda_runtime_max": 12999,
            "cublaslt_abi": 12,
            "cudnn_abi": 9,
        },
    }
    def cubin(arch: int) -> bytes:
        header = bytearray(64)
        header[:16] = b"\x7fELF\x02\x01\x01\x33\x07" + b"\0" * 7
        struct.pack_into("<HHI", header, 16, 2, 190, 124)
        struct.pack_into("<I", header, 48, arch | (arch << 16))
        struct.pack_into("<H", header, 52, 64)
        return bytes(header)

    AimWriter.write(
        Path(sys.argv[1]),
        platform=Platform.LINUX_X86_64_GNU,
        manifest=manifest,
        graph={"opset": 1},
        tensors={"count": 0, "items": []},
        variants=[
            VariantPayload(CudaArch.SM89, cubin(89), b"weights", plan(CudaArch.SM89)),
            VariantPayload(CudaArch.SM120, cubin(120), b"weights", plan(CudaArch.SM120)),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
