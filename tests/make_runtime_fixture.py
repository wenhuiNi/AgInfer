from __future__ import annotations

import json
import sys
import struct
from pathlib import Path

from aginfer.aim import AimWriter, VariantPayload
from aginfer.schema import CudaArch, Platform


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    plan = json.dumps(
        {
            "cuda_arch": "sm89",
            "arena_bytes": 4096,
            "workspace_bytes": 2048,
            "shape_dispatch": [{"profile": "default", "kernel": 0}],
            "cuda_graph_templates": [{"profile": "default"}],
        },
        separators=(",", ":"),
    ).encode()
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
            VariantPayload(CudaArch.SM89, cubin(89), b"weights", plan),
            VariantPayload(CudaArch.SM120, cubin(120), b"weights", plan.replace(b"sm89", b"sm120")),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
