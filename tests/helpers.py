from __future__ import annotations

import json
import struct
from pathlib import Path


def fake_cubin(arch: str) -> bytes:
    number = int(arch.removeprefix("sm"))
    header = bytearray(64)
    header[:16] = b"\x7fELF\x02\x01\x01\x33\x07" + b"\0" * 7
    struct.pack_into("<H", header, 16, 2)
    struct.pack_into("<H", header, 18, 190)
    struct.pack_into("<I", header, 20, 124)
    struct.pack_into("<I", header, 48, number | (number << 16))
    struct.pack_into("<H", header, 52, 64)
    return bytes(header)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {}
    content = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        begin = len(content)
        content.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(content)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + content)


def create_checkpoint(root: Path, model_type: str = "groot") -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [f"{model_type.title()}ForConditionalGeneration"]})
    )
    write_safetensors(root / "model.safetensors", {"layer.weight": ("F16", [2, 2], bytes(range(8)))})
    return root


def create_artifacts(root: Path, arches: tuple[str, ...] = ("sm89",)) -> Path:
    root.mkdir(parents=True)
    (root / "toolchain.json").write_text(
        json.dumps(
            {
                "cuda_driver_min": 12000,
                "cuda_runtime_min": 12000,
                "cuda_runtime_max": 12999,
                "cublaslt_abi": 12,
                "cudnn_abi": 9,
            }
        )
    )
    for arch in arches:
        target = root / arch
        target.mkdir()
        (target / "kernels.cubin").write_bytes(fake_cubin(arch))
        (target / "plan.json").write_text(
            json.dumps(
                {
                    "cuda_arch": arch,
                    "arena_bytes": 4096,
                    "workspace_bytes": 2048,
                    "shape_dispatch": [{"profile": "default", "kernel": 0}],
                    "cuda_graph_templates": [{"profile": "default"}],
                }
            )
        )
        (target / "tactics.json").write_text(
            json.dumps({"cuda_arch": arch, "tactics": [{"op": "gemm", "algorithm": "verified-1"}]})
        )
    return root


def create_profile(path: Path) -> Path:
    bounds = {"min": 1, "opt": 2, "max": 4}
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "default",
                        "batch": 1,
                        **{key: bounds for key in (
                            "sequence_length", "image_height", "image_width", "image_count",
                            "state_length", "action_horizon"
                        )},
                    }
                ],
                "default_denoising": {"steps": 4},
            }
        )
    )
    return path
