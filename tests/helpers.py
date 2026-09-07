from __future__ import annotations

import json
import struct
from pathlib import Path

from aginfer.aim import Compatibility, ProviderRequirement


def fake_cubin(arch: str) -> bytes:
    number = int(arch.removeprefix("sm"))
    header = bytearray(64)
    header[:16] = b"\x7fELF\x02\x01\x01\x33\x07" + b"\0" * 7
    struct.pack_into("<H", header, 16, 2)
    struct.pack_into("<H", header, 18, 190)
    struct.pack_into("<I", header, 20, 124)
    flags = (number << 8) | 2 if number >= 100 else number | (number << 16)
    struct.pack_into("<I", header, 48, flags)
    struct.pack_into("<H", header, 52, 64)
    return bytes(header)


def runtime_compatibility() -> Compatibility:
    return Compatibility(
        cuda_driver_min=12000,
        cuda_runtime_min=12000,
        cuda_runtime_max=12999,
        providers=(
            ProviderRequirement(provider_id=1, abi_min=12, abi_max=12),
            ProviderRequirement(provider_id=2, abi_min=9, abi_max=9),
        ),
    )


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
