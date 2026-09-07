from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .schema import DType, REJECTED_DTYPES, SUPPORTED_CHECKPOINT_DTYPES

_PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}


def reject_unsafe_checkpoint_files(root: Path) -> None:
    unsafe = sorted(path.relative_to(root) for path in root.rglob("*") if path.suffix.lower() in _PICKLE_SUFFIXES)
    if unsafe:
        names = ", ".join(str(path) for path in unsafe[:5])
        raise ValidationError(f"unsafe pickle checkpoint files are forbidden: {names}")


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source: Path
    source_offset: int
    byte_length: int


@dataclass(frozen=True)
class Checkpoint:
    root: Path
    revision: str | None
    config: dict[str, Any]
    tensors: tuple[TensorRecord, ...]
    safetensor_files: tuple[Path, ...]

    @property
    def dtypes(self) -> frozenset[str]:
        return frozenset(tensor.dtype for tensor in self.tensors)


def resolve_source(source: str, revision: str | None, offline: bool) -> tuple[Path, str | None]:
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve(), revision
    if offline:
        raise ValidationError("offline mode only accepts an existing local Hugging Face snapshot")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValidationError("remote source support requires: pip install aginfer[hub]") from exc
    try:
        downloaded = snapshot_download(
            repo_id=source,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.yml", "*.model", "*.txt"],
        )
    except Exception as exc:  # third-party client has a broad exception surface
        location = f"{source}@{revision}" if revision else source
        raise ValidationError(f"failed to download snapshot {location}: {exc}") from exc
    return Path(downloaded), revision


def load_checkpoint(source: str, revision: str | None = None, offline: bool = False) -> Checkpoint:
    root, resolved_revision = resolve_source(source, revision, offline)
    reject_unsafe_checkpoint_files(root)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValidationError(f"missing Hugging Face config.json in {root}")
    config = _load_json(config_path)
    if not isinstance(config, dict):
        raise ValidationError("config.json must contain an object")
    files = tuple(sorted(root.rglob("*.safetensors")))
    if not files:
        raise ValidationError(f"no safetensors files found in {root}")
    records: list[TensorRecord] = []
    seen: set[str] = set()
    for path in files:
        for record in inspect_safetensors(path):
            if record.name in seen:
                raise ValidationError(f"duplicate tensor name across checkpoint shards: {record.name}")
            seen.add(record.name)
            records.append(record)
    if not records:
        raise ValidationError("checkpoint contains no tensors")
    _validate_fp8_config(config, records)
    return Checkpoint(root, resolved_revision, config, tuple(sorted(records, key=lambda item: item.name)), files)


def inspect_safetensors(path: Path) -> tuple[TensorRecord, ...]:
    size = path.stat().st_size
    if size < 8:
        raise ValidationError(f"truncated safetensors file: {path}")
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size <= 1 or header_size > 100 * 1024 * 1024 or header_size > size - 8:
            raise ValidationError(f"invalid safetensors header size in {path}")
        try:
            header = json.loads(stream.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid safetensors JSON header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValidationError(f"safetensors header must be an object: {path}")
    data_start = 8 + header_size
    data_size = size - data_start
    spans: list[tuple[int, int, str]] = []
    records: list[TensorRecord] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValidationError(f"invalid tensor entry in {path}")
        try:
            dtype = str(entry["dtype"])
            shape = tuple(int(item) for item in entry["shape"])
            begin, end = (int(item) for item in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"invalid metadata for tensor {name} in {path}") from exc
        if dtype in REJECTED_DTYPES or dtype not in SUPPORTED_CHECKPOINT_DTYPES:
            raise ValidationError(f"tensor {name} uses unsupported dtype {dtype}")
        if any(dimension < 0 for dimension in shape):
            raise ValidationError(f"tensor {name} has a negative dimension")
        if begin < 0 or end <= begin or end > data_size:
            raise ValidationError(f"tensor {name} has invalid data offsets")
        expected = _element_count(shape) * _dtype_size(dtype)
        if end - begin != expected:
            raise ValidationError(
                f"tensor {name} byte length mismatch: metadata={end - begin}, expected={expected}"
            )
        spans.append((begin, end, name))
        records.append(TensorRecord(name, dtype, shape, path, data_start + begin, end - begin))
    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise ValidationError(f"overlapping tensor data in {path}: {previous[2]} and {current[2]}")
    return tuple(records)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse {path}: {exc}") from exc


def _element_count(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _dtype_size(dtype: str) -> int:
    return {DType.FP32.value: 4, DType.FP16.value: 2, DType.BF16.value: 2, DType.FP8_E4M3.value: 1}[dtype]


def _validate_fp8_config(config: dict[str, Any], records: list[TensorRecord]) -> None:
    if not any(record.dtype == DType.FP8_E4M3.value for record in records):
        return
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValidationError("FP8 tensors require a ModelOpt Unified HF quantization_config")
    method = str(quantization.get("quant_method", "")).lower()
    fmt = str(quantization.get("quantization_format", quantization.get("format", ""))).lower()
    scales = str(quantization.get("scales", quantization.get("scale_mode", ""))).lower()
    if "modelopt" not in method or fmt not in {"fp8", "e4m3", "fp8_e4m3"} or scales != "explicit":
        raise ValidationError("only ModelOpt Unified HF E4M3 FP8 with explicit scales is accepted")
