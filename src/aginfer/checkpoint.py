from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import ValidationError
from .schema import DType, REJECTED_DTYPES, SUPPORTED_CHECKPOINT_DTYPES

_PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
_METADATA_NAMES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
}


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
    revision: str
    config: dict[str, Any]
    tensors: tuple[TensorRecord, ...]
    safetensor_files: tuple[Path, ...]

    @property
    def dtypes(self) -> frozenset[str]:
        return frozenset(tensor.dtype for tensor in self.tensors)


def resolve_source(source: str, revision: str | None, offline: bool) -> tuple[Path, str]:
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve(), revision or _local_revision(local)
    if offline:
        raise ValidationError("offline mode only accepts an existing local Hugging Face snapshot")
    if not revision or revision in {"main", "master", "latest"}:
        raise ValidationError("remote Hugging Face sources require an immutable --revision commit hash")
    if len(revision) < 7 or any(character not in "0123456789abcdefABCDEF" for character in revision):
        raise ValidationError("--revision must be an immutable hexadecimal commit hash")
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
        raise ValidationError(f"failed to download immutable snapshot {source}@{revision}: {exc}") from exc
    return Path(downloaded), revision.lower()


def load_checkpoint(source: str, revision: str | None = None, offline: bool = False) -> Checkpoint:
    root, resolved_revision = resolve_source(source, revision, offline)
    unsafe = sorted(path.relative_to(root) for path in root.rglob("*") if path.suffix.lower() in _PICKLE_SUFFIXES)
    if unsafe:
        names = ", ".join(str(path) for path in unsafe[:5])
        raise ValidationError(f"unsafe pickle checkpoint files are forbidden: {names}")
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


def pack_weights(tensors: Iterable[TensorRecord], alignment: int = 256) -> tuple[bytes, list[dict[str, Any]]]:
    output = bytearray()
    table: list[dict[str, Any]] = []
    for tensor in sorted(tensors, key=lambda item: item.name):
        output.extend(b"\0" * ((-len(output)) % alignment))
        offset = len(output)
        with tensor.source.open("rb") as stream:
            stream.seek(tensor.source_offset)
            content = stream.read(tensor.byte_length)
        if len(content) != tensor.byte_length:
            raise ValidationError(f"truncated tensor data while reading {tensor.name}")
        output.extend(content)
        table.append(
            {
                "name": tensor.name,
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "stride": _contiguous_stride(tensor.shape),
                "layout": "checkpoint_contiguous",
                "blob_offset": offset,
                "byte_length": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "quantization": {"format": "e4m3", "scale": "explicit"}
                if tensor.dtype == DType.FP8_E4M3.value
                else None,
            }
        )
    return bytes(output), table


def collect_metadata(root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in sorted(_METADATA_NAMES):
        path = root / name
        if path.is_file():
            metadata[name] = _load_json(path)
    for path in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
        metadata[path.name] = path.read_text(encoding="utf-8")
    return metadata


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse {path}: {exc}") from exc


def _local_revision(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return f"local-sha256:{digest.hexdigest()}"


def _element_count(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _dtype_size(dtype: str) -> int:
    return {DType.FP32.value: 4, DType.FP16.value: 2, DType.BF16.value: 2, DType.FP8_E4M3.value: 1}[dtype]


def _contiguous_stride(shape: tuple[int, ...]) -> list[int]:
    stride: list[int] = []
    value = 1
    for dimension in reversed(shape):
        stride.append(value)
        value *= dimension
    return list(reversed(stride))


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
