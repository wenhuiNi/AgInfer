from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from .checkpoint import TensorRecord, inspect_safetensors, reject_unsafe_checkpoint_files
from .errors import ValidationError
from .schema import SUPPORTED_CHECKPOINT_DTYPES


class AssetRole(str, Enum):
    MODEL_SHARD = "model_shard"
    PREPROCESSOR_STATE = "preprocessor_state"
    POSTPROCESSOR_STATE = "postprocessor_state"
    TOKENIZER = "tokenizer"
    METADATA = "metadata"
    EXTERNAL_BACKBONE = "external_backbone"


@dataclass(slots=True)
class SourceAsset:
    asset_id: str
    role: str
    namespace: str
    path: str
    byte_size: int
    required: bool = True
    shard_set: str | None = None
    index_path: str | None = None


@dataclass(slots=True)
class TensorSource:
    namespace: str
    shard_set: str
    name: str
    asset_id: str
    dtype: str
    shape: list[int]
    source_offset: int
    byte_length: int


@dataclass(slots=True)
class SourceManifest:
    source: str
    revision: str | None
    assets: list[SourceAsset]
    tensors: list[TensorSource]
    version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "revision": self.revision,
            "assets": [asdict(asset) for asset in self.assets],
            "tensors": [asdict(tensor) for tensor in self.tensors],
        }

    def write(self, path: str | Path) -> None:
        _validate_manifest(self)
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "SourceManifest":
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"cannot parse source manifest {source}: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Any) -> "SourceManifest":
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("version"), int)
            or isinstance(data.get("version"), bool)
            or data.get("version") != 0
        ):
            raise ValidationError("source manifest must be a version 0 object")
        if not isinstance(data.get("source"), str) or not data["source"]:
            raise ValidationError("source manifest source must be a non-empty string")
        revision = data.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise ValidationError("source manifest revision must be a string or null")
        assets_data = data.get("assets")
        tensors_data = data.get("tensors")
        if not isinstance(assets_data, list) or not isinstance(tensors_data, list):
            raise ValidationError("source manifest assets and tensors must be arrays")
        try:
            assets = [SourceAsset(**item) for item in assets_data]
            tensors = [TensorSource(**item) for item in tensors_data]
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"invalid source manifest entry: {exc}") from exc
        manifest = cls(data["source"], revision, assets, tensors)
        _validate_manifest(manifest)
        return manifest


def build_source_manifest(root: str | Path, *, revision: str | None = None) -> SourceManifest:
    """Describe a local checkpoint without reading tensor payload bytes."""

    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise ValidationError(f"source manifest root is not a directory: {source_root}")
    reject_unsafe_checkpoint_files(source_root)
    safetensor_paths = tuple(sorted(source_root.rglob("*.safetensors")))
    if not safetensor_paths:
        raise ValidationError(f"no safetensors files found in {source_root}")

    index_paths = tuple(sorted(source_root.rglob("*.safetensors.index.json")))
    indexed: dict[Path, tuple[str, str]] = {}
    index_maps: dict[Path, dict[str, str]] = {}
    for index_path in index_paths:
        weight_map = _load_weight_map(source_root, index_path)
        index_relative = index_path.relative_to(source_root).as_posix()
        shard_set = index_relative
        namespace = _namespace_for_path(index_path)
        declared_paths: set[Path] = set()
        for tensor_name, shard_name in weight_map.items():
            shard_path = _resolve_index_shard(source_root, index_path, shard_name)
            if not shard_path.is_file():
                raise ValidationError(f"safetensors index {index_relative} references missing shard: {shard_name}")
            declared_paths.add(shard_path)
            previous = indexed.get(shard_path)
            if previous is not None and previous != (namespace, shard_set):
                raise ValidationError(f"safetensors shard is declared by multiple indexes: {shard_name}")
            indexed[shard_path] = (namespace, shard_set)
        _reject_extra_index_shards(source_root, index_path, declared_paths)
        index_maps[index_path] = weight_map

    assets: list[SourceAsset] = []
    tensors: list[TensorSource] = []
    records_by_path: dict[Path, tuple[TensorRecord, ...]] = {}
    for safetensor_path in safetensor_paths:
        relative = safetensor_path.relative_to(source_root).as_posix()
        namespace, shard_set = indexed.get(
            safetensor_path,
            (_namespace_for_path(safetensor_path), relative),
        )
        role = _state_role_for_namespace(namespace)
        asset_id = f"{namespace}:{relative}"
        index_path = shard_set if safetensor_path in indexed else None
        assets.append(
            SourceAsset(
                asset_id=asset_id,
                role=role.value,
                namespace=namespace,
                path=relative,
                byte_size=safetensor_path.stat().st_size,
                shard_set=shard_set,
                index_path=index_path,
            )
        )
        records = inspect_safetensors(safetensor_path)
        records_by_path[safetensor_path] = records
        for record in records:
            tensors.append(
                TensorSource(
                    namespace=namespace,
                    shard_set=shard_set,
                    name=record.name,
                    asset_id=asset_id,
                    dtype=record.dtype,
                    shape=list(record.shape),
                    source_offset=record.source_offset,
                    byte_length=record.byte_length,
                )
            )

    for index_path, weight_map in index_maps.items():
        _validate_weight_map(source_root, index_path, weight_map, records_by_path)

    tensor_paths = set(safetensor_paths)
    for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
        if path in tensor_paths or path.suffix.lower() not in {".json", ".yaml", ".yml", ".model", ".txt"}:
            continue
        relative = path.relative_to(source_root).as_posix()
        role, namespace = _metadata_role_and_namespace(path)
        assets.append(
            SourceAsset(
                asset_id=f"{namespace}:{relative}",
                role=role.value,
                namespace=namespace,
                path=relative,
                byte_size=path.stat().st_size,
            )
        )

    manifest = SourceManifest(
        source=str(source_root),
        revision=revision,
        assets=sorted(assets, key=lambda item: item.asset_id),
        tensors=sorted(tensors, key=lambda item: (item.namespace, item.shard_set, item.name)),
    )
    _validate_manifest(manifest)
    return manifest


def _load_weight_map(root: Path, index_path: Path) -> dict[str, str]:
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse safetensors index {index_path.relative_to(root)}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("weight_map"), dict):
        raise ValidationError(f"safetensors index has no weight_map object: {index_path.relative_to(root)}")
    result: dict[str, str] = {}
    for tensor_name, shard_name in data["weight_map"].items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValidationError(f"safetensors index contains an invalid tensor name: {index_path.relative_to(root)}")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValidationError(f"safetensors index contains an invalid shard name: {index_path.relative_to(root)}")
        result[tensor_name] = shard_name
    return result


def _resolve_index_shard(root: Path, index_path: Path, shard_name: str) -> Path:
    logical = PurePosixPath(shard_name)
    if logical.is_absolute() or ".." in logical.parts or "\\" in shard_name:
        raise ValidationError(f"unsafe shard path in {index_path.relative_to(root)}: {shard_name}")
    candidate = (index_path.parent / Path(*logical.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValidationError(f"shard path escapes source root in {index_path.relative_to(root)}: {shard_name}") from exc
    return candidate


def _reject_extra_index_shards(root: Path, index_path: Path, declared: set[Path]) -> None:
    base = index_path.name.removesuffix(".safetensors.index.json")
    candidates = set(index_path.parent.glob(f"{base}-*.safetensors"))
    exact = index_path.parent / f"{base}.safetensors"
    if exact.is_file():
        candidates.add(exact)
    extras = sorted(path.relative_to(root).as_posix() for path in candidates - declared)
    if extras:
        raise ValidationError(
            f"safetensors index {index_path.relative_to(root)} has undeclared matching shards: {', '.join(extras)}"
        )


def _validate_weight_map(
    root: Path,
    index_path: Path,
    weight_map: dict[str, str],
    records_by_path: dict[Path, tuple[TensorRecord, ...]],
) -> None:
    actual: dict[str, str] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = _resolve_index_shard(root, index_path, shard_name)
        for record in records_by_path[shard_path]:
            if record.name in actual:
                raise ValidationError(
                    f"duplicate tensor in logical shard set {index_path.relative_to(root)}: {record.name}"
                )
            actual[record.name] = shard_name
    missing = sorted(set(weight_map) - set(actual))
    extra = sorted(set(actual) - set(weight_map))
    wrong = sorted(name for name in set(actual) & set(weight_map) if actual[name] != weight_map[name])
    if missing or extra or wrong:
        details = []
        if missing:
            details.append(f"missing tensors={missing[:5]}")
        if extra:
            details.append(f"undeclared tensors={extra[:5]}")
        if wrong:
            details.append(f"wrong shard={wrong[:5]}")
        raise ValidationError(f"invalid weight_map in {index_path.relative_to(root)}: {'; '.join(details)}")


def _validate_manifest(manifest: SourceManifest) -> None:
    if not isinstance(manifest.version, int) or isinstance(manifest.version, bool) or manifest.version != 0:
        raise ValidationError("source manifest version must be integer 0")
    asset_ids: set[str] = set()
    asset_paths: set[str] = set()
    assets_by_id: dict[str, SourceAsset] = {}
    for asset in manifest.assets:
        if not isinstance(asset.asset_id, str) or not asset.asset_id or asset.asset_id in asset_ids:
            raise ValidationError(f"duplicate or empty source asset ID: {asset.asset_id}")
        asset_ids.add(asset.asset_id)
        assets_by_id[asset.asset_id] = asset
        if not isinstance(asset.namespace, str) or not asset.namespace:
            raise ValidationError(f"source asset namespace must be non-empty: {asset.asset_id}")
        if not isinstance(asset.role, str):
            raise ValidationError(f"source asset role must be a string: {asset.asset_id}")
        if asset.role not in {role.value for role in AssetRole}:
            raise ValidationError(f"unknown source asset role: {asset.role}")
        _validate_relative_path(asset.path)
        if asset.path in asset_paths:
            raise ValidationError(f"source asset path is declared more than once: {asset.path}")
        asset_paths.add(asset.path)
        if asset.shard_set is not None:
            _validate_relative_path(asset.shard_set)
        if asset.index_path is not None:
            _validate_relative_path(asset.index_path)
        if not isinstance(asset.byte_size, int) or isinstance(asset.byte_size, bool) or asset.byte_size < 0:
            raise ValidationError(f"negative source asset size: {asset.asset_id}")
        if not isinstance(asset.required, bool):
            raise ValidationError(f"source asset required flag must be boolean: {asset.asset_id}")
    tensor_keys: set[tuple[str, str]] = set()
    tensor_roles = {
        AssetRole.MODEL_SHARD.value,
        AssetRole.PREPROCESSOR_STATE.value,
        AssetRole.POSTPROCESSOR_STATE.value,
        AssetRole.EXTERNAL_BACKBONE.value,
    }
    for tensor in manifest.tensors:
        key = (tensor.namespace, tensor.name)
        if key in tensor_keys:
            raise ValidationError(f"duplicate tensor in source manifest: {key}")
        tensor_keys.add(key)
        if not all(isinstance(value, str) and value for value in (tensor.namespace, tensor.shard_set, tensor.name)):
            raise ValidationError("source tensor namespace, shard set, and name must be non-empty")
        if not isinstance(tensor.asset_id, str) or not tensor.asset_id:
            raise ValidationError("source tensor asset ID must be a non-empty string")
        _validate_relative_path(tensor.shard_set)
        if not isinstance(tensor.dtype, str) or tensor.dtype not in SUPPORTED_CHECKPOINT_DTYPES:
            raise ValidationError(f"unsupported source tensor dtype: {tensor.dtype}")
        asset = assets_by_id.get(tensor.asset_id)
        if asset is None:
            raise ValidationError(f"tensor references unknown source asset: {tensor.asset_id}")
        if asset.role not in tensor_roles:
            raise ValidationError(f"tensor references a non-tensor source asset: {tensor.asset_id}")
        if asset.namespace != tensor.namespace or asset.shard_set != tensor.shard_set:
            raise ValidationError(f"tensor namespace or shard set disagrees with asset: {tensor.name}")
        if not isinstance(tensor.shape, list) or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
            for dimension in tensor.shape
        ):
            raise ValidationError(f"invalid tensor shape in source manifest: {tensor.name}")
        if (
            not isinstance(tensor.source_offset, int)
            or isinstance(tensor.source_offset, bool)
            or not isinstance(tensor.byte_length, int)
            or isinstance(tensor.byte_length, bool)
            or tensor.source_offset < 0
            or tensor.byte_length <= 0
        ):
            raise ValidationError(f"invalid tensor byte range in source manifest: {tensor.name}")
        if tensor.source_offset + tensor.byte_length > asset.byte_size:
            raise ValidationError(f"tensor byte range exceeds source asset: {tensor.name}")


def _validate_relative_path(path: str) -> None:
    if not isinstance(path, str):
        raise ValidationError(f"source asset path must be a string: {path!r}")
    logical = PurePosixPath(path)
    if not path or logical.is_absolute() or ".." in logical.parts or "\\" in path:
        raise ValidationError(f"source asset path must be safe and relative: {path}")


def _namespace_for_path(path: Path) -> str:
    value = path.as_posix().lower()
    if "preprocessor" in value:
        return "preprocessor"
    if "postprocessor" in value:
        return "postprocessor"
    if any(token in value for token in ("tokenizer", "special_tokens", "vocab", "sentencepiece")):
        return "tokenizer"
    return "model"


def _state_role_for_namespace(namespace: str) -> AssetRole:
    if namespace == "preprocessor":
        return AssetRole.PREPROCESSOR_STATE
    if namespace == "postprocessor":
        return AssetRole.POSTPROCESSOR_STATE
    return AssetRole.MODEL_SHARD


def _metadata_role_and_namespace(path: Path) -> tuple[AssetRole, str]:
    namespace = _namespace_for_path(path)
    if namespace == "tokenizer" or path.suffix.lower() in {".model", ".txt"}:
        return AssetRole.TOKENIZER, "tokenizer"
    return AssetRole.METADATA, namespace
