from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .errors import ValidationError
from .safetensors import SafetensorsReader
from .source_manifest import SourceManifest, TensorSource


@dataclass(frozen=True, order=True, slots=True)
class ConstantKey:
    """A logical tensor address independent of files and shard layout."""

    namespace: str
    name: str


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    key: ConstantKey
    disposition: str
    detail: str


class ConstantStore:
    """Namespace-aware, bounded access to manifest-declared safetensors.

    Construction checks asset presence and exact file sizes but never reads
    tensor payload bytes. Each payload access revalidates the safetensors
    record against the manifest before exposing a bounded copy or mmap view.
    """

    def __init__(self, manifest: SourceManifest, *, root: str | Path | None = None) -> None:
        self.manifest = SourceManifest.from_dict(manifest.to_dict())
        selected_root = self.manifest.source if root is None else root
        self.root = Path(selected_root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValidationError(f"constant store root is not a directory: {self.root}")

        self._paths: dict[str, Path] = {}
        for asset in self.manifest.assets:
            path = (self.root / asset.path).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise ValidationError(f"source asset escapes constant store root: {asset.path}") from exc
            if not path.is_file():
                if asset.required:
                    raise ValidationError(f"required source asset is missing: {asset.path}")
                continue
            actual_size = path.stat().st_size
            if actual_size != asset.byte_size:
                raise ValidationError(
                    f"source asset size changed for {asset.path}: manifest={asset.byte_size}, actual={actual_size}"
                )
            self._paths[asset.asset_id] = path

        self._tensors: dict[ConstantKey, TensorSource] = {}
        for tensor in self.manifest.tensors:
            key = ConstantKey(tensor.namespace, tensor.name)
            if key in self._tensors:
                raise ValidationError(f"duplicate logical constant: {key.namespace}:{key.name}")
            if tensor.asset_id not in self._paths:
                raise ValidationError(f"tensor source asset is unavailable: {tensor.asset_id}")
            self._tensors[key] = tensor

    @property
    def keys(self) -> tuple[ConstantKey, ...]:
        return tuple(sorted(self._tensors))

    @property
    def namespaces(self) -> tuple[str, ...]:
        return tuple(sorted({key.namespace for key in self._tensors}))

    def tensor(self, namespace: str, name: str) -> TensorSource:
        key = ConstantKey(namespace, name)
        try:
            return self._tensors[key]
        except KeyError as exc:
            raise ValidationError(f"constant not found: {namespace}:{name}") from exc

    def readinto(self, namespace: str, name: str, destination: object, *, offset: int = 0) -> int:
        tensor, path = self._resolve(namespace, name)
        with SafetensorsReader(path) as reader:
            self._check_record(reader, tensor)
            return reader.readinto(name, destination, offset=offset)

    def iter_chunks(
        self,
        namespace: str,
        name: str,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        tensor, path = self._resolve(namespace, name)
        with SafetensorsReader(path) as reader:
            self._check_record(reader, tensor)
            yield from reader.iter_chunks(name, chunk_size=chunk_size)

    @contextmanager
    def mmap_slice(
        self,
        namespace: str,
        name: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> Iterator[memoryview]:
        tensor, path = self._resolve(namespace, name)
        with SafetensorsReader(path) as reader:
            self._check_record(reader, tensor)
            with reader.mmap_slice(name, offset=offset, length=length) as view:
                yield view

    def _resolve(self, namespace: str, name: str) -> tuple[TensorSource, Path]:
        tensor = self.tensor(namespace, name)
        return tensor, self._paths[tensor.asset_id]

    @staticmethod
    def _check_record(reader: SafetensorsReader, tensor: TensorSource) -> None:
        record = reader.tensor(tensor.name)
        actual = (record.dtype, record.shape, record.source_offset, record.byte_length)
        expected = (tensor.dtype, tuple(tensor.shape), tensor.source_offset, tensor.byte_length)
        if actual != expected:
            raise ValidationError(f"safetensors metadata changed for {tensor.namespace}:{tensor.name}")


class ConstantCoverage:
    """Fail-closed accounting for recipe consumption of source constants."""

    def __init__(self, store: ConstantStore) -> None:
        self._store = store
        self._expected = frozenset(store.keys)
        self._records: dict[ConstantKey, CoverageRecord] = {}

    @property
    def store(self) -> ConstantStore:
        return self._store

    @property
    def unresolved(self) -> tuple[ConstantKey, ...]:
        return tuple(sorted(self._expected - self._records.keys()))

    @property
    def records(self) -> tuple[CoverageRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def consume(self, namespace: str, name: str, *, consumer: str) -> None:
        self._resolve(ConstantKey(namespace, name), "consumed", consumer)

    def ignore(self, namespace: str, name: str, *, reason: str) -> None:
        self._resolve(ConstantKey(namespace, name), "ignored", reason)

    def require_complete(self) -> None:
        missing = self.unresolved
        if missing:
            preview = ", ".join(f"{key.namespace}:{key.name}" for key in missing[:5])
            raise ValidationError(f"unresolved source constants ({len(missing)}): {preview}")

    def _resolve(self, key: ConstantKey, disposition: str, detail: str) -> None:
        if key not in self._expected:
            raise ValidationError(f"coverage references unknown constant: {key.namespace}:{key.name}")
        if key in self._records:
            raise ValidationError(f"source constant already resolved: {key.namespace}:{key.name}")
        if not isinstance(detail, str) or not detail.strip():
            raise ValidationError(f"{disposition} source constant requires a non-empty detail")
        self._records[key] = CoverageRecord(key, disposition, detail)
