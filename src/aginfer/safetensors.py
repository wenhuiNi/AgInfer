from __future__ import annotations

import mmap
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from .checkpoint import TensorRecord, inspect_safetensors
from .errors import ValidationError


class SafetensorsReader:
    """Bounded, zero-deserialization access to one safetensors file.

    Metadata is validated when the reader is created. Payload bytes are exposed
    only while the reader context is open, either through a caller-owned buffer,
    bounded chunks, or a scoped read-only mmap view.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records = inspect_safetensors(self.path)
        self._records_by_name = {record.name: record for record in self.records}
        self._stream: BinaryIO | None = None
        self._mapping: mmap.mmap | None = None

    def __enter__(self) -> "SafetensorsReader":
        if self._mapping is not None:
            raise ValidationError(f"safetensors reader is already open: {self.path}")
        try:
            self._stream = self.path.open("rb")
            self._mapping = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as exc:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            raise ValidationError(f"cannot mmap safetensors file {self.path}: {exc}") from exc
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self.records)

    def tensor(self, name: str) -> TensorRecord:
        try:
            return self._records_by_name[name]
        except KeyError as exc:
            raise ValidationError(f"tensor not found in {self.path}: {name}") from exc

    def readinto(self, name: str, destination: object, *, offset: int = 0) -> int:
        """Copy at most ``len(destination)`` bytes from one tensor.

        Returns the number of bytes copied. ``offset`` is relative to the start
        of the tensor and may equal its byte length, in which case zero is
        returned.
        """

        mapping = self._require_open()
        record = self.tensor(name)
        if offset < 0 or offset > record.byte_length:
            raise ValidationError(f"tensor byte offset is out of range for {name}: {offset}")
        try:
            target = memoryview(destination)
            if target.readonly:
                raise ValidationError("readinto destination must be writable")
            target_bytes = target.cast("B")
        except (TypeError, ValueError) as exc:
            raise ValidationError("readinto destination must be a contiguous writable buffer") from exc

        count = min(target_bytes.nbytes, record.byte_length - offset)
        if count:
            start = record.source_offset + offset
            source = memoryview(mapping)[start : start + count]
            try:
                target_bytes[:count] = source
            finally:
                source.release()
        return count

    def iter_chunks(self, name: str, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        """Yield one tensor as copies bounded by ``chunk_size`` bytes."""

        mapping = self._require_open()
        if chunk_size <= 0:
            raise ValidationError("chunk_size must be positive")
        record = self.tensor(name)
        position = 0
        while position < record.byte_length:
            count = min(chunk_size, record.byte_length - position)
            start = record.source_offset + position
            yield mapping[start : start + count]
            position += count

    @contextmanager
    def mmap_slice(
        self,
        name: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> Iterator[memoryview]:
        """Expose a scoped read-only mmap view for one tensor range."""

        mapping = self._require_open()
        record = self.tensor(name)
        if offset < 0 or offset > record.byte_length:
            raise ValidationError(f"tensor byte offset is out of range for {name}: {offset}")
        available = record.byte_length - offset
        selected_length = available if length is None else length
        if selected_length < 0 or selected_length > available:
            raise ValidationError(f"tensor byte length is out of range for {name}: {selected_length}")
        start = record.source_offset + offset
        view = memoryview(mapping)[start : start + selected_length]
        try:
            yield view
        finally:
            view.release()

    def close(self) -> None:
        if self._mapping is not None:
            try:
                self._mapping.close()
            except BufferError as exc:
                raise ValidationError(
                    f"cannot close safetensors reader while an mmap view is still exported: {self.path}"
                ) from exc
            finally:
                self._mapping = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _require_open(self) -> mmap.mmap:
        if self._mapping is None:
            raise ValidationError("safetensors payload access requires an open reader context")
        return self._mapping
