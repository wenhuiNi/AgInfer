from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass

from ..errors import ValidationError
from .inventory import (
    LoweringInventory,
    LoweringKind,
    RequirementStatus,
    dump_lowering_inventory,
)
from .schedule import (
    ExecutionSchedule,
    ValueStorage,
    dump_execution_schedule,
)


LITERAL_MATERIALIZATION_SCHEMA = "aginfer.literal-materialization.v1"
LITERAL_MATERIALIZATION_ALIGNMENT = 16
DEFAULT_MAX_MATERIALIZED_BYTES = 64 * 1024 * 1024
_DTYPE_BYTES = {"bool": 1, "i32": 4, "f32": 4, "bf16": 2}


@dataclass(frozen=True, slots=True)
class MaterializedLiteralBlob:
    blob_id: int
    offset: int
    byte_size: int
    allocation_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_id": self.blob_id,
            "offset": self.offset,
            "byte_size": self.byte_size,
            "allocation_bytes": self.allocation_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class LiteralMaterializationRecord:
    execution_index: int
    site: str
    source_value_id: int
    output_value_id: int
    source_identity: str
    dtype: str
    layout: str
    device: str
    shape: tuple[int, ...]
    broadcast_dimensions: tuple[int, ...]
    blob_id: int
    blob_offset: int
    byte_size: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index,
            "site": self.site,
            "source_value_id": self.source_value_id,
            "output_value_id": self.output_value_id,
            "source_identity": self.source_identity,
            "dtype": self.dtype,
            "layout": self.layout,
            "device": self.device,
            "shape": list(self.shape),
            "broadcast_dimensions": list(self.broadcast_dimensions),
            "blob_id": self.blob_id,
            "blob_offset": self.blob_offset,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class LiteralMaterialization:
    inventory_sha256: str
    schedule_sha256: str
    alignment: int
    data: bytes
    blobs: tuple[MaterializedLiteralBlob, ...]
    records: tuple[LiteralMaterializationRecord, ...]

    def __post_init__(self) -> None:
        _digest(self.inventory_sha256, "inventory SHA-256")
        _digest(self.schedule_sha256, "schedule SHA-256")
        if (
            not isinstance(self.alignment, int)
            or isinstance(self.alignment, bool)
            or self.alignment <= 0
            or self.alignment > 2**20
            or self.alignment & (self.alignment - 1)
        ):
            raise ValidationError(
                "literal materialization alignment must be a bounded power of two"
            )
        if not isinstance(self.data, bytes):
            raise ValidationError("literal materialization data must be immutable bytes")
        if len(self.data) > DEFAULT_MAX_MATERIALIZED_BYTES:
            raise ValidationError("literal materialization data exceeds its hard bound")
        if not isinstance(self.blobs, tuple) or not isinstance(self.records, tuple):
            raise ValidationError(
                "literal materialization blobs and records must be immutable tuples"
            )

        cursor = 0
        blobs_by_id: dict[int, MaterializedLiteralBlob] = {}
        for expected_id, blob in enumerate(self.blobs):
            if not isinstance(blob, MaterializedLiteralBlob):
                raise ValidationError("literal materialization blob is invalid")
            if (
                not isinstance(blob.blob_id, int)
                or isinstance(blob.blob_id, bool)
                or not isinstance(blob.offset, int)
                or isinstance(blob.offset, bool)
                or not isinstance(blob.byte_size, int)
                or isinstance(blob.byte_size, bool)
                or not isinstance(blob.allocation_bytes, int)
                or isinstance(blob.allocation_bytes, bool)
                or blob.blob_id != expected_id
                or blob.offset != cursor
            ):
                raise ValidationError("literal materialization blobs are not canonical")
            if blob.offset % self.alignment:
                raise ValidationError("literal materialization blob is misaligned")
            if blob.byte_size <= 0 or blob.allocation_bytes != _align(
                blob.byte_size, self.alignment
            ):
                raise ValidationError("literal materialization blob bounds are invalid")
            end = blob.offset + blob.byte_size
            allocation_end = blob.offset + blob.allocation_bytes
            if allocation_end > len(self.data):
                raise ValidationError("literal materialization blob exceeds its data")
            if hashlib.sha256(self.data[blob.offset:end]).hexdigest() != blob.sha256:
                raise ValidationError("literal materialization blob digest differs")
            if any(self.data[end:allocation_end]):
                raise ValidationError("literal materialization padding must be zero")
            _digest(blob.sha256, "blob SHA-256")
            blobs_by_id[blob.blob_id] = blob
            cursor = allocation_end
        if cursor != len(self.data):
            raise ValidationError("literal materialization has trailing data")

        previous_execution = -1
        output_values: set[int] = set()
        for record in self.records:
            if not isinstance(record, LiteralMaterializationRecord):
                raise ValidationError("literal materialization record is invalid")
            if (
                not isinstance(record.execution_index, int)
                or isinstance(record.execution_index, bool)
                or record.execution_index < 0
                or record.execution_index <= previous_execution
            ):
                raise ValidationError(
                    "literal materialization records are not in execution order"
                )
            previous_execution = record.execution_index
            if (
                not isinstance(record.site, str)
                or not record.site
                or not isinstance(record.source_value_id, int)
                or isinstance(record.source_value_id, bool)
                or record.source_value_id < 0
                or not isinstance(record.output_value_id, int)
                or isinstance(record.output_value_id, bool)
                or record.output_value_id < 0
            ):
                raise ValidationError(
                    "literal materialization record identity is invalid"
                )
            if record.output_value_id in output_values:
                raise ValidationError("literal materialization repeats an output value")
            output_values.add(record.output_value_id)
            if (
                not isinstance(record.source_identity, str)
                or not record.source_identity.startswith("literal:")
                or not record.source_identity[len("literal:") :]
            ):
                raise ValidationError("literal materialization source is not a literal")
            if (
                record.dtype not in _DTYPE_BYTES
                or record.layout != "row_major"
                or record.device != "cuda"
                or not isinstance(record.shape, tuple)
                or not record.shape
                or not all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in record.shape
                )
                or not isinstance(record.broadcast_dimensions, tuple)
                or any(
                    not isinstance(axis, int)
                    or isinstance(axis, bool)
                    or axis < 0
                    or axis >= len(record.shape)
                    for axis in record.broadcast_dimensions
                )
                or tuple(sorted(record.broadcast_dimensions))
                != record.broadcast_dimensions
                or len(set(record.broadcast_dimensions))
                != len(record.broadcast_dimensions)
            ):
                raise ValidationError("literal materialization tensor contract is invalid")
            if (
                not isinstance(record.blob_id, int)
                or isinstance(record.blob_id, bool)
                or not isinstance(record.blob_offset, int)
                or isinstance(record.blob_offset, bool)
                or not isinstance(record.byte_size, int)
                or isinstance(record.byte_size, bool)
                or record.blob_offset < 0
                or record.byte_size <= 0
            ):
                raise ValidationError(
                    "literal materialization record bounds are invalid"
                )
            _digest(record.sha256, "record SHA-256")
            expected_bytes = math.prod(record.shape) * _DTYPE_BYTES[record.dtype]
            try:
                blob = blobs_by_id[record.blob_id]
            except KeyError as exc:
                raise ValidationError(
                    "literal materialization record names an unknown blob"
                ) from exc
            if (
                record.byte_size != expected_bytes
                or record.byte_size != blob.byte_size
                or record.blob_offset != blob.offset
                or record.sha256 != blob.sha256
            ):
                raise ValidationError(
                    "literal materialization record/blob contract differs"
                )

    @property
    def materialized_ops(self) -> tuple[int, ...]:
        return tuple(record.execution_index for record in self.records)

    @property
    def digest(self) -> str:
        return hashlib.sha256(dump_literal_materialization(self).encode()).hexdigest()

    def record_for_value(self, value_id: int) -> LiteralMaterializationRecord | None:
        return next(
            (record for record in self.records if record.output_value_id == value_id),
            None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": LITERAL_MATERIALIZATION_SCHEMA,
            "inventory_sha256": self.inventory_sha256,
            "schedule_sha256": self.schedule_sha256,
            "alignment": self.alignment,
            "summary": {
                "records": len(self.records),
                "unique_blobs": len(self.blobs),
                "data_bytes": len(self.data),
                "logical_bytes": sum(record.byte_size for record in self.records),
                "deduplicated_bytes": sum(blob.byte_size for blob in self.blobs),
            },
            "data_sha256": hashlib.sha256(self.data).hexdigest(),
            "blobs": [blob.to_dict() for blob in self.blobs],
            "records": [record.to_dict() for record in self.records],
        }


def build_literal_materialization(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    *,
    excluded_execution_indices: tuple[int, ...] | set[int] = (),
    max_materialized_bytes: int = DEFAULT_MAX_MATERIALIZED_BYTES,
) -> LiteralMaterialization:
    """Materialize residual literal broadcasts into a deduplicated constant blob."""

    if not isinstance(schedule, ExecutionSchedule) or not isinstance(
        inventory, LoweringInventory
    ):
        raise ValidationError(
            "literal materialization requires a schedule and lowering inventory"
        )
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode()
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("literal materialization stage identities do not match")
    if (
        not isinstance(max_materialized_bytes, int)
        or isinstance(max_materialized_bytes, bool)
        or max_materialized_bytes <= 0
    ):
        raise ValidationError("literal materialization byte bound must be positive")
    try:
        excluded = set(excluded_execution_indices)
    except TypeError as exc:
        raise ValidationError(
            "literal materialization exclusions must be execution indices"
        ) from exc
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(schedule.ops)
        for index in excluded
    ):
        raise ValidationError(
            "literal materialization exclusion is not a scheduled execution"
        )

    inventory_by_site = {item.site_id: item for item in inventory.ops}
    candidates: list[
        tuple[
            int,
            str,
            int,
            int,
            str,
            str,
            str,
            str,
            tuple[int, ...],
            tuple[int, ...],
            bytes,
        ]
    ] = []
    for op in schedule.ops:
        if op.execution_index in excluded or op.opcode != "broadcast_in_dim":
            continue
        item = inventory_by_site.get(op.site)
        if (
            item is None
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.MEMORY
            or len(op.inputs) != 1
            or len(op.outputs) != 1
        ):
            continue
        source = schedule.values[op.inputs[0]]
        output = schedule.values[op.outputs[0]]
        identity = source.constant_identity
        if (
            source.storage != ValueStorage.CONSTANT
            or identity is None
            or not identity.startswith("literal:")
            or output.type.dtype not in _DTYPE_BYTES
        ):
            continue
        literal = inventory_by_site.get(identity[len("literal:") :])
        if (
            literal is None
            or literal.opcode != "constant"
            or literal.input_types
            or literal.output_types != (source.type,)
            or len(literal.attributes) != 1
            or literal.attributes[0][0] != "value"
        ):
            raise ValidationError(
                "literal materialization cannot resolve its source constant"
            )
        if not all(isinstance(dimension, int) for dimension in output.type.shape):
            raise ValidationError(
                "literal materialization requires a specialized static output"
            )
        if not all(isinstance(dimension, int) for dimension in source.type.shape):
            raise ValidationError(
                "literal materialization requires a specialized static source"
            )
        attributes = dict(op.attributes)
        dimensions = attributes.get("broadcast_dimensions")
        shape = attributes.get("shape")
        if (
            not isinstance(dimensions, tuple)
            or not isinstance(shape, tuple)
            or shape != output.type.shape
        ):
            raise ValidationError("literal materialization broadcast contract is invalid")
        values = literal.attributes[0][1]
        if not isinstance(values, tuple):
            raise ValidationError("literal materialization source payload is invalid")
        payload = _broadcast_payload(
            values,
            source.type.dtype,
            tuple(int(dimension) for dimension in source.type.shape),
            tuple(int(dimension) for dimension in output.type.shape),
            dimensions,
        )
        if len(payload) > max_materialized_bytes:
            raise ValidationError(
                "literal materialization exceeds its bounded data size"
            )
        candidates.append(
            (
                op.execution_index,
                op.site,
                op.inputs[0],
                op.outputs[0],
                identity,
                output.type.dtype,
                output.type.layout,
                output.type.device,
                tuple(int(dimension) for dimension in output.type.shape),
                dimensions,
                payload,
            )
        )

    data = bytearray()
    blobs: list[MaterializedLiteralBlob] = []
    blob_by_payload: dict[bytes, MaterializedLiteralBlob] = {}
    records: list[LiteralMaterializationRecord] = []
    for (
        execution_index,
        site,
        source_value_id,
        output_value_id,
        source_identity,
        dtype,
        layout,
        device,
        shape,
        dimensions,
        payload,
    ) in candidates:
        blob = blob_by_payload.get(payload)
        if blob is None:
            allocation_bytes = _align(len(payload), LITERAL_MATERIALIZATION_ALIGNMENT)
            if len(data) > max_materialized_bytes - allocation_bytes:
                raise ValidationError(
                    "literal materialization exceeds its bounded data size"
                )
            blob = MaterializedLiteralBlob(
                len(blobs),
                len(data),
                len(payload),
                allocation_bytes,
                hashlib.sha256(payload).hexdigest(),
            )
            blobs.append(blob)
            blob_by_payload[payload] = blob
            data.extend(payload)
            data.extend(bytes(allocation_bytes - len(payload)))
        records.append(
            LiteralMaterializationRecord(
                execution_index,
                site,
                source_value_id,
                output_value_id,
                source_identity,
                dtype,
                layout,
                device,
                shape,
                dimensions,
                blob.blob_id,
                blob.offset,
                blob.byte_size,
                blob.sha256,
            )
        )
    return LiteralMaterialization(
        inventory_sha256,
        schedule_sha256,
        LITERAL_MATERIALIZATION_ALIGNMENT,
        bytes(data),
        tuple(blobs),
        tuple(records),
    )


def dump_literal_materialization(materialization: LiteralMaterialization) -> str:
    if not isinstance(materialization, LiteralMaterialization):
        raise ValidationError(
            "literal materialization dump requires a LiteralMaterialization"
        )
    return json.dumps(
        materialization.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ) + "\n"


def encode_literal_tensor(
    values: tuple[object, ...], dtype: str, shape: tuple[int, ...]
) -> bytes:
    """Encode a verified row-major literal with ProgramIR scalar semantics."""

    if (
        dtype not in _DTYPE_BYTES
        or not isinstance(values, tuple)
        or not isinstance(shape, tuple)
        or not shape
        or not all(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension > 0
            for dimension in shape
        )
        or len(values) != math.prod(shape)
    ):
        raise ValidationError("literal tensor payload differs from its type")
    return b"".join(_encode(value, dtype) for value in values)


def _broadcast_payload(
    values: tuple[object, ...],
    dtype: str,
    source_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    dimensions: tuple[int, ...],
) -> bytes:
    if (
        not source_shape
        or not output_shape
        or any(dimension <= 0 for dimension in source_shape + output_shape)
        or len(values) != math.prod(source_shape)
        or len(dimensions) != len(source_shape)
        or tuple(sorted(dimensions)) != dimensions
        or len(set(dimensions)) != len(dimensions)
        or any(axis < 0 or axis >= len(output_shape) for axis in dimensions)
        or any(
            source_shape[input_axis] not in (1, output_shape[output_axis])
            for input_axis, output_axis in enumerate(dimensions)
        )
    ):
        raise ValidationError("literal materialization source shape differs")
    encoded = tuple(_encode(value, dtype) for value in values)
    output_elements = math.prod(output_shape)
    if len(encoded) == 1:
        return encoded[0] * output_elements
    source_strides = _strides(source_shape)
    result = bytearray()
    for output_index in range(output_elements):
        output_coordinates = _coordinates(output_index, output_shape)
        source_coordinates = tuple(
            0 if source_shape[input_axis] == 1 else output_coordinates[output_axis]
            for input_axis, output_axis in enumerate(dimensions)
        )
        source_index = sum(
            coordinate * stride
            for coordinate, stride in zip(source_coordinates, source_strides)
        )
        result.extend(encoded[source_index])
    return bytes(result)


def _encode(value: object, dtype: str) -> bytes:
    if dtype == "bool":
        if not isinstance(value, bool):
            raise ValidationError("literal BOOL materialization has invalid data")
        return bytes((int(value),))
    if dtype == "i32":
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not -(2**31) <= value < 2**31
        ):
            raise ValidationError("literal I32 materialization has invalid data")
        return struct.pack("<i", value)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValidationError("literal floating materialization has invalid data")
    try:
        f32 = struct.pack("<f", float(value))
    except (OverflowError, struct.error) as exc:
        raise ValidationError("literal floating materialization overflows F32") from exc
    bits = struct.unpack("<I", f32)[0]
    if (bits & 0x7F800000) == 0x7F800000:
        raise ValidationError("literal floating materialization is non-finite in F32")
    if dtype == "f32":
        return f32
    if dtype == "bf16":
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        bf16 = (rounded >> 16) & 0xFFFF
        if (bf16 & 0x7F80) == 0x7F80:
            raise ValidationError("literal BF16 materialization rounded to non-finite")
        return struct.pack("<H", bf16)
    raise ValidationError(f"literal materialization does not support dtype {dtype}")


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    stride = 1
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


def _coordinates(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = index % shape[axis]
        index //= shape[axis]
    return tuple(result)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or not any(character != "0" for character in value)
    ):
        raise ValidationError(f"literal materialization {label} is invalid")
