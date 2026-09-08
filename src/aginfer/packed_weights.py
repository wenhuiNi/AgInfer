from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .constant_store import ConstantStore
from .errors import ValidationError
from .executable import PackedWeightSpan
from .lowering.command import CommandStream
from .lowering.inventory import LoweringInventory, dump_lowering_inventory
from .lowering.materialize import LiteralMaterialization, encode_literal_tensor
from .lowering.memory import AllocationRegion, MemoryPlan
from .lowering.schedule import ExecutionSchedule


@dataclass(frozen=True, slots=True)
class PackedWeights:
    path: Path
    byte_size: int
    sha256: str
    spans: tuple[PackedWeightSpan, ...]


def pack_command_weights(
    destination: str | os.PathLike[str],
    schedule: ExecutionSchedule,
    memory_plan: MemoryPlan,
    inventory: LoweringInventory,
    command_stream: CommandStream,
    *,
    constants: ConstantStore | None = None,
    literal_materialization: LiteralMaterialization | None = None,
    computed_constants: dict[int, bytes] | None = None,
    widened_constants: dict[int, int] | None = None,
    chunk_size: int = 4 * 1024 * 1024,
) -> PackedWeights:
    """Stream command-referenced constants into one deterministic weight blob."""

    if (
        not isinstance(schedule, ExecutionSchedule)
        or not isinstance(memory_plan, MemoryPlan)
        or not isinstance(inventory, LoweringInventory)
        or not isinstance(command_stream, CommandStream)
    ):
        raise ValidationError("packed weights require schedule, memory, inventory, and commands")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValidationError("packed weight chunk_size must be positive")
    if (
        command_stream.value_count != len(schedule.values)
        or len(memory_plan.allocations) != len(schedule.values)
        or schedule.inventory_sha256
        != hashlib.sha256(dump_lowering_inventory(inventory).encode()).hexdigest()
    ):
        raise ValidationError("packed weight inputs have inconsistent identities")
    allocations = {item.value_id: item for item in memory_plan.allocations}
    roots = _roots(memory_plan)
    referenced_roots = sorted(
        {
            roots[operand.value_id]
            for command in command_stream.commands
            for operand in command.operands
            if allocations[roots[operand.value_id]].region == AllocationRegion.CONSTANT
        }
    )
    computed_constants = {} if computed_constants is None else computed_constants
    widened_constants = {} if widened_constants is None else widened_constants
    if (not isinstance(widened_constants, dict) or constants is None and widened_constants
            or any(type(k) is not int or k not in referenced_roots or k in computed_constants
                   or type(v) is not int or v not in allocations or k == v
                   or allocations[v].region != AllocationRegion.CONSTANT
                   or allocations[k].byte_size != 2 * allocations[v].byte_size
                   or schedule.values[k].constant_identity is not None
                   for k, v in widened_constants.items())):
        raise ValidationError('widened constants require new constant roots and declared source weights')
    if (not isinstance(computed_constants, dict)
            or any(type(k) is not int or k not in referenced_roots or not isinstance(v, bytes)
                   or len(v) != allocations[k].byte_size or schedule.values[k].constant_identity is not None
                   for k, v in computed_constants.items())):
        raise ValidationError("computed constants must exactly sized new command-referenced constant roots")
    materialized = (
        {}
        if literal_materialization is None
        else {
            record.output_value_id: record
            for record in literal_materialization.records
        }
    )
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    spans: list[PackedWeightSpan] = []
    whole_digest = hashlib.sha256()
    position = 0
    try:
        with temporary.open("wb") as output:
            for value_id in referenced_roots:
                aligned = _align(position, memory_plan.alignment)
                padding = aligned - position
                if padding:
                    zeros = bytes(padding)
                    output.write(zeros)
                    whole_digest.update(zeros)
                    position = aligned
                value = schedule.values[value_id]
                allocation = allocations[value_id]
                span_digest = hashlib.sha256()
                written = 0
                record = materialized.get(value_id)
                if value_id in widened_constants:
                    from .constant_conversion import cast_chunks
                    source_id = widened_constants[value_id]
                    for chunk in cast_chunks(constants, schedule.values[source_id], value,
                            allocations[source_id].byte_size, chunk_size=chunk_size):
                        output.write(chunk)
                        span_digest.update(chunk)
                        whole_digest.update(chunk)
                        written += len(chunk)
                elif value_id in computed_constants:
                    payload = computed_constants[value_id]
                    output.write(payload)
                    span_digest.update(payload)
                    whole_digest.update(payload)
                    written = len(payload)
                elif record is not None:
                    if literal_materialization is None:
                        raise AssertionError("materialized record lost its owner")
                    payload = literal_materialization.data[
                        record.blob_offset : record.blob_offset + record.byte_size
                    ]
                    output.write(payload)
                    span_digest.update(payload)
                    whole_digest.update(payload)
                    written = len(payload)
                elif value.constant_identity is not None and value.constant_identity.startswith("literal:"):
                    site = value.constant_identity[len("literal:") :]
                    try:
                        literal = inventory_by_site[site]
                    except KeyError as exc:
                        raise ValidationError(
                            f"packed weights cannot resolve literal site {site}"
                        ) from exc
                    attributes = dict(literal.attributes)
                    raw_values = attributes.get("value")
                    if not isinstance(raw_values, tuple):
                        raise ValidationError("packed weight literal has no tuple payload")
                    if not all(isinstance(dimension, int) for dimension in value.type.shape):
                        raise ValidationError("packed weight literal shape is not static")
                    payload = encode_literal_tensor(
                        raw_values,
                        value.type.dtype,
                        tuple(int(dimension) for dimension in value.type.shape),
                    )
                    output.write(payload)
                    span_digest.update(payload)
                    whole_digest.update(payload)
                    written = len(payload)
                elif value.constant_identity is not None and ":" in value.constant_identity:
                    if constants is None:
                        raise ValidationError(
                            f"packed weights require a ConstantStore for {value.constant_identity}"
                        )
                    namespace, name = value.constant_identity.split(":", 1)
                    source = constants.tensor(namespace, name)
                    expected_dtype = value.type.dtype.upper()
                    if (
                        source.dtype != expected_dtype
                        or tuple(source.shape) != tuple(value.type.shape)
                        or source.byte_length != allocation.byte_size
                    ):
                        raise ValidationError(
                            f"packed weight source contract differs for {value.constant_identity}"
                        )
                    for chunk in constants.iter_chunks(namespace, name, chunk_size=chunk_size):
                        output.write(chunk)
                        span_digest.update(chunk)
                        whole_digest.update(chunk)
                        written += len(chunk)
                else:
                    raise ValidationError(
                        f"packed weights cannot resolve constant value {value_id}"
                    )
                if written != allocation.byte_size:
                    raise ValidationError(
                        f"packed weight byte size differs for value {value_id}: {written} != {allocation.byte_size}"
                    )
                spans.append(
                    PackedWeightSpan(value_id, position, written, span_digest.hexdigest())
                )
                position += written
            final_size = _align(position, memory_plan.alignment)
            if final_size == 0:
                raise ValidationError("packed weights contain no command-referenced constants")
            padding = final_size - position
            if padding:
                zeros = bytes(padding)
                output.write(zeros)
                whole_digest.update(zeros)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return PackedWeights(
        target,
        final_size,
        whole_digest.hexdigest(),
        tuple(sorted(spans, key=lambda item: item.value_id)),
    )


def _roots(memory_plan: MemoryPlan) -> tuple[int, ...]:
    allocations = {item.value_id: item for item in memory_plan.allocations}
    result: list[int] = []
    for value_id in range(len(memory_plan.allocations)):
        root = value_id
        seen: set[int] = set()
        while allocations[root].region == AllocationRegion.ALIAS:
            if root in seen or allocations[root].alias_of is None:
                raise ValidationError("packed weight alias graph is invalid")
            seen.add(root)
            root = allocations[root].alias_of
            if root not in allocations:
                raise ValidationError("packed weight alias target is out of range")
        result.append(root)
    return tuple(result)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)
