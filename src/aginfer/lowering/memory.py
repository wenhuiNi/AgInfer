from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..errors import ValidationError
from .schedule import (
    ExecutionSchedule,
    ScheduledOp,
    ScheduledValue,
    ValueStorage,
    dump_execution_schedule,
)

if TYPE_CHECKING:
    from .materialize import LiteralMaterialization, LiteralMaterializationRecord


MEMORY_PLAN_SCHEMA = "aginfer.memory-plan.v1"
DEFAULT_ARENA_ALIGNMENT = 256
_UINT64_MAX = 2**64 - 1
_DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "bool": 1}


class AllocationRegion(str, Enum):
    UNUSED = "unused"
    EXTERNAL = "external"
    CONSTANT = "constant"
    STATE = "state"
    ARENA = "arena"
    ALIAS = "alias"


@dataclass(frozen=True, slots=True)
class ValueAllocation:
    value_id: int
    region: AllocationRegion
    byte_size: int
    offset: int | None = None
    allocation_bytes: int | None = None
    alias_of: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "value_id": self.value_id,
            "region": self.region.value,
            "byte_size": self.byte_size,
        }
        if self.offset is not None:
            result["offset"] = self.offset
        if self.allocation_bytes is not None:
            result["allocation_bytes"] = self.allocation_bytes
        if self.alias_of is not None:
            result["alias_of"] = self.alias_of
        return result


@dataclass(frozen=True, slots=True)
class LivenessInterval:
    value_id: int
    start: int
    end: int
    byte_size: int
    allocation_bytes: int
    offset: int

    def to_dict(self) -> dict[str, int]:
        return {
            "value_id": self.value_id,
            "start": self.start,
            "end": self.end,
            "byte_size": self.byte_size,
            "allocation_bytes": self.allocation_bytes,
            "offset": self.offset,
        }


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    schedule_sha256: str
    literal_materialization_sha256: str | None
    alignment: int
    arena_bytes: int
    state_bytes: int
    peak_live_arena_bytes: int
    naive_arena_bytes: int
    entry_input_bytes: int
    entry_output_bytes: int
    constant_bytes: int
    allocations: tuple[ValueAllocation, ...]
    intervals: tuple[LivenessInterval, ...]
    elided_ops: tuple[int, ...]
    command_schedule_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        region_counts: dict[str, int] = {}
        for allocation in self.allocations:
            region_counts[allocation.region.value] = (
                region_counts.get(allocation.region.value, 0) + 1
            )
        result: dict[str, object] = {
            "schema": MEMORY_PLAN_SCHEMA,
            "schedule_sha256": self.schedule_sha256,
            "alignment": self.alignment,
            "summary": {
                "arena_bytes": self.arena_bytes,
                "state_bytes": self.state_bytes,
                "peak_live_arena_bytes": self.peak_live_arena_bytes,
                "naive_arena_bytes": self.naive_arena_bytes,
                "arena_reuse_bytes": self.naive_arena_bytes - self.arena_bytes,
                "entry_input_bytes": self.entry_input_bytes,
                "entry_output_bytes": self.entry_output_bytes,
                "constant_bytes": self.constant_bytes,
                "values": len(self.allocations),
                "values_by_region": dict(sorted(region_counts.items())),
                "intervals": len(self.intervals),
                "elided_ops": len(self.elided_ops),
            },
            "allocations": [item.to_dict() for item in self.allocations],
            "intervals": [item.to_dict() for item in self.intervals],
            "elided_ops": list(self.elided_ops),
        }
        if self.literal_materialization_sha256 is not None:
            result["literal_materialization_sha256"] = (
                self.literal_materialization_sha256
            )
        if self.command_schedule_sha256 is not None:
            result["schema"] = "aginfer.command-memory-plan.v1"
            result["command_schedule_sha256"] = self.command_schedule_sha256
            result["liveness_index_space"] = "provider_command"
        return result


def build_memory_plan(
    schedule: ExecutionSchedule,
    *,
    alignment: int = DEFAULT_ARENA_ALIGNMENT,
    literal_materialization: LiteralMaterialization | None = None,
) -> MemoryPlan:
    """Plan persistent state and reusable temporary storage for a static schedule."""

    if not isinstance(schedule, ExecutionSchedule):
        raise ValidationError("memory planning requires an ExecutionSchedule")
    if (
        not isinstance(alignment, int)
        or isinstance(alignment, bool)
        or alignment <= 0
        or alignment > 2**20
        or alignment & (alignment - 1)
    ):
        raise ValidationError("memory plan alignment must be a bounded positive power of two")

    schedule_sha256 = hashlib.sha256(
        dump_execution_schedule(schedule).encode("utf-8")
    ).hexdigest()
    materialized_by_value: dict[int, LiteralMaterializationRecord] = {}
    materialized_ops: set[int] = set()
    materialization_digest: str | None = None
    materialization_bytes = 0
    materialization_alignment = 1
    if literal_materialization is not None:
        from .materialize import LiteralMaterialization

        if not isinstance(literal_materialization, LiteralMaterialization):
            raise ValidationError(
                "memory planning literal materialization has the wrong type"
            )
        if (
            literal_materialization.schedule_sha256 != schedule_sha256
            or literal_materialization.inventory_sha256 != schedule.inventory_sha256
        ):
            raise ValidationError(
                "memory plan literal materialization identities do not match"
            )
        materialized_by_value = {
            record.output_value_id: record
            for record in literal_materialization.records
        }
        materialized_ops = set(literal_materialization.materialized_ops)
        materialization_digest = literal_materialization.digest
        materialization_bytes = len(literal_materialization.data)
        materialization_alignment = literal_materialization.alignment
        for record in literal_materialization.records:
            if record.execution_index >= len(schedule.ops):
                raise ValidationError(
                    "memory plan literal materialization names an unknown execution"
                )
            op = schedule.ops[record.execution_index]
            if (
                op.site != record.site
                or op.opcode != "broadcast_in_dim"
                or op.inputs != (record.source_value_id,)
                or op.outputs != (record.output_value_id,)
                or schedule.values[record.source_value_id].constant_identity
                != record.source_identity
                or schedule.values[record.output_value_id].type.dtype != record.dtype
                or schedule.values[record.output_value_id].type.layout != record.layout
                or schedule.values[record.output_value_id].type.device != record.device
                or schedule.values[record.output_value_id].type.shape != record.shape
                or dict(op.attributes).get("broadcast_dimensions")
                != record.broadcast_dimensions
            ):
                raise ValidationError(
                    "memory plan literal materialization differs from its scheduled op"
                )

    byte_sizes = tuple(_byte_size(value) for value in schedule.values)
    aliases: dict[int, int] = {}
    for value in schedule.values:
        if value.storage == ValueStorage.STATE_ALIAS:
            if value.alias_of is None:
                raise AssertionError("verified schedule lost state alias target")
            aliases[value.value_id] = value.alias_of
    elided_ops: list[int] = []
    for op in schedule.ops:
        if op.execution_index in materialized_ops:
            elided_ops.append(op.execution_index)
        elif op.opcode == "reshape":
            if len(op.inputs) != 1 or len(op.outputs) != 1:
                raise AssertionError("verified reshape lost unary contract")
            aliases[op.outputs[0]] = op.inputs[0]
            elided_ops.append(op.execution_index)
        elif op.opcode == "transpose" and _is_row_major_identity_transpose(
            schedule, op
        ):
            aliases[op.outputs[0]] = op.inputs[0]
            elided_ops.append(op.execution_index)
        elif op.opcode == "broadcast_in_dim" and _is_row_major_identity_broadcast(
            schedule, op
        ):
            aliases[op.outputs[0]] = op.inputs[0]
            elided_ops.append(op.execution_index)
        elif op.opcode == "state_read":
            elided_ops.append(op.execution_index)

    roots: list[int] = []
    for value in schedule.values:
        root = value.value_id
        visited: set[int] = set()
        while root in aliases:
            if root in visited:
                raise ValidationError("memory plan alias graph contains a cycle")
            visited.add(root)
            root = aliases[root]
        if byte_sizes[root] != byte_sizes[value.value_id]:
            raise ValidationError("memory plan alias changes the tensor byte size")
        roots.append(root)

    uses: dict[int, int] = {}
    for op in schedule.ops:
        for value_id in op.inputs:
            root = roots[value_id]
            uses[root] = max(uses.get(root, -1), op.execution_index)
    for _, value_id in schedule.entry_outputs:
        root = roots[value_id]
        uses[root] = max(uses.get(root, -1), len(schedule.ops))

    state_offsets: dict[int, tuple[int, int]] = {}
    state_end = 0
    for _, value_id in schedule.states:
        size = _align(byte_sizes[value_id], alignment)
        state_end = _align(state_end, alignment)
        _checked_add(state_end, size, "state region")
        state_offsets[value_id] = (state_end, size)
        state_end += size

    pending: list[tuple[int, int, int, int]] = []
    for value in schedule.values:
        if (
            roots[value.value_id] != value.value_id
            or value.storage != ValueStorage.TEMPORARY
            or value.value_id in materialized_by_value
        ):
            continue
        if value.producer is None:
            raise AssertionError("temporary schedule value has no producer")
        end = max(value.producer, uses.get(value.value_id, value.producer))
        pending.append(
            (
                value.producer,
                value.value_id,
                end,
                _align(byte_sizes[value.value_id], alignment),
            )
        )

    placed, arena_bytes, peak_live = _allocate_intervals(pending, alignment)
    intervals = tuple(
        LivenessInterval(
            value_id=value_id,
            start=start,
            end=end,
            byte_size=byte_sizes[value_id],
            allocation_bytes=allocation_bytes,
            offset=offset,
        )
        for start, value_id, end, allocation_bytes, offset in placed
    )
    interval_by_value = {item.value_id: item for item in intervals}

    allocations: list[ValueAllocation] = []
    for value in schedule.values:
        value_id = value.value_id
        root = roots[value_id]
        if root != value_id:
            allocations.append(
                ValueAllocation(
                    value_id,
                    AllocationRegion.ALIAS,
                    byte_sizes[value_id],
                    alias_of=root,
                )
            )
        elif value.storage == ValueStorage.ENTRY_INPUT:
            allocations.append(
                ValueAllocation(value_id, AllocationRegion.EXTERNAL, byte_sizes[value_id])
            )
        elif value.storage == ValueStorage.CONSTANT:
            allocations.append(
                ValueAllocation(value_id, AllocationRegion.CONSTANT, byte_sizes[value_id])
            )
        elif value_id in materialized_by_value:
            record = materialized_by_value[value_id]
            allocations.append(
                ValueAllocation(
                    value_id,
                    AllocationRegion.CONSTANT,
                    byte_sizes[value_id],
                    record.blob_offset,
                    _align(record.byte_size, materialization_alignment),
                )
            )
        elif value.storage == ValueStorage.STATE:
            offset, allocation_bytes = state_offsets[value_id]
            allocations.append(
                ValueAllocation(
                    value_id,
                    AllocationRegion.STATE,
                    byte_sizes[value_id],
                    offset,
                    allocation_bytes,
                )
            )
        elif value.storage == ValueStorage.TEMPORARY:
            interval = interval_by_value[value_id]
            allocations.append(
                ValueAllocation(
                    value_id,
                    AllocationRegion.ARENA,
                    byte_sizes[value_id],
                    interval.offset,
                    interval.allocation_bytes,
                )
            )
        else:
            raise AssertionError("unhandled root storage class in memory planner")

    plan = MemoryPlan(
        schedule_sha256=schedule_sha256,
        literal_materialization_sha256=materialization_digest,
        alignment=alignment,
        arena_bytes=arena_bytes,
        state_bytes=state_end,
        peak_live_arena_bytes=peak_live,
        naive_arena_bytes=sum(item.allocation_bytes for item in intervals),
        entry_input_bytes=sum(byte_sizes[value_id] for _, value_id in schedule.entry_inputs),
        entry_output_bytes=sum(byte_sizes[value_id] for _, value_id in schedule.entry_outputs),
        constant_bytes=materialization_bytes + sum(
            byte_sizes[value.value_id]
            for value in schedule.values
            if value.storage == ValueStorage.CONSTANT
        ),
        allocations=tuple(allocations),
        intervals=intervals,
        elided_ops=tuple(elided_ops),
    )
    _verify_memory_plan(
        schedule,
        plan,
        roots,
        materialized_by_value,
        materialization_bytes,
    )
    return plan


def dump_memory_plan(plan: MemoryPlan) -> str:
    if not isinstance(plan, MemoryPlan):
        raise ValidationError("memory plan dump requires a MemoryPlan")
    return json.dumps(
        plan.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ) + "\n"


def replan_memory_for_commands(
    schedule: ExecutionSchedule,
    base_plan: MemoryPlan,
    placements: object,
) -> MemoryPlan:
    """Recompute liveness at the actual fused-command boundary.

    Original op intervals cannot safely place outputs emitted earlier by a fused
    command. This pass verifies read-before-write ordering, allocates only live
    command boundary values, and makes terminal output roots caller-owned.
    """

    from .assemble import CommandPlacement, order_command_placements
    from .command import OperandAccess

    if not isinstance(schedule, ExecutionSchedule) or not isinstance(base_plan, MemoryPlan):
        raise ValidationError("command memory planning requires a schedule and base plan")
    if base_plan.schedule_sha256 != hashlib.sha256(
        dump_execution_schedule(schedule).encode()
    ).hexdigest():
        raise ValidationError("command memory plan schedule identity differs")
    try:
        records = order_command_placements(schedule, base_plan, placements)
    except (TypeError, AttributeError) as exc:
        raise ValidationError("command memory planning requires command placements") from exc
    if not records or not all(isinstance(item, CommandPlacement) for item in records):
        raise ValidationError("command memory planning requires command placements")
    allocations = {item.value_id: item for item in base_plan.allocations}
    if set(allocations) != set(range(len(schedule.values))):
        raise ValidationError("command memory plan has incomplete base allocations")
    roots: list[int] = []
    for value in schedule.values:
        root = value.value_id
        seen: set[int] = set()
        while allocations[root].region == AllocationRegion.ALIAS:
            if root in seen or allocations[root].alias_of is None:
                raise ValidationError("command memory plan alias graph is invalid")
            seen.add(root)
            root = allocations[root].alias_of
            if root not in allocations:
                raise ValidationError("command memory plan alias target is out of range")
        roots.append(root)
    input_roots = {roots[value_id] for _, value_id in schedule.entry_inputs}
    output_roots = {roots[value_id] for _, value_id in schedule.entry_outputs}
    if input_roots & output_roots:
        raise ValidationError("command memory plan cannot bind one root as both input and output")
    if any(allocations[root].region != AllocationRegion.ARENA for root in output_roots):
        raise ValidationError("command memory plan outputs must be temporary boundary values")

    written: set[int] = set()
    first_write: dict[int, int] = {}
    last_use: dict[int, int] = {}
    command_receipts: list[dict[str, object]] = []
    for command_index, placement in enumerate(records):
        command = placement.command
        reads: set[int] = set()
        writes: set[int] = set()
        for operand in command.operands:
            if operand.value_id >= len(roots):
                raise ValidationError("command memory planning found an unknown operand")
            root = roots[operand.value_id]
            if operand.byte_offset >= allocations[root].byte_size:
                raise ValidationError("command memory planning found an out-of-bounds operand")
            if operand.access in {OperandAccess.READ, OperandAccess.READ_WRITE}:
                reads.add(root)
            if operand.access in {OperandAccess.WRITE, OperandAccess.READ_WRITE}:
                if root in writes:
                    raise ValidationError("command memory planning found duplicate output storage")
                writes.add(root)
            last_use[root] = command_index
        for root in reads:
            if allocations[root].region == AllocationRegion.ARENA and root not in written:
                raise ValidationError(
                    "command order reads an unproduced temporary: "
                    f"command={command_index} anchor={placement.execution_index} value={root}"
                )
        for root in writes:
            if allocations[root].region in {AllocationRegion.CONSTANT, AllocationRegion.EXTERNAL}:
                raise ValidationError("command order writes a constant or entry input")
            if allocations[root].region == AllocationRegion.ARENA and root in written:
                raise ValidationError("command order writes an SSA temporary more than once")
            first_write.setdefault(root, command_index)
            written.add(root)
        command_receipts.append(
            {
                "execution_index": placement.execution_index,
                "covered_execution_indices": list(placement.covered_execution_indices),
                "provider_id": command.provider_id,
                "abi": [command.abi_major, command.abi_minor],
                "tag": int(command.tag),
                "capability_digest": command.capability_digest,
                "payload_sha256": hashlib.sha256(command.payload).hexdigest(),
                "operands": [
                    [operand.value_id, int(operand.access), operand.byte_offset]
                    for operand in command.operands
                ],
                "workspace": [command.workspace_offset, command.workspace_bytes],
            }
        )
    if not output_roots.issubset(written):
        raise ValidationError("command order does not produce every entry output")

    pending: list[tuple[int, int, int, int]] = []
    for root, allocation in allocations.items():
        if (
            roots[root] != root
            or allocation.region != AllocationRegion.ARENA
            or root in output_roots
            or root not in last_use
        ):
            continue
        if root not in first_write:
            raise ValidationError("live command temporary has no write command")
        pending.append(
            (
                first_write[root],
                root,
                last_use[root],
                _align(allocation.byte_size, base_plan.alignment),
            )
        )
    placed, arena_bytes, peak_live = _allocate_intervals(pending, base_plan.alignment)
    intervals = tuple(
        LivenessInterval(
            value_id,
            start,
            end,
            allocations[value_id].byte_size,
            allocation_bytes,
            offset,
        )
        for start, value_id, end, allocation_bytes, offset in placed
    )
    interval_by_value = {item.value_id: item for item in intervals}
    final_allocations: list[ValueAllocation] = []
    for value_id in range(len(schedule.values)):
        allocation = allocations[value_id]
        if roots[value_id] != value_id:
            final_allocations.append(allocation)
        elif value_id in output_roots:
            final_allocations.append(
                ValueAllocation(value_id, AllocationRegion.EXTERNAL, allocation.byte_size)
            )
        elif allocation.region == AllocationRegion.ARENA:
            interval = interval_by_value.get(value_id)
            if interval is None:
                final_allocations.append(
                    ValueAllocation(value_id, AllocationRegion.UNUSED, allocation.byte_size)
                )
            else:
                final_allocations.append(
                    ValueAllocation(
                        value_id,
                        AllocationRegion.ARENA,
                        allocation.byte_size,
                        interval.offset,
                        interval.allocation_bytes,
                    )
                )
        else:
            final_allocations.append(allocation)
    command_digest = hashlib.sha256(
        json.dumps(command_receipts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MemoryPlan(
        schedule_sha256=base_plan.schedule_sha256,
        literal_materialization_sha256=base_plan.literal_materialization_sha256,
        alignment=base_plan.alignment,
        arena_bytes=arena_bytes,
        state_bytes=base_plan.state_bytes,
        peak_live_arena_bytes=peak_live,
        naive_arena_bytes=sum(item.allocation_bytes for item in intervals),
        entry_input_bytes=base_plan.entry_input_bytes,
        entry_output_bytes=base_plan.entry_output_bytes,
        constant_bytes=base_plan.constant_bytes,
        allocations=tuple(final_allocations),
        intervals=intervals,
        elided_ops=base_plan.elided_ops,
        command_schedule_sha256=command_digest,
    )


def _is_row_major_identity_transpose(
    schedule: ExecutionSchedule, op: ScheduledOp
) -> bool:
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        return False
    source = schedule.values[op.inputs[0]].type
    target = schedule.values[op.outputs[0]].type
    permutation = dict(op.attributes).get("permutation")
    if (
        source.dtype != target.dtype
        or source.device != target.device
        or source.layout != "row_major"
        or target.layout != "row_major"
        or not isinstance(permutation, tuple)
        or len(permutation) != len(source.shape)
        or sorted(permutation) != list(range(len(source.shape)))
        or not all(isinstance(dimension, int) for dimension in source.shape)
        or target.shape != tuple(source.shape[axis] for axis in permutation)
    ):
        return False
    source_order = tuple(
        axis for axis, dimension in enumerate(source.shape) if dimension != 1
    )
    target_order = tuple(
        axis for axis in permutation if source.shape[axis] != 1
    )
    return source_order == target_order


def _is_row_major_identity_broadcast(
    schedule: ExecutionSchedule, op: ScheduledOp
) -> bool:
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        return False
    source = schedule.values[op.inputs[0]].type
    target = schedule.values[op.outputs[0]].type
    dimensions = dict(op.attributes).get("broadcast_dimensions")
    if (
        source.dtype != target.dtype
        or source.device != target.device
        or source.layout != "row_major"
        or target.layout != "row_major"
        or not isinstance(dimensions, tuple)
        or len(dimensions) != len(source.shape)
        or tuple(sorted(dimensions)) != dimensions
        or len(set(dimensions)) != len(dimensions)
        or not all(isinstance(dimension, int) for dimension in source.shape)
        or not all(isinstance(dimension, int) for dimension in target.shape)
    ):
        return False
    mapped = set(dimensions)
    if any(axis < 0 or axis >= len(target.shape) for axis in dimensions):
        return False
    if any(target.shape[axis] != 1 for axis in range(len(target.shape)) if axis not in mapped):
        return False
    return all(
        source.shape[input_axis] == target.shape[output_axis]
        for input_axis, output_axis in enumerate(dimensions)
    )


def _byte_size(value: ScheduledValue) -> int:
    try:
        element_bytes = _DTYPE_BYTES[value.type.dtype]
    except KeyError as exc:
        raise ValidationError(f"memory plan does not know dtype {value.type.dtype}") from exc
    if not all(isinstance(dimension, int) for dimension in value.type.shape):
        raise ValidationError(
            f"memory plan requires specialized static shape for value {value.value_id}"
        )
    numel = math.prod(value.type.shape)
    if numel <= 0 or numel > _UINT64_MAX // element_bytes:
        raise ValidationError(f"memory plan tensor byte size overflows uint64: value {value.value_id}")
    return numel * element_bytes


def _allocate_intervals(
    pending: list[tuple[int, int, int, int]],
    alignment: int,
) -> tuple[list[tuple[int, int, int, int, int]], int, int]:
    active: list[tuple[int, int, int, int]] = []
    free: list[tuple[int, int]] = []
    arena_end = 0
    peak_live = 0
    placed: list[tuple[int, int, int, int, int]] = []
    for start, value_id, end, allocation_bytes in sorted(pending):
        retained: list[tuple[int, int, int, int]] = []
        for active_end, offset, size, active_value in active:
            if active_end < start:
                free.append((offset, size))
            else:
                retained.append((active_end, offset, size, active_value))
        active = retained
        free = _merge_free(free)

        offset = -1
        for index, (candidate_offset, candidate_size) in enumerate(free):
            if candidate_size < allocation_bytes:
                continue
            offset = candidate_offset
            remainder = candidate_size - allocation_bytes
            if remainder:
                free[index] = (candidate_offset + allocation_bytes, remainder)
            else:
                free.pop(index)
            break
        if offset < 0:
            offset = _align(arena_end, alignment)
            _checked_add(offset, allocation_bytes, "temporary arena")
            arena_end = offset + allocation_bytes
        for _, active_offset, active_size, _ in active:
            if _overlap(offset, allocation_bytes, active_offset, active_size):
                raise AssertionError("arena allocator overlapped simultaneously live values")
        active.append((end, offset, allocation_bytes, value_id))
        peak_live = max(peak_live, sum(item[2] for item in active))
        placed.append((start, value_id, end, allocation_bytes, offset))
    return placed, arena_end, peak_live


def _merge_free(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for offset, size in sorted(blocks):
        if merged and merged[-1][0] + merged[-1][1] == offset:
            previous_offset, previous_size = merged[-1]
            merged[-1] = (previous_offset, previous_size + size)
        else:
            merged.append((offset, size))
    return merged


def _verify_memory_plan(
    schedule: ExecutionSchedule,
    plan: MemoryPlan,
    roots: list[int],
    materialized_by_value: dict[int, LiteralMaterializationRecord],
    materialization_bytes: int,
) -> None:
    if len(plan.allocations) != len(schedule.values):
        raise AssertionError("memory plan does not dispose every scheduled value")
    if [item.value_id for item in plan.allocations] != list(range(len(schedule.values))):
        raise AssertionError("memory plan allocations are not in value ID order")
    for allocation in plan.allocations:
        if allocation.region == AllocationRegion.ALIAS:
            if allocation.alias_of is None or roots[allocation.value_id] != allocation.alias_of:
                raise AssertionError("memory plan alias receipt is inconsistent")
            continue
        if allocation.alias_of is not None:
            raise AssertionError("owned memory allocation has an alias target")
        if allocation.value_id in materialized_by_value:
            record = materialized_by_value[allocation.value_id]
            if (
                allocation.region != AllocationRegion.CONSTANT
                or allocation.offset != record.blob_offset
                or allocation.byte_size != record.byte_size
                or allocation.allocation_bytes is None
                or allocation.offset is None
                or allocation.offset > materialization_bytes
                or allocation.allocation_bytes
                > materialization_bytes - allocation.offset
            ):
                raise AssertionError(
                    "memory plan materialized constant allocation is inconsistent"
                )
            continue
        if allocation.region in {AllocationRegion.ARENA, AllocationRegion.STATE}:
            if allocation.offset is None or allocation.allocation_bytes is None:
                raise AssertionError("owned memory allocation is missing bounds")
            limit = plan.arena_bytes if allocation.region == AllocationRegion.ARENA else plan.state_bytes
            if allocation.offset > limit or allocation.allocation_bytes > limit - allocation.offset:
                raise AssertionError("owned memory allocation exceeds its region")

    intervals = sorted(plan.intervals, key=lambda item: (item.start, item.value_id))
    active: list[LivenessInterval] = []
    for interval in intervals:
        active = [item for item in active if item.end >= interval.start]
        for item in active:
            if _overlap(
                interval.offset,
                interval.allocation_bytes,
                item.offset,
                item.allocation_bytes,
            ):
                raise AssertionError("memory plan overlaps live arena intervals")
        active.append(interval)


def _align(value: int, alignment: int) -> int:
    if value < 0 or value > _UINT64_MAX - (alignment - 1):
        raise ValidationError("memory plan alignment overflows uint64")
    return (value + alignment - 1) & ~(alignment - 1)


def _checked_add(left: int, right: int, label: str) -> None:
    if left < 0 or right < 0 or left > _UINT64_MAX - right:
        raise ValidationError(f"memory plan {label} size overflows uint64")


def _overlap(left_offset: int, left_size: int, right_offset: int, right_size: int) -> bool:
    return left_offset < right_offset + right_size and right_offset < left_offset + left_size
