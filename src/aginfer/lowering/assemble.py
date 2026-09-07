from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from typing import Iterable

from ..errors import ValidationError
from ..schema import CudaArch
from .command import CommandStream, OperandAccess, ProviderCommand
from .memory import AllocationRegion, MemoryPlan, dump_memory_plan
from .schedule import ExecutionSchedule, dump_execution_schedule


@dataclass(frozen=True, slots=True)
class CommandPlacement:
    execution_index: int
    covered_execution_indices: tuple[int, ...]
    command: ProviderCommand

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution_index, int)
            or isinstance(self.execution_index, bool)
            or self.execution_index < 0
        ):
            raise ValidationError("command placement execution_index must be non-negative")
        if (
            not isinstance(self.covered_execution_indices, tuple)
            or not self.covered_execution_indices
            or tuple(sorted(set(self.covered_execution_indices)))
            != self.covered_execution_indices
            or self.execution_index not in self.covered_execution_indices
        ):
            raise ValidationError(
                "command placement coverage must be sorted, unique, and contain its anchor"
            )
        if not isinstance(self.command, ProviderCommand):
            raise ValidationError("command placement requires a ProviderCommand")

    @classmethod
    def from_lowered(cls, lowered: object) -> "CommandPlacement":
        """Normalize one provider lowering record without depending on its class."""

        execution_index = getattr(lowered, "execution_index", None)
        command = getattr(lowered, "command", None)
        covered = getattr(lowered, "fused_execution_indices", None)
        if covered is None and isinstance(execution_index, int):
            covered = (execution_index,)
        try:
            normalized = tuple(sorted(set(covered)))
            return cls(execution_index, normalized, command)
        except (TypeError, ValidationError) as exc:
            raise ValidationError(f"invalid provider lowering record: {exc}") from exc


def order_command_placements(
    schedule: ExecutionSchedule,
    memory_plan: MemoryPlan,
    placements: Iterable[CommandPlacement],
) -> tuple[CommandPlacement, ...]:
    """Stable topological order of fused boundary operands, not source anchors.

    A fused normalization may consume a projection produced after its source
    anchor. Mutable state accesses additionally retain source read/write order;
    an interleaved fusion that cannot preserve that order is rejected as a cycle.
    """
    records = tuple(placements)
    if not records or not all(isinstance(p, CommandPlacement) for p in records):
        raise ValidationError("command ordering requires command placements")
    if len({p.execution_index for p in records}) != len(records):
        raise ValidationError("command ordering has duplicate command anchors")
    allocations = {a.value_id: a for a in memory_plan.allocations}
    if set(allocations) != set(range(len(schedule.values))):
        raise ValidationError("command ordering has incomplete allocations")
    roots = []
    for value_id in range(len(schedule.values)):
        root, seen = value_id, set()
        while allocations[root].region == AllocationRegion.ALIAS:
            if root in seen:
                raise ValidationError("command ordering has cyclic aliases")
            seen.add(root)
            root = allocations[root].alias_of
            if root not in allocations:
                raise ValidationError("command ordering has an invalid alias")
        roots.append(root)
    dependencies = [set() for _ in records]
    writers: dict[int, int] = {}
    reads: list[tuple[int, int]] = []
    states = dict(schedule.states)
    state_events: dict[int, set[tuple[int, int]]] = {}
    inputs = {roots[value_id] for _, value_id in schedule.entry_inputs}
    outputs = {roots[value_id] for _, value_id in schedule.entry_outputs}
    for index, placement in enumerate(records):
        if any(i >= len(schedule.ops) for i in placement.covered_execution_indices):
            raise ValidationError("command placement covers an out-of-range execution")
        for operand in placement.command.operands:
            if operand.value_id >= len(roots):
                raise ValidationError("command ordering references an unknown operand")
            root = roots[operand.value_id]
            region = allocations[root].region
            read = operand.access in {OperandAccess.READ, OperandAccess.READ_WRITE}
            write = operand.access in {OperandAccess.WRITE, OperandAccess.READ_WRITE}
            if region == AllocationRegion.UNUSED:
                raise ValidationError("command ordering references unused storage")
            if region == AllocationRegion.STATE:
                events = set()
                for execution in placement.covered_execution_indices:
                    op = schedule.ops[execution]
                    if (states.get(dict(op.attributes).get("state")) == root
                        and ((read and op.opcode == "state_read")
                             or (write and op.opcode in {"state_write", "state_update"}))):
                        events.add(execution)
                producer = schedule.values[operand.value_id].producer
                if read and producer is not None and schedule.ops[producer].opcode == "state_read":
                    events.add(producer)
                if not events:
                    raise ValidationError("command state access has no source read/write receipt")
                state_events.setdefault(root, set()).update((event, index) for event in events)
                continue
            if write:
                if region == AllocationRegion.CONSTANT or root in inputs:
                    raise ValidationError("command order writes a constant or entry input")
                if root in writers:
                    raise ValidationError("command order has duplicate SSA output storage")
                writers[root] = index
            if read and (region == AllocationRegion.ARENA or root in outputs):
                reads.append((index, root))
    for index, root in reads:
        if root not in writers:
            raise ValidationError(f"command order reads an unproduced temporary: value={root}")
        dependencies[index].add(writers[root])
    for events in state_events.values():
        ordered = sorted(events)
        for (_, previous), (_, following) in zip(ordered, ordered[1:]):
            if previous != following:
                dependencies[following].add(previous)
    users = [set() for _ in records]
    for index, deps in enumerate(dependencies):
        for dependency in deps:
            users[dependency].add(index)
    ready = [(p.execution_index, i) for i, p in enumerate(records) if not dependencies[i]]
    heapq.heapify(ready)
    result = []
    while ready:
        _, index = heapq.heappop(ready)
        result.append(records[index])
        for user in users[index]:
            dependencies[user].remove(index)
            if not dependencies[user]:
                heapq.heappush(ready, (records[user].execution_index, user))
    if len(result) != len(records):
        raise ValidationError("fused command dependencies contain a cycle")
    return tuple(result)


def assemble_command_stream(
    schedule: ExecutionSchedule,
    memory_plan: MemoryPlan,
    placements: Iterable[CommandPlacement],
    *,
    target_arch: CudaArch,
) -> CommandStream:
    """Create the one canonical command stream for a fully lowered schedule."""

    if not isinstance(schedule, ExecutionSchedule):
        raise ValidationError("command assembly requires an ExecutionSchedule")
    if not isinstance(memory_plan, MemoryPlan):
        raise ValidationError("command assembly requires a MemoryPlan")
    if not isinstance(target_arch, CudaArch):
        raise ValidationError("command assembly requires a target CudaArch")
    schedule_digest = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    memory_digest = hashlib.sha256(dump_memory_plan(memory_plan).encode()).hexdigest()
    if memory_plan.schedule_sha256 != schedule_digest:
        raise ValidationError("command assembly memory plan does not match its schedule")
    records = tuple(placements)
    if not records or not all(isinstance(item, CommandPlacement) for item in records):
        raise ValidationError("command assembly needs a non-empty placement sequence")
    anchors: set[int] = set()
    covered: set[int] = set()
    elided = set(memory_plan.elided_ops)
    for placement in records:
        if placement.execution_index in anchors:
            raise ValidationError("command assembly has duplicate command anchors")
        anchors.add(placement.execution_index)
        selected = set(placement.covered_execution_indices)
        if any(index >= len(schedule.ops) for index in selected):
            raise ValidationError("command placement covers an out-of-range execution")
        if selected & covered:
            raise ValidationError("command placements overlap each other")
        if selected & elided:
            raise ValidationError("command placement overlaps a memory-elided execution")
        covered.update(selected)
        for operand in placement.command.operands:
            if operand.value_id >= len(schedule.values):
                raise ValidationError("command placement references an unknown scheduled value")
    expected = set(range(len(schedule.ops)))
    if covered | elided != expected:
        missing = sorted(expected - covered - elided)
        raise ValidationError(
            "command assembly does not cover the complete schedule "
            f"({len(missing)} missing; first={missing[:16]})"
        )
    records = order_command_placements(schedule, memory_plan, records)
    workspace_bytes = max(
        (
            placement.command.workspace_offset
            + placement.command.workspace_bytes
            for placement in records
        ),
        default=0,
    )
    return CommandStream(
        target_arch=target_arch,
        memory_plan_sha256=memory_digest,
        value_count=len(schedule.values),
        arena_bytes=memory_plan.arena_bytes,
        state_bytes=memory_plan.state_bytes,
        workspace_bytes=workspace_bytes,
        commands=tuple(placement.command for placement in records),
    )


def placements_from_partial_lowering(
    schedule: ExecutionSchedule, lowering: object
) -> tuple[CommandPlacement, ...]:
    """Normalize a partial lowering, assigning shared fused producers once.

    Region lowerers may eliminate a shared mask/broadcast producer that is not
    repeated in every command record. Such an execution is assigned to the
    first following command in the same inlined invocation. This preserves
    schedule order while keeping coverage ownership unique.
    """

    raw_commands = getattr(lowering, "commands", None)
    if not isinstance(raw_commands, tuple) or not raw_commands:
        raise ValidationError("partial lowering has no immutable command records")
    placements = [CommandPlacement.from_lowered(item) for item in raw_commands]
    item_execution_indices = [getattr(item, "execution_index", None) for item in raw_commands]
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(schedule.ops)
        for index in item_execution_indices
    ):
        raise ValidationError("partial lowering command anchor is out of range")
    covered = {
        execution
        for placement in placements
        for execution in placement.covered_execution_indices
    }
    raw_global = getattr(lowering, "fused_execution_indices", tuple(sorted(covered)))
    try:
        global_fused = tuple(sorted(set(raw_global)))
    except TypeError as exc:
        raise ValidationError("partial lowering fused coverage is not iterable") from exc
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(schedule.ops)
        for index in global_fused
    ):
        raise ValidationError("partial lowering fused coverage is out of range")
    if not covered.issubset(global_fused):
        raise ValidationError("partial lowering command coverage exceeds its fused receipt")
    expanded = [set(item.covered_execution_indices) for item in placements]
    for execution in sorted(set(global_fused) - covered):
        invocation = schedule.ops[execution].invocation_id
        candidates = [
            (command_execution, index)
            for index, command_execution in enumerate(item_execution_indices)
            if schedule.ops[command_execution].invocation_id == invocation
            and command_execution >= execution
        ]
        if not candidates:
            raise ValidationError(
                "shared fused execution has no following command in its invocation"
            )
        _, selected = min(candidates)
        expanded[selected].add(execution)
    return tuple(
        CommandPlacement(placement.execution_index, tuple(sorted(selected)), placement.command)
        for placement, selected in zip(placements, expanded)
    )
