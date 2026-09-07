from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from .errors import FormatError, ValidationError
from .lowering.command import CommandStream, CommandTag
from .lowering.memory import AllocationRegion, MemoryPlan, dump_memory_plan
from .lowering.schedule import ExecutionSchedule, dump_execution_schedule
from .schema import CudaArch


EXECUTABLE_PLAN_MAGIC = b"AIMEXE2\0"
EXECUTABLE_PLAN_SCHEMA_MAJOR = 2
EXECUTABLE_PLAN_SCHEMA_MINOR = 0
MAX_TENSOR_RANK = 8
MAX_VALUES = 8_000_000
MAX_PORTS = 4096
MAX_STATES = 1_000_000
MAX_PROVIDERS = 256

EXECUTABLE_PLAN_HEADER = struct.Struct(
    "<8sHH" + "I" * 9 + "Q" * 11 + "32s" * 4 + "56s"
)
EXECUTABLE_VALUE_RECORD = struct.Struct("<" + "I" * 8 + "Q" * 3 + "q" * 8 + "32s8s")
EXECUTABLE_PORT_RECORD = struct.Struct("<" + "I" * 8)
EXECUTABLE_STATE_RECORD = struct.Struct("<" + "I" * 4 + "Q" * 2)
EXECUTABLE_PROVIDER_RECORD = struct.Struct("<" + "I" * 8 + "32s")

assert EXECUTABLE_PLAN_HEADER.size == 320
assert EXECUTABLE_VALUE_RECORD.size == 160
assert EXECUTABLE_PORT_RECORD.size == 32
assert EXECUTABLE_STATE_RECORD.size == 32
assert EXECUTABLE_PROVIDER_RECORD.size == 64


class ExecutableValueRegion(IntEnum):
    UNUSED = 0
    EXTERNAL_INPUT = 1
    EXTERNAL_OUTPUT = 2
    WEIGHTS = 3
    STATE = 4
    ARENA = 5
    ALIAS = 6


class ExecutableDType(IntEnum):
    F32 = 1
    F16 = 2
    BF16 = 3
    I32 = 4
    BOOL = 5


class ExecutablePortKind(IntEnum):
    INPUT = 1
    OUTPUT = 2


class ExecutableStateInit(IntEnum):
    ZERO = 1


_DTYPE_IDS = {
    "f32": ExecutableDType.F32,
    "f16": ExecutableDType.F16,
    "bf16": ExecutableDType.BF16,
    "i32": ExecutableDType.I32,
    "bool": ExecutableDType.BOOL,
}
_DTYPE_BYTES = {
    ExecutableDType.F32: 4,
    ExecutableDType.F16: 2,
    ExecutableDType.BF16: 2,
    ExecutableDType.I32: 4,
    ExecutableDType.BOOL: 1,
}
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


@dataclass(frozen=True, slots=True)
class PackedWeightSpan:
    value_id: int
    offset: int
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        _uint(self.value_id, 32, "packed weight value_id")
        _uint(self.offset, 64, "packed weight offset")
        _positive_uint(self.byte_size, 64, "packed weight byte_size")
        _digest(self.sha256, "packed weight SHA-256")


@dataclass(frozen=True, slots=True)
class ExecutableValue:
    value_id: int
    region: ExecutableValueRegion
    dtype: ExecutableDType
    shape: tuple[int, ...]
    byte_size: int
    offset: int
    allocation_bytes: int
    alias_of: int | None
    producer: int | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class ExecutablePort:
    port_id: int
    kind: ExecutablePortKind
    value_id: int


@dataclass(frozen=True, slots=True)
class ExecutableState:
    state_id: int
    value_id: int
    init: ExecutableStateInit


@dataclass(frozen=True, slots=True)
class ExecutableProvider:
    provider_id: int
    abi_major: int
    abi_minor: int
    tag_mask: int
    command_count: int
    capture_safe_count: int
    usage_sha256: str


@dataclass(frozen=True, slots=True)
class ExecutablePlan:
    target_arch: CudaArch
    alignment: int
    arena_bytes: int
    state_bytes: int
    workspace_bytes: int
    weights_bytes: int
    schedule_sha256: str
    memory_plan_sha256: str
    command_stream_sha256: str
    values: tuple[ExecutableValue, ...]
    ports: tuple[ExecutablePort, ...]
    states: tuple[ExecutableState, ...]
    providers: tuple[ExecutableProvider, ...]
    command_stream: CommandStream
    data: bytes

    def summary(self) -> dict[str, int | str]:
        return {
            "schema": f"{EXECUTABLE_PLAN_SCHEMA_MAJOR}.{EXECUTABLE_PLAN_SCHEMA_MINOR}",
            "cuda_arch": self.target_arch.name_string,
            "values": len(self.values),
            "ports": len(self.ports),
            "states": len(self.states),
            "providers": len(self.providers),
            "commands": len(self.command_stream.commands),
            "arena_bytes": self.arena_bytes,
            "state_bytes": self.state_bytes,
            "workspace_bytes": self.workspace_bytes,
            "weights_bytes": self.weights_bytes,
            "bytes": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }


def compile_executable_plan(
    schedule: ExecutionSchedule,
    memory_plan: MemoryPlan,
    command_stream: CommandStream,
    packed_weights: Iterable[PackedWeightSpan],
    *,
    weights_bytes: int,
) -> ExecutablePlan:
    """Compile a complete, numeric, host-independent execution plan.

    The returned bytes contain no model names or host paths. Entry outputs are
    explicitly mapped to caller-owned buffers; no implicit terminal copy is
    introduced. Packed-weight spans cover only constants actually referenced by
    provider commands, while dead/fused-away constants are encoded as UNUSED.
    """

    if not isinstance(schedule, ExecutionSchedule):
        raise ValidationError("executable plan requires an ExecutionSchedule")
    if not isinstance(memory_plan, MemoryPlan):
        raise ValidationError("executable plan requires a MemoryPlan")
    if not isinstance(command_stream, CommandStream):
        raise ValidationError("executable plan requires a CommandStream")
    _positive_uint(weights_bytes, 64, "executable plan weights_bytes")
    schedule_digest = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    memory_digest = hashlib.sha256(dump_memory_plan(memory_plan).encode()).hexdigest()
    if memory_plan.schedule_sha256 != schedule_digest:
        raise ValidationError("executable plan memory plan does not match its schedule")
    if (
        command_stream.target_arch not in {CudaArch.SM89, CudaArch.SM110, CudaArch.SM120}
        or command_stream.value_count != len(schedule.values)
        or command_stream.memory_plan_sha256 != memory_digest
        or command_stream.arena_bytes != memory_plan.arena_bytes
        or command_stream.state_bytes != memory_plan.state_bytes
    ):
        raise ValidationError("executable plan command stream differs from its schedule or memory plan")
    if len(schedule.values) == 0 or len(schedule.values) > MAX_VALUES:
        raise ValidationError("executable plan value count is outside its bounds")
    if len(schedule.entry_inputs) + len(schedule.entry_outputs) > MAX_PORTS:
        raise ValidationError("executable plan port count exceeds its bound")
    if len(schedule.states) > MAX_STATES:
        raise ValidationError("executable plan state count exceeds its bound")

    allocations = {item.value_id: item for item in memory_plan.allocations}
    if set(allocations) != set(range(len(schedule.values))):
        raise ValidationError("executable plan memory allocation table is incomplete")
    roots = _alias_roots(memory_plan)
    input_roots = {roots[value_id] for _, value_id in schedule.entry_inputs}
    output_roots = {roots[value_id] for _, value_id in schedule.entry_outputs}
    if input_roots & output_roots:
        raise ValidationError("executable plan v2 does not allow one value to be both input and output")
    if len(output_roots) != len(schedule.entry_outputs):
        raise ValidationError("executable plan v2 output ports must have distinct storage roots")

    required_values = {
        operand.value_id
        for command in command_stream.commands
        for operand in command.operands
    }
    required_roots = {roots[value_id] for value_id in required_values}
    spans = tuple(sorted(packed_weights, key=lambda item: item.value_id))
    if not all(isinstance(item, PackedWeightSpan) for item in spans):
        raise ValidationError("executable plan packed weight table has an invalid record")
    span_by_value: dict[int, PackedWeightSpan] = {}
    previous_end = 0
    for span in sorted(spans, key=lambda item: (item.offset, item.value_id)):
        if span.value_id in span_by_value:
            raise ValidationError("executable plan packed weight table repeats a value")
        if span.value_id not in allocations or span.byte_size != allocations[span.value_id].byte_size:
            raise ValidationError("executable plan packed weight span has the wrong tensor size")
        if span.offset % memory_plan.alignment:
            raise ValidationError("executable plan packed weight span is misaligned")
        if span.offset < previous_end or span.offset > weights_bytes or span.byte_size > weights_bytes - span.offset:
            raise ValidationError("executable plan packed weight spans overlap or exceed the blob")
        previous_end = span.offset + span.byte_size
        span_by_value[span.value_id] = span

    required_constant_roots = {
        value_id
        for value_id in required_roots
        if allocations[value_id].region == AllocationRegion.CONSTANT
    }
    if set(span_by_value) != required_constant_roots:
        raise ValidationError(
            "executable plan packed weights do not exactly cover command-referenced constants"
        )

    value_records: list[ExecutableValue] = []
    for scheduled in schedule.values:
        value_id = scheduled.value_id
        allocation = allocations[value_id]
        dtype = _dtype(scheduled.type.dtype)
        shape = _static_shape(scheduled.type.shape, f"value {value_id}")
        if scheduled.type.device != "cuda" or scheduled.type.layout != "row_major":
            raise ValidationError("executable plan requires CUDA row-major tensors")
        byte_size = _tensor_bytes(dtype, shape, f"value {value_id}")
        if byte_size != allocation.byte_size:
            raise ValidationError("executable plan value size differs from its memory allocation")
        root = roots[value_id]
        span = span_by_value.get(value_id)
        region: ExecutableValueRegion
        offset = 0
        allocation_bytes = 0
        alias_of: int | None = None
        sha256: str | None = None
        if root != value_id:
            region = ExecutableValueRegion.ALIAS
            alias_of = root
        elif value_id in input_roots:
            region = ExecutableValueRegion.EXTERNAL_INPUT
            allocation_bytes = byte_size
        elif value_id in output_roots:
            region = ExecutableValueRegion.EXTERNAL_OUTPUT
            allocation_bytes = byte_size
        elif allocation.region == AllocationRegion.CONSTANT:
            if span is None:
                region = ExecutableValueRegion.UNUSED
            else:
                region = ExecutableValueRegion.WEIGHTS
                offset = span.offset
                allocation_bytes = span.byte_size
                sha256 = span.sha256
        elif allocation.region == AllocationRegion.UNUSED:
            region = ExecutableValueRegion.UNUSED
        elif allocation.region == AllocationRegion.STATE:
            region = ExecutableValueRegion.STATE
            offset = _present(allocation.offset, "state allocation offset")
            allocation_bytes = _present(allocation.allocation_bytes, "state allocation size")
        elif allocation.region == AllocationRegion.ARENA:
            region = ExecutableValueRegion.ARENA
            offset = _present(allocation.offset, "arena allocation offset")
            allocation_bytes = _present(allocation.allocation_bytes, "arena allocation size")
        elif allocation.region == AllocationRegion.EXTERNAL:
            raise ValidationError("executable plan found an external value that is not an entry port")
        else:
            raise ValidationError("executable plan found an unsupported allocation region")
        value_records.append(
            ExecutableValue(
                value_id,
                region,
                dtype,
                shape,
                byte_size,
                offset,
                allocation_bytes,
                alias_of,
                scheduled.producer,
                sha256,
            )
        )

    for command_index, command in enumerate(command_stream.commands):
        for operand in command.operands:
            root = roots[operand.value_id]
            record = value_records[root]
            if record.region == ExecutableValueRegion.UNUSED:
                raise ValidationError(
                    f"executable plan command {command_index} references an unused value"
                )
            if operand.byte_offset >= record.byte_size:
                raise ValidationError(
                    f"executable plan command {command_index} operand byte offset is out of bounds"
                )

    ports = tuple(
        ExecutablePort(index, ExecutablePortKind.INPUT, value_id)
        for index, (_, value_id) in enumerate(schedule.entry_inputs)
    ) + tuple(
        ExecutablePort(len(schedule.entry_inputs) + index, ExecutablePortKind.OUTPUT, value_id)
        for index, (_, value_id) in enumerate(schedule.entry_outputs)
    )
    states = tuple(
        ExecutableState(index, value_id, ExecutableStateInit.ZERO)
        for index, (_, value_id) in enumerate(schedule.states)
    )
    providers = _providers(command_stream)
    if len(providers) > MAX_PROVIDERS:
        raise ValidationError("executable plan provider count exceeds its bound")

    command_bytes = command_stream.to_bytes()
    values_offset = EXECUTABLE_PLAN_HEADER.size
    ports_offset = values_offset + len(value_records) * EXECUTABLE_VALUE_RECORD.size
    states_offset = ports_offset + len(ports) * EXECUTABLE_PORT_RECORD.size
    providers_offset = states_offset + len(states) * EXECUTABLE_STATE_RECORD.size
    commands_offset = providers_offset + len(providers) * EXECUTABLE_PROVIDER_RECORD.size
    section_size = commands_offset + len(command_bytes)
    body = bytearray()
    for value in value_records:
        padded_shape = value.shape + (0,) * (MAX_TENSOR_RANK - len(value.shape))
        body.extend(
            EXECUTABLE_VALUE_RECORD.pack(
                value.value_id,
                int(value.region),
                int(value.dtype),
                len(value.shape),
                0,
                _UINT32_MAX if value.alias_of is None else value.alias_of,
                _UINT32_MAX if value.producer is None else value.producer,
                0,
                value.offset,
                value.byte_size,
                value.allocation_bytes,
                *padded_shape,
                b"\0" * 32 if value.sha256 is None else bytes.fromhex(value.sha256),
                b"\0" * 8,
            )
        )
    for port in ports:
        body.extend(
            EXECUTABLE_PORT_RECORD.pack(
                port.port_id, int(port.kind), port.value_id, 2, 0, 0, 0, 0
            )
        )
    for state in states:
        body.extend(
            EXECUTABLE_STATE_RECORD.pack(
                state.state_id, state.value_id, int(state.init), 0, 0, 0
            )
        )
    for provider in providers:
        body.extend(
            EXECUTABLE_PROVIDER_RECORD.pack(
                provider.provider_id,
                provider.abi_major,
                provider.abi_minor,
                provider.tag_mask,
                provider.command_count,
                provider.capture_safe_count,
                0,
                0,
                bytes.fromhex(provider.usage_sha256),
            )
        )
    body.extend(command_bytes)
    if len(body) != section_size - EXECUTABLE_PLAN_HEADER.size:
        raise AssertionError("executable plan body size differs from its canonical tables")
    header = EXECUTABLE_PLAN_HEADER.pack(
        EXECUTABLE_PLAN_MAGIC,
        EXECUTABLE_PLAN_SCHEMA_MAJOR,
        EXECUTABLE_PLAN_SCHEMA_MINOR,
        EXECUTABLE_PLAN_HEADER.size,
        int(command_stream.target_arch),
        0,
        len(value_records),
        len(ports),
        len(states),
        len(providers),
        memory_plan.alignment,
        0,
        memory_plan.arena_bytes,
        memory_plan.state_bytes,
        command_stream.workspace_bytes,
        weights_bytes,
        values_offset,
        ports_offset,
        states_offset,
        providers_offset,
        commands_offset,
        len(command_bytes),
        section_size,
        bytes.fromhex(schedule_digest),
        bytes.fromhex(memory_digest),
        hashlib.sha256(command_bytes).digest(),
        hashlib.sha256(body).digest(),
        b"\0" * 56,
    )
    return parse_executable_plan(header + body, expected_weights_bytes=weights_bytes)


def parse_executable_plan(
    data: bytes | bytearray | memoryview,
    *,
    expected_arch: CudaArch | None = None,
    expected_weights_bytes: int | None = None,
) -> ExecutablePlan:
    view = memoryview(data)
    if len(view) < EXECUTABLE_PLAN_HEADER.size:
        raise FormatError("executable plan is smaller than its fixed header")
    fields = EXECUTABLE_PLAN_HEADER.unpack_from(view)
    (
        magic,
        major,
        minor,
        header_size,
        raw_arch,
        flags,
        value_count,
        port_count,
        state_count,
        provider_count,
        alignment,
        reserved_u32,
        arena_bytes,
        state_bytes,
        workspace_bytes,
        weights_bytes,
        values_offset,
        ports_offset,
        states_offset,
        providers_offset,
        commands_offset,
        command_bytes,
        section_size,
        schedule_digest,
        memory_digest,
        command_digest,
        body_digest,
        reserved,
    ) = fields
    if magic != EXECUTABLE_PLAN_MAGIC:
        raise FormatError("bad executable plan magic")
    if major != EXECUTABLE_PLAN_SCHEMA_MAJOR or minor > EXECUTABLE_PLAN_SCHEMA_MINOR:
        raise FormatError(f"unsupported executable plan schema {major}.{minor}")
    if header_size != EXECUTABLE_PLAN_HEADER.size or flags != 0 or reserved_u32 != 0 or any(reserved):
        raise FormatError("invalid executable plan header, flags, or reserved bytes")
    try:
        arch = CudaArch(raw_arch)
    except ValueError as exc:
        raise FormatError(f"unknown executable plan architecture sm{raw_arch}") from exc
    if expected_arch is not None and arch != expected_arch:
        raise FormatError("executable plan architecture differs from its AIM variant")
    if expected_weights_bytes is not None and weights_bytes != expected_weights_bytes:
        raise FormatError("executable plan weight size differs from its AIM variant")
    if (
        value_count == 0
        or value_count > MAX_VALUES
        or port_count == 0
        or port_count > MAX_PORTS
        or state_count > MAX_STATES
        or provider_count == 0
        or provider_count > MAX_PROVIDERS
        or alignment == 0
        or alignment > 2**20
        or alignment & (alignment - 1)
        or weights_bytes == 0
    ):
        raise FormatError("executable plan counts, alignment, or sizes are outside their bounds")
    expected_ports = EXECUTABLE_PLAN_HEADER.size + value_count * EXECUTABLE_VALUE_RECORD.size
    expected_states = expected_ports + port_count * EXECUTABLE_PORT_RECORD.size
    expected_providers = expected_states + state_count * EXECUTABLE_STATE_RECORD.size
    expected_commands = expected_providers + provider_count * EXECUTABLE_PROVIDER_RECORD.size
    if (
        values_offset != EXECUTABLE_PLAN_HEADER.size
        or ports_offset != expected_ports
        or states_offset != expected_states
        or providers_offset != expected_providers
        or commands_offset != expected_commands
        or commands_offset > len(view)
        or command_bytes != len(view) - commands_offset
        or section_size != len(view)
    ):
        raise FormatError("executable plan tables have invalid or non-canonical bounds")
    if not all(any(item) for item in (schedule_digest, memory_digest, command_digest, body_digest)):
        raise FormatError("executable plan has a zero identity digest")
    if hashlib.sha256(view[values_offset:]).digest() != body_digest:
        raise FormatError("executable plan body checksum mismatch")
    raw_command_stream = bytes(view[commands_offset:])
    if hashlib.sha256(raw_command_stream).digest() != command_digest:
        raise FormatError("executable plan command-stream checksum mismatch")
    try:
        command_stream = CommandStream.from_bytes(raw_command_stream)
    except FormatError as exc:
        raise FormatError(f"invalid executable plan command stream: {exc}") from exc
    if (
        command_stream.target_arch != arch
        or command_stream.value_count != value_count
        or command_stream.memory_plan_sha256 != memory_digest.hex()
        or command_stream.arena_bytes != arena_bytes
        or command_stream.state_bytes != state_bytes
        or command_stream.workspace_bytes != workspace_bytes
    ):
        raise FormatError("executable plan command stream differs from its header")

    values: list[ExecutableValue] = []
    for index in range(value_count):
        record = EXECUTABLE_VALUE_RECORD.unpack_from(
            view, values_offset + index * EXECUTABLE_VALUE_RECORD.size
        )
        (
            value_id,
            raw_region,
            raw_dtype,
            rank,
            value_flags,
            raw_alias,
            raw_producer,
            value_reserved,
            offset,
            byte_size,
            allocation_bytes,
            *tail,
        ) = record
        shape = tuple(tail[:MAX_TENSOR_RANK])
        digest = tail[MAX_TENSOR_RANK]
        trailing_reserved = tail[MAX_TENSOR_RANK + 1]
        try:
            region = ExecutableValueRegion(raw_region)
            dtype = ExecutableDType(raw_dtype)
        except ValueError as exc:
            raise FormatError(f"executable plan value {index} has an unknown enum") from exc
        try:
            tensor_bytes = _tensor_bytes(dtype, shape[:rank], f"value {index}")
        except ValidationError as exc:
            raise FormatError(str(exc)) from exc
        if (
            value_id != index
            or value_flags != 0
            or value_reserved != 0
            or any(trailing_reserved)
            or rank == 0
            or rank > MAX_TENSOR_RANK
            or any(dimension <= 0 for dimension in shape[:rank])
            or any(shape[rank:])
            or byte_size != tensor_bytes
        ):
            raise FormatError(f"executable plan value {index} has an invalid tensor contract")
        alias_of = None if raw_alias == _UINT32_MAX else raw_alias
        producer = None if raw_producer == _UINT32_MAX else raw_producer
        if region == ExecutableValueRegion.ALIAS:
            if alias_of is None or alias_of >= value_count or alias_of == value_id:
                raise FormatError(f"executable plan value {index} has an invalid alias")
            if offset != 0 or allocation_bytes != 0 or any(digest):
                raise FormatError(f"executable plan alias value {index} owns storage")
        else:
            if alias_of is not None:
                raise FormatError(f"executable plan owned value {index} has an alias target")
            if region == ExecutableValueRegion.UNUSED:
                if offset != 0 or allocation_bytes != 0 or any(digest):
                    raise FormatError(f"executable plan unused value {index} owns storage")
            elif region in {ExecutableValueRegion.EXTERNAL_INPUT, ExecutableValueRegion.EXTERNAL_OUTPUT}:
                if offset != 0 or allocation_bytes != byte_size or any(digest):
                    raise FormatError(f"executable plan external value {index} has invalid storage")
            else:
                limit = {
                    ExecutableValueRegion.WEIGHTS: weights_bytes,
                    ExecutableValueRegion.STATE: state_bytes,
                    ExecutableValueRegion.ARENA: arena_bytes,
                }[region]
                if (
                    allocation_bytes < byte_size
                    or offset > limit
                    or allocation_bytes > limit - offset
                    or (region == ExecutableValueRegion.WEIGHTS) != bool(any(digest))
                ):
                    raise FormatError(f"executable plan value {index} exceeds its storage region")
                if offset % alignment:
                    raise FormatError(f"executable plan value {index} is misaligned")
        values.append(
            ExecutableValue(
                value_id,
                region,
                dtype,
                shape[:rank],
                byte_size,
                offset,
                allocation_bytes,
                alias_of,
                producer,
                digest.hex() if any(digest) else None,
            )
        )
    weight_ranges = sorted(
        (value.offset, value.offset + value.allocation_bytes)
        for value in values if value.region == ExecutableValueRegion.WEIGHTS
    )
    if any(left[1] > right[0] for left, right in zip(weight_ranges, weight_ranges[1:])):
        raise FormatError("executable plan packed weight spans overlap")
    roots = _parsed_alias_roots(values)
    for value in values:
        if value.region == ExecutableValueRegion.ALIAS and value.byte_size != values[roots[value.value_id]].byte_size:
            raise FormatError("executable plan alias changes tensor byte size")

    ports: list[ExecutablePort] = []
    port_values: set[int] = set()
    for index in range(port_count):
        record = EXECUTABLE_PORT_RECORD.unpack_from(
            view, ports_offset + index * EXECUTABLE_PORT_RECORD.size
        )
        port_id, raw_kind, value_id, location, port_flags, r0, r1, r2 = record
        try:
            kind = ExecutablePortKind(raw_kind)
        except ValueError as exc:
            raise FormatError(f"executable plan port {index} has an unknown kind") from exc
        expected_region = (
            ExecutableValueRegion.EXTERNAL_INPUT
            if kind == ExecutablePortKind.INPUT
            else ExecutableValueRegion.EXTERNAL_OUTPUT
        )
        if (
            port_id != index
            or value_id >= value_count
            or location != 2
            or port_flags != 0
            or r0 != 0
            or r1 != 0
            or r2 != 0
            or values[roots[value_id]].region != expected_region
            or roots[value_id] in port_values
        ):
            raise FormatError(f"executable plan port {index} has an invalid value contract")
        port_values.add(roots[value_id])
        ports.append(ExecutablePort(port_id, kind, value_id))
    if not any(item.kind == ExecutablePortKind.INPUT for item in ports) or not any(
        item.kind == ExecutablePortKind.OUTPUT for item in ports
    ):
        raise FormatError("executable plan requires input and output ports")
    if port_values != {v.value_id for v in values if v.region in {
        ExecutableValueRegion.EXTERNAL_INPUT, ExecutableValueRegion.EXTERNAL_OUTPUT
    }}:
        raise FormatError("executable plan does not bind every external storage root")

    states: list[ExecutableState] = []
    state_values: set[int] = set()
    for index in range(state_count):
        state_id, value_id, raw_init, state_flags, initial_offset, state_reserved = (
            EXECUTABLE_STATE_RECORD.unpack_from(
                view, states_offset + index * EXECUTABLE_STATE_RECORD.size
            )
        )
        try:
            init = ExecutableStateInit(raw_init)
        except ValueError as exc:
            raise FormatError(f"executable plan state {index} has an unknown initializer") from exc
        if (
            state_id != index
            or value_id >= value_count
            or values[roots[value_id]].region != ExecutableValueRegion.STATE
            or roots[value_id] in state_values
            or state_flags != 0
            or initial_offset != 0
            or state_reserved != 0
        ):
            raise FormatError(f"executable plan state {index} has an invalid value contract")
        state_values.add(roots[value_id])
        states.append(ExecutableState(state_id, value_id, init))

    if state_values != {v.value_id for v in values if v.region == ExecutableValueRegion.STATE}:
        raise FormatError("executable plan does not initialize every state storage root")
    providers: list[ExecutableProvider] = []
    previous_provider = 0
    for index in range(provider_count):
        record = EXECUTABLE_PROVIDER_RECORD.unpack_from(
            view, providers_offset + index * EXECUTABLE_PROVIDER_RECORD.size
        )
        provider_id, abi_major, abi_minor, tag_mask, count, capture_count, p0, p1, usage_digest = record
        if (
            provider_id <= previous_provider
            or abi_major == 0
            or tag_mask == 0
            or tag_mask & ~sum(1 << int(tag) for tag in CommandTag)
            or count == 0
            or capture_count > count
            or p0 != 0
            or p1 != 0
            or not any(usage_digest)
        ):
            raise FormatError(f"executable plan provider {index} has an invalid descriptor")
        providers.append(
            ExecutableProvider(
                provider_id,
                abi_major,
                abi_minor,
                tag_mask,
                count,
                capture_count,
                usage_digest.hex(),
            )
        )
        previous_provider = provider_id
    if tuple(providers) != _providers(command_stream):
        raise FormatError("executable plan provider table differs from its command stream")

    for command_index, command in enumerate(command_stream.commands):
        for operand in command.operands:
            root = roots[operand.value_id]
            if values[root].region == ExecutableValueRegion.UNUSED or operand.byte_offset >= values[root].byte_size:
                raise FormatError(
                    f"executable plan command {command_index} has an out-of-range value operand"
                )
    return ExecutablePlan(
        arch,
        alignment,
        arena_bytes,
        state_bytes,
        workspace_bytes,
        weights_bytes,
        schedule_digest.hex(),
        memory_digest.hex(),
        command_digest.hex(),
        tuple(values),
        tuple(ports),
        tuple(states),
        tuple(providers),
        command_stream,
        bytes(view),
    )


def _providers(command_stream: CommandStream) -> tuple[ExecutableProvider, ...]:
    grouped: dict[int, list[object]] = {}
    for command in command_stream.commands:
        grouped.setdefault(command.provider_id, []).append(command)
    result: list[ExecutableProvider] = []
    for provider_id, raw_commands in sorted(grouped.items()):
        commands = tuple(raw_commands)
        abi_pairs = {(item.abi_major, item.abi_minor) for item in commands}  # type: ignore[attr-defined]
        if len(abi_pairs) != 1:
            raise ValidationError(f"provider {provider_id} uses inconsistent ABI versions")
        abi_major, abi_minor = next(iter(abi_pairs))
        tag_mask = 0
        usage = bytearray()
        for command in commands:
            tag_mask |= 1 << int(command.tag)  # type: ignore[attr-defined]
            usage.extend(bytes.fromhex(command.capability_digest))  # type: ignore[attr-defined]
            usage.extend(hashlib.sha256(command.payload).digest())  # type: ignore[attr-defined]
        result.append(
            ExecutableProvider(
                provider_id,
                abi_major,
                abi_minor,
                tag_mask,
                len(commands),
                sum(bool(item.capture_safe) for item in commands),  # type: ignore[attr-defined]
                hashlib.sha256(usage).hexdigest(),
            )
        )
    return tuple(result)


def _alias_roots(memory_plan: MemoryPlan) -> tuple[int, ...]:
    allocations = {item.value_id: item for item in memory_plan.allocations}
    roots: list[int] = []
    for value_id in range(len(memory_plan.allocations)):
        root = value_id
        seen: set[int] = set()
        while allocations[root].region == AllocationRegion.ALIAS:
            if root in seen or allocations[root].alias_of is None:
                raise ValidationError("executable plan memory aliases contain a cycle or missing target")
            seen.add(root)
            root = allocations[root].alias_of
            if root not in allocations:
                raise ValidationError("executable plan memory alias target is out of range")
        roots.append(root)
    return tuple(roots)


def _parsed_alias_roots(values: list[ExecutableValue]) -> tuple[int, ...]:
    roots: list[int] = []
    for value in values:
        root = value.value_id
        seen: set[int] = set()
        while values[root].region == ExecutableValueRegion.ALIAS:
            if root in seen or values[root].alias_of is None:
                raise FormatError("executable plan alias graph contains a cycle")
            seen.add(root)
            root = values[root].alias_of
        roots.append(root)
    return tuple(roots)


def _dtype(value: str) -> ExecutableDType:
    try:
        return _DTYPE_IDS[value]
    except KeyError as exc:
        raise ValidationError(f"executable plan does not support dtype {value}") from exc


def _static_shape(value: tuple[int | str, ...], label: str) -> tuple[int, ...]:
    if not 1 <= len(value) <= MAX_TENSOR_RANK or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value
    ):
        raise ValidationError(f"executable plan {label} requires a bounded static shape")
    return tuple(value)  # type: ignore[return-value]


def _tensor_bytes(dtype: ExecutableDType, shape: tuple[int, ...], label: str) -> int:
    count = math.prod(shape)
    item_bytes = _DTYPE_BYTES[dtype]
    if count <= 0 or count > _UINT64_MAX // item_bytes:
        raise ValidationError(f"executable plan {label} byte size overflows uint64")
    return count * item_bytes


def _present(value: int | None, label: str) -> int:
    if value is None:
        raise ValidationError(f"executable plan is missing {label}")
    return value


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationError(f"{label} must be a canonical nonzero SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValidationError(f"{label} must be a canonical nonzero SHA-256 digest") from exc
    if len(decoded) != 32 or not any(decoded) or decoded.hex() != value:
        raise ValidationError(f"{label} must be a canonical nonzero SHA-256 digest")


def _positive_uint(value: object, bits: int, label: str) -> None:
    _uint(value, bits, label)
    if value == 0:
        raise ValidationError(f"{label} must be positive")


def _uint(value: object, bits: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 2**bits - 1:
        raise ValidationError(f"{label} must be a uint{bits}")
