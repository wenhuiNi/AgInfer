from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

from .errors import FormatError, ValidationError
from .schema import CudaArch

PLAN_MAGIC = b"AIMPLN1\0"
PLAN_SCHEMA_MAJOR = 1
PLAN_SCHEMA_MINOR = 0
MAX_TENSOR_RANK = 8

PLAN_HEADER_STRUCT = struct.Struct("<8sHHIIIIIIIQQQQQQQQ24s")
PROFILE_STRUCT = struct.Struct("<IIIIII40s")
TENSOR_STRUCT = struct.Struct("<IIIIIIQ8q8q")
LAUNCH_STRUCT = struct.Struct("<IIIIIIIIIIII16s")
ARG_STRUCT = struct.Struct("<IIQQQ")

assert PLAN_HEADER_STRUCT.size == 128
assert PROFILE_STRUCT.size == 64
assert TENSOR_STRUCT.size == 160
assert LAUNCH_STRUCT.size == 64
assert ARG_STRUCT.size == 32

DTYPE_IDS = {"F32": 1, "F16": 2, "BF16": 3, "F8_E4M3": 4, "I32": 5}
DTYPE_SIZES = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "I32": 4}
LOCATION_IDS = {"host": 1, "device": 2}
IO_IDS = {"input": 1, "output": 2}
ARG_IDS = {"tensor": 1, "scalar_u32": 2, "scalar_f32": 3, "arena_offset": 4, "weights_offset": 5}


@dataclass(frozen=True)
class CompiledPlan:
    data: bytes
    profiles: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    launch_count: int


class _StringTable:
    def __init__(self) -> None:
        self.data = bytearray(b"\0")
        self.offsets: dict[str, int] = {"": 0}

    def add(self, value: str, label: str) -> int:
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValidationError(f"{label} must be a non-empty string without NUL bytes")
        encoded = value.encode("utf-8")
        if len(encoded) > 4096:
            raise ValidationError(f"{label} is too long")
        if value not in self.offsets:
            self.offsets[value] = len(self.data)
            self.data.extend(encoded)
            self.data.append(0)
        return self.offsets[value]


def compile_execution_plan(plan: dict[str, Any], arch: CudaArch, *, weight_size: int) -> CompiledPlan:
    if not isinstance(plan, dict):
        raise ValidationError("execution plan must be an object")
    if plan.get("cuda_arch") != arch.name_string:
        raise ValidationError(f"execution plan target must be exactly {arch.name_string}")
    arena_bytes = _unsigned(plan.get("arena_bytes"), "arena_bytes")
    workspace_bytes = _unsigned(plan.get("workspace_bytes"), "workspace_bytes")
    dispatches = plan.get("shape_dispatch")
    if not isinstance(dispatches, list) or not dispatches:
        raise ValidationError("execution plan needs a non-empty shape_dispatch list")
    graph_profiles = _validate_graph_templates(plan.get("cuda_graph_templates"), dispatches)

    strings = _StringTable()
    profile_records: list[tuple[int, int, int, int, int, int]] = []
    tensor_records: list[tuple[int, int, int, int, int, int, int, tuple[int, ...], tuple[int, ...]]] = []
    launch_records: list[tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int], int, int]] = []
    argument_records: list[tuple[int, int, int, int]] = []
    profile_names: list[str] = []
    first_input_names: tuple[str, ...] = ()
    first_output_names: tuple[str, ...] = ()

    for profile_index, dispatch in enumerate(dispatches):
        if not isinstance(dispatch, dict):
            raise ValidationError(f"shape_dispatch[{profile_index}] must be an object")
        profile = dispatch.get("profile")
        profile_offset = strings.add(profile, f"shape_dispatch[{profile_index}].profile")
        if profile in profile_names:
            raise ValidationError(f"duplicate execution profile: {profile}")
        profile_names.append(profile)

        first_tensor = len(tensor_records)
        names: dict[str, int] = {}
        device_names: set[str] = set()
        input_names: list[str] = []
        output_names: list[str] = []
        for io_name, io_kind, collected_names in (
            ("inputs", "input", input_names),
            ("outputs", "output", output_names),
        ):
            specs = dispatch.get(io_name)
            if not isinstance(specs, list) or not specs:
                raise ValidationError(f"profile {profile}: {io_name} must be a non-empty list")
            for item_index, spec in enumerate(specs):
                record = _compile_tensor(
                    spec,
                    io_kind,
                    strings,
                    f"profile {profile} {io_name}[{item_index}]",
                )
                name = spec["name"]
                if name in names:
                    raise ValidationError(f"profile {profile}: duplicate tensor name {name}")
                names[name] = len(tensor_records)
                if record[2] == LOCATION_IDS["device"]:
                    device_names.add(name)
                collected_names.append(name)
                tensor_records.append(record)
        if profile_index == 0:
            first_input_names = tuple(input_names)
            first_output_names = tuple(output_names)

        first_launch = len(launch_records)
        launches = dispatch.get("launches")
        if not isinstance(launches, list) or not launches:
            raise ValidationError(f"profile {profile}: launches must be a non-empty list")
        for launch_index, launch in enumerate(launches):
            if not isinstance(launch, dict):
                raise ValidationError(f"profile {profile} launch {launch_index} must be an object")
            kernel_offset = strings.add(launch.get("kernel"), f"profile {profile} launch {launch_index}.kernel")
            grid = _dimensions(launch.get("grid"), f"profile {profile} launch {launch_index}.grid")
            block = _dimensions(launch.get("block"), f"profile {profile} launch {launch_index}.block")
            if math.prod(block) > 1024:
                raise ValidationError(f"profile {profile} launch {launch_index}: block has more than 1024 threads")
            shared_bytes = _unsigned(launch.get("shared_bytes", 0), "shared_bytes", maximum=2**32 - 1)
            arguments = launch.get("arguments", [])
            if not isinstance(arguments, list):
                raise ValidationError(f"profile {profile} launch {launch_index}.arguments must be a list")
            first_arg = len(argument_records)
            for arg_index, argument in enumerate(arguments):
                argument_records.append(
                    _compile_argument(
                        argument,
                        names,
                        device_names,
                        arena_bytes,
                        weight_size,
                        f"profile {profile} launch {launch_index} argument {arg_index}",
                    )
                )
            launch_records.append(
                (kernel_offset, profile_index, first_arg, len(arguments), grid, block, shared_bytes, 0)
            )
        flags = 1 if profile in graph_profiles else 0
        profile_records.append(
            (profile_offset, first_tensor, len(tensor_records) - first_tensor, first_launch, len(launches), flags)
        )

    profiles_offset = PLAN_HEADER_STRUCT.size
    tensors_offset = profiles_offset + len(profile_records) * PROFILE_STRUCT.size
    launches_offset = tensors_offset + len(tensor_records) * TENSOR_STRUCT.size
    args_offset = launches_offset + len(launch_records) * LAUNCH_STRUCT.size
    strings_offset = args_offset + len(argument_records) * ARG_STRUCT.size
    file_size = strings_offset + len(strings.data)
    header = PLAN_HEADER_STRUCT.pack(
        PLAN_MAGIC,
        PLAN_SCHEMA_MAJOR,
        PLAN_SCHEMA_MINOR,
        PLAN_HEADER_STRUCT.size,
        int(arch),
        len(profile_records),
        len(tensor_records),
        len(launch_records),
        len(argument_records),
        len(strings.data),
        arena_bytes,
        workspace_bytes,
        profiles_offset,
        tensors_offset,
        launches_offset,
        args_offset,
        strings_offset,
        file_size,
        b"\0" * 24,
    )
    output = bytearray(header)
    for record in profile_records:
        output.extend(PROFILE_STRUCT.pack(*record, b"\0" * 40))
    for name, dtype, location, io_kind, rank, flags, byte_size, shape, stride in tensor_records:
        padded_shape = shape + (0,) * (MAX_TENSOR_RANK - rank)
        padded_stride = stride + (0,) * (MAX_TENSOR_RANK - rank)
        output.extend(
            TENSOR_STRUCT.pack(name, dtype, location, io_kind, rank, flags, byte_size, *padded_shape, *padded_stride)
        )
    for kernel, profile, first_arg, arg_count, grid, block, shared, flags in launch_records:
        output.extend(
            LAUNCH_STRUCT.pack(
                kernel,
                profile,
                first_arg,
                arg_count,
                *grid,
                *block,
                shared,
                flags,
                b"\0" * 16,
            )
        )
    for kind, index, offset, value in argument_records:
        output.extend(ARG_STRUCT.pack(kind, index, offset, value, 0))
    output.extend(strings.data)
    if len(output) != file_size:
        raise AssertionError("internal execution plan size mismatch")
    return CompiledPlan(bytes(output), tuple(profile_names), first_input_names, first_output_names, len(launch_records))


def inspect_plan(data: bytes) -> dict[str, int]:
    """Validate the fixed Plan v1 header and return its summary fields."""
    if len(data) < PLAN_HEADER_STRUCT.size:
        raise FormatError("execution plan is smaller than its fixed header")
    values = PLAN_HEADER_STRUCT.unpack_from(data)
    if values[0] != PLAN_MAGIC:
        raise FormatError("bad AIM execution plan magic")
    if values[1] != PLAN_SCHEMA_MAJOR or values[3] != PLAN_HEADER_STRUCT.size:
        raise FormatError("unsupported AIM execution plan schema")
    if values[17] != len(data):
        raise FormatError("execution plan size mismatch")
    if any(values[18]):
        raise FormatError("non-zero execution plan reserved bytes")
    return {
        "arch": values[4],
        "profile_count": values[5],
        "tensor_count": values[6],
        "launch_count": values[7],
        "argument_count": values[8],
        "arena_bytes": values[10],
        "workspace_bytes": values[11],
    }


def _compile_tensor(
    spec: Any,
    io_kind: str,
    strings: _StringTable,
    label: str,
) -> tuple[int, int, int, int, int, int, int, tuple[int, ...], tuple[int, ...]]:
    if not isinstance(spec, dict):
        raise ValidationError(f"{label} must be an object")
    name_offset = strings.add(spec.get("name"), f"{label}.name")
    dtype_name = spec.get("dtype")
    if dtype_name not in DTYPE_IDS:
        raise ValidationError(f"{label}.dtype must be one of {', '.join(DTYPE_IDS)}")
    location_name = spec.get("location", "device")
    if location_name not in LOCATION_IDS:
        raise ValidationError(f"{label}.location must be host or device")
    shape_value = spec.get("shape")
    if not isinstance(shape_value, list) or not 1 <= len(shape_value) <= MAX_TENSOR_RANK:
        raise ValidationError(f"{label}.shape must contain 1 to {MAX_TENSOR_RANK} dimensions")
    shape = tuple(_positive_int(item, f"{label}.shape") for item in shape_value)
    stride_value = spec.get("stride")
    if stride_value is None:
        stride = _contiguous_stride(shape)
    else:
        if not isinstance(stride_value, list) or len(stride_value) != len(shape):
            raise ValidationError(f"{label}.stride must have the same rank as shape")
        stride = tuple(_positive_int(item, f"{label}.stride") for item in stride_value)
    required_bytes = _required_bytes(shape, stride, DTYPE_SIZES[dtype_name])
    byte_size = _unsigned(spec.get("byte_size", required_bytes), f"{label}.byte_size")
    if byte_size < required_bytes:
        raise ValidationError(f"{label}.byte_size is smaller than the declared shape and stride")
    return (
        name_offset,
        DTYPE_IDS[dtype_name],
        LOCATION_IDS[location_name],
        IO_IDS[io_kind],
        len(shape),
        0,
        byte_size,
        shape,
        stride,
    )


def _compile_argument(
    argument: Any,
    names: dict[str, int],
    device_names: set[str],
    arena_bytes: int,
    weight_size: int,
    label: str,
) -> tuple[int, int, int, int]:
    if not isinstance(argument, dict) or len(argument) != 1:
        raise ValidationError(f"{label} must contain exactly one argument kind")
    kind, raw_value = next(iter(argument.items()))
    if kind not in ARG_IDS:
        raise ValidationError(f"{label} has unsupported kind {kind!r}")
    if kind == "tensor":
        if raw_value not in names:
            raise ValidationError(f"{label} references unknown tensor {raw_value!r}")
        if raw_value not in device_names:
            raise ValidationError(f"{label} references a host tensor; Kernel arguments must be device tensors")
        return ARG_IDS[kind], names[raw_value], 0, 0
    if kind == "scalar_u32":
        return ARG_IDS[kind], 0, 0, _unsigned(raw_value, label, maximum=2**32 - 1)
    if kind == "scalar_f32":
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)) or not math.isfinite(raw_value):
            raise ValidationError(f"{label} must be a finite float")
        try:
            bits = struct.unpack("<I", struct.pack("<f", float(raw_value)))[0]
        except OverflowError as exc:
            raise ValidationError(f"{label} is outside the FP32 range") from exc
        return ARG_IDS[kind], 0, 0, bits
    offset = _unsigned(raw_value, label)
    limit = arena_bytes if kind == "arena_offset" else weight_size
    if offset >= limit:
        raise ValidationError(f"{label} offset {offset} is outside its {limit}-byte allocation")
    return ARG_IDS[kind], 0, offset, 0


def _validate_graph_templates(value: Any, dispatches: list[Any]) -> set[str]:
    if not isinstance(value, list):
        raise ValidationError("cuda_graph_templates must be a list")
    known = {
        item.get("profile")
        for item in dispatches
        if isinstance(item, dict) and isinstance(item.get("profile"), str)
    }
    result: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or item.get("profile") not in known:
            raise ValidationError(f"cuda_graph_templates[{index}] references an unknown profile")
        profile = item["profile"]
        if profile in result:
            raise ValidationError(f"duplicate CUDA Graph template for profile {profile}")
        result.add(profile)
    return result


def _dimensions(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValidationError(f"{label} must contain exactly three dimensions")
    return tuple(_positive_int(item, label, maximum=2**32 - 1) for item in value)  # type: ignore[return-value]


def _unsigned(value: Any, label: str, maximum: int = 2**64 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label} must be an unsigned integer")
    converted = value
    if converted < 0 or converted > maximum:
        raise ValidationError(f"{label} must be an unsigned integer no greater than {maximum}")
    return converted


def _positive_int(value: Any, label: str, maximum: int = 2**63 - 1) -> int:
    result = _unsigned(value, label, maximum)
    if result == 0:
        raise ValidationError(f"{label} must be positive")
    return result


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    value = 1
    for dimension in reversed(shape):
        result.append(value)
        value *= dimension
    return tuple(reversed(result))


def _required_bytes(shape: tuple[int, ...], stride: tuple[int, ...], item_size: int) -> int:
    last_element = sum((dimension - 1) * step for dimension, step in zip(shape, stride))
    return (last_element + 1) * item_size
