from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import TypeAlias

from ..errors import ValidationError

PROGRAM_IR_SCHEMA_MAJOR = 1
PROGRAM_IR_SCHEMA_MINOR = 0


class DType(str, Enum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    I32 = "i32"
    BOOL = "bool"

    @property
    def is_float(self) -> bool:
        return self in {DType.F32, DType.F16, DType.BF16}


class Layout(str, Enum):
    ROW_MAJOR = "row_major"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


Dimension: TypeAlias = int | str
Attribute: TypeAlias = bool | int | float | str | tuple[object, ...]


@dataclass(frozen=True)
class DimensionRange:
    symbol: str
    minimum: int
    preferred: int
    maximum: int

    def __post_init__(self) -> None:
        _validate_identifier(self.symbol, "shape symbol")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in self.bounds):
            raise ValidationError(f"shape range {self.symbol} bounds must be integers")
        if self.minimum <= 0 or not self.minimum <= self.preferred <= self.maximum:
            raise ValidationError(
                f"shape range {self.symbol} must satisfy 0 < minimum <= preferred <= maximum"
            )

    @property
    def bounds(self) -> tuple[int, int, int]:
        return self.minimum, self.preferred, self.maximum


@dataclass(frozen=True)
class ShapeDomain:
    dimensions: tuple[DimensionRange, ...] = ()

    def __post_init__(self) -> None:
        names = [dimension.symbol for dimension in self.dimensions]
        if len(names) != len(set(names)):
            raise ValidationError("shape domain contains duplicate symbols")

    def get(self, symbol: str) -> DimensionRange | None:
        return next((dimension for dimension in self.dimensions if dimension.symbol == symbol), None)


@dataclass(frozen=True)
class TensorType:
    dtype: DType
    shape: tuple[Dimension, ...]
    layout: Layout = Layout.ROW_MAJOR
    device: Device = Device.CPU

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise ValidationError("tensor dtype must be a ProgramIR DType")
        if not isinstance(self.layout, Layout):
            raise ValidationError("tensor layout must be a ProgramIR Layout")
        if not isinstance(self.device, Device):
            raise ValidationError("tensor device must be a ProgramIR Device")
        for dimension in self.shape:
            if isinstance(dimension, bool) or not isinstance(dimension, (int, str)):
                raise ValidationError("tensor dimensions must be positive integers or shape symbols")
            if isinstance(dimension, int) and dimension <= 0:
                raise ValidationError("static tensor dimensions must be positive")
            if isinstance(dimension, str):
                _validate_identifier(dimension, "tensor shape symbol")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def static_numel(self) -> int | None:
        return prod(self.shape) if all(isinstance(dimension, int) for dimension in self.shape) else None


@dataclass(frozen=True)
class Value:
    value_id: str
    type: TensorType

    def __post_init__(self) -> None:
        _validate_identifier(self.value_id, "SSA value ID")


@dataclass(frozen=True)
class Op:
    opcode: str
    inputs: tuple[str, ...]
    outputs: tuple[Value, ...]
    attributes: tuple[tuple[str, Attribute], ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.opcode, "opcode")
        for value_id in self.inputs:
            _validate_identifier(value_id, "op input ID")
        for key, _ in self.attributes:
            _validate_identifier(key, "attribute name")
        keys = [key for key, _ in self.attributes]
        if len(keys) != len(set(keys)):
            raise ValidationError(f"{self.opcode} contains duplicate attributes")

    def attribute(self, name: str, default: object = None) -> object:
        return next((value for key, value in self.attributes if key == name), default)


@dataclass(frozen=True)
class Region:
    ops: tuple[Op, ...]


@dataclass(frozen=True)
class Function:
    name: str
    inputs: tuple[Value, ...]
    outputs: tuple[str, ...]
    body: Region

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "function name")
        for value_id in self.outputs:
            _validate_identifier(value_id, "function output ID")


class StateAccess(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class State:
    name: str
    type: TensorType
    access: StateAccess

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "state name")
        if not isinstance(self.access, StateAccess):
            raise ValidationError("state access must be a ProgramIR StateAccess")


@dataclass(frozen=True)
class Program:
    functions: tuple[Function, ...]
    entry: str
    states: tuple[State, ...] = ()
    shape_domain: ShapeDomain = ShapeDomain()
    schema_major: int = PROGRAM_IR_SCHEMA_MAJOR
    schema_minor: int = PROGRAM_IR_SCHEMA_MINOR

    def __post_init__(self) -> None:
        _validate_identifier(self.entry, "entry function")


def attributes(**values: Attribute) -> tuple[tuple[str, Attribute], ...]:
    """Build canonically ordered immutable op attributes."""

    return tuple(sorted(values.items()))


def _validate_identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValidationError(f"{label} must be a non-empty string without NUL bytes")
    if not (value[0].isalpha() or value[0] == "_") or not all(
        character.isalnum() or character in {"_", "."} for character in value
    ):
        raise ValidationError(f"{label} has invalid characters: {value!r}")
