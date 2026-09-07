from __future__ import annotations

from .build_binding import BuildBinding, is_build_binding

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from ..errors import FormatError, ValidationError
from ..lowering.capability import ProviderCapability, dump_provider_capabilities
from ..lowering.command import CommandOperand, CommandTag, OperandAccess, ProviderCommand
from ..lowering.inventory import (
    InventoryOp,
    LoweringInventory,
    LoweringKind,
    RequirementStatus,
    dump_lowering_inventory,
)
from ..lowering.memory import MemoryPlan
from ..lowering.schedule import ExecutionSchedule, dump_execution_schedule
from ..schema import CudaArch


CUDA_KERNEL_PAYLOAD_MAGIC = b"AICUKR1\0"
CUDA_KERNEL_SCHEMA_MAJOR = 1
CUDA_KERNEL_SCHEMA_MINOR = 0
CUDA_KERNEL_PAYLOAD = struct.Struct(
    "<8sHH" + "I" * 9 + "Q" * 2 + "32s" + "I" * 4 + "16s"
)
CUDA_KERNEL_FLAG_CONTIGUOUS = 1
CUDA_KERNEL_MODULE_CUBIN = 1
CUDA_KERNEL_LAUNCH_UNARY_POINTERS_NUMEL = 1
CUDA_KERNEL_LAUNCH_BINARY_POINTERS_NUMEL = 2
CUDA_KERNEL_PROVIDER_ID = 2
CUDA_KERNEL_PROVIDER_ABI_MAJOR = 1
CUDA_KERNEL_PROVIDER_ABI_MINOR = 0
CUDA_KERNEL_PROVIDER_VERSION = "aginfer-aot-cuda=1"
CUDA_KERNEL_MAX_GRID_X = 65_535
CUDA_KERNEL_BLOCK_X = 256
CUDA_KERNEL_MAX_SHARED_BYTES = 96 * 1024
CUDA_KERNEL_MAX_ALIGNMENT = 1 << 20
AOT_CAST_RECEIPT_SCHEMA = "aginfer.aot-cast-validation.v1"
AOT_CAST_PARTIAL_LOWERING_SCHEMA = "aginfer.aot-cast-partial-lowering.v1"
AOT_POINTWISE_RECEIPT_SCHEMA = "aginfer.aot-pointwise-validation.v1"
AOT_POINTWISE_PARTIAL_LOWERING_SCHEMA = "aginfer.aot-pointwise-partial-lowering.v1"

assert CUDA_KERNEL_PAYLOAD.size == 128


class CudaKernelId(IntEnum):
    CAST_BF16_TO_F32 = 1
    CAST_F32_TO_BF16 = 2
    CAST_BOOL_TO_I32 = 3
    ADD_F32 = 4
    ADD_BF16 = 5
    ADD_I32 = 6
    MUL_F32 = 7
    MUL_BF16 = 8
    GELU_F32 = 9
    GELU_BF16 = 10
    SILU_F32 = 11

    @property
    def implementation_id(self) -> str:
        return {
            CudaKernelId.CAST_BF16_TO_F32: "aginfer.cast.bf16_to_f32.v1",
            CudaKernelId.CAST_F32_TO_BF16: "aginfer.cast.f32_to_bf16.v1",
            CudaKernelId.CAST_BOOL_TO_I32: "aginfer.cast.bool_to_i32.v1",
            CudaKernelId.ADD_F32: "aginfer.add.f32.v1",
            CudaKernelId.ADD_BF16: "aginfer.add.bf16.v1",
            CudaKernelId.ADD_I32: "aginfer.add.i32.v1",
            CudaKernelId.MUL_F32: "aginfer.mul.f32.v1",
            CudaKernelId.MUL_BF16: "aginfer.mul.bf16.v1",
            CudaKernelId.GELU_F32: "aginfer.gelu.tanh.f32.v1",
            CudaKernelId.GELU_BF16: "aginfer.gelu.tanh.bf16.v1",
            CudaKernelId.SILU_F32: "aginfer.silu.f32.v1",
        }[self]

    @property
    def symbol(self) -> str:
        return {
            CudaKernelId.CAST_BF16_TO_F32: "aginfer_cast_bf16_to_f32",
            CudaKernelId.CAST_F32_TO_BF16: "aginfer_cast_f32_to_bf16",
            CudaKernelId.CAST_BOOL_TO_I32: "aginfer_cast_bool_to_i32",
            CudaKernelId.ADD_F32: "aginfer_add_f32",
            CudaKernelId.ADD_BF16: "aginfer_add_bf16",
            CudaKernelId.ADD_I32: "aginfer_add_i32",
            CudaKernelId.MUL_F32: "aginfer_mul_f32",
            CudaKernelId.MUL_BF16: "aginfer_mul_bf16",
            CudaKernelId.GELU_F32: "aginfer_gelu_tanh_f32",
            CudaKernelId.GELU_BF16: "aginfer_gelu_tanh_bf16",
            CudaKernelId.SILU_F32: "aginfer_silu_f32",
        }[self]

    @property
    def launch_abi(self) -> int:
        return (
            CUDA_KERNEL_LAUNCH_UNARY_POINTERS_NUMEL
            if self in _CAST_KERNEL_IDS + _ACTIVATION_KERNEL_IDS
            else CUDA_KERNEL_LAUNCH_BINARY_POINTERS_NUMEL
        )


class CudaKernelDType(IntEnum):
    F32 = 1
    BF16 = 2
    I32 = 3
    BOOL = 4

    @classmethod
    def from_ir(cls, value: str) -> "CudaKernelDType":
        try:
            return {
                "f32": cls.F32,
                "bf16": cls.BF16,
                "i32": cls.I32,
                "bool": cls.BOOL,
            }[value]
        except KeyError as exc:
            raise ValidationError(f"AOT CUDA kernels do not support dtype {value}") from exc

    @property
    def byte_size(self) -> int:
        return {
            CudaKernelDType.F32: 4,
            CudaKernelDType.BF16: 2,
            CudaKernelDType.I32: 4,
            CudaKernelDType.BOOL: 1,
        }[self]


_KERNEL_BY_DTYPES = {
    (CudaKernelDType.BF16, CudaKernelDType.F32): CudaKernelId.CAST_BF16_TO_F32,
    (CudaKernelDType.F32, CudaKernelDType.BF16): CudaKernelId.CAST_F32_TO_BF16,
    (CudaKernelDType.BOOL, CudaKernelDType.I32): CudaKernelId.CAST_BOOL_TO_I32,
}
_CAST_KERNEL_IDS = (
    CudaKernelId.CAST_BF16_TO_F32,
    CudaKernelId.CAST_F32_TO_BF16,
    CudaKernelId.CAST_BOOL_TO_I32,
)
_POINTWISE_KERNEL_BY_OPCODE_DTYPE = {
    ("add", CudaKernelDType.F32): CudaKernelId.ADD_F32,
    ("add", CudaKernelDType.BF16): CudaKernelId.ADD_BF16,
    ("add", CudaKernelDType.I32): CudaKernelId.ADD_I32,
    ("mul", CudaKernelDType.F32): CudaKernelId.MUL_F32,
    ("mul", CudaKernelDType.BF16): CudaKernelId.MUL_BF16,
}
_POINTWISE_KERNEL_IDS = tuple(_POINTWISE_KERNEL_BY_OPCODE_DTYPE.values())
_ACTIVATION_KERNEL_BY_OPCODE_DTYPE = {
    ("gelu", CudaKernelDType.F32): CudaKernelId.GELU_F32,
    ("gelu", CudaKernelDType.BF16): CudaKernelId.GELU_BF16,
    ("silu", CudaKernelDType.F32): CudaKernelId.SILU_F32,
}
_ACTIVATION_KERNEL_IDS = tuple(_ACTIVATION_KERNEL_BY_OPCODE_DTYPE.values())
_ACTIVATION_SHAPES = {
    ("gelu", CudaKernelDType.F32, 1_101_824): (1, 256, 4304),
    ("gelu", CudaKernelDType.BF16, 204_800): (1, 50, 4096),
    ("gelu", CudaKernelDType.BF16, 15_859_712): (1, 968, 16384),
    ("silu", CudaKernelDType.F32, 1_024): (1, 1024),
}


@dataclass(frozen=True, slots=True)
class AotCastProblem:
    target_arch: CudaArch
    kernel_id: CudaKernelId
    input_dtype: CudaKernelDType
    output_dtype: CudaKernelDType
    numel: int

    def __post_init__(self) -> None:
        if not isinstance(self.target_arch, CudaArch):
            raise ValidationError("AOT cast problem target_arch must be a CudaArch")
        if not isinstance(self.kernel_id, CudaKernelId):
            raise ValidationError("AOT cast problem kernel ID is unknown")
        if not isinstance(self.input_dtype, CudaKernelDType) or not isinstance(
            self.output_dtype, CudaKernelDType
        ):
            raise ValidationError("AOT cast problem dtype is unknown")
        if _KERNEL_BY_DTYPES.get((self.input_dtype, self.output_dtype)) != self.kernel_id:
            raise ValidationError("AOT cast problem kernel ID does not match its dtypes")
        _positive_uint(self.numel, 64, "AOT cast problem numel")
        _checked_bytes(self.numel, self.input_dtype.byte_size, "input")
        _checked_bytes(self.numel, self.output_dtype.byte_size, "output")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "AotCastProblem":
        if not isinstance(item, InventoryOp):
            raise ValidationError("AOT cast problem requires an InventoryOp")
        if (
            item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.AOT_CUDA
            or item.opcode != "cast"
            or len(item.input_types) != 1
            or len(item.output_types) != 1
        ):
            raise ValidationError("AOT cast problem requires a required unary cast site")
        source = item.input_types[0]
        output = item.output_types[0]
        if (
            source.device != "cuda"
            or output.device != "cuda"
            or source.layout != "row_major"
            or output.layout != "row_major"
        ):
            raise ValidationError("AOT cast requires contiguous CUDA row-major tensors")
        if source.shape != output.shape:
            raise ValidationError("AOT cast input and output shapes differ")
        if item.attributes != (("dtype", output.dtype),):
            raise ValidationError("AOT cast dtype attribute does not match its output")
        input_dtype = CudaKernelDType.from_ir(source.dtype)
        output_dtype = CudaKernelDType.from_ir(output.dtype)
        kernel_id = _KERNEL_BY_DTYPES.get((input_dtype, output_dtype))
        if kernel_id is None:
            raise ValidationError("AOT cast dtype conversion has no implemented kernel")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in source.shape):
            raise ValidationError("AOT cast requires a positive static shape")
        return cls(
            target_arch,
            kernel_id,
            input_dtype,
            output_dtype,
            math.prod(source.shape),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "kernel_id": int(self.kernel_id),
            "input_dtype": self.input_dtype.name.lower(),
            "output_dtype": self.output_dtype.name.lower(),
            "numel": self.numel,
        }


@dataclass(frozen=True, slots=True)
class AotPointwiseProblem:
    target_arch: CudaArch
    kernel_id: CudaKernelId
    opcode: str
    dtype: CudaKernelDType
    numel: int

    def __post_init__(self) -> None:
        if not isinstance(self.target_arch, CudaArch):
            raise ValidationError("AOT pointwise problem target_arch must be a CudaArch")
        if self.opcode not in {"add", "mul"}:
            raise ValidationError("AOT pointwise problem opcode must be add or mul")
        if not isinstance(self.dtype, CudaKernelDType):
            raise ValidationError("AOT pointwise problem dtype is unknown")
        if _POINTWISE_KERNEL_BY_OPCODE_DTYPE.get((self.opcode, self.dtype)) != self.kernel_id:
            raise ValidationError("AOT pointwise problem kernel ID does not match its semantics")
        _positive_uint(self.numel, 64, "AOT pointwise problem numel")
        _checked_bytes(self.numel, self.dtype.byte_size, "pointwise tensor")

    @property
    def input_dtype(self) -> CudaKernelDType:
        return self.dtype

    @property
    def output_dtype(self) -> CudaKernelDType:
        return self.dtype

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "AotPointwiseProblem":
        if not isinstance(item, InventoryOp):
            raise ValidationError("AOT pointwise problem requires an InventoryOp")
        if (
            item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.AOT_CUDA
            or item.opcode not in {"add", "mul"}
            or len(item.input_types) != 2
            or len(item.output_types) != 1
        ):
            raise ValidationError("AOT pointwise problem requires a required binary add/mul site")
        lhs, rhs = item.input_types
        output = item.output_types[0]
        if lhs != rhs or lhs != output:
            raise ValidationError("AOT pointwise requires identical input/output tensor signatures")
        if lhs.device != "cuda" or lhs.layout != "row_major":
            raise ValidationError("AOT pointwise requires contiguous CUDA row-major tensors")
        if item.attributes:
            raise ValidationError("AOT pointwise add/mul does not accept semantic attributes")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in lhs.shape
        ):
            raise ValidationError("AOT pointwise requires a positive static shape")
        dtype = CudaKernelDType.from_ir(lhs.dtype)
        kernel_id = _POINTWISE_KERNEL_BY_OPCODE_DTYPE.get((item.opcode, dtype))
        if kernel_id is None:
            raise ValidationError("AOT pointwise opcode/dtype has no implemented kernel")
        return cls(target_arch, kernel_id, item.opcode, dtype, math.prod(lhs.shape))

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "kernel_id": int(self.kernel_id),
            "opcode": self.opcode,
            "dtype": self.dtype.name.lower(),
            "numel": self.numel,
        }


@dataclass(frozen=True, slots=True)
class AotActivationProblem:
    target_arch: CudaArch
    kernel_id: CudaKernelId
    dtype: CudaKernelDType
    numel: int
    opcode: str = "gelu"

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("AOT GELU exact problems require SM120")
        if not isinstance(self.dtype, CudaKernelDType):
            raise ValidationError("AOT GELU problem dtype is unknown")
        if _ACTIVATION_KERNEL_BY_OPCODE_DTYPE.get((self.opcode, self.dtype)) != self.kernel_id:
            raise ValidationError("AOT activation kernel ID does not match its semantics")
        _positive_uint(self.numel, 64, "AOT activation problem numel")
        if (self.opcode, self.dtype, self.numel) not in _ACTIVATION_SHAPES:
            raise ValidationError(
                "AOT activation problem is outside the delivered exact envelopes"
            )
        _checked_bytes(self.numel, self.dtype.byte_size, "activation tensor")

    @property
    def input_dtype(self) -> CudaKernelDType:
        return self.dtype

    @property
    def output_dtype(self) -> CudaKernelDType:
        return self.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return _ACTIVATION_SHAPES[(self.opcode, self.dtype, self.numel)]

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "AotActivationProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.AOT_CUDA
            or item.opcode not in {"gelu", "silu"}
            or item.attributes
            != ((("approximation", "tanh"),) if item.opcode == "gelu" else ())
            or len(item.input_types) != 1
            or len(item.output_types) != 1
            or item.input_types != item.output_types
        ):
            raise ValidationError(
                "AOT activation site is outside the delivered exact envelopes"
            )
        tensor = item.input_types[0]
        if tensor.device != "cuda" or tensor.layout != "row_major":
            raise ValidationError(
                "AOT activation requires a contiguous CUDA row-major tensor"
            )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in tensor.shape
        ):
            raise ValidationError("AOT activation requires a positive static shape")
        dtype = CudaKernelDType.from_ir(tensor.dtype)
        kernel_id = _ACTIVATION_KERNEL_BY_OPCODE_DTYPE.get((item.opcode, dtype))
        numel = math.prod(tensor.shape)
        if (
            kernel_id is None
            or _ACTIVATION_SHAPES.get((item.opcode, dtype, numel)) != tensor.shape
        ):
            raise ValidationError(
                "AOT activation site is outside the delivered exact envelopes"
            )
        return cls(target_arch, kernel_id, dtype, numel, item.opcode)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "kernel_id": int(self.kernel_id),
            "opcode": self.opcode,
            "dtype": self.dtype.name.lower(),
            "shape": list(self.shape),
            "numel": self.numel,
            "attributes": {"approximation": "tanh"} if self.opcode == "gelu" else {},
        }


@dataclass(frozen=True, slots=True)
class CudaKernelPayload:
    problem: AotCastProblem | AotPointwiseProblem | AotActivationProblem
    module_bytes: int
    module_sha256: str
    grid_x: int
    block_x: int = CUDA_KERNEL_BLOCK_X
    shared_bytes: int = 0
    input_alignment: int = 1
    output_alignment: int = 1

    def __post_init__(self) -> None:
        if not isinstance(
            self.problem, (AotCastProblem, AotPointwiseProblem, AotActivationProblem)
        ):
            raise ValidationError("CUDA kernel payload problem is invalid")
        _positive_uint(self.module_bytes, 64, "CUDA kernel module_bytes")
        _digest(self.module_sha256, "CUDA kernel module_sha256")
        for value, label, maximum in (
            (self.grid_x, "grid_x", CUDA_KERNEL_MAX_GRID_X),
            (self.block_x, "block_x", 1024),
            (self.shared_bytes, "shared_bytes", CUDA_KERNEL_MAX_SHARED_BYTES),
        ):
            _uint(value, 32, f"CUDA kernel {label}")
            if (label != "shared_bytes" and value == 0) or value > maximum:
                raise ValidationError(f"CUDA kernel {label} is outside the supported launch range")
        expected_grid = min(
            (self.problem.numel + self.block_x - 1) // self.block_x,
            CUDA_KERNEL_MAX_GRID_X,
        )
        if self.grid_x != expected_grid or self.shared_bytes != 0:
            raise ValidationError("CUDA kernel launch geometry is not canonical")
        for value, label, natural in (
            (self.input_alignment, "input", self.problem.input_dtype.byte_size),
            (self.output_alignment, "output", self.problem.output_dtype.byte_size),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < natural
                or value > CUDA_KERNEL_MAX_ALIGNMENT
                or value & (value - 1)
            ):
                raise ValidationError(
                    f"CUDA kernel {label} alignment must be a bounded power of two at least its dtype size"
                )

    @classmethod
    def for_problem(
        cls,
        problem: AotCastProblem | AotPointwiseProblem | AotActivationProblem,
        *,
        module_bytes: int,
        module_sha256: str,
    ) -> "CudaKernelPayload":
        return cls(
            problem=problem,
            module_bytes=module_bytes,
            module_sha256=module_sha256,
            grid_x=min(
                (problem.numel + CUDA_KERNEL_BLOCK_X - 1) // CUDA_KERNEL_BLOCK_X,
                CUDA_KERNEL_MAX_GRID_X,
            ),
            input_alignment=problem.input_dtype.byte_size,
            output_alignment=problem.output_dtype.byte_size,
        )

    def to_bytes(self) -> bytes:
        return CUDA_KERNEL_PAYLOAD.pack(
            CUDA_KERNEL_PAYLOAD_MAGIC,
            CUDA_KERNEL_SCHEMA_MAJOR,
            CUDA_KERNEL_SCHEMA_MINOR,
            CUDA_KERNEL_PAYLOAD.size,
            int(self.problem.target_arch),
            int(self.problem.kernel_id),
            int(self.problem.input_dtype),
            int(self.problem.output_dtype),
            CUDA_KERNEL_FLAG_CONTIGUOUS,
            self.grid_x,
            self.block_x,
            self.shared_bytes,
            self.problem.numel,
            self.module_bytes,
            bytes.fromhex(self.module_sha256),
            self.input_alignment,
            self.output_alignment,
            CUDA_KERNEL_MODULE_CUBIN,
            self.problem.kernel_id.launch_abi,
            b"\0" * 16,
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "CudaKernelPayload":
        if len(data) != CUDA_KERNEL_PAYLOAD.size:
            raise FormatError("CUDA kernel payload must have its fixed size")
        fields = CUDA_KERNEL_PAYLOAD.unpack(data)
        if fields[0] != CUDA_KERNEL_PAYLOAD_MAGIC:
            raise FormatError("bad CUDA kernel payload magic")
        if (
            fields[1] != CUDA_KERNEL_SCHEMA_MAJOR
            or fields[2] > CUDA_KERNEL_SCHEMA_MINOR
            or fields[3] != CUDA_KERNEL_PAYLOAD.size
        ):
            raise FormatError("unsupported CUDA kernel payload schema or size")
        try:
            arch = CudaArch(fields[4])
            kernel_id = CudaKernelId(fields[5])
            input_dtype = CudaKernelDType(fields[6])
            output_dtype = CudaKernelDType(fields[7])
        except ValueError as exc:
            raise FormatError("CUDA kernel payload has an unknown target, kernel, or dtype") from exc
        if (
            fields[8] != CUDA_KERNEL_FLAG_CONTIGUOUS
            or fields[17] != CUDA_KERNEL_MODULE_CUBIN
            or any(fields[19])
        ):
            raise FormatError("CUDA kernel payload has invalid fixed fields")
        try:
            if kernel_id in _CAST_KERNEL_IDS:
                problem: AotCastProblem | AotPointwiseProblem | AotActivationProblem = AotCastProblem(
                    arch, kernel_id, input_dtype, output_dtype, fields[12]
                )
            elif kernel_id in _ACTIVATION_KERNEL_IDS:
                if input_dtype != output_dtype:
                    raise ValidationError("AOT activation payload dtypes differ")
                opcode = "silu" if kernel_id == CudaKernelId.SILU_F32 else "gelu"
                problem = AotActivationProblem(
                    arch, kernel_id, input_dtype, fields[12], opcode
                )
            else:
                opcode = (
                    "add"
                    if kernel_id
                    in {CudaKernelId.ADD_F32, CudaKernelId.ADD_BF16, CudaKernelId.ADD_I32}
                    else "mul"
                )
                if input_dtype != output_dtype:
                    raise ValidationError("AOT pointwise payload dtypes differ")
                problem = AotPointwiseProblem(
                    arch, kernel_id, opcode, input_dtype, fields[12]
                )
            if fields[18] != kernel_id.launch_abi:
                raise ValidationError("CUDA kernel payload launch ABI does not match kernel ID")
            return cls(
                problem,
                fields[13],
                fields[14].hex(),
                fields[9],
                fields[10],
                fields[11],
                fields[15],
                fields[16],
            )
        except ValidationError as exc:
            raise FormatError(f"CUDA kernel payload contract is invalid: {exc}") from exc


@dataclass(frozen=True, slots=True)
class AotCastValidationReceipt:
    target_arch: CudaArch
    module_bytes: int
    module_sha256: str
    cuda_driver_version: int
    cuda_runtime_version: int
    problems: tuple[AotCastProblem, ...]
    symbols: tuple[str, ...]
    normal_launches_per_problem: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    full_output_compared: bool
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.target_arch, CudaArch):
            raise ValidationError("AOT cast receipt target_arch must be a CudaArch")
        _positive_uint(self.module_bytes, 64, "AOT cast receipt module_bytes")
        _digest(self.module_sha256, "AOT cast receipt module_sha256")
        _positive_uint(self.cuda_driver_version, 32, "AOT cast receipt CUDA driver version")
        _positive_uint(self.cuda_runtime_version, 32, "AOT cast receipt CUDA runtime version")
        if not self.problems or not all(isinstance(item, AotCastProblem) for item in self.problems):
            raise ValidationError("AOT cast receipt needs validated exact problems")
        if len(set(self.problems)) != len(self.problems):
            raise ValidationError("AOT cast receipt contains duplicate exact problems")
        if any(item.target_arch != self.target_arch for item in self.problems):
            raise ValidationError("AOT cast receipt problem target differs from module target")
        expected_symbols = tuple(item.symbol for item in _CAST_KERNEL_IDS)
        if self.symbols != expected_symbols:
            raise ValidationError("AOT cast receipt symbol inventory is incomplete or non-canonical")
        if (
            not isinstance(self.normal_launches_per_problem, int)
            or isinstance(self.normal_launches_per_problem, bool)
            or self.normal_launches_per_problem < 2
            or self.normal_launches_per_problem > 2**32 - 1
        ):
            raise ValidationError("AOT cast receipt requires two launches per exact problem")
        if self.normal_repeat_bit_exact is not True:
            raise ValidationError("AOT cast receipt requires bit-exact normal repeats")
        if self.capture_replay_bit_exact is not True:
            raise ValidationError("AOT cast receipt requires bit-exact capture replay")
        if self.full_output_compared is not True:
            raise ValidationError("AOT cast receipt requires full-output reference comparison")
        if self.contains_ptx is not False:
            raise ValidationError("AOT cast receipt requires a CUBIN without embedded PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": AOT_CAST_RECEIPT_SCHEMA,
            "target_arch": self.target_arch.name_string,
            "module_bytes": self.module_bytes,
            "module_sha256": self.module_sha256,
            "cuda_driver_version": self.cuda_driver_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "problems": [item.to_dict() for item in self.problems],
            "symbols": list(self.symbols),
            "normal_launches_per_problem": self.normal_launches_per_problem,
            "normal_repeat_bit_exact": self.normal_repeat_bit_exact,
            "capture_replay_bit_exact": self.capture_replay_bit_exact,
            "full_output_compared": self.full_output_compared,
            "contains_ptx": self.contains_ptx,
        }

    @classmethod
    def from_dict(cls, value: object) -> "AotCastValidationReceipt":
        expected_fields = {
            "schema",
            "target_arch",
            "module_bytes",
            "module_sha256",
            "cuda_driver_version",
            "cuda_runtime_version",
            "problems",
            "symbols",
            "normal_launches_per_problem",
            "normal_repeat_bit_exact",
            "capture_replay_bit_exact",
            "full_output_compared",
            "contains_ptx",
        }
        if not isinstance(value, dict) or set(value) != expected_fields:
            raise ValidationError("AOT cast receipt has unknown or missing fields")
        if value["schema"] != AOT_CAST_RECEIPT_SCHEMA:
            raise ValidationError("AOT cast receipt schema is unknown")
        try:
            target_arch = CudaArch.parse(value["target_arch"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValidationError("AOT cast receipt target_arch is unknown") from exc
        raw_problems = value["problems"]
        if not isinstance(raw_problems, list):
            raise ValidationError("AOT cast receipt problems must be a list")
        problems: list[AotCastProblem] = []
        for raw in raw_problems:
            if not isinstance(raw, dict) or set(raw) != {
                "target_arch",
                "kernel_id",
                "input_dtype",
                "output_dtype",
                "numel",
            }:
                raise ValidationError("AOT cast receipt problem has unknown or missing fields")
            if raw["target_arch"] != target_arch.name_string:
                raise ValidationError("AOT cast receipt problem target differs from module target")
            try:
                problem = AotCastProblem(
                    target_arch,
                    CudaKernelId(raw["kernel_id"]),  # type: ignore[arg-type]
                    CudaKernelDType[str(raw["input_dtype"]).upper()],
                    CudaKernelDType[str(raw["output_dtype"]).upper()],
                    raw["numel"],  # type: ignore[arg-type]
                )
            except (KeyError, ValueError, ValidationError) as exc:
                raise ValidationError(f"AOT cast receipt problem is invalid: {exc}") from exc
            problems.append(problem)
        raw_symbols = value["symbols"]
        if not isinstance(raw_symbols, list) or not all(
            isinstance(item, str) for item in raw_symbols
        ):
            raise ValidationError("AOT cast receipt symbols must be a string list")
        return cls(
            target_arch=target_arch,
            module_bytes=value["module_bytes"],  # type: ignore[arg-type]
            module_sha256=value["module_sha256"],  # type: ignore[arg-type]
            cuda_driver_version=value["cuda_driver_version"],  # type: ignore[arg-type]
            cuda_runtime_version=value["cuda_runtime_version"],  # type: ignore[arg-type]
            problems=tuple(problems),
            symbols=tuple(raw_symbols),
            normal_launches_per_problem=value["normal_launches_per_problem"],  # type: ignore[arg-type]
            normal_repeat_bit_exact=value["normal_repeat_bit_exact"],  # type: ignore[arg-type]
            capture_replay_bit_exact=value["capture_replay_bit_exact"],  # type: ignore[arg-type]
            full_output_compared=value["full_output_compared"],  # type: ignore[arg-type]
            contains_ptx=value["contains_ptx"],  # type: ignore[arg-type]
        )

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = AotCastProblem.from_inventory(item, target_arch=self.target_arch)
        if problem not in self.problems:
            raise ValidationError("AOT cast receipt does not cover the inventory problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=CUDA_KERNEL_PROVIDER_ID,
            abi_major=CUDA_KERNEL_PROVIDER_ABI_MAJOR,
            abi_minor=CUDA_KERNEL_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{CUDA_KERNEL_PROVIDER_VERSION};cuda-runtime={self.cuda_runtime_version};"
                f"driver={self.cuda_driver_version}"
            ),
            implementation_id=problem.kernel_id.implementation_id,
            implementation_digest=self.module_sha256,
            target_arch=self.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )


@dataclass(frozen=True, slots=True)
class AotCastLoweredCommand:
    execution_index: int
    site: str
    command: ProviderCommand

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index,
            "site": self.site,
            "tag": self.command.tag.name.lower(),
            "provider_id": self.command.provider_id,
            "abi": [self.command.abi_major, self.command.abi_minor],
            "capability_digest": self.command.capability_digest,
            "operands": [
                {
                    "value_id": item.value_id,
                    "access": item.access.name.lower(),
                    "byte_offset": item.byte_offset,
                }
                for item in self.command.operands
            ],
            "payload_sha256": hashlib.sha256(self.command.payload).hexdigest(),
            "capture_safe": self.command.capture_safe,
        }


@dataclass(frozen=True, slots=True)
class AotCastPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[AotCastLoweredCommand, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": AOT_CAST_PARTIAL_LOWERING_SCHEMA,
            "inventory_sha256": self.inventory_sha256,
            "schedule_sha256": self.schedule_sha256,
            "memory_plan_sha256": self.memory_plan_sha256,
            "complete": self.complete,
            "summary": {
                "capabilities": len(self.capabilities),
                "commands": len(self.commands),
                "elided_executions": len(self.elided_execution_indices),
                "unhandled_executions": len(self.unhandled_execution_indices),
                "workspace_bytes": 0,
            },
            "capability_registry_sha256": hashlib.sha256(
                dump_provider_capabilities(self.capabilities).encode("utf-8")
            ).hexdigest(),
            "commands": [item.to_dict() for item in self.commands],
            "elided_execution_indices": list(self.elided_execution_indices),
            "unhandled_execution_indices": list(self.unhandled_execution_indices),
        }


def make_aot_cast_capabilities(
    inventory: LoweringInventory,
    receipt: AotCastValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if not isinstance(inventory, LoweringInventory):
        raise ValidationError("AOT cast capability construction requires a LoweringInventory")
    if not (isinstance(receipt, AotCastValidationReceipt) or is_build_binding(receipt, AotCastProblem)):
        raise ValidationError("AOT cast capability construction requires a validation receipt")
    if receipt.target_arch != target_arch:
        raise ValidationError("AOT cast receipt target differs from capability target")
    sites_by_problem: dict[AotCastProblem, list[InventoryOp]] = {}
    for item in inventory.ops:
        if item.status == RequirementStatus.REQUIRED and item.opcode == "cast":
            problem = AotCastProblem.from_inventory(item, target_arch=target_arch)
            sites_by_problem.setdefault(problem, []).append(item)
    if set(sites_by_problem) != set(receipt.problems):
        raise ValidationError(
            "AOT cast receipt coverage mismatch: "
            f"missing={len(set(sites_by_problem) - set(receipt.problems))} "
            f"extra={len(set(receipt.problems) - set(sites_by_problem))}"
        )
    capabilities_by_digest: dict[str, ProviderCapability] = {}
    for sites in sites_by_problem.values():
        for item in sites:
            capability = receipt.capability_for(item)
            capabilities_by_digest.setdefault(capability.digest, capability)
    return tuple(sorted(capabilities_by_digest.values(), key=lambda item: item.digest))


def lower_aot_cast_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: AotCastValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> AotCastPartialLowering:
    if not isinstance(schedule, ExecutionSchedule):
        raise ValidationError("AOT cast lowering requires an ExecutionSchedule")
    if not isinstance(memory_plan, MemoryPlan):
        raise ValidationError("AOT cast lowering requires a MemoryPlan")
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode("utf-8")
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(
        dump_execution_schedule(schedule).encode("utf-8")
    ).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("AOT cast lowering inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("AOT cast lowering memory plan does not match its schedule")
    capabilities = make_aot_cast_capabilities(
        inventory, receipt, target_arch=target_arch
    )
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    capability_by_site: dict[str, ProviderCapability] = {}
    for item in inventory.ops:
        if item.status == RequirementStatus.REQUIRED and item.opcode == "cast":
            capability_by_site[item.site_id] = receipt.capability_for(item)
    if {item.value_id for item in memory_plan.allocations} != set(range(len(schedule.values))):
        raise ValidationError("AOT cast lowering memory plan has incomplete value coverage")

    commands: list[AotCastLoweredCommand] = []
    unhandled: list[int] = []
    elided = set(memory_plan.elided_ops)
    for op in schedule.ops:
        if op.execution_index in elided:
            continue
        if op.opcode != "cast":
            unhandled.append(op.execution_index)
            continue
        item = inventory_by_site.get(op.site)
        if item is None:
            raise ValidationError("AOT cast schedule site is absent from inventory")
        problem = AotCastProblem.from_inventory(item, target_arch=target_arch)
        capability = capability_by_site[item.site_id]
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise ValidationError("AOT cast scheduled operation has invalid arity")
        if (
            schedule.values[op.inputs[0]].type != item.input_types[0]
            or schedule.values[op.outputs[0]].type != item.output_types[0]
        ):
            raise ValidationError("AOT cast scheduled operand types differ from inventory")
        payload = CudaKernelPayload.for_problem(
            problem,
            module_bytes=receipt.module_bytes,
            module_sha256=receipt.module_sha256,
        ).to_bytes()
        commands.append(
            AotCastLoweredCommand(
                op.execution_index,
                op.site,
                ProviderCommand(
                    tag=CommandTag.CUDA_KERNEL,
                    provider_id=capability.provider_id,
                    abi_major=capability.abi_major,
                    abi_minor=capability.abi_minor,
                    capability_digest=capability.digest,
                    operands=(
                        CommandOperand(op.inputs[0], OperandAccess.READ),
                        CommandOperand(op.outputs[0], OperandAccess.WRITE),
                    ),
                    payload=payload,
                    capture_safe=capability.supports_capture,
                ),
            )
        )
    memory_plan_sha256 = hashlib.sha256(
        _canonical_json(memory_plan.to_dict()).encode("utf-8") + b"\n"
    ).hexdigest()
    return AotCastPartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(elided)),
        tuple(unhandled),
    )


def dump_aot_cast_partial_lowering(lowering: AotCastPartialLowering) -> str:
    if not isinstance(lowering, AotCastPartialLowering):
        raise ValidationError("AOT cast lowering dump requires an AotCastPartialLowering")
    return _canonical_json(lowering.to_dict()) + "\n"


@dataclass(frozen=True, slots=True)
class AotPointwiseValidationReceipt:
    target_arch: CudaArch
    module_bytes: int
    module_sha256: str
    cuda_driver_version: int
    cuda_runtime_version: int
    problems: tuple[AotPointwiseProblem, ...]
    symbols: tuple[str, ...]
    normal_launches_per_problem: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    full_output_compared: bool
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.target_arch, CudaArch):
            raise ValidationError("AOT pointwise receipt target_arch must be a CudaArch")
        _positive_uint(self.module_bytes, 64, "AOT pointwise receipt module_bytes")
        _digest(self.module_sha256, "AOT pointwise receipt module_sha256")
        _positive_uint(
            self.cuda_driver_version, 32, "AOT pointwise receipt CUDA driver version"
        )
        _positive_uint(
            self.cuda_runtime_version, 32, "AOT pointwise receipt CUDA runtime version"
        )
        if not self.problems or not all(
            isinstance(item, AotPointwiseProblem) for item in self.problems
        ):
            raise ValidationError("AOT pointwise receipt needs validated exact problems")
        if len(set(self.problems)) != len(self.problems):
            raise ValidationError("AOT pointwise receipt contains duplicate exact problems")
        if any(item.target_arch != self.target_arch for item in self.problems):
            raise ValidationError("AOT pointwise receipt problem target differs from module target")
        if self.symbols != tuple(item.symbol for item in _POINTWISE_KERNEL_IDS):
            raise ValidationError(
                "AOT pointwise receipt symbol inventory is incomplete or non-canonical"
            )
        if (
            not isinstance(self.normal_launches_per_problem, int)
            or isinstance(self.normal_launches_per_problem, bool)
            or not 2 <= self.normal_launches_per_problem <= 2**32 - 1
        ):
            raise ValidationError("AOT pointwise receipt requires two launches per exact problem")
        if self.normal_repeat_bit_exact is not True:
            raise ValidationError("AOT pointwise receipt requires bit-exact normal repeats")
        if self.capture_replay_bit_exact is not True:
            raise ValidationError("AOT pointwise receipt requires bit-exact capture replay")
        if self.full_output_compared is not True:
            raise ValidationError("AOT pointwise receipt requires full-output reference comparison")
        if self.contains_ptx is not False:
            raise ValidationError("AOT pointwise receipt requires a CUBIN without embedded PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": AOT_POINTWISE_RECEIPT_SCHEMA,
            "target_arch": self.target_arch.name_string,
            "module_bytes": self.module_bytes,
            "module_sha256": self.module_sha256,
            "cuda_driver_version": self.cuda_driver_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "problems": [item.to_dict() for item in self.problems],
            "symbols": list(self.symbols),
            "normal_launches_per_problem": self.normal_launches_per_problem,
            "normal_repeat_bit_exact": self.normal_repeat_bit_exact,
            "capture_replay_bit_exact": self.capture_replay_bit_exact,
            "full_output_compared": self.full_output_compared,
            "contains_ptx": self.contains_ptx,
        }

    @classmethod
    def from_dict(cls, value: object) -> "AotPointwiseValidationReceipt":
        expected = {
            "schema", "target_arch", "module_bytes", "module_sha256",
            "cuda_driver_version", "cuda_runtime_version", "problems", "symbols",
            "normal_launches_per_problem", "normal_repeat_bit_exact",
            "capture_replay_bit_exact", "full_output_compared", "contains_ptx",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValidationError("AOT pointwise receipt has unknown or missing fields")
        if value["schema"] != AOT_POINTWISE_RECEIPT_SCHEMA:
            raise ValidationError("AOT pointwise receipt schema is unknown")
        try:
            target_arch = CudaArch.parse(value["target_arch"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValidationError("AOT pointwise receipt target_arch is unknown") from exc
        raw_problems = value["problems"]
        if not isinstance(raw_problems, list):
            raise ValidationError("AOT pointwise receipt problems must be a list")
        problems: list[AotPointwiseProblem] = []
        for raw in raw_problems:
            if not isinstance(raw, dict) or set(raw) != {
                "target_arch", "kernel_id", "opcode", "dtype", "numel",
            }:
                raise ValidationError(
                    "AOT pointwise receipt problem has unknown or missing fields"
                )
            if raw["target_arch"] != target_arch.name_string:
                raise ValidationError(
                    "AOT pointwise receipt problem target differs from module target"
                )
            try:
                problems.append(
                    AotPointwiseProblem(
                        target_arch,
                        CudaKernelId(raw["kernel_id"]),  # type: ignore[arg-type]
                        raw["opcode"],  # type: ignore[arg-type]
                        CudaKernelDType[str(raw["dtype"]).upper()],
                        raw["numel"],  # type: ignore[arg-type]
                    )
                )
            except (KeyError, ValueError, ValidationError) as exc:
                raise ValidationError(
                    f"AOT pointwise receipt problem is invalid: {exc}"
                ) from exc
        symbols = value["symbols"]
        if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
            raise ValidationError("AOT pointwise receipt symbols must be a string list")
        return cls(
            target_arch,
            value["module_bytes"],  # type: ignore[arg-type]
            value["module_sha256"],  # type: ignore[arg-type]
            value["cuda_driver_version"],  # type: ignore[arg-type]
            value["cuda_runtime_version"],  # type: ignore[arg-type]
            tuple(problems),
            tuple(symbols),
            value["normal_launches_per_problem"],  # type: ignore[arg-type]
            value["normal_repeat_bit_exact"],  # type: ignore[arg-type]
            value["capture_replay_bit_exact"],  # type: ignore[arg-type]
            value["full_output_compared"],  # type: ignore[arg-type]
            value["contains_ptx"],  # type: ignore[arg-type]
        )

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = AotPointwiseProblem.from_inventory(item, target_arch=self.target_arch)
        if problem not in self.problems:
            raise ValidationError("AOT pointwise receipt does not cover the inventory problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=CUDA_KERNEL_PROVIDER_ID,
            abi_major=CUDA_KERNEL_PROVIDER_ABI_MAJOR,
            abi_minor=CUDA_KERNEL_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{CUDA_KERNEL_PROVIDER_VERSION};cuda-runtime={self.cuda_runtime_version};"
                f"driver={self.cuda_driver_version}"
            ),
            implementation_id=problem.kernel_id.implementation_id,
            implementation_digest=self.module_sha256,
            target_arch=self.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )


@dataclass(frozen=True, slots=True)
class AotPointwiseLoweredCommand:
    execution_index: int
    site: str
    command: ProviderCommand

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index,
            "site": self.site,
            "tag": self.command.tag.name.lower(),
            "provider_id": self.command.provider_id,
            "abi": [self.command.abi_major, self.command.abi_minor],
            "capability_digest": self.command.capability_digest,
            "operands": [
                {
                    "value_id": item.value_id,
                    "access": item.access.name.lower(),
                    "byte_offset": item.byte_offset,
                }
                for item in self.command.operands
            ],
            "payload_sha256": hashlib.sha256(self.command.payload).hexdigest(),
            "capture_safe": self.command.capture_safe,
        }


@dataclass(frozen=True, slots=True)
class AotPointwisePartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[AotPointwiseLoweredCommand, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": AOT_POINTWISE_PARTIAL_LOWERING_SCHEMA,
            "inventory_sha256": self.inventory_sha256,
            "schedule_sha256": self.schedule_sha256,
            "memory_plan_sha256": self.memory_plan_sha256,
            "complete": self.complete,
            "summary": {
                "capabilities": len(self.capabilities),
                "commands": len(self.commands),
                "elided_executions": len(self.elided_execution_indices),
                "unhandled_executions": len(self.unhandled_execution_indices),
                "workspace_bytes": 0,
            },
            "capability_registry_sha256": hashlib.sha256(
                dump_provider_capabilities(self.capabilities).encode("utf-8")
            ).hexdigest(),
            "commands": [item.to_dict() for item in self.commands],
            "elided_execution_indices": list(self.elided_execution_indices),
            "unhandled_execution_indices": list(self.unhandled_execution_indices),
        }


def make_aot_pointwise_capabilities(
    inventory: LoweringInventory,
    receipt: AotPointwiseValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if not isinstance(inventory, LoweringInventory):
        raise ValidationError(
            "AOT pointwise capability construction requires a LoweringInventory"
        )
    if not (isinstance(receipt, AotPointwiseValidationReceipt) or is_build_binding(receipt, AotPointwiseProblem)):
        raise ValidationError(
            "AOT pointwise capability construction requires a validation receipt"
        )
    if receipt.target_arch != target_arch:
        raise ValidationError("AOT pointwise receipt target differs from capability target")
    sites_by_problem: dict[AotPointwiseProblem, list[InventoryOp]] = {}
    for item in inventory.ops:
        if item.status == RequirementStatus.REQUIRED and item.opcode in {"add", "mul"}:
            problem = AotPointwiseProblem.from_inventory(item, target_arch=target_arch)
            sites_by_problem.setdefault(problem, []).append(item)
    if set(sites_by_problem) != set(receipt.problems):
        raise ValidationError(
            "AOT pointwise receipt coverage mismatch: "
            f"missing={len(set(sites_by_problem) - set(receipt.problems))} "
            f"extra={len(set(receipt.problems) - set(sites_by_problem))}"
        )
    capabilities: dict[str, ProviderCapability] = {}
    for sites in sites_by_problem.values():
        for item in sites:
            capability = receipt.capability_for(item)
            capabilities.setdefault(capability.digest, capability)
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_aot_pointwise_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: AotPointwiseValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> AotPointwisePartialLowering:
    if not isinstance(schedule, ExecutionSchedule):
        raise ValidationError("AOT pointwise lowering requires an ExecutionSchedule")
    if not isinstance(memory_plan, MemoryPlan):
        raise ValidationError("AOT pointwise lowering requires a MemoryPlan")
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode("utf-8")
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(
        dump_execution_schedule(schedule).encode("utf-8")
    ).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("AOT pointwise lowering inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("AOT pointwise lowering memory plan does not match its schedule")
    capabilities = make_aot_pointwise_capabilities(
        inventory, receipt, target_arch=target_arch
    )
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    capability_by_site = {
        item.site_id: receipt.capability_for(item)
        for item in inventory.ops
        if item.status == RequirementStatus.REQUIRED and item.opcode in {"add", "mul"}
    }
    if {item.value_id for item in memory_plan.allocations} != set(range(len(schedule.values))):
        raise ValidationError("AOT pointwise lowering memory plan has incomplete value coverage")

    commands: list[AotPointwiseLoweredCommand] = []
    unhandled: list[int] = []
    elided = set(memory_plan.elided_ops)
    for op in schedule.ops:
        if op.execution_index in elided:
            continue
        if op.opcode not in {"add", "mul"}:
            unhandled.append(op.execution_index)
            continue
        item = inventory_by_site.get(op.site)
        if item is None:
            raise ValidationError("AOT pointwise schedule site is absent from inventory")
        problem = AotPointwiseProblem.from_inventory(item, target_arch=target_arch)
        capability = capability_by_site[item.site_id]
        if len(op.inputs) != 2 or len(op.outputs) != 1:
            raise ValidationError("AOT pointwise scheduled operation has invalid arity")
        actual_inputs = tuple(schedule.values[value_id].type for value_id in op.inputs)
        actual_outputs = tuple(schedule.values[value_id].type for value_id in op.outputs)
        if actual_inputs != item.input_types or actual_outputs != item.output_types:
            raise ValidationError("AOT pointwise scheduled operand types differ from inventory")
        payload = CudaKernelPayload.for_problem(
            problem,
            module_bytes=receipt.module_bytes,
            module_sha256=receipt.module_sha256,
        ).to_bytes()
        commands.append(
            AotPointwiseLoweredCommand(
                op.execution_index,
                op.site,
                ProviderCommand(
                    tag=CommandTag.CUDA_KERNEL,
                    provider_id=capability.provider_id,
                    abi_major=capability.abi_major,
                    abi_minor=capability.abi_minor,
                    capability_digest=capability.digest,
                    operands=(
                        CommandOperand(op.inputs[0], OperandAccess.READ),
                        CommandOperand(op.inputs[1], OperandAccess.READ),
                        CommandOperand(op.outputs[0], OperandAccess.WRITE),
                    ),
                    payload=payload,
                    capture_safe=capability.supports_capture,
                ),
            )
        )
    memory_plan_sha256 = hashlib.sha256(
        _canonical_json(memory_plan.to_dict()).encode("utf-8") + b"\n"
    ).hexdigest()
    return AotPointwisePartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(elided)),
        tuple(unhandled),
    )


def dump_aot_pointwise_partial_lowering(
    lowering: AotPointwisePartialLowering,
) -> str:
    if not isinstance(lowering, AotPointwisePartialLowering):
        raise ValidationError(
            "AOT pointwise lowering dump requires an AotPointwisePartialLowering"
        )
    return _canonical_json(lowering.to_dict()) + "\n"


def _checked_bytes(numel: int, item_bytes: int, label: str) -> None:
    if numel > (2**64 - 1) // item_bytes:
        raise ValidationError(f"AOT cast {label} byte size overflows uint64")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationError(f"{label} must be a canonical SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValidationError(f"{label} must be a canonical SHA-256 digest") from exc
    if len(decoded) != 32 or not any(decoded) or value != decoded.hex():
        raise ValidationError(f"{label} must be a canonical SHA-256 digest")


def _positive_uint(value: object, bits: int, label: str) -> None:
    _uint(value, bits, label)
    if value == 0:
        raise ValidationError(f"{label} must be positive")


def _uint(value: object, bits: int, label: str) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > 2**bits - 1
    ):
        raise ValidationError(f"{label} must be a uint{bits}")
