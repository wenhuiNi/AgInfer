from __future__ import annotations

from .build_binding import LinearBuildBinding

import math
import hashlib
import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from ..errors import FormatError, ValidationError
from ..lowering.capability import ProviderCapability, dump_provider_capabilities
from ..lowering.command import CommandOperand, OperandAccess, ProviderCommand, CommandTag
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


CUBLASLT_LINEAR_PAYLOAD_MAGIC = b"AILTMM1\0"
CUBLASLT_LINEAR_SCHEMA_MAJOR = 1
CUBLASLT_LINEAR_SCHEMA_MINOR = 1
CUBLASLT_LINEAR_PAYLOAD = struct.Struct(
    "<8sHH" + "I" * 14 + "Q" * 8 + "iIiIIIIHH" + "I" * 5 + "8s"
)
CUBLASLT_LINEAR_FLAG_BIAS = 1
CUBLASLT_COMPUTE_F32 = 1
CUBLASLT_COMPUTE_FAST_TF32 = 2
CUBLASLT_SCALE_F32 = 1
CUBLASLT_EPILOGUE_BIAS = 1
CUBLASLT_OP_N = 0
CUBLASLT_OP_T = 1
CUBLASLT_ORDER_COLUMN = 0
MAX_POINTER_ALIGNMENT = 1 << 20
CUBLASLT_PROVIDER_ID = 1
CUBLASLT_PROVIDER_ABI_MAJOR = 1
CUBLASLT_PROVIDER_ABI_MINOR = 0
CUBLASLT_IMPLEMENTATION_ID = "cublaslt.linear.bias.v1"
CUBLASLT_RECEIPT_SCHEMA = "aginfer.cublaslt-validation.v1"
CUBLASLT_PARTIAL_LOWERING_SCHEMA = "aginfer.cublaslt-partial-lowering.v1"

assert CUBLASLT_LINEAR_PAYLOAD.size == 192


class CublasLtDType(IntEnum):
    F32 = 1
    BF16 = 2

    @classmethod
    def from_ir(cls, value: str) -> "CublasLtDType":
        try:
            return {"f32": cls.F32, "bf16": cls.BF16}[value]
        except KeyError as exc:
            raise ValidationError(
                f"cuBLASLt linear only supports f32 or bf16, got {value}"
            ) from exc


@dataclass(frozen=True, slots=True)
class CublasLtLinearProblem:
    target_arch: CudaArch
    dtype: CublasLtDType
    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        if not isinstance(self.target_arch, CudaArch):
            raise ValidationError("cuBLASLt problem target_arch must be a CudaArch")
        if not isinstance(self.dtype, CublasLtDType):
            raise ValidationError("cuBLASLt problem dtype is unknown")
        for value, label in ((self.m, "M"), (self.n, "N"), (self.k, "K")):
            _positive_uint(value, 64, f"cuBLASLt problem {label}")
        element_bytes = 4 if self.dtype == CublasLtDType.F32 else 2
        _checked_tensor_bytes(self.m, self.k, element_bytes, "X")
        _checked_tensor_bytes(self.n, self.k, element_bytes, "weight")
        _checked_tensor_bytes(self.m, self.n, element_bytes, "output")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "CublasLtLinearProblem":
        if not isinstance(item, InventoryOp):
            raise ValidationError("cuBLASLt problem requires an InventoryOp")
        if (
            item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.GEMM
            or item.opcode != "linear"
        ):
            raise ValidationError("cuBLASLt problem requires a required GEMM linear site")
        if item.attributes or len(item.input_types) != 3 or len(item.output_types) != 1:
            raise ValidationError("cuBLASLt linear requires X, weight, bias and one output")
        x, weight, bias = item.input_types
        output = item.output_types[0]
        tensors = (x, weight, bias, output)
        if any(value.device != "cuda" or value.layout != "row_major" for value in tensors):
            raise ValidationError("cuBLASLt linear requires CUDA row-major tensors")
        if len(x.shape) < 1 or len(weight.shape) != 2 or len(bias.shape) != 1:
            raise ValidationError("cuBLASLt linear has invalid tensor ranks")
        if not all(
            isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
            for value in tensors
            for dimension in value.shape
        ):
            raise ValidationError("cuBLASLt linear requires positive static shapes")
        if len({value.dtype for value in tensors}) != 1:
            raise ValidationError("cuBLASLt linear requires one exact tensor dtype")
        dtype = CublasLtDType.from_ir(x.dtype)
        k = x.shape[-1]
        n, weight_k = weight.shape
        assert isinstance(k, int) and isinstance(n, int) and isinstance(weight_k, int)
        if weight_k != k or bias.shape != (n,) or output.shape != x.shape[:-1] + (n,):
            raise ValidationError("cuBLASLt linear tensor shapes do not close X*W^T+bias")
        m = math.prod(x.shape[:-1]) if len(x.shape) > 1 else 1
        return cls(target_arch, dtype, m, n, k)


@dataclass(frozen=True, slots=True)
class CublasLtAlgorithm:
    algorithm_id: int
    tile_id: int
    split_k: int
    reduction_scheme: int
    cta_swizzling: int
    custom_option: int
    stages_id: int
    inner_shape_id: int
    cluster_shape_id: int

    def __post_init__(self) -> None:
        _uint(self.algorithm_id, 31, "cuBLASLt algorithm ID")
        _uint(self.tile_id, 32, "cuBLASLt tile ID")
        _uint(self.split_k, 31, "cuBLASLt split-K")
        for value, label in (
            (self.reduction_scheme, "reduction scheme"),
            (self.cta_swizzling, "CTA swizzling"),
            (self.custom_option, "custom option"),
            (self.stages_id, "stages ID"),
        ):
            _uint(value, 32, f"cuBLASLt {label}")
        _uint(self.inner_shape_id, 16, "cuBLASLt inner-shape ID")
        _uint(self.cluster_shape_id, 16, "cuBLASLt cluster-shape ID")


@dataclass(frozen=True, slots=True)
class CublasLtLinearPayload:
    problem: CublasLtLinearProblem
    cublaslt_version: int
    algorithm: CublasLtAlgorithm
    workspace_bytes: int
    x_alignment: int
    weight_alignment: int
    bias_alignment: int
    output_alignment: int
    workspace_alignment: int
    compute_mode: int = CUBLASLT_COMPUTE_F32

    def __post_init__(self) -> None:
        if not isinstance(self.problem, CublasLtLinearProblem):
            raise ValidationError("cuBLASLt payload problem is invalid")
        if type(self.compute_mode) is not int or self.compute_mode not in (CUBLASLT_COMPUTE_F32,CUBLASLT_COMPUTE_FAST_TF32) or (self.compute_mode==CUBLASLT_COMPUTE_FAST_TF32 and self.problem.dtype!=CublasLtDType.F32):
            raise ValidationError("cuBLASLt compute mode does not match the tensor dtype")
        _positive_uint(self.cublaslt_version, 32, "cuBLASLt library version")
        if not isinstance(self.algorithm, CublasLtAlgorithm):
            raise ValidationError("cuBLASLt payload algorithm is invalid")
        _uint(self.workspace_bytes, 64, "cuBLASLt workspace bytes")
        for value, label in zip(
            self.alignments,
            ("X", "weight", "bias", "output", "workspace"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value > MAX_POINTER_ALIGNMENT
                or value & (value - 1)
            ):
                raise ValidationError(
                    f"cuBLASLt {label} alignment must be a bounded power of two"
                )

    @property
    def alignments(self) -> tuple[int, int, int, int, int]:
        return (
            self.x_alignment,
            self.weight_alignment,
            self.bias_alignment,
            self.output_alignment,
            self.workspace_alignment,
        )

    def to_bytes(self) -> bytes:
        problem = self.problem
        algorithm = self.algorithm
        return CUBLASLT_LINEAR_PAYLOAD.pack(
            CUBLASLT_LINEAR_PAYLOAD_MAGIC,
            CUBLASLT_LINEAR_SCHEMA_MAJOR,
            1 if self.compute_mode==CUBLASLT_COMPUTE_FAST_TF32 else 0,
            CUBLASLT_LINEAR_PAYLOAD.size,
            int(problem.target_arch),
            self.cublaslt_version,
            int(problem.dtype),
            self.compute_mode,
            CUBLASLT_SCALE_F32,
            CUBLASLT_LINEAR_FLAG_BIAS,
            CUBLASLT_EPILOGUE_BIAS,
            CUBLASLT_OP_T,
            CUBLASLT_OP_N,
            CUBLASLT_ORDER_COLUMN,
            CUBLASLT_ORDER_COLUMN,
            CUBLASLT_ORDER_COLUMN,
            CUBLASLT_ORDER_COLUMN,
            problem.m,
            problem.n,
            problem.k,
            problem.k,
            problem.k,
            problem.n,
            problem.n,
            self.workspace_bytes,
            algorithm.algorithm_id,
            algorithm.tile_id,
            algorithm.split_k,
            algorithm.reduction_scheme,
            algorithm.cta_swizzling,
            algorithm.custom_option,
            algorithm.stages_id,
            algorithm.inner_shape_id,
            algorithm.cluster_shape_id,
            *self.alignments,
            b"\0" * 8,
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "CublasLtLinearPayload":
        if len(data) != CUBLASLT_LINEAR_PAYLOAD.size:
            raise FormatError("cuBLASLt linear payload must have its fixed size")
        fields = CUBLASLT_LINEAR_PAYLOAD.unpack(data)
        if fields[0] != CUBLASLT_LINEAR_PAYLOAD_MAGIC:
            raise FormatError("bad cuBLASLt linear payload magic")
        if (
            fields[1] != CUBLASLT_LINEAR_SCHEMA_MAJOR
            or fields[2] > CUBLASLT_LINEAR_SCHEMA_MINOR
        ):
            raise FormatError(f"unsupported cuBLASLt payload schema {fields[1]}.{fields[2]}")
        if fields[3] != CUBLASLT_LINEAR_PAYLOAD.size:
            raise FormatError("cuBLASLt payload declares the wrong fixed size")
        try:
            target_arch = CudaArch(fields[4])
            dtype = CublasLtDType(fields[6])
        except ValueError as exc:
            raise FormatError("cuBLASLt payload has unknown target or dtype") from exc
        if (
            fields[7] not in (CUBLASLT_COMPUTE_F32,CUBLASLT_COMPUTE_FAST_TF32)
            or (fields[7]==CUBLASLT_COMPUTE_FAST_TF32 and fields[2]!=1)
            or fields[8:17]
            != (
                CUBLASLT_SCALE_F32,
                CUBLASLT_LINEAR_FLAG_BIAS,
                CUBLASLT_EPILOGUE_BIAS,
                CUBLASLT_OP_T,
                CUBLASLT_OP_N,
                CUBLASLT_ORDER_COLUMN,
                CUBLASLT_ORDER_COLUMN,
                CUBLASLT_ORDER_COLUMN,
                CUBLASLT_ORDER_COLUMN,
            )
            or fields[20:24] != (fields[19], fields[19], fields[18], fields[18])
            or any(fields[39])
        ):
            raise FormatError("cuBLASLt payload has invalid fixed layout or reserved fields")
        try:
            return cls(
                problem=CublasLtLinearProblem(
                    target_arch, dtype, fields[17], fields[18], fields[19]
                ),
                cublaslt_version=fields[5],
                algorithm=CublasLtAlgorithm(*fields[25:34]),
                workspace_bytes=fields[24],
                x_alignment=fields[34],
                weight_alignment=fields[35],
                bias_alignment=fields[36],
                output_alignment=fields[37],
                workspace_alignment=fields[38],
                compute_mode=fields[7],
            )
        except ValidationError as exc:
            raise FormatError(f"cuBLASLt payload contract is invalid: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CublasLtValidationReceipt:
    payload: CublasLtLinearPayload
    cuda_driver_version: int
    cuda_runtime_version: int
    algo_check: bool
    normal_launches: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    sample_count: int
    max_abs: float
    sample_tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.payload, CublasLtLinearPayload):
            raise ValidationError("cuBLASLt receipt payload is invalid")
        _positive_uint(self.cuda_driver_version, 32, "CUDA driver version")
        _positive_uint(self.cuda_runtime_version, 32, "CUDA runtime version")
        if self.algo_check is not True:
            raise ValidationError("cuBLASLt receipt requires a successful AlgoCheck")
        if (
            not isinstance(self.normal_launches, int)
            or isinstance(self.normal_launches, bool)
            or self.normal_launches < 2
            or self.normal_launches > 2**32 - 1
        ):
            raise ValidationError("cuBLASLt receipt requires at least two normal launches")
        if self.normal_repeat_bit_exact is not True:
            raise ValidationError("cuBLASLt receipt requires bit-exact normal repeats")
        if self.capture_replay_bit_exact is not True:
            raise ValidationError("cuBLASLt receipt requires bit-exact capture replay")
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or not 64 <= self.sample_count <= 2**32 - 1
        ):
            raise ValidationError("cuBLASLt receipt requires at least 64 CPU samples")
        expected_tolerance = (
            0.0005 if self.payload.problem.dtype == CublasLtDType.F32 else 0.005
        )
        for value, label in (
            (self.max_abs, "max_abs"),
            (self.sample_tolerance, "sample_tolerance"),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValidationError(f"cuBLASLt receipt {label} must be finite and non-negative")
        if self.sample_tolerance != expected_tolerance:
            raise ValidationError("cuBLASLt receipt uses an unlocked sample tolerance")
        if self.max_abs > self.sample_tolerance:
            raise ValidationError("cuBLASLt receipt exceeds its sampled correctness gate")

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload.to_bytes()).hexdigest()

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @property
    def provider_version(self) -> str:
        return (
            f"cublaslt={self.payload.cublaslt_version};"
            f"cuda-runtime={self.cuda_runtime_version};"
            f"driver={self.cuda_driver_version}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CUBLASLT_RECEIPT_SCHEMA,
            "payload_hex": self.payload.to_bytes().hex(),
            "payload_sha256": self.payload_sha256,
            "cuda_driver_version": self.cuda_driver_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "algo_check": self.algo_check,
            "normal_launches": self.normal_launches,
            "normal_repeat_bit_exact": self.normal_repeat_bit_exact,
            "capture_replay_bit_exact": self.capture_replay_bit_exact,
            "sample_count": self.sample_count,
            "max_abs": self.max_abs,
            "sample_tolerance": self.sample_tolerance,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CublasLtValidationReceipt":
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "payload_hex",
            "payload_sha256",
            "cuda_driver_version",
            "cuda_runtime_version",
            "algo_check",
            "normal_launches",
            "normal_repeat_bit_exact",
            "capture_replay_bit_exact",
            "sample_count",
            "max_abs",
            "sample_tolerance",
        }:
            raise ValidationError("cuBLASLt receipt has unknown or missing fields")
        if value["schema"] != CUBLASLT_RECEIPT_SCHEMA:
            raise ValidationError("cuBLASLt receipt schema is unknown")
        payload_hex = value["payload_hex"]
        if not isinstance(payload_hex, str) or len(payload_hex) != 384:
            raise ValidationError("cuBLASLt receipt payload_hex is invalid")
        try:
            payload_bytes = bytes.fromhex(payload_hex)
        except ValueError as exc:
            raise ValidationError("cuBLASLt receipt payload_hex is invalid") from exc
        if payload_hex != payload_bytes.hex():
            raise ValidationError("cuBLASLt receipt payload_hex is not canonical")
        payload_sha256 = value["payload_sha256"]
        _digest(payload_sha256, "cuBLASLt receipt payload_sha256")
        if hashlib.sha256(payload_bytes).hexdigest() != payload_sha256:
            raise ValidationError("cuBLASLt receipt payload digest does not match")
        try:
            payload = CublasLtLinearPayload.from_bytes(payload_bytes)
        except FormatError as exc:
            raise ValidationError(f"cuBLASLt receipt payload is invalid: {exc}") from exc
        return cls(
            payload=payload,
            cuda_driver_version=value["cuda_driver_version"],  # type: ignore[arg-type]
            cuda_runtime_version=value["cuda_runtime_version"],  # type: ignore[arg-type]
            algo_check=value["algo_check"],  # type: ignore[arg-type]
            normal_launches=value["normal_launches"],  # type: ignore[arg-type]
            normal_repeat_bit_exact=value["normal_repeat_bit_exact"],  # type: ignore[arg-type]
            capture_replay_bit_exact=value["capture_replay_bit_exact"],  # type: ignore[arg-type]
            sample_count=value["sample_count"],  # type: ignore[arg-type]
            max_abs=value["max_abs"],  # type: ignore[arg-type]
            sample_tolerance=value["sample_tolerance"],  # type: ignore[arg-type]
        )

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        expected = CublasLtLinearProblem.from_inventory(
            item, target_arch=self.payload.problem.target_arch
        )
        if expected != self.payload.problem:
            raise ValidationError("cuBLASLt receipt problem does not match inventory site")
        return ProviderCapability.exact_for(
            item,
            provider_id=CUBLASLT_PROVIDER_ID,
            abi_major=CUBLASLT_PROVIDER_ABI_MAJOR,
            abi_minor=CUBLASLT_PROVIDER_ABI_MINOR,
            provider_version=self.provider_version,
            implementation_id=CUBLASLT_IMPLEMENTATION_ID,
            implementation_digest=self.payload_sha256,
            target_arch=self.payload.problem.target_arch,
            supports_capture=True,
            workspace_bytes=self.payload.workspace_bytes,
        )


@dataclass(frozen=True, slots=True)
class CublasLtLoweredCommand:
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
                    "value_id": operand.value_id,
                    "access": operand.access.name.lower(),
                    "byte_offset": operand.byte_offset,
                }
                for operand in self.command.operands
            ],
            "payload_sha256": hashlib.sha256(self.command.payload).hexdigest(),
            "workspace_offset": self.command.workspace_offset,
            "workspace_bytes": self.command.workspace_bytes,
            "capture_safe": self.command.capture_safe,
        }


@dataclass(frozen=True, slots=True)
class CublasLtPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[CublasLtLoweredCommand, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]
    workspace_bytes: int

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CUBLASLT_PARTIAL_LOWERING_SCHEMA,
            "inventory_sha256": self.inventory_sha256,
            "schedule_sha256": self.schedule_sha256,
            "memory_plan_sha256": self.memory_plan_sha256,
            "complete": self.complete,
            "summary": {
                "capabilities": len(self.capabilities),
                "commands": len(self.commands),
                "elided_executions": len(self.elided_execution_indices),
                "unhandled_executions": len(self.unhandled_execution_indices),
                "workspace_bytes": self.workspace_bytes,
            },
            "capability_registry_sha256": hashlib.sha256(
                dump_provider_capabilities(self.capabilities).encode("utf-8")
            ).hexdigest(),
            "commands": [command.to_dict() for command in self.commands],
            "elided_execution_indices": list(self.elided_execution_indices),
            "unhandled_execution_indices": list(self.unhandled_execution_indices),
        }


def make_cublaslt_capabilities(
    inventory: LoweringInventory,
    receipts: Iterable[CublasLtValidationReceipt],
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if not isinstance(inventory, LoweringInventory):
        raise ValidationError("cuBLASLt capability construction requires a LoweringInventory")
    if not isinstance(target_arch, CudaArch):
        raise ValidationError("cuBLASLt capability target_arch must be a CudaArch")
    receipt_records = tuple(receipts)
    if not all(isinstance(item, (CublasLtValidationReceipt, LinearBuildBinding)) for item in receipt_records):
        raise ValidationError("cuBLASLt capability input contains an invalid receipt")
    receipt_by_problem: dict[CublasLtLinearProblem, CublasLtValidationReceipt] = {}
    for receipt in receipt_records:
        problem = receipt.payload.problem
        if problem.target_arch != target_arch:
            raise ValidationError("cuBLASLt receipt target does not match capability target")
        if problem in receipt_by_problem:
            raise ValidationError("cuBLASLt receipts contain a duplicate exact problem")
        receipt_by_problem[problem] = receipt

    sites_by_problem: dict[CublasLtLinearProblem, list[InventoryOp]] = {}
    for item in inventory.ops:
        if (
            item.status == RequirementStatus.REQUIRED
            and item.lowering_kind == LoweringKind.GEMM
            and item.opcode == "linear"
        ):
            problem = CublasLtLinearProblem.from_inventory(item, target_arch=target_arch)
            sites_by_problem.setdefault(problem, []).append(item)
    missing = set(sites_by_problem) - set(receipt_by_problem)
    extra = set(receipt_by_problem) - set(sites_by_problem)
    if missing or extra:
        raise ValidationError(
            f"cuBLASLt receipt coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    capabilities = tuple(
        receipt_by_problem[problem].capability_for(
            sorted(sites_by_problem[problem], key=lambda item: item.site_id)[0]
        )
        for problem in sorted(
            sites_by_problem,
            key=lambda item: (int(item.target_arch), int(item.dtype), item.m, item.n, item.k),
        )
    )
    return tuple(sorted(capabilities, key=lambda item: item.digest))


def lower_cublaslt_linear_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipts: Iterable[CublasLtValidationReceipt],
    *,
    target_arch: CudaArch,
) -> CublasLtPartialLowering:
    if not isinstance(schedule, ExecutionSchedule):
        raise ValidationError("cuBLASLt lowering requires an ExecutionSchedule")
    if not isinstance(memory_plan, MemoryPlan):
        raise ValidationError("cuBLASLt lowering requires a MemoryPlan")
    inventory_dump = dump_lowering_inventory(inventory)
    inventory_sha256 = hashlib.sha256(inventory_dump.encode("utf-8")).hexdigest()
    schedule_dump = dump_execution_schedule(schedule)
    schedule_sha256 = hashlib.sha256(schedule_dump.encode("utf-8")).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("cuBLASLt lowering inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("cuBLASLt lowering memory plan does not match its schedule")

    receipt_records = tuple(receipts)
    capabilities = make_cublaslt_capabilities(
        inventory, receipt_records, target_arch=target_arch
    )
    receipt_by_problem = {item.payload.problem: item for item in receipt_records}
    capability_by_problem: dict[CublasLtLinearProblem, ProviderCapability] = {}
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    for problem, receipt in receipt_by_problem.items():
        site = next(
            item
            for item in inventory.ops
            if item.opcode == "linear"
            and CublasLtLinearProblem.from_inventory(item, target_arch=target_arch) == problem
        )
        capability_by_problem[problem] = receipt.capability_for(site)
    allocation_ids = {item.value_id for item in memory_plan.allocations}
    if allocation_ids != set(range(len(schedule.values))):
        raise ValidationError("cuBLASLt lowering memory plan has incomplete value coverage")

    commands: list[CublasLtLoweredCommand] = []
    unhandled: list[int] = []
    elided = set(memory_plan.elided_ops)
    for op in schedule.ops:
        if op.execution_index in elided:
            continue
        if op.opcode != "linear":
            unhandled.append(op.execution_index)
            continue
        item = inventory_by_site.get(op.site)
        if item is None:
            raise ValidationError("cuBLASLt schedule site is absent from inventory")
        problem = CublasLtLinearProblem.from_inventory(item, target_arch=target_arch)
        receipt = receipt_by_problem[problem]
        capability = capability_by_problem[problem]
        actual_inputs = tuple(schedule.values[value_id].type for value_id in op.inputs)
        actual_outputs = tuple(schedule.values[value_id].type for value_id in op.outputs)
        if actual_inputs != item.input_types or actual_outputs != item.output_types:
            raise ValidationError("cuBLASLt scheduled operand types differ from inventory")
        if len(op.inputs) != 3 or len(op.outputs) != 1:
            raise ValidationError("cuBLASLt scheduled linear has invalid arity")
        payload = receipt.payload.to_bytes()
        if hashlib.sha256(payload).hexdigest() != capability.implementation_digest:
            raise ValidationError("cuBLASLt capability does not bind its provider payload")
        commands.append(
            CublasLtLoweredCommand(
                op.execution_index,
                op.site,
                ProviderCommand(
                    tag=CommandTag.CUBLASLT_MATMUL,
                    provider_id=capability.provider_id,
                    abi_major=capability.abi_major,
                    abi_minor=capability.abi_minor,
                    capability_digest=capability.digest,
                    operands=(
                        CommandOperand(op.inputs[0], OperandAccess.READ),
                        CommandOperand(op.inputs[1], OperandAccess.READ),
                        CommandOperand(op.inputs[2], OperandAccess.READ),
                        CommandOperand(op.outputs[0], OperandAccess.WRITE),
                    ),
                    payload=payload,
                    workspace_offset=0,
                    workspace_bytes=capability.workspace_bytes,
                    capture_safe=capability.supports_capture,
                ),
            )
        )
    memory_plan_sha256 = hashlib.sha256(
        _canonical_json(memory_plan.to_dict()).encode("utf-8") + b"\n"
    ).hexdigest()
    return CublasLtPartialLowering(
        inventory_sha256=inventory_sha256,
        schedule_sha256=schedule_sha256,
        memory_plan_sha256=memory_plan_sha256,
        capabilities=capabilities,
        commands=tuple(commands),
        elided_execution_indices=tuple(sorted(elided)),
        unhandled_execution_indices=tuple(unhandled),
        workspace_bytes=max(
            (receipt.payload.workspace_bytes for receipt in receipt_records), default=0
        ),
    )


def dump_cublaslt_partial_lowering(lowering: CublasLtPartialLowering) -> str:
    if not isinstance(lowering, CublasLtPartialLowering):
        raise ValidationError("cuBLASLt lowering dump requires a partial lowering")
    return _canonical_json(lowering.to_dict()) + "\n"


def _checked_tensor_bytes(rows: int, columns: int, item_bytes: int, label: str) -> None:
    maximum = 2**64 - 1
    if rows > maximum // columns or rows * columns > maximum // item_bytes:
        raise ValidationError(f"cuBLASLt {label} tensor byte size overflows uint64")


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
