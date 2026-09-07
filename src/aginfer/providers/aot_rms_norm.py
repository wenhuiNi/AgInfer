from __future__ import annotations

from .build_binding import BuildBinding, is_build_binding

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from enum import IntEnum

from ..errors import FormatError, ValidationError
from ..lowering.capability import ProviderCapability, dump_provider_capabilities
from ..lowering.command import CommandOperand, CommandTag, OperandAccess, ProviderCommand
from ..lowering.inventory import (
    InventoryOp,
    LoweringInventory,
    LoweringKind,
    RequirementStatus,
    TensorSignature,
    dump_lowering_inventory,
)
from ..lowering.memory import MemoryPlan
from ..lowering.schedule import ExecutionSchedule, dump_execution_schedule
from ..schema import CudaArch


RMS_NORM_PAYLOAD_MAGIC = b"AIRMS1\0\0"
RMS_NORM_SCHEMA_MAJOR = 1
RMS_NORM_SCHEMA_MINOR = 0
RMS_NORM_PAYLOAD = struct.Struct("<8sHH" + "I" * 20 + "f" + "Q" * 4 + "32s32s")
RMS_NORM_PROVIDER_ID = 2
RMS_NORM_PROVIDER_ABI_MAJOR = 1
RMS_NORM_PROVIDER_ABI_MINOR = 0
RMS_NORM_PROVIDER_VERSION = "aginfer-aot-cuda=1"
RMS_NORM_RECEIPT_SCHEMA = "aginfer.rms-norm-validation.v1"
RMS_NORM_PARTIAL_LOWERING_SCHEMA = "aginfer.rms-norm-partial-lowering.v1"

assert RMS_NORM_PAYLOAD.size == 192


class RmsNormVariant(IntEnum):
    F32_ROWS50_WIDTH1024 = 1
    BF16_ROWS968_WIDTH2048 = 2


_VARIANT_CONTRACTS: dict[RmsNormVariant, tuple[str, tuple[int, ...], tuple[int, ...]]] = {
    RmsNormVariant.F32_ROWS50_WIDTH1024: ("f32", (1, 50, 1024), (1024,)),
    RmsNormVariant.BF16_ROWS968_WIDTH2048: ("bf16", (1, 968, 2048), (2048,)),
}


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


@dataclass(frozen=True, slots=True)
class RmsNormProblem:
    target_arch: CudaArch
    variant: RmsNormVariant

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("AOT RMSNorm exact variants require SM120")
        if not isinstance(self.variant, RmsNormVariant):
            raise ValidationError("AOT RMSNorm variant is invalid")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "RmsNormProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.AOT_CUDA
            or item.opcode != "rms_norm"
            or item.attributes != (("epsilon", 1.0e-6),)
            or len(item.input_types) != 2
            or len(item.output_types) != 1
        ):
            raise ValidationError("AOT RMSNorm site is outside the delivered exact envelopes")
        for variant, (dtype, tensor_shape, weight_shape) in _VARIANT_CONTRACTS.items():
            if (
                item.input_types
                == (_sig(dtype, tensor_shape), _sig("f32", weight_shape))
                and item.output_types == (_sig(dtype, tensor_shape),)
            ):
                return cls(target_arch, variant)
        raise ValidationError("AOT RMSNorm site is outside the delivered exact envelopes")

    @property
    def dtype(self) -> str:
        return _VARIANT_CONTRACTS[self.variant][0]

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        return _VARIANT_CONTRACTS[self.variant][1]

    @property
    def weight_shape(self) -> tuple[int, ...]:
        return _VARIANT_CONTRACTS[self.variant][2]

    @property
    def rows(self) -> int:
        return math.prod(self.tensor_shape[:-1])

    @property
    def width(self) -> int:
        return self.tensor_shape[-1]

    @property
    def input_bytes(self) -> int:
        return math.prod(self.tensor_shape) * (4 if self.dtype == "f32" else 2)

    @property
    def weight_bytes(self) -> int:
        return math.prod(self.weight_shape) * 4

    @property
    def implementation_id(self) -> str:
        return (
            "aginfer.rms_norm.f32.rows50.width1024.v1"
            if self.variant == RmsNormVariant.F32_ROWS50_WIDTH1024
            else "aginfer.rms_norm.bf16.rows968.width2048.f32_weight.v1"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "variant": self.variant.name.lower(),
            "input": list(self.tensor_shape),
            "weight": list(self.weight_shape),
            "output": list(self.tensor_shape),
            "dtype": self.dtype,
            "weight_dtype": "f32",
            "epsilon": 1.0e-6,
        }


@dataclass(frozen=True, slots=True)
class RmsNormPayload:
    problem: RmsNormProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, RmsNormProblem):
            raise ValidationError("AOT RMSNorm payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        dtype = 1 if self.problem.dtype == "f32" else 2
        return RMS_NORM_PAYLOAD.pack(
            RMS_NORM_PAYLOAD_MAGIC,
            RMS_NORM_SCHEMA_MAJOR,
            RMS_NORM_SCHEMA_MINOR,
            int(self.problem.target_arch),
            int(self.problem.variant),
            dtype,
            1,  # F32 weight
            dtype,
            1,  # row major
            1,
            self.problem.rows,
            self.problem.width,
            self.problem.rows,
            1,
            1,
            256,
            1,
            1,
            0,
            16,
            16,
            16,
            1,  # F32 accumulation
            1.0e-6,
            self.problem.input_bytes,
            self.problem.weight_bytes,
            self.problem.input_bytes,
            self.module_bytes,
            bytes.fromhex(self.module_sha256),
            bytes(32),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "RmsNormPayload":
        view = memoryview(data)
        if len(view) != RMS_NORM_PAYLOAD.size:
            raise FormatError("AOT RMSNorm payload must have its exact fixed size")
        fields = RMS_NORM_PAYLOAD.unpack(view)
        if fields[0] != RMS_NORM_PAYLOAD_MAGIC:
            raise FormatError("AOT RMSNorm payload magic is invalid")
        if fields[1] != RMS_NORM_SCHEMA_MAJOR or fields[2] > RMS_NORM_SCHEMA_MINOR:
            raise FormatError("AOT RMSNorm payload schema is unsupported")
        try:
            target_arch = CudaArch(fields[3])
            variant = RmsNormVariant(fields[4])
            parsed = cls(RmsNormProblem(target_arch, variant), fields[27], fields[28].hex())
        except (ValueError, ValidationError) as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError("AOT RMSNorm payload is outside the delivered exact variants")
        return parsed


@dataclass(frozen=True, slots=True)
class RmsNormValidationReceipt:
    problem: RmsNormProblem
    implementation_sha256: str
    implementation_bytes: int
    cuda_compiler_version: int
    cuda_runtime_version: int
    cuda_driver_version: int
    normal_launches: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    capture_matches_normal: bool
    full_output_compared: bool
    racecheck_hazards: int
    cosine: float
    max_abs: float
    p99_abs: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, RmsNormProblem):
            raise ValidationError("AOT RMSNorm receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("AOT RMSNorm receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_output_compared, "full-output comparison"),
        ):
            if value is not True:
                raise ValidationError(f"AOT RMSNorm receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("AOT RMSNorm receipt requires zero racecheck hazards")
        max_abs_gate = (
            0.00025
            if self.problem.variant == RmsNormVariant.F32_ROWS50_WIDTH1024
            else 0.03125
        )
        p99_gate = (
            0.000125
            if self.problem.variant == RmsNormVariant.F32_ROWS50_WIDTH1024
            else 0.015625
        )
        if (
            not math.isfinite(self.cosine)
            or self.cosine < 0.99999
            or not math.isfinite(self.max_abs)
            or self.max_abs > max_abs_gate
            or not math.isfinite(self.p99_abs)
            or self.p99_abs > p99_gate
        ):
            raise ValidationError("AOT RMSNorm correctness gate failed")
        if self.contains_ptx is not False:
            raise ValidationError("AOT RMSNorm implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = RmsNormProblem.from_inventory(item, target_arch=self.problem.target_arch)
        if problem != self.problem:
            raise ValidationError("AOT RMSNorm receipt does not cover this problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=RMS_NORM_PROVIDER_ID,
            abi_major=RMS_NORM_PROVIDER_ABI_MAJOR,
            abi_minor=RMS_NORM_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{RMS_NORM_PROVIDER_VERSION};"
                f"cuda-runtime={self.cuda_runtime_version};driver={self.cuda_driver_version}"
            ),
            implementation_id=self.problem.implementation_id,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> RmsNormPayload:
        return RmsNormPayload(self.problem, self.implementation_bytes, self.implementation_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RMS_NORM_RECEIPT_SCHEMA,
            "problem": self.problem.to_dict(),
            "implementation_sha256": self.implementation_sha256,
            "implementation_bytes": self.implementation_bytes,
            "cuda_compiler_version": self.cuda_compiler_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "normal_launches": self.normal_launches,
            "normal_repeat_bit_exact": self.normal_repeat_bit_exact,
            "capture_replay_bit_exact": self.capture_replay_bit_exact,
            "capture_matches_normal": self.capture_matches_normal,
            "full_output_compared": self.full_output_compared,
            "racecheck_hazards": self.racecheck_hazards,
            "cosine": self.cosine,
            "max_abs": self.max_abs,
            "p99_abs": self.p99_abs,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class RmsNormLoweredCommand:
    execution_index: int
    site: str
    command: ProviderCommand

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index,
            "site": self.site,
            "capability_digest": self.command.capability_digest,
            "operands": [
                {"value_id": item.value_id, "access": item.access.name.lower()}
                for item in self.command.operands
            ],
            "payload_sha256": hashlib.sha256(self.command.payload).hexdigest(),
            "capture_safe": self.command.capture_safe,
        }


@dataclass(frozen=True, slots=True)
class RmsNormPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[RmsNormLoweredCommand, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RMS_NORM_PARTIAL_LOWERING_SCHEMA,
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
                dump_provider_capabilities(self.capabilities).encode()
            ).hexdigest(),
            "commands": [item.to_dict() for item in self.commands],
            "elided_execution_indices": list(self.elided_execution_indices),
            "unhandled_execution_indices": list(self.unhandled_execution_indices),
        }


def _receipt_map(
    receipts: tuple[RmsNormValidationReceipt | BuildBinding, ...], *, target_arch: CudaArch
) -> dict[RmsNormProblem, RmsNormValidationReceipt]:
    if not isinstance(receipts, tuple) or not receipts:
        raise ValidationError("AOT RMSNorm receipts must be a non-empty tuple")
    result: dict[RmsNormProblem, RmsNormValidationReceipt] = {}
    for receipt in receipts:
        if not (isinstance(receipt, RmsNormValidationReceipt) or is_build_binding(receipt, RmsNormProblem)):
            raise ValidationError("AOT RMSNorm receipt is invalid")
        if receipt.problem.target_arch != target_arch:
            raise ValidationError("AOT RMSNorm receipt target differs from capability target")
        if receipt.problem in result:
            raise ValidationError("AOT RMSNorm receipts contain a duplicate problem")
        result[receipt.problem] = receipt
    return result


def make_rms_norm_capabilities(
    inventory: LoweringInventory,
    receipts: tuple[RmsNormValidationReceipt | BuildBinding, ...],
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    by_problem = _receipt_map(receipts, target_arch=target_arch)
    capabilities: dict[str, ProviderCapability] = {}
    matched: set[RmsNormProblem] = set()
    for item in inventory.ops:
        if item.opcode != "rms_norm":
            continue
        try:
            problem = RmsNormProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        receipt = by_problem.get(problem)
        if receipt is None:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
        matched.add(problem)
    missing = set(by_problem) - matched
    if missing:
        raise ValidationError("AOT RMSNorm receipt matched no inventory site for one problem")
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_rms_norm_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipts: tuple[RmsNormValidationReceipt | BuildBinding, ...],
    *,
    target_arch: CudaArch,
) -> RmsNormPartialLowering:
    inventory_sha256 = hashlib.sha256(dump_lowering_inventory(inventory).encode()).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("AOT RMSNorm inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("AOT RMSNorm memory plan does not match its schedule")
    if {item.value_id for item in memory_plan.allocations} != set(range(len(schedule.values))):
        raise ValidationError("AOT RMSNorm memory plan has incomplete value coverage")

    by_problem = _receipt_map(receipts, target_arch=target_arch)
    capabilities = make_rms_norm_capabilities(
        inventory, receipts, target_arch=target_arch
    )
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    commands: list[RmsNormLoweredCommand] = []
    handled: set[int] = set()
    for op in schedule.ops:
        item = inventory_by_site.get(op.site)
        if item is None or item.opcode != "rms_norm":
            continue
        try:
            problem = RmsNormProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        receipt = by_problem.get(problem)
        if receipt is None:
            continue
        if len(op.inputs) != 2 or len(op.outputs) != 1:
            raise ValidationError("AOT RMSNorm scheduled arity is invalid")
        capability = receipt.capability_for(item)
        commands.append(
            RmsNormLoweredCommand(
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
                    payload=receipt.payload().to_bytes(),
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        handled.add(op.execution_index)

    elided = set(memory_plan.elided_ops)
    if handled & elided:
        raise ValidationError("AOT RMSNorm command overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in handled and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return RmsNormPartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_rms_norm_partial_lowering(lowering: RmsNormPartialLowering) -> str:
    if not isinstance(lowering, RmsNormPartialLowering):
        raise ValidationError("AOT RMSNorm dump requires a RmsNormPartialLowering")
    return _canonical_json(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"AOT RMSNorm {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or not any(char != "0" for char in value)
    ):
        raise ValidationError(f"AOT RMSNorm {label} must be a SHA-256 digest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
