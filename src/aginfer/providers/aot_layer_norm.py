from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass

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


LAYER_NORM_PAYLOAD_MAGIC = b"AILNR1\0\0"
LAYER_NORM_SCHEMA_MAJOR = 1
LAYER_NORM_SCHEMA_MINOR = 0
LAYER_NORM_PAYLOAD = struct.Struct(
    "<8sHH" + "I" * 19 + "f" + "Q" * 5 + "32s28s"
)
LAYER_NORM_PROVIDER_ID = 2
LAYER_NORM_PROVIDER_ABI_MAJOR = 1
LAYER_NORM_PROVIDER_ABI_MINOR = 0
LAYER_NORM_PROVIDER_VERSION = "aginfer-aot-cuda=1"
LAYER_NORM_IMPLEMENTATION = "aginfer.layer_norm.f32.rows256.width1152.v1"
LAYER_NORM_RECEIPT_SCHEMA = "aginfer.layer-norm-validation.v1"
LAYER_NORM_PARTIAL_LOWERING_SCHEMA = "aginfer.layer-norm-partial-lowering.v1"
LAYER_NORM_TENSOR_BYTES = 1 * 256 * 1152 * 4
LAYER_NORM_PARAMETER_BYTES = 1152 * 4

assert LAYER_NORM_PAYLOAD.size == 192


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


@dataclass(frozen=True, slots=True)
class LayerNormProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("AOT LayerNorm exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "LayerNormProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.AOT_CUDA
            or item.opcode != "layer_norm"
            or item.input_types
            != (
                _sig("f32", (1, 256, 1152)),
                _sig("f32", (1152,)),
                _sig("f32", (1152,)),
            )
            or item.output_types != (_sig("f32", (1, 256, 1152)),)
            or item.attributes != (("epsilon", 1.0e-6),)
        ):
            raise ValidationError(
                "AOT LayerNorm site is outside the F32 rows256/width1152 envelope"
            )
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "input": [1, 256, 1152],
            "parameters": [1152],
            "output": [1, 256, 1152],
            "dtype": "f32",
            "epsilon": 1.0e-6,
        }


@dataclass(frozen=True, slots=True)
class LayerNormPayload:
    problem: LayerNormProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, LayerNormProblem):
            raise ValidationError("AOT LayerNorm payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        return LAYER_NORM_PAYLOAD.pack(
            LAYER_NORM_PAYLOAD_MAGIC,
            LAYER_NORM_SCHEMA_MAJOR,
            LAYER_NORM_SCHEMA_MINOR,
            int(self.problem.target_arch),
            1,  # exact variant
            1,  # F32
            1,  # row major
            1,
            256,
            1152,
            256,
            1,
            1,
            256,
            1,
            1,
            0,
            16,
            16,
            16,
            1,  # affine weight and bias
            1,  # two-pass centered variance
            1.0e-6,
            LAYER_NORM_TENSOR_BYTES,
            LAYER_NORM_PARAMETER_BYTES,
            LAYER_NORM_PARAMETER_BYTES,
            LAYER_NORM_TENSOR_BYTES,
            self.module_bytes,
            bytes.fromhex(self.module_sha256),
            bytes(28),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "LayerNormPayload":
        view = memoryview(data)
        if len(view) != LAYER_NORM_PAYLOAD.size:
            raise FormatError("AOT LayerNorm payload must have its exact fixed size")
        fields = LAYER_NORM_PAYLOAD.unpack(view)
        if fields[0] != LAYER_NORM_PAYLOAD_MAGIC:
            raise FormatError("AOT LayerNorm payload magic is invalid")
        if fields[1] != LAYER_NORM_SCHEMA_MAJOR or fields[2] > LAYER_NORM_SCHEMA_MINOR:
            raise FormatError("AOT LayerNorm payload schema is unsupported")
        try:
            target_arch = CudaArch(fields[3])
        except ValueError as exc:
            raise FormatError("AOT LayerNorm payload target arch is unknown") from exc
        try:
            parsed = cls(LayerNormProblem(target_arch), fields[27], fields[28].hex())
        except ValidationError as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError("AOT LayerNorm payload is outside the delivered exact variant")
        return parsed


@dataclass(frozen=True, slots=True)
class LayerNormValidationReceipt:
    problem: LayerNormProblem
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
        if not isinstance(self.problem, LayerNormProblem):
            raise ValidationError("AOT LayerNorm receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("AOT LayerNorm receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_output_compared, "full-output comparison"),
        ):
            if value is not True:
                raise ValidationError(f"AOT LayerNorm receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("AOT LayerNorm receipt requires zero racecheck hazards")
        if (
            not math.isfinite(self.cosine)
            or self.cosine < 0.99999
            or not math.isfinite(self.max_abs)
            or self.max_abs > 0.00025
            or not math.isfinite(self.p99_abs)
            or self.p99_abs > 0.000125
        ):
            raise ValidationError("AOT LayerNorm correctness gate failed")
        if self.contains_ptx is not False:
            raise ValidationError("AOT LayerNorm implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = LayerNormProblem.from_inventory(
            item, target_arch=self.problem.target_arch
        )
        if problem != self.problem:
            raise ValidationError("AOT LayerNorm receipt does not cover this problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=LAYER_NORM_PROVIDER_ID,
            abi_major=LAYER_NORM_PROVIDER_ABI_MAJOR,
            abi_minor=LAYER_NORM_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{LAYER_NORM_PROVIDER_VERSION};"
                f"cuda-runtime={self.cuda_runtime_version};driver={self.cuda_driver_version}"
            ),
            implementation_id=LAYER_NORM_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> LayerNormPayload:
        return LayerNormPayload(
            self.problem, self.implementation_bytes, self.implementation_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": LAYER_NORM_RECEIPT_SCHEMA,
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
class LayerNormLoweredCommand:
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
class LayerNormPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[LayerNormLoweredCommand, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": LAYER_NORM_PARTIAL_LOWERING_SCHEMA,
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


def make_layer_norm_capabilities(
    inventory: LoweringInventory,
    receipt: LayerNormValidationReceipt,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if receipt.problem.target_arch != target_arch:
        raise ValidationError("AOT LayerNorm receipt target differs from capability target")
    capabilities: dict[str, ProviderCapability] = {}
    matched = 0
    for item in inventory.ops:
        if item.opcode != "layer_norm":
            continue
        try:
            LayerNormProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
        matched += 1
    if matched == 0:
        raise ValidationError("AOT LayerNorm receipt matched no inventory sites")
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_layer_norm_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: LayerNormValidationReceipt,
    *,
    target_arch: CudaArch,
) -> LayerNormPartialLowering:
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode()
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("AOT LayerNorm inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("AOT LayerNorm memory plan does not match its schedule")
    if {item.value_id for item in memory_plan.allocations} != set(range(len(schedule.values))):
        raise ValidationError("AOT LayerNorm memory plan has incomplete value coverage")

    capabilities = make_layer_norm_capabilities(
        inventory, receipt, target_arch=target_arch
    )
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    commands: list[LayerNormLoweredCommand] = []
    handled: set[int] = set()
    for op in schedule.ops:
        item = inventory_by_site.get(op.site)
        if item is None or item.opcode != "layer_norm":
            continue
        try:
            LayerNormProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        if len(op.inputs) != 3 or len(op.outputs) != 1:
            raise ValidationError("AOT LayerNorm scheduled arity is invalid")
        capability = receipt.capability_for(item)
        commands.append(
            LayerNormLoweredCommand(
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
                        CommandOperand(op.inputs[2], OperandAccess.READ),
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
        raise ValidationError("AOT LayerNorm command overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in handled and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return LayerNormPartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_layer_norm_partial_lowering(lowering: LayerNormPartialLowering) -> str:
    if not isinstance(lowering, LayerNormPartialLowering):
        raise ValidationError("AOT LayerNorm dump requires a LayerNormPartialLowering")
    return _canonical_json(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"AOT LayerNorm {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or not any(char != "0" for char in value)
    ):
        raise ValidationError(f"AOT LayerNorm {label} must be a SHA-256 digest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
