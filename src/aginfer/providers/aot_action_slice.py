from __future__ import annotations

from .build_binding import BuildBinding, is_build_binding

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
from ..lowering.schedule import ExecutionSchedule, ScheduledOp, dump_execution_schedule
from ..schema import CudaArch


ACTION_SLICE_PAYLOAD_MAGIC = b"AISLC1\0\0"
ACTION_SLICE_PAYLOAD = struct.Struct("<8sHH" + "I" * 20 + "Q" * 6 + "32s20s")
ACTION_SLICE_RECEIPT_SCHEMA = "aginfer.action-slice-validation.v1"
ACTION_SLICE_PARTIAL_LOWERING_SCHEMA = "aginfer.action-slice-partial-lowering.v1"
ACTION_SLICE_IMPLEMENTATION = "aginfer.action_slice.f32.r50.w32.o7.v1"

assert ACTION_SLICE_PAYLOAD.size == 192


def _sig(shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature("f32", shape, "row_major", "cuda")


@dataclass(frozen=True, slots=True)
class ActionSliceProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("action slice exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "ActionSliceProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.MEMORY
            or item.opcode != "slice"
            or item.input_types != (_sig((1, 50, 32)),)
            or item.output_types != (_sig((1, 50, 7)),)
            or item.attributes != (("axis", 2), ("start", 0), ("stop", 7))
        ):
            raise ValidationError(
                "action slice operation is outside the delivered exact envelope"
            )
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "dtype": "f32",
            "input_shape": [1, 50, 32],
            "output_shape": [1, 50, 7],
            "axis": 2,
            "start": 0,
            "stop": 7,
        }


@dataclass(frozen=True, slots=True)
class ActionSlicePayload:
    problem: ActionSliceProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, ActionSliceProblem):
            raise ValidationError("action slice payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        return ACTION_SLICE_PAYLOAD.pack(
            ACTION_SLICE_PAYLOAD_MAGIC,
            1,
            0,
            int(self.problem.target_arch),
            1,
            1,
            3,
            50,
            32,
            7,
            350,
            2,
            256,
            4,
            4,
            1,
            1,
            2,
            0,
            7,
            1,
            0,
            0,
            6400,
            1400,
            self.module_bytes,
            0,
            0,
            0,
            bytes.fromhex(self.module_sha256),
            bytes(20),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "ActionSlicePayload":
        view = memoryview(data)
        if len(view) != ACTION_SLICE_PAYLOAD.size:
            raise FormatError("action slice payload must have its exact fixed size")
        fields = ACTION_SLICE_PAYLOAD.unpack(view)
        if fields[0] != ACTION_SLICE_PAYLOAD_MAGIC or fields[1:3] != (1, 0):
            raise FormatError("action slice payload header is invalid")
        try:
            parsed = cls(
                ActionSliceProblem(CudaArch(fields[3])), fields[25], fields[29].hex()
            )
        except (ValueError, ValidationError) as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError("action slice payload is outside the delivered exact variant")
        return parsed


@dataclass(frozen=True, slots=True)
class ActionSliceValidationReceipt:
    problem: ActionSliceProblem
    implementation_sha256: str
    implementation_bytes: int
    cuda_compiler_version: int
    cuda_runtime_version: int
    cuda_driver_version: int
    normal_launches: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    capture_matches_normal: bool
    full_output_bit_exact: bool
    contiguous_prefix_negative_detected: bool
    racecheck_hazards: int
    latency_ms: float
    semantic_baseline_latency_ms: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, ActionSliceProblem):
            raise ValidationError("action slice receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("action slice receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_output_bit_exact, "bit-exact full output"),
            (self.contiguous_prefix_negative_detected, "contiguous-prefix negative control"),
        ):
            if value is not True:
                raise ValidationError(f"action slice receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("action slice receipt requires zero racecheck hazards")
        if (
            not math.isfinite(self.latency_ms)
            or not math.isfinite(self.semantic_baseline_latency_ms)
            or self.latency_ms <= 0
            or self.semantic_baseline_latency_ms <= 0
        ):
            raise ValidationError("action slice latency evidence is invalid")
        if self.contains_ptx is not False:
            raise ValidationError("action slice implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        ActionSliceProblem.from_inventory(item, target_arch=self.problem.target_arch)
        return ProviderCapability.exact_for(
            item,
            provider_id=2,
            abi_major=1,
            abi_minor=0,
            provider_version=(
                f"aginfer-aot-cuda=1;cuda-runtime={self.cuda_runtime_version};"
                f"driver={self.cuda_driver_version}"
            ),
            implementation_id=ACTION_SLICE_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> ActionSlicePayload:
        return ActionSlicePayload(
            self.problem, self.implementation_bytes, self.implementation_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ACTION_SLICE_RECEIPT_SCHEMA,
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
            "full_output_bit_exact": self.full_output_bit_exact,
            "contiguous_prefix_negative_detected": self.contiguous_prefix_negative_detected,
            "racecheck_hazards": self.racecheck_hazards,
            "latency_ms": self.latency_ms,
            "semantic_baseline_latency_ms": self.semantic_baseline_latency_ms,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class ActionSliceLoweredCommand:
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
class ActionSlicePartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[ActionSliceLoweredCommand, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ACTION_SLICE_PARTIAL_LOWERING_SCHEMA,
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


def make_action_slice_capabilities(
    inventory: LoweringInventory,
    receipt: ActionSliceValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if not isinstance(inventory, LoweringInventory):
        raise ValidationError("action slice capabilities require an inventory")
    if not (isinstance(receipt, ActionSliceValidationReceipt) or is_build_binding(receipt, ActionSliceProblem)):
        raise ValidationError("action slice receipt is invalid")
    if receipt.problem.target_arch != target_arch:
        raise ValidationError("action slice receipt target differs from capability target")
    result: dict[str, ProviderCapability] = {}
    for item in inventory.ops:
        try:
            ActionSliceProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        result.setdefault(capability.digest, capability)
    if not result:
        raise ValidationError("action slice receipt matched no inventory site")
    return tuple(sorted(result.values(), key=lambda item: item.digest))


def lower_action_slice_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: ActionSliceValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> ActionSlicePartialLowering:
    if not isinstance(schedule, ExecutionSchedule) or not isinstance(memory_plan, MemoryPlan):
        raise ValidationError("action slice lowering needs a schedule and memory plan")
    inventory_sha = hashlib.sha256(dump_lowering_inventory(inventory).encode()).hexdigest()
    schedule_sha = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha or memory_plan.schedule_sha256 != schedule_sha:
        raise ValidationError("action slice stage identities do not match")
    if {item.value_id for item in memory_plan.allocations} != set(range(len(schedule.values))):
        raise ValidationError("action slice memory plan has incomplete value coverage")
    capabilities = make_action_slice_capabilities(
        inventory, receipt, target_arch=target_arch
    )
    by_site = {item.site_id: item for item in inventory.ops}
    consumers: dict[int, list[ScheduledOp]] = {}
    for scheduled in schedule.ops:
        for value_id in scheduled.inputs:
            consumers.setdefault(value_id, []).append(scheduled)
    entry_outputs = {value_id for _, value_id in schedule.entry_outputs}
    commands: list[ActionSliceLoweredCommand] = []
    handled: set[int] = set()
    for op in schedule.ops:
        item = by_site.get(op.site)
        if item is None:
            continue
        try:
            ActionSliceProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise ValidationError("action slice scheduled arity is invalid")
        if op.outputs[0] not in entry_outputs or consumers.get(op.outputs[0]):
            raise ValidationError("action slice exact variant requires a terminal entry output")
        if (
            tuple(schedule.values[index].type for index in op.inputs) != item.input_types
            or tuple(schedule.values[index].type for index in op.outputs) != item.output_types
        ):
            raise ValidationError("action slice scheduled types differ from inventory")
        capability = receipt.capability_for(item)
        commands.append(
            ActionSliceLoweredCommand(
                op.execution_index,
                op.site,
                ProviderCommand(
                    CommandTag.CUDA_KERNEL,
                    capability.provider_id,
                    capability.abi_major,
                    capability.abi_minor,
                    capability.digest,
                    (
                        CommandOperand(op.inputs[0], OperandAccess.READ),
                        CommandOperand(op.outputs[0], OperandAccess.WRITE),
                    ),
                    receipt.payload().to_bytes(),
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        handled.add(op.execution_index)
    if not commands:
        raise ValidationError("action slice found no terminal exact operation")
    elided = set(memory_plan.elided_ops)
    if handled & elided:
        raise ValidationError("action slice overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in handled and op.execution_index not in elided
    )
    memory_sha = hashlib.sha256((_canonical(memory_plan.to_dict()) + "\n").encode()).hexdigest()
    return ActionSlicePartialLowering(
        inventory_sha,
        schedule_sha,
        memory_sha,
        capabilities,
        tuple(commands),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_action_slice_partial_lowering(lowering: ActionSlicePartialLowering) -> str:
    if not isinstance(lowering, ActionSlicePartialLowering):
        raise ValidationError("action slice dump requires its partial lowering")
    return _canonical(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"action slice {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or not any(character != "0" for character in value)
    ):
        raise ValidationError(f"action slice {label} must be a SHA-256 digest")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
