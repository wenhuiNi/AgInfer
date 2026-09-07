from __future__ import annotations

from .build_binding import BuildBinding, is_build_binding

import hashlib
import json
import math
from dataclasses import dataclass

from ..errors import ValidationError
from ..lowering.capability import ProviderCapability, dump_provider_capabilities
from ..lowering.command import CommandOperand, CommandTag, OperandAccess, ProviderCommand
from ..lowering.inventory import (
    InventoryOp,
    LoweringInventory,
    dump_lowering_inventory,
)
from ..lowering.memory import MemoryPlan
from ..lowering.schedule import ExecutionSchedule, dump_execution_schedule
from ..schema import CudaArch
from .aot_cuda import (
    CUDA_KERNEL_PROVIDER_ABI_MAJOR,
    CUDA_KERNEL_PROVIDER_ABI_MINOR,
    CUDA_KERNEL_PROVIDER_ID,
    CUDA_KERNEL_PROVIDER_VERSION,
    AotActivationProblem,
    CudaKernelDType,
    CudaKernelPayload,
)


AOT_ACTIVATION_RECEIPT_SCHEMA = "aginfer.aot-activation-validation.v1"
AOT_ACTIVATION_PARTIAL_LOWERING_SCHEMA = (
    "aginfer.aot-activation-partial-lowering.v1"
)


@dataclass(frozen=True, slots=True)
class AotActivationValidationReceipt:
    problem: AotActivationProblem
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
        if not isinstance(self.problem, AotActivationProblem):
            raise ValidationError("AOT activation receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("AOT activation receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_output_compared, "full-output comparison"),
        ):
            if value is not True:
                raise ValidationError(f"AOT activation receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("AOT activation receipt requires zero racecheck hazards")
        f32 = self.problem.dtype == CudaKernelDType.F32
        max_abs_gate = 0.00001 if f32 else 0.015625
        p99_gate = 0.000002 if f32 else 0.0078125
        if (
            not math.isfinite(self.cosine)
            or self.cosine < 0.99999
            or not math.isfinite(self.max_abs)
            or self.max_abs > max_abs_gate
            or not math.isfinite(self.p99_abs)
            or self.p99_abs > p99_gate
        ):
            raise ValidationError("AOT activation correctness gate failed")
        if self.contains_ptx is not False:
            raise ValidationError("AOT activation implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = AotActivationProblem.from_inventory(
            item, target_arch=self.problem.target_arch
        )
        if problem != self.problem:
            raise ValidationError("AOT activation receipt does not cover this problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=CUDA_KERNEL_PROVIDER_ID,
            abi_major=CUDA_KERNEL_PROVIDER_ABI_MAJOR,
            abi_minor=CUDA_KERNEL_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{CUDA_KERNEL_PROVIDER_VERSION};"
                f"cuda-runtime={self.cuda_runtime_version};"
                f"driver={self.cuda_driver_version}"
            ),
            implementation_id=self.problem.kernel_id.implementation_id,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> CudaKernelPayload:
        return CudaKernelPayload.for_problem(
            self.problem,
            module_bytes=self.implementation_bytes,
            module_sha256=self.implementation_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": AOT_ACTIVATION_RECEIPT_SCHEMA,
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
class AotActivationLoweredCommand:
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
class AotActivationPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[AotActivationLoweredCommand, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": AOT_ACTIVATION_PARTIAL_LOWERING_SCHEMA,
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
    receipts: tuple[AotActivationValidationReceipt | BuildBinding, ...], *, target_arch: CudaArch
) -> dict[AotActivationProblem, AotActivationValidationReceipt]:
    if not isinstance(receipts, tuple) or not receipts:
        raise ValidationError("AOT activation receipts must be a non-empty tuple")
    result: dict[AotActivationProblem, AotActivationValidationReceipt] = {}
    for receipt in receipts:
        if not (isinstance(receipt, AotActivationValidationReceipt) or is_build_binding(receipt, AotActivationProblem)):
            raise ValidationError("AOT activation receipt is invalid")
        if receipt.problem.target_arch != target_arch:
            raise ValidationError("AOT activation receipt target differs from capability target")
        if receipt.problem in result:
            raise ValidationError("AOT activation receipts contain a duplicate problem")
        result[receipt.problem] = receipt
    return result


def make_aot_activation_capabilities(
    inventory: LoweringInventory,
    receipts: tuple[AotActivationValidationReceipt | BuildBinding, ...],
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if not isinstance(inventory, LoweringInventory):
        raise ValidationError("AOT activation capabilities require a LoweringInventory")
    by_problem = _receipt_map(receipts, target_arch=target_arch)
    capabilities: dict[str, ProviderCapability] = {}
    matched: set[AotActivationProblem] = set()
    for item in inventory.ops:
        if item.opcode not in {"gelu", "silu"}:
            continue
        try:
            problem = AotActivationProblem.from_inventory(
                item, target_arch=target_arch
            )
        except ValidationError:
            continue
        receipt = by_problem.get(problem)
        if receipt is None:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
        matched.add(problem)
    if set(by_problem) - matched:
        raise ValidationError(
            "AOT activation receipt matched no inventory site for one problem"
        )
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_aot_activation_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipts: tuple[AotActivationValidationReceipt | BuildBinding, ...],
    *,
    target_arch: CudaArch,
) -> AotActivationPartialLowering:
    if not isinstance(schedule, ExecutionSchedule):
        raise ValidationError("AOT activation lowering requires an ExecutionSchedule")
    if not isinstance(memory_plan, MemoryPlan):
        raise ValidationError("AOT activation lowering requires a MemoryPlan")
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode()
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(
        dump_execution_schedule(schedule).encode()
    ).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("AOT activation inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("AOT activation memory plan does not match its schedule")
    if {item.value_id for item in memory_plan.allocations} != set(
        range(len(schedule.values))
    ):
        raise ValidationError("AOT activation memory plan has incomplete value coverage")

    by_problem = _receipt_map(receipts, target_arch=target_arch)
    capabilities = make_aot_activation_capabilities(
        inventory, receipts, target_arch=target_arch
    )
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    commands: list[AotActivationLoweredCommand] = []
    handled: set[int] = set()
    for op in schedule.ops:
        item = inventory_by_site.get(op.site)
        if item is None or item.opcode not in {"gelu", "silu"}:
            continue
        try:
            problem = AotActivationProblem.from_inventory(
                item, target_arch=target_arch
            )
        except ValidationError:
            continue
        receipt = by_problem.get(problem)
        if receipt is None:
            continue
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise ValidationError("AOT activation scheduled arity is invalid")
        actual_inputs = tuple(schedule.values[index].type for index in op.inputs)
        actual_outputs = tuple(schedule.values[index].type for index in op.outputs)
        if actual_inputs != item.input_types or actual_outputs != item.output_types:
            raise ValidationError(
                "AOT activation scheduled operand types differ from inventory"
            )
        capability = receipt.capability_for(item)
        commands.append(
            AotActivationLoweredCommand(
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
                    payload=receipt.payload().to_bytes(),
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        handled.add(op.execution_index)

    elided = set(memory_plan.elided_ops)
    if handled & elided:
        raise ValidationError("AOT activation command overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in handled and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return AotActivationPartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_aot_activation_partial_lowering(
    lowering: AotActivationPartialLowering,
) -> str:
    if not isinstance(lowering, AotActivationPartialLowering):
        raise ValidationError(
            "AOT activation dump requires an AotActivationPartialLowering"
        )
    return _canonical_json(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"AOT activation {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or not any(char != "0" for char in value)
    ):
        raise ValidationError(f"AOT activation {label} must be a SHA-256 digest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
