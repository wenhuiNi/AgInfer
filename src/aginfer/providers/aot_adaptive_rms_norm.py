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


ADAPTIVE_RMS_NORM_PAYLOAD_MAGIC = b"AIARM1\0\0"
ADAPTIVE_RMS_NORM_SCHEMA_MAJOR = 1
ADAPTIVE_RMS_NORM_SCHEMA_MINOR = 0
ADAPTIVE_RMS_NORM_PAYLOAD = struct.Struct(
    "<8sHH" + "I" * 24 + "f" + "Q" * 5 + "32s8s"
)
ADAPTIVE_RMS_NORM_PROVIDER_ID = 2
ADAPTIVE_RMS_NORM_PROVIDER_ABI_MAJOR = 1
ADAPTIVE_RMS_NORM_PROVIDER_ABI_MINOR = 0
ADAPTIVE_RMS_NORM_PROVIDER_VERSION = "aginfer-aot-cuda=1"
ADAPTIVE_RMS_NORM_IMPLEMENTATION = "aginfer.adaptive_rms_norm.bf16_f32.width1024.v1"
ADAPTIVE_RMS_NORM_RECEIPT_SCHEMA = "aginfer.adaptive-rms-norm-validation.v1"
ADAPTIVE_RMS_NORM_PARTIAL_LOWERING_SCHEMA = (
    "aginfer.adaptive-rms-norm-partial-lowering.v1"
)
ADAPTIVE_RMS_NORM_VARIANT = 1

assert ADAPTIVE_RMS_NORM_PAYLOAD.size == 192


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


_BF16_TENSOR = _sig("bf16", (1, 50, 1024))
_F32_TENSOR = _sig("f32", (1, 50, 1024))
_F32_VECTOR = _sig("f32", (1, 1024))
_F32_WEIGHT = _sig("f32", (1024,))
_F32_MODULATION = _sig("f32", (1, 3072))
_F32_SCALAR = _sig("f32", (1,))


@dataclass(frozen=True, slots=True)
class AdaptiveRmsNormProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("adaptive RMSNorm exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "AdaptiveRmsNormProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.AOT_CUDA
            or item.opcode != "rms_norm"
            or item.input_types != (_F32_TENSOR, _F32_WEIGHT)
            or item.output_types != (_F32_TENSOR,)
            or item.attributes != (("epsilon", 1.0e-6),)
        ):
            raise ValidationError(
                "adaptive RMSNorm anchor is outside the delivered exact envelope"
            )
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "hidden": [1, 50, 1024],
            "hidden_dtype": "bf16",
            "modulation": [1, 3072],
            "modulation_dtype": "f32",
            "normalized": [1, 50, 1024],
            "gate": [1, 50, 1024],
            "output_dtype": "bf16",
            "modulation_layout": ["scale", "shift", "gate"],
            "epsilon": 1.0e-6,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveRmsNormPayload:
    problem: AdaptiveRmsNormProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AdaptiveRmsNormProblem):
            raise ValidationError("adaptive RMSNorm payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        tensor_bytes = 1 * 50 * 1024 * 2
        modulation_bytes = 1 * 3072 * 4
        return ADAPTIVE_RMS_NORM_PAYLOAD.pack(
            ADAPTIVE_RMS_NORM_PAYLOAD_MAGIC,
            ADAPTIVE_RMS_NORM_SCHEMA_MAJOR,
            ADAPTIVE_RMS_NORM_SCHEMA_MINOR,
            int(self.problem.target_arch),
            ADAPTIVE_RMS_NORM_VARIANT,
            2,  # hidden BF16
            1,  # modulation F32
            2,  # normalized BF16
            2,  # gate BF16
            1,  # row major
            1,
            50,
            1024,
            3072,
            2,
            50,
            1,
            1,
            256,
            1,
            1,
            0,
            16,
            16,
            16,
            16,
            1,  # F32 accumulation
            1.0e-6,
            tensor_bytes,
            modulation_bytes,
            tensor_bytes,
            tensor_bytes,
            self.module_bytes,
            bytes.fromhex(self.module_sha256),
            bytes(8),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "AdaptiveRmsNormPayload":
        view = memoryview(data)
        if len(view) != ADAPTIVE_RMS_NORM_PAYLOAD.size:
            raise FormatError("adaptive RMSNorm payload must have its exact fixed size")
        fields = ADAPTIVE_RMS_NORM_PAYLOAD.unpack(view)
        if fields[0] != ADAPTIVE_RMS_NORM_PAYLOAD_MAGIC:
            raise FormatError("adaptive RMSNorm payload magic is invalid")
        if (
            fields[1] != ADAPTIVE_RMS_NORM_SCHEMA_MAJOR
            or fields[2] > ADAPTIVE_RMS_NORM_SCHEMA_MINOR
        ):
            raise FormatError("adaptive RMSNorm payload schema is unsupported")
        try:
            target_arch = CudaArch(fields[3])
            parsed = cls(
                AdaptiveRmsNormProblem(target_arch), fields[32], fields[33].hex()
            )
        except (ValueError, ValidationError) as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError(
                "adaptive RMSNorm payload is outside the delivered exact variant"
            )
        return parsed


@dataclass(frozen=True, slots=True)
class AdaptiveRmsNormValidationReceipt:
    problem: AdaptiveRmsNormProblem
    implementation_sha256: str
    implementation_bytes: int
    cuda_compiler_version: int
    cuda_runtime_version: int
    cuda_driver_version: int
    normal_launches: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    capture_matches_normal: bool
    full_outputs_compared: bool
    gate_bit_exact: bool
    racecheck_hazards: int
    cosine: float
    max_abs: float
    p99_abs: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AdaptiveRmsNormProblem):
            raise ValidationError("adaptive RMSNorm receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("adaptive RMSNorm receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_outputs_compared, "full-output comparison"),
            (self.gate_bit_exact, "bit-exact gate output"),
        ):
            if value is not True:
                raise ValidationError(f"adaptive RMSNorm receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("adaptive RMSNorm receipt requires zero racecheck hazards")
        if (
            not math.isfinite(self.cosine)
            or self.cosine < 0.99999
            or not math.isfinite(self.max_abs)
            or self.max_abs > 0.03125
            or not math.isfinite(self.p99_abs)
            or self.p99_abs > 0.015625
        ):
            raise ValidationError("adaptive RMSNorm correctness gate failed")
        if self.contains_ptx is not False:
            raise ValidationError("adaptive RMSNorm implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = AdaptiveRmsNormProblem.from_inventory(
            item, target_arch=self.problem.target_arch
        )
        if problem != self.problem:
            raise ValidationError("adaptive RMSNorm receipt does not cover this problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=ADAPTIVE_RMS_NORM_PROVIDER_ID,
            abi_major=ADAPTIVE_RMS_NORM_PROVIDER_ABI_MAJOR,
            abi_minor=ADAPTIVE_RMS_NORM_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{ADAPTIVE_RMS_NORM_PROVIDER_VERSION};"
                f"cuda-runtime={self.cuda_runtime_version};driver={self.cuda_driver_version}"
            ),
            implementation_id=ADAPTIVE_RMS_NORM_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> AdaptiveRmsNormPayload:
        return AdaptiveRmsNormPayload(
            self.problem, self.implementation_bytes, self.implementation_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ADAPTIVE_RMS_NORM_RECEIPT_SCHEMA,
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
            "full_outputs_compared": self.full_outputs_compared,
            "gate_bit_exact": self.gate_bit_exact,
            "racecheck_hazards": self.racecheck_hazards,
            "cosine": self.cosine,
            "max_abs": self.max_abs,
            "p99_abs": self.p99_abs,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveRmsNormLoweredCommand:
    execution_index: int
    site: str
    fused_execution_indices: tuple[int, ...]
    command: ProviderCommand

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index,
            "site": self.site,
            "fused_execution_indices": list(self.fused_execution_indices),
            "capability_digest": self.command.capability_digest,
            "operands": [
                {"value_id": item.value_id, "access": item.access.name.lower()}
                for item in self.command.operands
            ],
            "payload_sha256": hashlib.sha256(self.command.payload).hexdigest(),
            "capture_safe": self.command.capture_safe,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveRmsNormPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[AdaptiveRmsNormLoweredCommand, ...]
    fused_execution_indices: tuple[int, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ADAPTIVE_RMS_NORM_PARTIAL_LOWERING_SCHEMA,
            "inventory_sha256": self.inventory_sha256,
            "schedule_sha256": self.schedule_sha256,
            "memory_plan_sha256": self.memory_plan_sha256,
            "complete": self.complete,
            "summary": {
                "capabilities": len(self.capabilities),
                "commands": len(self.commands),
                "fused_executions": len(self.fused_execution_indices),
                "elided_executions": len(self.elided_execution_indices),
                "unhandled_executions": len(self.unhandled_execution_indices),
                "workspace_bytes": 0,
            },
            "capability_registry_sha256": hashlib.sha256(
                dump_provider_capabilities(self.capabilities).encode()
            ).hexdigest(),
            "commands": [item.to_dict() for item in self.commands],
            "fused_execution_indices": list(self.fused_execution_indices),
            "elided_execution_indices": list(self.elided_execution_indices),
            "unhandled_execution_indices": list(self.unhandled_execution_indices),
        }


def make_adaptive_rms_norm_capabilities(
    inventory: LoweringInventory,
    receipt: AdaptiveRmsNormValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if not (isinstance(receipt, AdaptiveRmsNormValidationReceipt) or is_build_binding(receipt, AdaptiveRmsNormProblem)):
        raise ValidationError("adaptive RMSNorm receipt is invalid")
    if receipt.problem.target_arch != target_arch:
        raise ValidationError("adaptive RMSNorm receipt target differs from capability target")
    capabilities: dict[str, ProviderCapability] = {}
    for item in inventory.ops:
        if item.opcode != "rms_norm":
            continue
        try:
            AdaptiveRmsNormProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
    if not capabilities:
        raise ValidationError("adaptive RMSNorm receipt matched no inventory anchor")
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_adaptive_rms_norm_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: AdaptiveRmsNormValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> AdaptiveRmsNormPartialLowering:
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode()
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("adaptive RMSNorm inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("adaptive RMSNorm memory plan does not match its schedule")
    if {item.value_id for item in memory_plan.allocations} != set(
        range(len(schedule.values))
    ):
        raise ValidationError("adaptive RMSNorm memory plan has incomplete value coverage")

    capabilities = make_adaptive_rms_norm_capabilities(
        inventory, receipt, target_arch=target_arch
    )
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    ops_by_index = {item.execution_index: item for item in schedule.ops}
    consumers: dict[int, list[ScheduledOp]] = {}
    entry_outputs = {value_id for _, value_id in schedule.entry_outputs}
    for op in schedule.ops:
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(op)

    def producer(value_id: int, opcode: str) -> ScheduledOp:
        producer_index = schedule.values[value_id].producer
        if producer_index is None:
            raise ValidationError(
                f"adaptive RMSNorm expected {opcode} producer for an intermediate"
            )
        result = ops_by_index[producer_index]
        if result.opcode != opcode:
            raise ValidationError(
                f"adaptive RMSNorm expected {opcode} producer for an intermediate"
            )
        return result

    def is_scalar_one(value_id: int) -> bool:
        value = schedule.values[value_id]
        if value.type != _F32_SCALAR or value.constant_identity is None:
            return False
        prefix = "literal:"
        if not value.constant_identity.startswith(prefix):
            return False
        item = inventory_by_site.get(value.constant_identity[len(prefix) :])
        return (
            item is not None
            and item.opcode == "constant"
            and item.input_types == ()
            and item.output_types == (_F32_SCALAR,)
            and item.attributes == (("value", (1.0,)),)
        )

    def check_broadcast(
        op: ScheduledOp, input_type: TensorSignature, output_type: TensorSignature
    ) -> None:
        if (
            op.attributes
            != (("broadcast_dimensions", (0, 2)), ("shape", (1, 50, 1024)))
            or len(op.inputs) != 1
            or len(op.outputs) != 1
            or schedule.values[op.inputs[0]].type != input_type
            or schedule.values[op.outputs[0]].type != output_type
        ):
            raise ValidationError("adaptive RMSNorm has a non-exact modulation broadcast")

    commands: list[AdaptiveRmsNormLoweredCommand] = []
    fused: set[int] = set()
    supported_rms: set[int] = set()
    supported_scale_adds: set[int] = set()
    shared_weight_broadcasts: dict[int, ScheduledOp] = {}
    shared_one_broadcasts: dict[int, ScheduledOp] = {}
    for rms in schedule.ops:
        item = inventory_by_site.get(rms.site)
        if item is None or item.opcode != "rms_norm":
            continue
        try:
            AdaptiveRmsNormProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        if len(rms.inputs) != 2 or len(rms.outputs) != 1:
            raise ValidationError("adaptive RMSNorm anchor arity is invalid")

        input_cast = producer(rms.inputs[0], "cast")
        if (
            input_cast.invocation_id != rms.invocation_id
            or input_cast.attributes != (("dtype", "f32"),)
            or len(input_cast.inputs) != 1
            or len(input_cast.outputs) != 1
            or schedule.values[input_cast.inputs[0]].type != _BF16_TENSOR
            or schedule.values[input_cast.outputs[0]].type != _F32_TENSOR
            or consumers.get(input_cast.outputs[0], []) != [rms]
        ):
            raise ValidationError("adaptive RMSNorm requires an exclusive BF16-to-F32 cast")

        weight_broadcast = producer(rms.inputs[1], "broadcast_in_dim")
        if (
            weight_broadcast.invocation_id != rms.invocation_id
            or weight_broadcast.attributes
            != (("broadcast_dimensions", (0,)), ("shape", (1024,)))
            or len(weight_broadcast.inputs) != 1
            or len(weight_broadcast.outputs) != 1
            or schedule.values[weight_broadcast.outputs[0]].type != _F32_WEIGHT
            or not is_scalar_one(weight_broadcast.inputs[0])
        ):
            raise ValidationError("adaptive RMSNorm requires a shared exact unit weight")

        rms_consumers = consumers.get(rms.outputs[0], [])
        if len(rms_consumers) != 1 or rms_consumers[0].opcode != "mul":
            raise ValidationError("adaptive RMSNorm output must feed one scale multiply")
        scaled = rms_consumers[0]
        if scaled.invocation_id != rms.invocation_id or len(scaled.inputs) != 2:
            raise ValidationError("adaptive RMSNorm scale multiply is invalid")
        scale_values = [value_id for value_id in scaled.inputs if value_id != rms.outputs[0]]
        if len(scale_values) != 1:
            raise ValidationError("adaptive RMSNorm scale multiply must use the RMS output once")
        scale_add = producer(scale_values[0], "add")

        scale_broadcast: ScheduledOp | None = None
        ones_tokens: ScheduledOp | None = None
        if len(scale_add.inputs) != 2 or len(scale_add.outputs) != 1:
            raise ValidationError("adaptive RMSNorm scale add arity is invalid")
        for value_id in scale_add.inputs:
            candidate = producer(value_id, "broadcast_in_dim")
            if len(candidate.inputs) == 1 and is_scalar_one(candidate.inputs[0]):
                ones_tokens = candidate
            else:
                scale_broadcast = candidate
        if scale_broadcast is None or ones_tokens is None:
            raise ValidationError("adaptive RMSNorm scale add must combine one and scale")
        check_broadcast(scale_broadcast, _F32_VECTOR, _F32_TENSOR)
        if (
            ones_tokens.attributes
            != (("broadcast_dimensions", (0,)), ("shape", (1, 50, 1024)))
            or len(ones_tokens.inputs) != 1
            or len(ones_tokens.outputs) != 1
            or schedule.values[ones_tokens.outputs[0]].type != _F32_TENSOR
        ):
            raise ValidationError("adaptive RMSNorm requires a shared scalar-one tensor")

        scaled_consumers = consumers.get(scaled.outputs[0], [])
        if len(scaled_consumers) != 1 or scaled_consumers[0].opcode != "add":
            raise ValidationError("adaptive RMSNorm scaled value must feed one shift add")
        shifted = scaled_consumers[0]
        shift_values = [value_id for value_id in shifted.inputs if value_id != scaled.outputs[0]]
        if len(shift_values) != 1:
            raise ValidationError("adaptive RMSNorm shift add must use the scaled value once")
        shift_broadcast = producer(shift_values[0], "broadcast_in_dim")
        check_broadcast(shift_broadcast, _F32_VECTOR, _F32_TENSOR)

        normalized_consumers = consumers.get(shifted.outputs[0], [])
        if len(normalized_consumers) != 1 or normalized_consumers[0].opcode != "cast":
            raise ValidationError("adaptive RMSNorm shifted value must feed one output cast")
        normalized_cast = normalized_consumers[0]
        if (
            normalized_cast.attributes != (("dtype", "bf16"),)
            or len(normalized_cast.outputs) != 1
            or schedule.values[normalized_cast.outputs[0]].type != _BF16_TENSOR
        ):
            raise ValidationError("adaptive RMSNorm normalized output cast is not exact")

        scale_slice = producer(scale_broadcast.inputs[0], "slice")
        shift_slice = producer(shift_broadcast.inputs[0], "slice")
        modulation_id = scale_slice.inputs[0] if len(scale_slice.inputs) == 1 else -1
        if (
            modulation_id < 0
            or schedule.values[modulation_id].type != _F32_MODULATION
            or len(shift_slice.inputs) != 1
            or shift_slice.inputs[0] != modulation_id
        ):
            raise ValidationError("adaptive RMSNorm scale and shift must share modulation")

        modulation_slices = consumers.get(modulation_id, [])
        if len(modulation_slices) != 3 or any(op.opcode != "slice" for op in modulation_slices):
            raise ValidationError("adaptive RMSNorm modulation must have exactly three slices")
        slices_by_start: dict[int, ScheduledOp] = {}
        broadcasts_by_start: dict[int, ScheduledOp] = {}
        for slice_op in modulation_slices:
            attributes = dict(slice_op.attributes)
            start = attributes.get("start")
            if (
                slice_op.invocation_id != rms.invocation_id
                or attributes.get("axis") != 1
                or start not in {0, 1024, 2048}
                or attributes.get("stop") != start + 1024
                or len(slice_op.outputs) != 1
                or schedule.values[slice_op.outputs[0]].type != _F32_VECTOR
                or len(consumers.get(slice_op.outputs[0], [])) != 1
            ):
                raise ValidationError("adaptive RMSNorm modulation slice is not exact")
            broadcast = consumers[slice_op.outputs[0]][0]
            if broadcast.opcode != "broadcast_in_dim":
                raise ValidationError("adaptive RMSNorm slice must feed one broadcast")
            check_broadcast(broadcast, _F32_VECTOR, _F32_TENSOR)
            slices_by_start[int(start)] = slice_op
            broadcasts_by_start[int(start)] = broadcast
        if set(slices_by_start) != {0, 1024, 2048}:
            raise ValidationError("adaptive RMSNorm modulation slices are incomplete")
        if broadcasts_by_start[0] != scale_broadcast or broadcasts_by_start[1024] != shift_broadcast:
            raise ValidationError("adaptive RMSNorm scale/shift slices are misrouted")
        if (
            consumers.get(scale_broadcast.outputs[0], []) != [scale_add]
            or consumers.get(shift_broadcast.outputs[0], []) != [shifted]
        ):
            raise ValidationError("adaptive RMSNorm scale/shift broadcast is not exclusive")

        gate_broadcast = broadcasts_by_start[2048]
        gate_consumers = consumers.get(gate_broadcast.outputs[0], [])
        if len(gate_consumers) != 1 or gate_consumers[0].opcode != "cast":
            raise ValidationError("adaptive RMSNorm gate must feed one output cast")
        gate_cast = gate_consumers[0]
        if (
            gate_cast.attributes != (("dtype", "bf16"),)
            or len(gate_cast.outputs) != 1
            or schedule.values[gate_cast.outputs[0]].type != _BF16_TENSOR
        ):
            raise ValidationError("adaptive RMSNorm gate output cast is not exact")

        region_ops = {
            input_cast.execution_index,
            rms.execution_index,
            scaled.execution_index,
            scale_add.execution_index,
            shifted.execution_index,
            normalized_cast.execution_index,
            gate_cast.execution_index,
            *(op.execution_index for op in slices_by_start.values()),
            *(op.execution_index for op in broadcasts_by_start.values()),
        }
        if len(region_ops) != 13 or fused & region_ops:
            raise ValidationError("adaptive RMSNorm regions overlap or are incomplete")
        intermediate_values = {
            input_cast.outputs[0],
            rms.outputs[0],
            scaled.outputs[0],
            scale_add.outputs[0],
            shifted.outputs[0],
            *(op.outputs[0] for op in slices_by_start.values()),
            *(op.outputs[0] for op in broadcasts_by_start.values()),
        }
        if entry_outputs & intermediate_values:
            raise ValidationError("adaptive RMSNorm fused intermediate is a public output")

        capability = receipt.capability_for(item)
        commands.append(
            AdaptiveRmsNormLoweredCommand(
                rms.execution_index,
                rms.site,
                tuple(sorted(region_ops)),
                ProviderCommand(
                    tag=CommandTag.CUDA_KERNEL,
                    provider_id=capability.provider_id,
                    abi_major=capability.abi_major,
                    abi_minor=capability.abi_minor,
                    capability_digest=capability.digest,
                    operands=(
                        CommandOperand(input_cast.inputs[0], OperandAccess.READ),
                        CommandOperand(modulation_id, OperandAccess.READ),
                        CommandOperand(normalized_cast.outputs[0], OperandAccess.WRITE),
                        CommandOperand(gate_cast.outputs[0], OperandAccess.WRITE),
                    ),
                    payload=receipt.payload().to_bytes(),
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        fused.update(region_ops)
        supported_rms.add(rms.execution_index)
        supported_scale_adds.add(scale_add.execution_index)
        shared_weight_broadcasts[weight_broadcast.execution_index] = weight_broadcast
        shared_one_broadcasts[ones_tokens.execution_index] = ones_tokens

    for broadcast in shared_weight_broadcasts.values():
        if all(op.execution_index in supported_rms for op in consumers[broadcast.outputs[0]]):
            fused.add(broadcast.execution_index)
    for broadcast in shared_one_broadcasts.values():
        if all(
            op.execution_index in supported_scale_adds
            for op in consumers[broadcast.outputs[0]]
        ):
            fused.add(broadcast.execution_index)

    elided = set(memory_plan.elided_ops)
    if fused & elided:
        raise ValidationError("adaptive RMSNorm fusion overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in fused and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return AdaptiveRmsNormPartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(fused)),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_adaptive_rms_norm_partial_lowering(
    lowering: AdaptiveRmsNormPartialLowering,
) -> str:
    if not isinstance(lowering, AdaptiveRmsNormPartialLowering):
        raise ValidationError(
            "adaptive RMSNorm dump requires an AdaptiveRmsNormPartialLowering"
        )
    return _canonical_json(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"adaptive RMSNorm {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or not any(char != "0" for char in value)
    ):
        raise ValidationError(f"adaptive RMSNorm {label} must be a SHA-256 digest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
