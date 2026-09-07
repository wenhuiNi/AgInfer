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
from ..lowering.schedule import ExecutionSchedule, ScheduledOp, dump_execution_schedule
from ..schema import CudaArch


VISION_ATTENTION_PAYLOAD_MAGIC = b"AIVAT1\0\0"
VISION_ATTENTION_SCHEMA_MAJOR = 1
VISION_ATTENTION_SCHEMA_MINOR = 0
VISION_ATTENTION_PAYLOAD = struct.Struct(
    "<8sHH" + "I" * 21 + "f" + "Q" * 6 + "32s12s"
)
VISION_ATTENTION_PROVIDER_ID = 2
VISION_ATTENTION_PROVIDER_ABI_MAJOR = 1
VISION_ATTENTION_PROVIDER_ABI_MINOR = 0
VISION_ATTENTION_PROVIDER_VERSION = "aginfer-aot-cuda=1"
VISION_ATTENTION_IMPLEMENTATION = "aginfer.vision_attention.f32_bshd.v1"
VISION_ATTENTION_VARIANT = 1
VISION_ATTENTION_RECEIPT_SCHEMA = "aginfer.vision-attention-validation.v1"
VISION_ATTENTION_PARTIAL_LOWERING_SCHEMA = (
    "aginfer.vision-attention-partial-lowering.v1"
)
VISION_ATTENTION_TENSOR_BYTES = 1 * 256 * 16 * 72 * 4

assert VISION_ATTENTION_PAYLOAD.size == 192


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


@dataclass(frozen=True, slots=True)
class VisionAttentionProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("AOT vision attention exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "VisionAttentionProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.ATTENTION
            or item.opcode != "scaled_dot_product_attention"
            or len(item.input_types) != 4
            or len(item.output_types) != 1
        ):
            raise ValidationError("AOT vision attention requires an attention inventory site")
        expected_inputs = (
            _sig("f32", (1, 16, 256, 72)),
            _sig("f32", (1, 16, 256, 72)),
            _sig("f32", (1, 16, 256, 72)),
            _sig("bool", (1, 16, 256, 256)),
        )
        expected_outputs = (_sig("f32", (1, 16, 256, 72)),)
        expected_attributes = (
            ("kv_group_size", 1),
            ("mask_fill", "dtype_min"),
            ("scale", 1.0 / math.sqrt(72.0)),
        )
        if (
            item.input_types != expected_inputs
            or item.output_types != expected_outputs
            or item.attributes != expected_attributes
        ):
            raise ValidationError(
                "AOT vision attention site is outside the F32 head-dim 72 envelope"
            )
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "query_key_value_bshd": [1, 256, 16, 72],
            "mask_scalar": [1],
            "output_bshd": [1, 256, 16, 72],
            "dtype": "f32",
            "mask_dtype": "bool",
            "scale": 1.0 / math.sqrt(72.0),
            "num_heads": 16,
        }


@dataclass(frozen=True, slots=True)
class VisionAttentionPayload:
    problem: VisionAttentionProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, VisionAttentionProblem):
            raise ValidationError("AOT vision attention payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        return VISION_ATTENTION_PAYLOAD.pack(
            VISION_ATTENTION_PAYLOAD_MAGIC,
            VISION_ATTENTION_SCHEMA_MAJOR,
            VISION_ATTENTION_SCHEMA_MINOR,
            int(self.problem.target_arch),
            VISION_ATTENTION_VARIANT,
            1,  # query dtype: F32
            1,  # KV dtype: F32
            1,  # output dtype: F32
            4,  # mask dtype: BOOL
            1,  # input layout: BSHD
            1,  # output layout: BSHD
            1,
            256,
            256,
            16,
            72,
            256,
            16,
            1,
            256,
            1,
            1,
            0,
            1,  # finite dtype_min mask semantics
            1.0 / math.sqrt(72.0),
            VISION_ATTENTION_TENSOR_BYTES,
            VISION_ATTENTION_TENSOR_BYTES,
            VISION_ATTENTION_TENSOR_BYTES,
            1,
            VISION_ATTENTION_TENSOR_BYTES,
            self.module_bytes,
            bytes.fromhex(self.module_sha256),
            bytes(12),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "VisionAttentionPayload":
        view = memoryview(data)
        if len(view) != VISION_ATTENTION_PAYLOAD.size:
            raise FormatError("AOT vision attention payload must have its exact fixed size")
        fields = VISION_ATTENTION_PAYLOAD.unpack(view)
        if fields[0] != VISION_ATTENTION_PAYLOAD_MAGIC:
            raise FormatError("AOT vision attention payload magic is invalid")
        if (
            fields[1] != VISION_ATTENTION_SCHEMA_MAJOR
            or fields[2] > VISION_ATTENTION_SCHEMA_MINOR
        ):
            raise FormatError("AOT vision attention payload schema is unsupported")
        try:
            target_arch = CudaArch(fields[3])
        except ValueError as exc:
            raise FormatError("AOT vision attention payload target arch is unknown") from exc
        module_bytes = fields[30]
        module_sha256 = fields[31].hex()
        try:
            parsed = cls(VisionAttentionProblem(target_arch), module_bytes, module_sha256)
        except ValidationError as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError(
                "AOT vision attention payload is outside the delivered exact variant"
            )
        return parsed


@dataclass(frozen=True, slots=True)
class VisionAttentionValidationReceipt:
    problem: VisionAttentionProblem
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
    false_mask_uniform_verified: bool
    cosine: float
    max_abs: float
    p99_abs: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, VisionAttentionProblem):
            raise ValidationError("AOT vision attention receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("AOT vision attention receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_output_compared, "full-output comparison"),
            (self.false_mask_uniform_verified, "false-mask dtype_min semantics"),
        ):
            if value is not True:
                raise ValidationError(f"AOT vision attention receipt requires {label}")
        if (
            not math.isfinite(self.cosine)
            or self.cosine < 0.99999
            or not math.isfinite(self.max_abs)
            or self.max_abs > 0.00025
            or not math.isfinite(self.p99_abs)
            or self.p99_abs > 0.000125
        ):
            raise ValidationError("AOT vision attention correctness gate failed")
        if self.contains_ptx is not False:
            raise ValidationError("AOT vision attention implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = VisionAttentionProblem.from_inventory(
            item, target_arch=self.problem.target_arch
        )
        if problem != self.problem:
            raise ValidationError("AOT vision attention receipt does not cover this problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=VISION_ATTENTION_PROVIDER_ID,
            abi_major=VISION_ATTENTION_PROVIDER_ABI_MAJOR,
            abi_minor=VISION_ATTENTION_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{VISION_ATTENTION_PROVIDER_VERSION};"
                f"cuda-runtime={self.cuda_runtime_version};driver={self.cuda_driver_version}"
            ),
            implementation_id=VISION_ATTENTION_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> VisionAttentionPayload:
        return VisionAttentionPayload(
            self.problem, self.implementation_bytes, self.implementation_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": VISION_ATTENTION_RECEIPT_SCHEMA,
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
            "false_mask_uniform_verified": self.false_mask_uniform_verified,
            "cosine": self.cosine,
            "max_abs": self.max_abs,
            "p99_abs": self.p99_abs,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class VisionAttentionLoweredCommand:
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
class VisionAttentionPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[VisionAttentionLoweredCommand, ...]
    fused_execution_indices: tuple[int, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": VISION_ATTENTION_PARTIAL_LOWERING_SCHEMA,
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


def make_vision_attention_capabilities(
    inventory: LoweringInventory,
    receipt: VisionAttentionValidationReceipt,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if receipt.problem.target_arch != target_arch:
        raise ValidationError(
            "AOT vision attention receipt target differs from capability target"
        )
    capabilities: dict[str, ProviderCapability] = {}
    matched = 0
    for item in inventory.ops:
        if item.opcode != "scaled_dot_product_attention":
            continue
        try:
            VisionAttentionProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
        matched += 1
    if matched == 0:
        raise ValidationError("AOT vision attention receipt matched no inventory sites")
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_vision_attention_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: VisionAttentionValidationReceipt,
    *,
    target_arch: CudaArch,
) -> VisionAttentionPartialLowering:
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode()
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("AOT vision attention inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("AOT vision attention memory plan does not match its schedule")
    if {item.value_id for item in memory_plan.allocations} != set(range(len(schedule.values))):
        raise ValidationError("AOT vision attention memory plan has incomplete value coverage")

    capabilities = make_vision_attention_capabilities(
        inventory, receipt, target_arch=target_arch
    )
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    ops_by_index = {item.execution_index: item for item in schedule.ops}
    consumers: dict[int, list[ScheduledOp]] = {}
    entry_output_ids = {value_id for _, value_id in schedule.entry_outputs}
    for op in schedule.ops:
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(op)

    supported: dict[int, tuple[ScheduledOp, ProviderCapability]] = {}
    for op in schedule.ops:
        item = inventory_by_site.get(op.site)
        if item is None or item.opcode != "scaled_dot_product_attention":
            continue
        try:
            VisionAttentionProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        supported[op.execution_index] = (op, receipt.capability_for(item))

    commands: list[VisionAttentionLoweredCommand] = []
    fused: set[int] = set()
    broadcast_ops: dict[int, ScheduledOp] = {}
    bshd_type = _sig("f32", (1, 256, 16, 72))
    bhsd_type = _sig("f32", (1, 16, 256, 72))
    for execution_index, (op, capability) in supported.items():
        if len(op.inputs) != 4 or len(op.outputs) != 1:
            raise ValidationError("AOT vision attention scheduled arity is invalid")

        input_transposes: list[ScheduledOp] = []
        input_values: list[int] = []
        for value_id in op.inputs[:3]:
            value = schedule.values[value_id]
            if value.producer is None:
                raise ValidationError(
                    "AOT vision attention input has no BSHD-to-BHSd transpose producer"
                )
            transpose = ops_by_index[value.producer]
            if (
                transpose.opcode != "transpose"
                or transpose.invocation_id != op.invocation_id
                or transpose.attributes != (("permutation", (0, 2, 1, 3)),)
                or len(transpose.inputs) != 1
                or len(transpose.outputs) != 1
                or transpose.outputs[0] != value_id
                or schedule.values[transpose.inputs[0]].type != bshd_type
                or value.type != bhsd_type
                or value_id in entry_output_ids
                or consumers.get(value_id, []) != [op]
            ):
                raise ValidationError(
                    "AOT vision attention requires exclusive exact BSHD input transposes"
                )
            input_transposes.append(transpose)
            input_values.append(transpose.inputs[0])

        mask_value = schedule.values[op.inputs[3]]
        if mask_value.producer is None:
            raise ValidationError("AOT vision attention mask has no broadcast producer")
        broadcast = ops_by_index[mask_value.producer]
        if (
            broadcast.opcode != "broadcast_in_dim"
            or broadcast.invocation_id != op.invocation_id
            or len(broadcast.inputs) != 1
            or len(broadcast.outputs) != 1
            or broadcast.outputs[0] != op.inputs[3]
            or broadcast.attributes
            != (("broadcast_dimensions", (0,)), ("shape", (1, 16, 256, 256)))
            or schedule.values[broadcast.inputs[0]].type != _sig("bool", (1,))
            or broadcast.outputs[0] in entry_output_ids
        ):
            raise ValidationError(
                "AOT vision attention requires the exact shared scalar-mask broadcast"
            )
        broadcast_ops[broadcast.execution_index] = broadcast

        output_consumers = consumers.get(op.outputs[0], [])
        if len(output_consumers) != 1:
            raise ValidationError(
                "AOT vision attention output must have one transpose consumer"
            )
        output_transpose = output_consumers[0]
        if (
            op.outputs[0] in entry_output_ids
            or output_transpose.opcode != "transpose"
            or output_transpose.invocation_id != op.invocation_id
            or output_transpose.attributes != (("permutation", (0, 2, 1, 3)),)
            or len(output_transpose.outputs) != 1
            or schedule.values[output_transpose.outputs[0]].type != bshd_type
        ):
            raise ValidationError(
                "AOT vision attention requires the exact BHSd-to-BSHD output transpose"
            )

        fused_indices = tuple(
            item.execution_index for item in input_transposes
        ) + (execution_index, output_transpose.execution_index)
        if fused.intersection(fused_indices):
            raise ValidationError("AOT vision attention fused regions overlap")
        payload = receipt.payload().to_bytes()
        commands.append(
            VisionAttentionLoweredCommand(
                execution_index,
                op.site,
                fused_indices,
                ProviderCommand(
                    tag=CommandTag.ATTENTION,
                    provider_id=capability.provider_id,
                    abi_major=capability.abi_major,
                    abi_minor=capability.abi_minor,
                    capability_digest=capability.digest,
                    operands=(
                        CommandOperand(input_values[0], OperandAccess.READ),
                        CommandOperand(input_values[1], OperandAccess.READ),
                        CommandOperand(input_values[2], OperandAccess.READ),
                        CommandOperand(broadcast.inputs[0], OperandAccess.READ),
                        CommandOperand(output_transpose.outputs[0], OperandAccess.WRITE),
                    ),
                    payload=payload,
                    workspace_bytes=capability.workspace_bytes,
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        fused.update(fused_indices)

    for broadcast in broadcast_ops.values():
        broadcast_consumers = consumers.get(broadcast.outputs[0], [])
        if not broadcast_consumers or any(
            consumer.execution_index not in supported for consumer in broadcast_consumers
        ):
            raise ValidationError(
                "AOT vision attention mask broadcast has an unsupported consumer"
            )
        fused.add(broadcast.execution_index)

    elided = set(memory_plan.elided_ops)
    if fused & elided:
        raise ValidationError("AOT vision attention fusion overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in fused and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return VisionAttentionPartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(fused)),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_vision_attention_partial_lowering(
    lowering: VisionAttentionPartialLowering,
) -> str:
    if not isinstance(lowering, VisionAttentionPartialLowering):
        raise ValidationError(
            "AOT vision attention dump requires a VisionAttentionPartialLowering"
        )
    return _canonical_json(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"AOT vision attention {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or not any(char != "0" for char in value)
    ):
        raise ValidationError(f"AOT vision attention {label} must be a SHA-256 digest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
