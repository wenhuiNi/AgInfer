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
from ..lowering.schedule import (
    ExecutionSchedule,
    ScheduledOp,
    ValueStorage,
    dump_execution_schedule,
)
from ..schema import CudaArch


PREFIX_INPUT_PAYLOAD_MAGIC = b"AIPIA1\0\0"
PREFIX_INPUT_PAYLOAD = struct.Struct("<8sHH" + "I" * 20 + "Q" * 6 + "32s20s")
PREFIX_INPUT_RECEIPT_SCHEMA = "aginfer.prefix-input-validation.v1"
PREFIX_INPUT_PARTIAL_LOWERING_SCHEMA = "aginfer.prefix-input-partial-lowering.v1"
PREFIX_INPUT_IMPLEMENTATION = "aginfer.prefix_input.f32_bf16.s968_d2048.v1"

assert PREFIX_INPUT_PAYLOAD.size == 192


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


_IMAGE = _sig("f32", (1, 256, 2048))
_EMBEDDING = _sig("bf16", (257152, 2048))
_TOKENS = _sig("i32", (1, 200))
_LANGUAGE_BF16 = _sig("bf16", (1, 200, 2048))
_LANGUAGE_F32 = _sig("f32", (1, 200, 2048))
_PREFIX = _sig("f32", (1, 968, 2048))
_MASK_SCALAR = _sig("bool", (1,))
_IMAGE_MASK = _sig("bool", (1, 256))
_TOKEN_MASK = _sig("bool", (1, 200))
_PAD_MASK = _sig("bool", (1, 968))
_PAD_I32 = _sig("i32", (1, 968))


@dataclass(frozen=True, slots=True)
class PrefixInputProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("prefix input exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "PrefixInputProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.MEMORY
            or item.opcode != "concat"
            or item.input_types != (_IMAGE, _IMAGE, _IMAGE, _LANGUAGE_F32)
            or item.output_types != (_PREFIX,)
            or item.attributes != (("axis", 1),)
        ):
            raise ValidationError(
                "prefix input anchor is outside the delivered exact envelope"
            )
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "image_count": 3,
            "image_tokens": 256,
            "language_tokens": 200,
            "width": 2048,
            "vocabulary": 257152,
        }


@dataclass(frozen=True, slots=True)
class PrefixInputPayload:
    problem: PrefixInputProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, PrefixInputProblem):
            raise ValidationError("prefix input payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        return PREFIX_INPUT_PAYLOAD.pack(
            PREFIX_INPUT_PAYLOAD_MAGIC,
            1,
            0,
            int(self.problem.target_arch),
            1,
            9,
            3,
            3,
            256,
            200,
            2048,
            257152,
            968,
            7744,
            256,
            4,
            2,
            4,
            1,
            4,
            4,
            0,
            0,
            2097152,
            1053294592,
            800,
            7929856,
            self.module_bytes,
            0,
            bytes.fromhex(self.module_sha256),
            bytes(20),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "PrefixInputPayload":
        view = memoryview(data)
        if len(view) != PREFIX_INPUT_PAYLOAD.size:
            raise FormatError("prefix input payload must have its exact fixed size")
        fields = PREFIX_INPUT_PAYLOAD.unpack(view)
        if fields[0] != PREFIX_INPUT_PAYLOAD_MAGIC or fields[1:3] != (1, 0):
            raise FormatError("prefix input payload header is invalid")
        try:
            parsed = cls(
                PrefixInputProblem(CudaArch(fields[3])), fields[27], fields[29].hex()
            )
        except (ValueError, ValidationError) as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError("prefix input payload is outside the delivered exact variant")
        return parsed


@dataclass(frozen=True, slots=True)
class PrefixInputValidationReceipt:
    problem: PrefixInputProblem
    implementation_sha256: str
    implementation_bytes: int
    cuda_compiler_version: int
    cuda_runtime_version: int
    cuda_driver_version: int
    normal_launches: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    capture_matches_normal: bool
    full_outputs_exact: bool
    bf16_rounding_negative_detected: bool
    mask_hole_negative_detected: bool
    invalid_token_guarded: bool
    racecheck_hazards: int
    latency_ms: float
    original_chain_latency_ms: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, PrefixInputProblem):
            raise ValidationError("prefix input receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("prefix input receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_outputs_exact, "exact full outputs"),
            (self.bf16_rounding_negative_detected, "BF16 rounding negative control"),
            (self.mask_hole_negative_detected, "mask-hole negative control"),
            (self.invalid_token_guarded, "invalid-token guard"),
        ):
            if value is not True:
                raise ValidationError(f"prefix input receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("prefix input receipt requires zero racecheck hazards")
        if (
            not math.isfinite(self.latency_ms)
            or not math.isfinite(self.original_chain_latency_ms)
            or self.latency_ms <= 0
            or self.original_chain_latency_ms <= self.latency_ms
        ):
            raise ValidationError("prefix input latency evidence is invalid")
        if self.contains_ptx is not False:
            raise ValidationError("prefix input implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        PrefixInputProblem.from_inventory(item, target_arch=self.problem.target_arch)
        return ProviderCapability.exact_for(
            item,
            provider_id=2,
            abi_major=1,
            abi_minor=0,
            provider_version=(
                f"aginfer-aot-cuda=1;cuda-runtime={self.cuda_runtime_version};"
                f"driver={self.cuda_driver_version}"
            ),
            implementation_id=PREFIX_INPUT_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> PrefixInputPayload:
        return PrefixInputPayload(
            self.problem, self.implementation_bytes, self.implementation_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PREFIX_INPUT_RECEIPT_SCHEMA,
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
            "full_outputs_exact": self.full_outputs_exact,
            "bf16_rounding_negative_detected": self.bf16_rounding_negative_detected,
            "mask_hole_negative_detected": self.mask_hole_negative_detected,
            "invalid_token_guarded": self.invalid_token_guarded,
            "racecheck_hazards": self.racecheck_hazards,
            "latency_ms": self.latency_ms,
            "original_chain_latency_ms": self.original_chain_latency_ms,
            "speedup": self.original_chain_latency_ms / self.latency_ms,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class PrefixInputLoweredCommand:
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
class PrefixInputPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[PrefixInputLoweredCommand, ...]
    fused_execution_indices: tuple[int, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PREFIX_INPUT_PARTIAL_LOWERING_SCHEMA,
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


def make_prefix_input_capabilities(
    inventory: LoweringInventory,
    receipt: PrefixInputValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if not (isinstance(receipt, PrefixInputValidationReceipt) or is_build_binding(receipt, PrefixInputProblem)):
        raise ValidationError("prefix input receipt is invalid")
    if receipt.problem.target_arch != target_arch:
        raise ValidationError("prefix input receipt target differs from capability target")
    result: dict[str, ProviderCapability] = {}
    for item in inventory.ops:
        try:
            PrefixInputProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        result.setdefault(capability.digest, capability)
    if not result:
        raise ValidationError("prefix input receipt matched no inventory anchor")
    return tuple(sorted(result.values(), key=lambda item: item.digest))


def lower_prefix_input_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: PrefixInputValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> PrefixInputPartialLowering:
    inventory_sha = hashlib.sha256(dump_lowering_inventory(inventory).encode()).hexdigest()
    schedule_sha = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha or memory_plan.schedule_sha256 != schedule_sha:
        raise ValidationError("prefix input stage identities do not match")
    if {item.value_id for item in memory_plan.allocations} != set(range(len(schedule.values))):
        raise ValidationError("prefix input memory plan has incomplete value coverage")
    capabilities = make_prefix_input_capabilities(inventory, receipt, target_arch=target_arch)
    by_site = {item.site_id: item for item in inventory.ops}
    by_index = {item.execution_index: item for item in schedule.ops}
    consumers: dict[int, list[ScheduledOp]] = {}
    for op in schedule.ops:
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(op)
    entry_outputs = {value_id for _, value_id in schedule.entry_outputs}

    def producer(value_id: int, opcode: str) -> ScheduledOp:
        index = schedule.values[value_id].producer
        if index is None or by_index[index].opcode != opcode:
            raise ValidationError(f"prefix input expected {opcode} producer")
        return by_index[index]

    def exclusive_consumer(value_id: int, opcode: str) -> ScheduledOp:
        matches = consumers.get(value_id, [])
        if len(matches) != 1 or matches[0].opcode != opcode:
            raise ValidationError(f"prefix input expected exclusive {opcode} consumer")
        return matches[0]

    def literal(value_id: int, expected: tuple[object, ...]) -> bool:
        value = schedule.values[value_id]
        if (
            value.storage != ValueStorage.CONSTANT
            or value.constant_identity is None
            or not value.constant_identity.startswith("literal:")
        ):
            return False
        item = by_site.get(value.constant_identity[len("literal:") :])
        return item is not None and item.opcode == "constant" and item.attributes == (("value", expected),)

    commands: list[PrefixInputLoweredCommand] = []
    fused: set[int] = set()
    for concat in schedule.ops:
        item = by_site.get(concat.site)
        if item is None:
            continue
        try:
            PrefixInputProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        if len(concat.inputs) != 4 or len(concat.outputs) != 1:
            raise ValidationError("prefix input embedding concat arity is invalid")
        language_cast = producer(concat.inputs[3], "cast")
        language_mul = producer(language_cast.inputs[0], "mul")
        if (
            language_cast.invocation_id != concat.invocation_id
            or language_cast.attributes != (("dtype", "f32"),)
            or language_mul.invocation_id != concat.invocation_id
            or len(language_mul.inputs) != 2
        ):
            raise ValidationError("prefix input language cast/mul chain is invalid")
        parents = [schedule.values[value_id].producer for value_id in language_mul.inputs]
        parent_ops = [by_index[index] for index in parents if index is not None]
        gathers = [op for op in parent_ops if op.opcode == "gather"]
        scales = [op for op in parent_ops if op.opcode == "broadcast_in_dim"]
        if len(gathers) != 1 or len(scales) != 1:
            raise ValidationError("prefix input gather/scale chain is invalid")
        gather, scale = gathers[0], scales[0]
        if (
            gather.invocation_id != concat.invocation_id
            or scale.invocation_id != concat.invocation_id
            or len(gather.inputs) != 2
            or len(scale.inputs) != 1
            or scale.attributes
            != (("broadcast_dimensions", (0,)), ("shape", (1, 200, 2048)))
            or not literal(scale.inputs[0], (math.sqrt(2048.0),))
            or schedule.values[gather.inputs[0]].type != _EMBEDDING
            or schedule.values[gather.inputs[1]].type != _TOKENS
            or schedule.values[gather.outputs[0]].type != _LANGUAGE_BF16
            or consumers.get(gather.outputs[0], []) != [language_mul]
            or consumers.get(scale.outputs[0], []) != [language_mul]
            or consumers.get(language_mul.outputs[0], []) != [language_cast]
            or consumers.get(language_cast.outputs[0], []) != [concat]
        ):
            raise ValidationError("prefix input language dataflow is not exact")

        pad_candidates = [
            op
            for op in schedule.ops
            if op.invocation_id == concat.invocation_id
            and op.opcode == "concat"
            and len(op.inputs) == 4
            and len(op.outputs) == 1
            and op.attributes == (("axis", 1),)
            and schedule.values[op.outputs[0]].type == _PAD_MASK
        ]
        if len(pad_candidates) != 1:
            raise ValidationError("prefix input pad-mask concat is not unique")
        pad_concat = pad_candidates[0]
        image_mask_broadcasts = [producer(value_id, "broadcast_in_dim") for value_id in pad_concat.inputs[:3]]
        image_mask_ids: list[int] = []
        for broadcast in image_mask_broadcasts:
            if (
                broadcast.invocation_id != concat.invocation_id
                or broadcast.attributes
                != (("broadcast_dimensions", (0,)), ("shape", (1, 256)))
                or len(broadcast.inputs) != 1
                or schedule.values[broadcast.inputs[0]].type != _MASK_SCALAR
                or consumers.get(broadcast.outputs[0], []) != [pad_concat]
            ):
                raise ValidationError("prefix input image-mask broadcast is invalid")
            image_mask_ids.append(broadcast.inputs[0])
        token_mask_id = pad_concat.inputs[3]
        if schedule.values[token_mask_id].type != _TOKEN_MASK:
            raise ValidationError("prefix input token-mask contract is invalid")

        pad_casts = [
            op
            for op in consumers.get(pad_concat.outputs[0], [])
            if op.invocation_id == concat.invocation_id
            and op.opcode == "cast"
            and op.attributes == (("dtype", "i32"),)
        ]
        if len(pad_casts) != 1:
            raise ValidationError("prefix input pad-mask cast is not unique")
        pad_cast = pad_casts[0]
        cumulative = exclusive_consumer(pad_cast.outputs[0], "cumulative_sum")
        position_add = exclusive_consumer(cumulative.outputs[0], "add")
        if (
            cumulative.invocation_id != concat.invocation_id
            or cumulative.attributes != (("axis", 1),)
            or position_add.invocation_id != concat.invocation_id
            or len(position_add.inputs) != 2
        ):
            raise ValidationError("prefix input position chain is invalid")
        negative_id = next(
            (value_id for value_id in position_add.inputs if value_id != cumulative.outputs[0]),
            None,
        )
        if negative_id is None:
            raise ValidationError("prefix input position adjustment is missing")
        negative = producer(negative_id, "broadcast_in_dim")
        if (
            negative.invocation_id != concat.invocation_id
            or negative.attributes
            != (("broadcast_dimensions", (0,)), ("shape", (1, 968)))
            or not literal(negative.inputs[0], (-1,))
            or consumers.get(negative.outputs[0], []) != [position_add]
            or schedule.values[position_add.outputs[0]].type != _PAD_I32
        ):
            raise ValidationError("prefix input negative-one adjustment is invalid")
        position_consumers = consumers.get(position_add.outputs[0], [])
        if not position_consumers or any(
            op.invocation_id != concat.invocation_id or op.opcode != "rope_default"
            for op in position_consumers
        ):
            raise ValidationError("prefix input positions have unsupported consumers")

        region = (
            gather,
            scale,
            language_mul,
            language_cast,
            concat,
            *image_mask_broadcasts,
            pad_concat,
            pad_cast,
            cumulative,
            negative,
            position_add,
        )
        if len({op.execution_index for op in region}) != 13 or any(
            op.invocation_id != concat.invocation_id for op in region
        ):
            raise ValidationError("prefix input region is not one exact invocation")
        indices = tuple(sorted(op.execution_index for op in region))
        public_outputs = {
            concat.outputs[0],
            pad_concat.outputs[0],
            position_add.outputs[0],
        }
        intermediates = {
            value_id for op in region for value_id in op.outputs
        } - public_outputs
        if intermediates & entry_outputs or fused.intersection(indices):
            raise ValidationError("prefix input region is public or overlaps")
        capability = receipt.capability_for(item)
        operands = tuple(
            CommandOperand(value_id, OperandAccess.READ)
            for value_id in (
                *concat.inputs[:3],
                gather.inputs[0],
                gather.inputs[1],
                *image_mask_ids,
                token_mask_id,
            )
        ) + tuple(
            CommandOperand(value_id, OperandAccess.WRITE)
            for value_id in (
                concat.outputs[0],
                pad_concat.outputs[0],
                position_add.outputs[0],
            )
        )
        commands.append(
            PrefixInputLoweredCommand(
                concat.execution_index,
                concat.site,
                indices,
                ProviderCommand(
                    CommandTag.CUDA_KERNEL,
                    capability.provider_id,
                    capability.abi_major,
                    capability.abi_minor,
                    capability.digest,
                    operands,
                    receipt.payload().to_bytes(),
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        fused.update(indices)
    if not commands:
        raise ValidationError("prefix input found no complete region")
    elided = set(memory_plan.elided_ops)
    if fused & elided:
        raise ValidationError("prefix input overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in fused and op.execution_index not in elided
    )
    memory_sha = hashlib.sha256((_canonical(memory_plan.to_dict()) + "\n").encode()).hexdigest()
    return PrefixInputPartialLowering(
        inventory_sha,
        schedule_sha,
        memory_sha,
        capabilities,
        tuple(commands),
        tuple(sorted(fused)),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_prefix_input_partial_lowering(lowering: PrefixInputPartialLowering) -> str:
    return _canonical(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"prefix input {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or not any(character != "0" for character in value)
    ):
        raise ValidationError(f"prefix input {label} must be a SHA-256 digest")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
