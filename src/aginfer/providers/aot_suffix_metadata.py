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
from ..lowering.schedule import ExecutionSchedule, ScheduledOp, ValueStorage, dump_execution_schedule
from ..schema import CudaArch


SUFFIX_METADATA_PAYLOAD_MAGIC = b"AISMD1\0\0"
SUFFIX_METADATA_PAYLOAD = struct.Struct("<8sHH" + "I" * 20 + "Q" * 6 + "32s20s")
SUFFIX_METADATA_RECEIPT_SCHEMA = "aginfer.suffix-metadata-validation.v1"
SUFFIX_METADATA_PARTIAL_LOWERING_SCHEMA = "aginfer.suffix-metadata-partial-lowering.v1"
SUFFIX_METADATA_IMPLEMENTATION = "aginfer.suffix_metadata.bool.s968_s50.v1"

assert SUFFIX_METADATA_PAYLOAD.size == 192


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


_PAD = _sig("bool", (1, 968))
_PREFIX_MASK = _sig("bool", (1, 50, 968))
_SUFFIX_MASK = _sig("bool", (1, 50, 50))
_MASK = _sig("bool", (1, 50, 1018))
_I32_50 = _sig("i32", (1, 50))
_MASK4 = _sig("bool", (1, 8, 50, 1018))


@dataclass(frozen=True, slots=True)
class SuffixMetadataProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("suffix metadata exact variant requires SM120")

    @classmethod
    def from_inventory(cls, item: InventoryOp, *, target_arch: CudaArch) -> "SuffixMetadataProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.MEMORY
            or item.opcode != "concat"
            or item.input_types != (_PREFIX_MASK, _SUFFIX_MASK)
            or item.output_types != (_MASK,)
            or item.attributes != (("axis", 2),)
        ):
            raise ValidationError("suffix metadata anchor is outside the delivered exact envelope")
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {"target_arch": self.target_arch.name_string, "prefix": 968, "suffix": 50}


@dataclass(frozen=True, slots=True)
class SuffixMetadataPayload:
    problem: SuffixMetadataProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, SuffixMetadataProblem):
            raise ValidationError("suffix metadata payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        return SUFFIX_METADATA_PAYLOAD.pack(
            SUFFIX_METADATA_PAYLOAD_MAGIC, 1, 0,
            int(self.problem.target_arch), 1, 5, 5, 3, 1, 1, 1, 1,
            968, 50, 1018, 199, 256, 1, 1, 4, 0, 0, 0,
            968, 50 * 1018, 50 * 4, self.module_bytes, 0, 0,
            bytes.fromhex(self.module_sha256), bytes(20),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "SuffixMetadataPayload":
        view = memoryview(data)
        if len(view) != SUFFIX_METADATA_PAYLOAD.size:
            raise FormatError("suffix metadata payload must have its exact fixed size")
        fields = SUFFIX_METADATA_PAYLOAD.unpack(view)
        if fields[0] != SUFFIX_METADATA_PAYLOAD_MAGIC or fields[1] != 1 or fields[2] != 0:
            raise FormatError("suffix metadata payload header is invalid")
        try:
            parsed = cls(SuffixMetadataProblem(CudaArch(fields[3])), fields[26], fields[29].hex())
        except (ValueError, ValidationError) as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError("suffix metadata payload is outside the delivered exact variant")
        return parsed


@dataclass(frozen=True, slots=True)
class SuffixMetadataValidationReceipt:
    problem: SuffixMetadataProblem
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
    hole_mask_negative_detected: bool
    racecheck_hazards: int
    latency_ms: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, SuffixMetadataProblem):
            raise ValidationError("suffix metadata receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("suffix metadata receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_outputs_exact, "exact full outputs"),
            (self.hole_mask_negative_detected, "hole-mask negative control"),
        ):
            if value is not True:
                raise ValidationError(f"suffix metadata receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("suffix metadata receipt requires zero racecheck hazards")
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0:
            raise ValidationError("suffix metadata latency must be finite and positive")
        if self.contains_ptx is not False:
            raise ValidationError("suffix metadata implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        SuffixMetadataProblem.from_inventory(item, target_arch=self.problem.target_arch)
        return ProviderCapability.exact_for(
            item, provider_id=2, abi_major=1, abi_minor=0,
            provider_version=f"aginfer-aot-cuda=1;cuda-runtime={self.cuda_runtime_version};driver={self.cuda_driver_version}",
            implementation_id=SUFFIX_METADATA_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch, supports_capture=True, workspace_bytes=0,
        )

    def payload(self) -> SuffixMetadataPayload:
        return SuffixMetadataPayload(self.problem, self.implementation_bytes, self.implementation_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SUFFIX_METADATA_RECEIPT_SCHEMA,
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
            "hole_mask_negative_detected": self.hole_mask_negative_detected,
            "racecheck_hazards": self.racecheck_hazards,
            "latency_ms": self.latency_ms,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class SuffixMetadataLoweredCommand:
    execution_index: int
    site: str
    fused_execution_indices: tuple[int, ...]
    command: ProviderCommand

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index, "site": self.site,
            "fused_execution_indices": list(self.fused_execution_indices),
            "capability_digest": self.command.capability_digest,
            "operands": [{"value_id": x.value_id, "access": x.access.name.lower()} for x in self.command.operands],
            "payload_sha256": hashlib.sha256(self.command.payload).hexdigest(),
            "capture_safe": self.command.capture_safe,
        }


@dataclass(frozen=True, slots=True)
class SuffixMetadataPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[SuffixMetadataLoweredCommand, ...]
    fused_execution_indices: tuple[int, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SUFFIX_METADATA_PARTIAL_LOWERING_SCHEMA,
            "inventory_sha256": self.inventory_sha256,
            "schedule_sha256": self.schedule_sha256,
            "memory_plan_sha256": self.memory_plan_sha256,
            "complete": self.complete,
            "summary": {"capabilities": len(self.capabilities), "commands": len(self.commands),
                        "fused_executions": len(self.fused_execution_indices),
                        "elided_executions": len(self.elided_execution_indices),
                        "unhandled_executions": len(self.unhandled_execution_indices), "workspace_bytes": 0},
            "capability_registry_sha256": hashlib.sha256(dump_provider_capabilities(self.capabilities).encode()).hexdigest(),
            "commands": [x.to_dict() for x in self.commands],
            "fused_execution_indices": list(self.fused_execution_indices),
            "elided_execution_indices": list(self.elided_execution_indices),
            "unhandled_execution_indices": list(self.unhandled_execution_indices),
        }


def make_suffix_metadata_capabilities(inventory: LoweringInventory, receipt: SuffixMetadataValidationReceipt, *, target_arch: CudaArch) -> tuple[ProviderCapability, ...]:
    if not (isinstance(receipt, SuffixMetadataValidationReceipt) or is_build_binding(receipt, SuffixMetadataProblem)):
        raise ValidationError("suffix metadata receipt is invalid")
    if receipt.problem.target_arch != target_arch:
        raise ValidationError("suffix metadata receipt target differs from capability target")
    result: dict[str, ProviderCapability] = {}
    for item in inventory.ops:
        try:
            SuffixMetadataProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        result.setdefault(capability.digest, capability)
    if not result:
        raise ValidationError("suffix metadata receipt matched no inventory anchor")
    return tuple(sorted(result.values(), key=lambda x: x.digest))


def lower_suffix_metadata_commands(schedule: ExecutionSchedule, inventory: LoweringInventory, memory_plan: MemoryPlan, receipt: SuffixMetadataValidationReceipt, *, target_arch: CudaArch) -> SuffixMetadataPartialLowering:
    inventory_sha = hashlib.sha256(dump_lowering_inventory(inventory).encode()).hexdigest()
    schedule_sha = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha or memory_plan.schedule_sha256 != schedule_sha:
        raise ValidationError("suffix metadata stage identities do not match")
    if {item.value_id for item in memory_plan.allocations} != set(
        range(len(schedule.values))
    ):
        raise ValidationError("suffix metadata memory plan has incomplete value coverage")
    capabilities = make_suffix_metadata_capabilities(inventory, receipt, target_arch=target_arch)
    by_site = {x.site_id: x for x in inventory.ops}
    by_index = {x.execution_index: x for x in schedule.ops}
    consumers: dict[int, list[ScheduledOp]] = {}
    for op in schedule.ops:
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(op)
    entry_outputs = {value_id for _, value_id in schedule.entry_outputs}

    def producer(value_id: int, opcode: str) -> ScheduledOp:
        index = schedule.values[value_id].producer
        if index is None or by_index[index].opcode != opcode:
            raise ValidationError(f"suffix metadata expected {opcode} producer")
        return by_index[index]

    def exclusive_consumer(value_id: int, opcode: str) -> ScheduledOp:
        matches = consumers.get(value_id, [])
        if len(matches) != 1 or matches[0].opcode != opcode:
            raise ValidationError(f"suffix metadata expected exclusive {opcode} consumer")
        return matches[0]

    def literal(value_id: int, expected: tuple[object, ...]) -> bool:
        value = schedule.values[value_id]
        if value.storage != ValueStorage.CONSTANT or value.constant_identity is None or not value.constant_identity.startswith("literal:"):
            return False
        item = by_site.get(value.constant_identity[len("literal:"):])
        return item is not None and item.opcode == "constant" and item.attributes == (("value", expected),)

    commands: list[SuffixMetadataLoweredCommand] = []
    fused: set[int] = set()
    for concat in schedule.ops:
        item = by_site.get(concat.site)
        if item is None:
            continue
        try:
            SuffixMetadataProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        if len(concat.inputs) != 2 or len(concat.outputs) != 1:
            raise ValidationError("suffix metadata concat arity is invalid")
        prefix_broadcast = producer(concat.inputs[0], "broadcast_in_dim")
        suffix_broadcast = producer(concat.inputs[1], "broadcast_in_dim")
        if (
            prefix_broadcast.invocation_id != concat.invocation_id
            or suffix_broadcast.invocation_id != concat.invocation_id
            or len(prefix_broadcast.inputs) != 1
            or len(suffix_broadcast.inputs) != 1
            or prefix_broadcast.attributes
            != (
                ("broadcast_dimensions", (0, 2)),
                ("shape", (1, 50, 968)),
            )
            or suffix_broadcast.attributes
            != (("broadcast_dimensions", (0,)), ("shape", (1, 50, 50)))
            or not literal(suffix_broadcast.inputs[0], (True,))
        ):
            raise ValidationError("suffix metadata mask broadcasts are not exact")
        pad_id = prefix_broadcast.inputs[0]
        if schedule.values[pad_id].type != _PAD:
            raise ValidationError("suffix metadata prefix pad contract is invalid")
        head_broadcast = exclusive_consumer(concat.outputs[0], "broadcast_in_dim")
        if (
            consumers.get(prefix_broadcast.outputs[0], []) != [concat]
            or consumers.get(suffix_broadcast.outputs[0], []) != [concat]
            or head_broadcast.invocation_id != concat.invocation_id
            or len(head_broadcast.inputs) != 1
            or len(head_broadcast.outputs) != 1
            or head_broadcast.attributes
            != (
                ("broadcast_dimensions", (0, 2, 3)),
                ("shape", (1, 8, 50, 1018)),
            )
            or schedule.values[head_broadcast.outputs[0]].type != _MASK4
        ):
            raise ValidationError("suffix metadata mask consumers are not exact")
        attention_consumers = consumers.get(head_broadcast.outputs[0], [])
        if not attention_consumers or any(
            op.invocation_id != concat.invocation_id
            or op.opcode != "scaled_dot_product_attention"
            or len(op.inputs) != 4
            or op.inputs[3] != head_broadcast.outputs[0]
            for op in attention_consumers
        ):
            raise ValidationError("suffix metadata mask has unsupported consumers")

        pad_casts = [
            op
            for op in consumers.get(pad_id, [])
            if op.invocation_id == concat.invocation_id and op.opcode == "cast"
        ]
        if len(pad_casts) != 1 or pad_casts[0].attributes != (("dtype", "i32"),):
            raise ValidationError("suffix metadata prefix pad cast is missing")
        pad_cast = pad_casts[0]
        reduce = exclusive_consumer(pad_cast.outputs[0], "reduce_sum")
        if (
            reduce.invocation_id != concat.invocation_id
            or reduce.attributes != (("axes", (1,)), ("keepdims", False))
        ):
            raise ValidationError("suffix metadata prefix reduction is invalid")
        offset_broadcast = exclusive_consumer(
            reduce.outputs[0], "broadcast_in_dim"
        )
        if (
            offset_broadcast.invocation_id != concat.invocation_id
            or offset_broadcast.attributes
            != (("broadcast_dimensions", (0,)), ("shape", (1, 50)))
        ):
            raise ValidationError("suffix metadata offset broadcast is invalid")

        suffix_1d = [
            op
            for op in schedule.ops
            if op.invocation_id == concat.invocation_id
            and op.opcode == "broadcast_in_dim"
            and op.attributes
            == (("broadcast_dimensions", (0,)), ("shape", (1, 50)))
            and op.inputs == suffix_broadcast.inputs
        ]
        if len(suffix_1d) != 1:
            raise ValidationError("suffix metadata one-dimensional suffix mask is invalid")
        suffix_cast = exclusive_consumer(suffix_1d[0].outputs[0], "cast")
        if (
            suffix_cast.invocation_id != concat.invocation_id
            or suffix_cast.attributes != (("dtype", "i32"),)
        ):
            raise ValidationError("suffix metadata suffix cast is invalid")
        cumulative = exclusive_consumer(suffix_cast.outputs[0], "cumulative_sum")
        if (
            cumulative.invocation_id != concat.invocation_id
            or cumulative.attributes != (("axis", 1),)
        ):
            raise ValidationError("suffix metadata cumulative sum is invalid")
        first_add = exclusive_consumer(offset_broadcast.outputs[0], "add")
        if (
            first_add.invocation_id != concat.invocation_id
            or len(first_add.inputs) != 2
            or first_add.inputs.count(offset_broadcast.outputs[0]) != 1
            or first_add.inputs.count(cumulative.outputs[0]) != 1
            or consumers.get(cumulative.outputs[0], []) != [first_add]
        ):
            raise ValidationError("suffix metadata offset add is invalid")
        final_add = exclusive_consumer(first_add.outputs[0], "add")
        if (
            final_add.invocation_id != concat.invocation_id
            or len(final_add.inputs) != 2
            or final_add.inputs.count(first_add.outputs[0]) != 1
        ):
            raise ValidationError("suffix metadata final add is invalid")
        negative_id = final_add.inputs[0] if final_add.inputs[1] == first_add.outputs[0] else final_add.inputs[1]
        negative_broadcast = producer(negative_id, "broadcast_in_dim")
        if (
            negative_broadcast.invocation_id != concat.invocation_id
            or negative_broadcast.attributes
            != (("broadcast_dimensions", (0,)), ("shape", (1, 50)))
            or not literal(negative_broadcast.inputs[0], (-1,))
            or consumers.get(negative_broadcast.outputs[0], []) != [final_add]
            or schedule.values[final_add.outputs[0]].type != _I32_50
        ):
            raise ValidationError("suffix metadata negative-one adjustment is invalid")
        position_consumers = consumers.get(final_add.outputs[0], [])
        if not position_consumers or any(
            op.opcode != "rope_default" or op.invocation_id != concat.invocation_id
            for op in position_consumers
        ):
            raise ValidationError("suffix metadata positions have unsupported consumers")
        region = (
            prefix_broadcast,
            suffix_broadcast,
            concat,
            pad_cast,
            reduce,
            offset_broadcast,
            suffix_1d[0],
            suffix_cast,
            cumulative,
            first_add,
            negative_broadcast,
            final_add,
        )
        indices = tuple(sorted(x.execution_index for x in region))
        intermediates = {value_id for op in region for value_id in op.outputs} - {concat.outputs[0], final_add.outputs[0]}
        if intermediates & entry_outputs or fused.intersection(indices):
            raise ValidationError("suffix metadata region is public or overlaps")
        capability = receipt.capability_for(item)
        commands.append(SuffixMetadataLoweredCommand(
            concat.execution_index, concat.site, indices,
            ProviderCommand(CommandTag.CUDA_KERNEL, capability.provider_id, capability.abi_major,
                            capability.abi_minor, capability.digest,
                            (CommandOperand(pad_id, OperandAccess.READ),
                             CommandOperand(concat.outputs[0], OperandAccess.WRITE),
                             CommandOperand(final_add.outputs[0], OperandAccess.WRITE)),
                            receipt.payload().to_bytes(), capture_safe=capability.supports_capture)))
        fused.update(indices)
    if not commands:
        raise ValidationError("suffix metadata found no complete region")
    elided = set(memory_plan.elided_ops)
    if fused & elided:
        raise ValidationError("suffix metadata overlaps a memory-elided op")
    unhandled = tuple(x.execution_index for x in schedule.ops if x.execution_index not in fused and x.execution_index not in elided)
    memory_sha = hashlib.sha256((_canonical(memory_plan.to_dict()) + "\n").encode()).hexdigest()
    return SuffixMetadataPartialLowering(inventory_sha, schedule_sha, memory_sha, capabilities,
                                         tuple(commands), tuple(sorted(fused)), tuple(sorted(elided)), unhandled)


def dump_suffix_metadata_partial_lowering(lowering: SuffixMetadataPartialLowering) -> str:
    return _canonical(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"suffix metadata {label} must be positive")


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) or not any(c != "0" for c in value):
        raise ValidationError(f"suffix metadata {label} must be a SHA-256 digest")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
