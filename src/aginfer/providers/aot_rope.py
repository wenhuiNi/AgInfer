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
from ..lowering.schedule import ExecutionSchedule, ScheduledOp, dump_execution_schedule
from ..schema import CudaArch


ROPE_PAYLOAD_MAGIC = b"AIROP1\0\0"
ROPE_SCHEMA_MAJOR = 1
ROPE_SCHEMA_MINOR = 0
ROPE_PAYLOAD = struct.Struct("<8sHH" + "I" * 24 + "f" + "Q" * 4 + "32s16s")
ROPE_PROVIDER_ID = 2
ROPE_PROVIDER_ABI_MAJOR = 1
ROPE_PROVIDER_ABI_MINOR = 0
ROPE_PROVIDER_VERSION = "aginfer-aot-cuda=1"
ROPE_RECEIPT_SCHEMA = "aginfer.rope-validation.v1"
ROPE_PARTIAL_LOWERING_SCHEMA = "aginfer.rope-partial-lowering.v1"
ROPE_HEAD_DIM = 256
ROPE_HALF_DIM = 128

assert ROPE_PAYLOAD.size == 192


class RopeVariant(IntEnum):
    BF16_SEQUENCE968_HEADS8 = 1
    BF16_SEQUENCE968_HEADS1 = 2
    BF16_SEQUENCE50_HEADS8 = 3
    BF16_SEQUENCE50_HEADS1 = 4


_VARIANT_SHAPES: dict[RopeVariant, tuple[int, int]] = {
    RopeVariant.BF16_SEQUENCE968_HEADS8: (968, 8),
    RopeVariant.BF16_SEQUENCE968_HEADS1: (968, 1),
    RopeVariant.BF16_SEQUENCE50_HEADS8: (50, 8),
    RopeVariant.BF16_SEQUENCE50_HEADS1: (50, 1),
}


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


@dataclass(frozen=True, slots=True)
class RopeProblem:
    target_arch: CudaArch
    variant: RopeVariant

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("AOT RoPE exact variants require SM120")
        if not isinstance(self.variant, RopeVariant):
            raise ValidationError("AOT RoPE variant is invalid")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "RopeProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.AOT_CUDA
            or item.opcode != "rope_default"
            or item.attributes
            != (
                ("frequency_dtype", "bf16"),
                ("pairing", "split_half"),
                ("theta", 10000.0),
            )
            or len(item.input_types) != 2
            or len(item.output_types) != 1
        ):
            raise ValidationError("AOT RoPE site is outside the delivered exact envelopes")
        for variant, (sequence, heads) in _VARIANT_SHAPES.items():
            semantic = _sig("bf16", (1, heads, sequence, ROPE_HEAD_DIM))
            if (
                item.input_types == (semantic, _sig("i32", (1, sequence)))
                and item.output_types == (semantic,)
            ):
                return cls(target_arch, variant)
        raise ValidationError("AOT RoPE site is outside the delivered exact envelopes")

    @property
    def sequence(self) -> int:
        return _VARIANT_SHAPES[self.variant][0]

    @property
    def heads(self) -> int:
        return _VARIANT_SHAPES[self.variant][1]

    @property
    def source_shape(self) -> tuple[int, ...]:
        return (1, self.sequence, self.heads, ROPE_HEAD_DIM)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, self.heads, self.sequence, ROPE_HEAD_DIM)

    @property
    def tensor_bytes(self) -> int:
        return math.prod(self.output_shape) * 2

    @property
    def position_bytes(self) -> int:
        return self.sequence * 4

    @property
    def grid_x(self) -> int:
        pairs = self.sequence * self.heads * ROPE_HALF_DIM
        return (pairs + 255) // 256

    @property
    def implementation_id(self) -> str:
        return f"aginfer.rope.bf16.s{self.sequence}.h{self.heads}.d256.split_half.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "variant": self.variant.name.lower(),
            "input_bshd": list(self.source_shape),
            "positions": [1, self.sequence],
            "output_bhsd": list(self.output_shape),
            "dtype": "bf16",
            "position_dtype": "i32",
            "head_dim": ROPE_HEAD_DIM,
            "frequency_dtype": "bf16",
            "pairing": "split_half",
            "theta": 10000.0,
        }


@dataclass(frozen=True, slots=True)
class RopePayload:
    problem: RopeProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, RopeProblem):
            raise ValidationError("AOT RoPE payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        return ROPE_PAYLOAD.pack(
            ROPE_PAYLOAD_MAGIC,
            ROPE_SCHEMA_MAJOR,
            ROPE_SCHEMA_MINOR,
            int(self.problem.target_arch),
            int(self.problem.variant),
            2,  # BF16 input
            3,  # I32 positions
            2,  # BF16 output
            2,  # BSHD input
            3,  # BHSD output
            1,
            self.problem.heads,
            self.problem.sequence,
            ROPE_HEAD_DIM,
            ROPE_HALF_DIM,
            self.problem.grid_x,
            1,
            1,
            256,
            1,
            1,
            0,
            16,
            16,
            16,
            2,  # BF16 inverse frequency
            1,  # split-half pairing
            10000.0,
            self.problem.tensor_bytes,
            self.problem.position_bytes,
            self.problem.tensor_bytes,
            self.module_bytes,
            bytes.fromhex(self.module_sha256),
            bytes(16),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "RopePayload":
        view = memoryview(data)
        if len(view) != ROPE_PAYLOAD.size:
            raise FormatError("AOT RoPE payload must have its exact fixed size")
        fields = ROPE_PAYLOAD.unpack(view)
        if fields[0] != ROPE_PAYLOAD_MAGIC:
            raise FormatError("AOT RoPE payload magic is invalid")
        if fields[1] != ROPE_SCHEMA_MAJOR or fields[2] > ROPE_SCHEMA_MINOR:
            raise FormatError("AOT RoPE payload schema is unsupported")
        try:
            target_arch = CudaArch(fields[3])
            variant = RopeVariant(fields[4])
            parsed = cls(RopeProblem(target_arch, variant), fields[31], fields[32].hex())
        except (ValueError, ValidationError) as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError("AOT RoPE payload is outside the delivered exact variants")
        return parsed


@dataclass(frozen=True, slots=True)
class RopeValidationReceipt:
    problem: RopeProblem
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
    split_half_negative_verified: bool
    racecheck_hazards: int
    cosine: float
    max_abs: float
    p99_abs: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, RopeProblem):
            raise ValidationError("AOT RoPE receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("AOT RoPE receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_output_compared, "full-output comparison"),
            (self.split_half_negative_verified, "split-half negative control"),
        ):
            if value is not True:
                raise ValidationError(f"AOT RoPE receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("AOT RoPE receipt requires zero racecheck hazards")
        if (
            not math.isfinite(self.cosine)
            or self.cosine < 0.99999
            or not math.isfinite(self.max_abs)
            or self.max_abs > 0.03125
            or not math.isfinite(self.p99_abs)
            or self.p99_abs > 0.015625
        ):
            raise ValidationError("AOT RoPE correctness gate failed")
        if self.contains_ptx is not False:
            raise ValidationError("AOT RoPE implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = RopeProblem.from_inventory(item, target_arch=self.problem.target_arch)
        if problem != self.problem:
            raise ValidationError("AOT RoPE receipt does not cover this problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=ROPE_PROVIDER_ID,
            abi_major=ROPE_PROVIDER_ABI_MAJOR,
            abi_minor=ROPE_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{ROPE_PROVIDER_VERSION};"
                f"cuda-runtime={self.cuda_runtime_version};driver={self.cuda_driver_version}"
            ),
            implementation_id=self.problem.implementation_id,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> RopePayload:
        return RopePayload(self.problem, self.implementation_bytes, self.implementation_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ROPE_RECEIPT_SCHEMA,
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
            "split_half_negative_verified": self.split_half_negative_verified,
            "racecheck_hazards": self.racecheck_hazards,
            "cosine": self.cosine,
            "max_abs": self.max_abs,
            "p99_abs": self.p99_abs,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class RopeLoweredCommand:
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
class RopePartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[RopeLoweredCommand, ...]
    fused_execution_indices: tuple[int, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ROPE_PARTIAL_LOWERING_SCHEMA,
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


def _receipt_map(
    receipts: tuple[RopeValidationReceipt | BuildBinding, ...], *, target_arch: CudaArch
) -> dict[RopeProblem, RopeValidationReceipt]:
    if not isinstance(receipts, tuple) or not receipts:
        raise ValidationError("AOT RoPE receipts must be a non-empty tuple")
    result: dict[RopeProblem, RopeValidationReceipt] = {}
    for receipt in receipts:
        if not (isinstance(receipt, RopeValidationReceipt) or is_build_binding(receipt, RopeProblem)):
            raise ValidationError("AOT RoPE receipt is invalid")
        if receipt.problem.target_arch != target_arch:
            raise ValidationError("AOT RoPE receipt target differs from capability target")
        if receipt.problem in result:
            raise ValidationError("AOT RoPE receipts contain a duplicate problem")
        result[receipt.problem] = receipt
    return result


def make_rope_capabilities(
    inventory: LoweringInventory,
    receipts: tuple[RopeValidationReceipt | BuildBinding, ...],
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    by_problem = _receipt_map(receipts, target_arch=target_arch)
    capabilities: dict[str, ProviderCapability] = {}
    matched: set[RopeProblem] = set()
    for item in inventory.ops:
        if item.opcode != "rope_default":
            continue
        try:
            problem = RopeProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        receipt = by_problem.get(problem)
        if receipt is None:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
        matched.add(problem)
    if set(by_problem) - matched:
        raise ValidationError("AOT RoPE receipt matched no inventory site for one problem")
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_rope_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipts: tuple[RopeValidationReceipt | BuildBinding, ...],
    *,
    target_arch: CudaArch,
) -> RopePartialLowering:
    inventory_sha256 = hashlib.sha256(dump_lowering_inventory(inventory).encode()).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("AOT RoPE inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("AOT RoPE memory plan does not match its schedule")
    if {item.value_id for item in memory_plan.allocations} != set(range(len(schedule.values))):
        raise ValidationError("AOT RoPE memory plan has incomplete value coverage")

    by_problem = _receipt_map(receipts, target_arch=target_arch)
    capabilities = make_rope_capabilities(inventory, receipts, target_arch=target_arch)
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    ops_by_index = {item.execution_index: item for item in schedule.ops}
    consumers: dict[int, list[ScheduledOp]] = {}
    entry_output_ids = {value_id for _, value_id in schedule.entry_outputs}
    for op in schedule.ops:
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(op)

    commands: list[RopeLoweredCommand] = []
    fused: set[int] = set()
    elided = set(memory_plan.elided_ops)
    for op in schedule.ops:
        item = inventory_by_site.get(op.site)
        if item is None or item.opcode != "rope_default":
            continue
        try:
            problem = RopeProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        receipt = by_problem.get(problem)
        if receipt is None:
            continue
        if len(op.inputs) != 2 or len(op.outputs) != 1:
            raise ValidationError("AOT RoPE scheduled arity is invalid")
        source_value = schedule.values[op.inputs[0]]
        if source_value.producer is None:
            raise ValidationError("AOT RoPE input has no BSHD-to-BHSD transpose producer")
        transpose = ops_by_index[source_value.producer]
        if (
            transpose.opcode != "transpose"
            or transpose.invocation_id != op.invocation_id
            or transpose.attributes != (("permutation", (0, 2, 1, 3)),)
            or len(transpose.inputs) != 1
            or len(transpose.outputs) != 1
            or transpose.outputs[0] != op.inputs[0]
            or schedule.values[transpose.inputs[0]].type != _sig("bf16", problem.source_shape)
            or source_value.type != _sig("bf16", problem.output_shape)
            or op.inputs[0] in entry_output_ids
            or consumers.get(op.inputs[0], []) != [op]
        ):
            raise ValidationError("AOT RoPE requires an exclusive exact BSHD input transpose")
        fused_indices = (
            (op.execution_index,)
            if transpose.execution_index in elided
            else (transpose.execution_index, op.execution_index)
        )
        if fused.intersection(fused_indices):
            raise ValidationError("AOT RoPE fused regions overlap")
        capability = receipt.capability_for(item)
        commands.append(
            RopeLoweredCommand(
                op.execution_index,
                op.site,
                fused_indices,
                ProviderCommand(
                    tag=CommandTag.CUDA_KERNEL,
                    provider_id=capability.provider_id,
                    abi_major=capability.abi_major,
                    abi_minor=capability.abi_minor,
                    capability_digest=capability.digest,
                    operands=(
                        CommandOperand(transpose.inputs[0], OperandAccess.READ),
                        CommandOperand(op.inputs[1], OperandAccess.READ),
                        CommandOperand(op.outputs[0], OperandAccess.WRITE),
                    ),
                    payload=receipt.payload().to_bytes(),
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        fused.update(fused_indices)

    if fused & elided:
        raise ValidationError("AOT RoPE fusion overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in fused and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return RopePartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(fused)),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_rope_partial_lowering(lowering: RopePartialLowering) -> str:
    if not isinstance(lowering, RopePartialLowering):
        raise ValidationError("AOT RoPE dump requires a RopePartialLowering")
    return _canonical_json(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"AOT RoPE {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or not any(char != "0" for char in value)
    ):
        raise ValidationError(f"AOT RoPE {label} must be a SHA-256 digest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
