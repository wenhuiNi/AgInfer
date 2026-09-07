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


KV_PACK_PAYLOAD_MAGIC = b"AIKVP1\0\0"
KV_PACK_SCHEMA_MAJOR = 1
KV_PACK_SCHEMA_MINOR = 0
KV_PACK_PAYLOAD = struct.Struct("<8sHH" + "I" * 20 + "Q" * 6 + "32s20s")
KV_PACK_PROVIDER_ID = 2
KV_PACK_PROVIDER_ABI_MAJOR = 1
KV_PACK_PROVIDER_ABI_MINOR = 0
KV_PACK_PROVIDER_VERSION = "aginfer-aot-cuda=1"
KV_PACK_IMPLEMENTATION = "aginfer.kv_pack.bf16.h1.s968_s50.d256.v1"
KV_PACK_RECEIPT_SCHEMA = "aginfer.kv-pack-validation.v1"
KV_PACK_PARTIAL_LOWERING_SCHEMA = "aginfer.kv-pack-partial-lowering.v1"
KV_PACK_VARIANT = 1

assert KV_PACK_PAYLOAD.size == 192


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


_PREFIX = _sig("bf16", (1, 1, 968, 256))
_CURRENT_BHSD = _sig("bf16", (1, 1, 50, 256))
_CURRENT_BSHD = _sig("bf16", (1, 50, 1, 256))
_PACKED = _sig("bf16", (1, 1, 1018, 256))
_QUERY = _sig("bf16", (1, 8, 50, 256))
_MASK = _sig("bool", (1, 8, 50, 1018))


@dataclass(frozen=True, slots=True)
class KvPackProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("KV pack exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "KvPackProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.MEMORY
            or item.opcode != "concat"
            or item.input_types != (_PREFIX, _CURRENT_BHSD)
            or item.output_types != (_PACKED,)
            or item.attributes != (("axis", 2),)
        ):
            raise ValidationError("KV pack anchor is outside the delivered exact envelope")
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "prefix_kv": [1, 1, 968, 256],
            "current_k_bhsd": [1, 1, 50, 256],
            "current_v_bshd": [1, 50, 1, 256],
            "packed_kv": [1, 1, 1018, 256],
            "dtype": "bf16",
        }


@dataclass(frozen=True, slots=True)
class KvPackPayload:
    problem: KvPackProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, KvPackProblem):
            raise ValidationError("KV pack payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        prefix_bytes = math.prod(_PREFIX.shape) * 2
        current_bytes = math.prod(_CURRENT_BHSD.shape) * 2
        packed_bytes = math.prod(_PACKED.shape) * 2
        return KV_PACK_PAYLOAD.pack(
            KV_PACK_PAYLOAD_MAGIC,
            KV_PACK_SCHEMA_MAJOR,
            KV_PACK_SCHEMA_MINOR,
            int(self.problem.target_arch),
            KV_PACK_VARIANT,
            2,  # BF16
            1,  # prefix BHSD
            1,  # current K BHSD
            2,  # current V BSHD
            1,  # output BHSD
            1,
            1,
            968,
            50,
            1018,
            256,
            255,
            1,
            1,
            256,
            1,
            1,
            16,
            prefix_bytes,
            current_bytes,
            current_bytes,
            packed_bytes,
            packed_bytes,
            self.module_bytes,
            bytes.fromhex(self.module_sha256),
            bytes(20),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "KvPackPayload":
        view = memoryview(data)
        if len(view) != KV_PACK_PAYLOAD.size:
            raise FormatError("KV pack payload must have its exact fixed size")
        fields = KV_PACK_PAYLOAD.unpack(view)
        if fields[0] != KV_PACK_PAYLOAD_MAGIC:
            raise FormatError("KV pack payload magic is invalid")
        if fields[1] != KV_PACK_SCHEMA_MAJOR or fields[2] > KV_PACK_SCHEMA_MINOR:
            raise FormatError("KV pack payload schema is unsupported")
        try:
            parsed = cls(
                KvPackProblem(CudaArch(fields[3])), fields[28], fields[29].hex()
            )
        except (ValueError, ValidationError) as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError("KV pack payload is outside the delivered exact variant")
        return parsed


@dataclass(frozen=True, slots=True)
class KvPackValidationReceipt:
    problem: KvPackProblem
    implementation_sha256: str
    implementation_bytes: int
    cuda_compiler_version: int
    cuda_runtime_version: int
    cuda_driver_version: int
    normal_launches: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    capture_matches_normal: bool
    full_outputs_bit_exact: bool
    racecheck_hazards: int
    latency_ms: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, KvPackProblem):
            raise ValidationError("KV pack receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("KV pack receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_outputs_bit_exact, "bit-exact full outputs"),
        ):
            if value is not True:
                raise ValidationError(f"KV pack receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("KV pack receipt requires zero racecheck hazards")
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0:
            raise ValidationError("KV pack receipt latency must be finite and positive")
        if self.contains_ptx is not False:
            raise ValidationError("KV pack implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = KvPackProblem.from_inventory(item, target_arch=self.problem.target_arch)
        if problem != self.problem:
            raise ValidationError("KV pack receipt does not cover this problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=KV_PACK_PROVIDER_ID,
            abi_major=KV_PACK_PROVIDER_ABI_MAJOR,
            abi_minor=KV_PACK_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{KV_PACK_PROVIDER_VERSION};cuda-runtime={self.cuda_runtime_version};"
                f"driver={self.cuda_driver_version}"
            ),
            implementation_id=KV_PACK_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> KvPackPayload:
        return KvPackPayload(
            self.problem, self.implementation_bytes, self.implementation_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": KV_PACK_RECEIPT_SCHEMA,
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
            "full_outputs_bit_exact": self.full_outputs_bit_exact,
            "racecheck_hazards": self.racecheck_hazards,
            "latency_ms": self.latency_ms,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class KvPackLoweredCommand:
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
class KvPackPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[KvPackLoweredCommand, ...]
    fused_execution_indices: tuple[int, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": KV_PACK_PARTIAL_LOWERING_SCHEMA,
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


def make_kv_pack_capabilities(
    inventory: LoweringInventory,
    receipt: KvPackValidationReceipt,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if not (isinstance(receipt, KvPackValidationReceipt) or is_build_binding(receipt, KvPackProblem)):
        raise ValidationError("KV pack receipt is invalid")
    if receipt.problem.target_arch != target_arch:
        raise ValidationError("KV pack receipt target differs from capability target")
    capabilities: dict[str, ProviderCapability] = {}
    for item in inventory.ops:
        if item.opcode != "concat":
            continue
        try:
            KvPackProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
    if not capabilities:
        raise ValidationError("KV pack receipt matched no inventory anchor")
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_kv_pack_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: KvPackValidationReceipt,
    *,
    target_arch: CudaArch,
) -> KvPackPartialLowering:
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode()
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("KV pack inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("KV pack memory plan does not match its schedule")
    if {item.value_id for item in memory_plan.allocations} != set(
        range(len(schedule.values))
    ):
        raise ValidationError("KV pack memory plan has incomplete value coverage")

    capabilities = make_kv_pack_capabilities(
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
        if producer_index is None or ops_by_index[producer_index].opcode != opcode:
            raise ValidationError(f"KV pack expected {opcode} producer")
        return ops_by_index[producer_index]

    def split_concat(concat: ScheduledOp) -> tuple[int, int, ProviderCapability]:
        item = inventory_by_site.get(concat.site)
        if item is None:
            raise ValidationError("KV pack concat has no inventory record")
        KvPackProblem.from_inventory(item, target_arch=target_arch)
        if len(concat.inputs) != 2 or len(concat.outputs) != 1:
            raise ValidationError("KV pack concat arity is invalid")
        prefix_id, current_id = concat.inputs
        if (
            schedule.values[prefix_id].type != _PREFIX
            or schedule.values[current_id].type != _CURRENT_BHSD
            or schedule.values[concat.outputs[0]].type != _PACKED
        ):
            raise ValidationError("KV pack concat scheduled types are invalid")
        return prefix_id, current_id, receipt.capability_for(item)

    commands: list[KvPackLoweredCommand] = []
    fused: set[int] = set()
    elided = set(memory_plan.elided_ops)
    for attention in schedule.ops:
        item = inventory_by_site.get(attention.site)
        if (
            item is None
            or item.opcode != "scaled_dot_product_attention"
            or item.input_types != (_QUERY, _PACKED, _PACKED, _MASK)
            or item.output_types != (_QUERY,)
            or item.attributes
            != (
                ("kv_group_size", 8),
                ("mask_fill", "dtype_min"),
                ("scale", 0.0625),
            )
        ):
            continue
        if len(attention.inputs) != 4:
            raise ValidationError("KV pack attention arity is invalid")
        key_concat = producer(attention.inputs[1], "concat")
        value_concat = producer(attention.inputs[2], "concat")
        prefix_k, current_k, capability = split_concat(key_concat)
        prefix_v, current_v_bhsd, value_capability = split_concat(value_concat)
        if capability.digest != value_capability.digest:
            raise ValidationError("KV pack K/V concat capabilities differ")
        if (
            key_concat.invocation_id != attention.invocation_id
            or value_concat.invocation_id != attention.invocation_id
            or consumers.get(key_concat.outputs[0], []) != [attention]
            or consumers.get(value_concat.outputs[0], []) != [attention]
        ):
            raise ValidationError("KV pack concat outputs must exclusively feed attention")

        for prefix_id in (prefix_k, prefix_v):
            value = schedule.values[prefix_id]
            state_read = producer(prefix_id, "state_read")
            if (
                value.storage != ValueStorage.STATE_ALIAS
                or value.alias_of is None
                or state_read.invocation_id != attention.invocation_id
                or len(state_read.inputs) != 0
                or len(state_read.outputs) != 1
                or state_read.outputs[0] != prefix_id
                or len(consumers.get(prefix_id, [])) != 1
            ):
                raise ValidationError("KV pack prefix must be an exclusive state-read alias")

        key_producer = producer(current_k, "rope_default")
        if (
            key_producer.invocation_id != attention.invocation_id
            or consumers.get(current_k, []) != [key_concat]
        ):
            raise ValidationError("KV pack current K must be one exclusive RoPE output")

        value_transpose = producer(current_v_bhsd, "transpose")
        if (
            value_transpose.invocation_id != attention.invocation_id
            or value_transpose.attributes != (("permutation", (0, 2, 1, 3)),)
            or len(value_transpose.inputs) != 1
            or len(value_transpose.outputs) != 1
            or schedule.values[value_transpose.inputs[0]].type != _CURRENT_BSHD
            or consumers.get(current_v_bhsd, []) != [value_concat]
        ):
            raise ValidationError("KV pack current V requires an exclusive singleton transpose")
        if any(
            value_id in entry_outputs
            for value_id in (
                current_v_bhsd,
                key_concat.outputs[0],
                value_concat.outputs[0],
            )
        ):
            raise ValidationError("KV pack fused intermediate is a public output")

        fused_indices = (
            (key_concat.execution_index, value_concat.execution_index)
            if value_transpose.execution_index in elided
            else (
                value_transpose.execution_index,
                key_concat.execution_index,
                value_concat.execution_index,
            )
        )
        if fused.intersection(fused_indices):
            raise ValidationError("KV pack fused regions overlap")
        commands.append(
            KvPackLoweredCommand(
                key_concat.execution_index,
                key_concat.site,
                tuple(sorted(fused_indices)),
                ProviderCommand(
                    tag=CommandTag.CUDA_KERNEL,
                    provider_id=capability.provider_id,
                    abi_major=capability.abi_major,
                    abi_minor=capability.abi_minor,
                    capability_digest=capability.digest,
                    operands=(
                        CommandOperand(prefix_k, OperandAccess.READ),
                        CommandOperand(prefix_v, OperandAccess.READ),
                        CommandOperand(current_k, OperandAccess.READ),
                        CommandOperand(value_transpose.inputs[0], OperandAccess.READ),
                        CommandOperand(key_concat.outputs[0], OperandAccess.WRITE),
                        CommandOperand(value_concat.outputs[0], OperandAccess.WRITE),
                    ),
                    payload=receipt.payload().to_bytes(),
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        fused.update(fused_indices)

    if not commands:
        raise ValidationError("KV pack found no complete attention-adjacent region")
    if fused & elided:
        raise ValidationError("KV pack fusion overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in fused and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return KvPackPartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(fused)),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_kv_pack_partial_lowering(lowering: KvPackPartialLowering) -> str:
    if not isinstance(lowering, KvPackPartialLowering):
        raise ValidationError("KV pack dump requires a KvPackPartialLowering")
    return _canonical_json(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"KV pack {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or not any(char != "0" for char in value)
    ):
        raise ValidationError(f"KV pack {label} must be a SHA-256 digest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
