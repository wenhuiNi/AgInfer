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
from ..lowering.schedule import (
    ExecutionSchedule,
    ScheduledOp,
    ValueStorage,
    dump_execution_schedule,
)
from ..schema import CudaArch


PREFIX_KV_STORE_PAYLOAD_MAGIC = b"AIKVS1\0\0"
PREFIX_KV_STORE_SCHEMA_MAJOR = 1
PREFIX_KV_STORE_SCHEMA_MINOR = 0
PREFIX_KV_STORE_PAYLOAD = struct.Struct("<8sHH" + "I" * 20 + "Q" * 6 + "32s20s")
PREFIX_KV_STORE_PROVIDER_ID = 2
PREFIX_KV_STORE_PROVIDER_ABI_MAJOR = 1
PREFIX_KV_STORE_PROVIDER_ABI_MINOR = 0
PREFIX_KV_STORE_PROVIDER_VERSION = "aginfer-aot-cuda=1"
PREFIX_KV_STORE_IMPLEMENTATION = "aginfer.prefix_kv_store.bf16.h1.s968.d256.v1"
PREFIX_KV_STORE_RECEIPT_SCHEMA = "aginfer.prefix-kv-store-validation.v1"
PREFIX_KV_STORE_PARTIAL_LOWERING_SCHEMA = "aginfer.prefix-kv-store-partial-lowering.v1"
PREFIX_KV_STORE_VARIANT = 1

assert PREFIX_KV_STORE_PAYLOAD.size == 192


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


_VALUE_BSHD = _sig("bf16", (1, 968, 1, 256))
_STATE_BHSD = _sig("bf16", (1, 1, 968, 256))
_QUERY = _sig("bf16", (1, 8, 968, 256))
_MASK = _sig("bool", (1, 8, 968, 968))
_POSITIONS = _sig("i32", (1, 968))
_ATTENTION_ATTRIBUTES = (
    ("kv_group_size", 8),
    ("mask_fill", "dtype_min"),
    ("scale", 0.0625),
)
_ROPE_ATTRIBUTES = (
    ("frequency_dtype", "bf16"),
    ("pairing", "split_half"),
    ("theta", 10000.0),
)


@dataclass(frozen=True, slots=True)
class PrefixKvStoreProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("prefix KV store exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "PrefixKvStoreProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.MEMORY
            or item.opcode != "transpose"
            or item.input_types != (_VALUE_BSHD,)
            or item.output_types != (_STATE_BHSD,)
            or item.attributes != (("permutation", (0, 2, 1, 3)),)
        ):
            raise ValidationError(
                "prefix KV store anchor is outside the delivered exact envelope"
            )
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "key_bhsd": [1, 1, 968, 256],
            "value_bshd": [1, 968, 1, 256],
            "state_kv_bhsd": [1, 1, 968, 256],
            "dtype": "bf16",
        }


@dataclass(frozen=True, slots=True)
class PrefixKvStorePayload:
    problem: PrefixKvStoreProblem
    module_bytes: int
    module_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, PrefixKvStoreProblem):
            raise ValidationError("prefix KV store payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")

    def to_bytes(self) -> bytes:
        tensor_bytes = math.prod(_STATE_BHSD.shape) * 2
        return PREFIX_KV_STORE_PAYLOAD.pack(
            PREFIX_KV_STORE_PAYLOAD_MAGIC,
            PREFIX_KV_STORE_SCHEMA_MAJOR,
            PREFIX_KV_STORE_SCHEMA_MINOR,
            int(self.problem.target_arch),
            PREFIX_KV_STORE_VARIANT,
            2,  # BF16
            1,  # key BHSD
            2,  # value BSHD
            1,  # state key BHSD
            1,  # state value BHSD
            1,
            1,
            968,
            256,
            242,
            1,
            1,
            256,
            1,
            1,
            16,
            0,
            0,
            tensor_bytes,
            tensor_bytes,
            tensor_bytes,
            tensor_bytes,
            self.module_bytes,
            0,
            bytes.fromhex(self.module_sha256),
            bytes(20),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "PrefixKvStorePayload":
        view = memoryview(data)
        if len(view) != PREFIX_KV_STORE_PAYLOAD.size:
            raise FormatError("prefix KV store payload must have its exact fixed size")
        fields = PREFIX_KV_STORE_PAYLOAD.unpack(view)
        if fields[0] != PREFIX_KV_STORE_PAYLOAD_MAGIC:
            raise FormatError("prefix KV store payload magic is invalid")
        if fields[1] != PREFIX_KV_STORE_SCHEMA_MAJOR or fields[2] > PREFIX_KV_STORE_SCHEMA_MINOR:
            raise FormatError("prefix KV store payload schema is unsupported")
        try:
            parsed = cls(
                PrefixKvStoreProblem(CudaArch(fields[3])), fields[27], fields[29].hex()
            )
        except (ValueError, ValidationError) as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError("prefix KV store payload is outside the delivered exact variant")
        return parsed


@dataclass(frozen=True, slots=True)
class PrefixKvStoreValidationReceipt:
    problem: PrefixKvStoreProblem
    implementation_sha256: str
    implementation_bytes: int
    cuda_compiler_version: int
    cuda_runtime_version: int
    cuda_driver_version: int
    normal_launches: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    capture_matches_normal: bool
    full_states_bit_exact: bool
    racecheck_hazards: int
    latency_ms: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, PrefixKvStoreProblem):
            raise ValidationError("prefix KV store receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("prefix KV store receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_states_bit_exact, "bit-exact full states"),
        ):
            if value is not True:
                raise ValidationError(f"prefix KV store receipt requires {label}")
        if self.racecheck_hazards != 0:
            raise ValidationError("prefix KV store receipt requires zero racecheck hazards")
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0:
            raise ValidationError(
                "prefix KV store receipt latency must be finite and positive"
            )
        if self.contains_ptx is not False:
            raise ValidationError("prefix KV store implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = PrefixKvStoreProblem.from_inventory(
            item, target_arch=self.problem.target_arch
        )
        if problem != self.problem:
            raise ValidationError("prefix KV store receipt does not cover this problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=PREFIX_KV_STORE_PROVIDER_ID,
            abi_major=PREFIX_KV_STORE_PROVIDER_ABI_MAJOR,
            abi_minor=PREFIX_KV_STORE_PROVIDER_ABI_MINOR,
            provider_version=(
                f"{PREFIX_KV_STORE_PROVIDER_VERSION};"
                f"cuda-runtime={self.cuda_runtime_version};driver={self.cuda_driver_version}"
            ),
            implementation_id=PREFIX_KV_STORE_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def payload(self) -> PrefixKvStorePayload:
        return PrefixKvStorePayload(
            self.problem, self.implementation_bytes, self.implementation_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PREFIX_KV_STORE_RECEIPT_SCHEMA,
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
            "full_states_bit_exact": self.full_states_bit_exact,
            "racecheck_hazards": self.racecheck_hazards,
            "latency_ms": self.latency_ms,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class PrefixKvStoreLoweredCommand:
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
class PrefixKvStorePartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[PrefixKvStoreLoweredCommand, ...]
    fused_execution_indices: tuple[int, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PREFIX_KV_STORE_PARTIAL_LOWERING_SCHEMA,
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


def make_prefix_kv_store_capabilities(
    inventory: LoweringInventory,
    receipt: PrefixKvStoreValidationReceipt,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if receipt.problem.target_arch != target_arch:
        raise ValidationError("prefix KV store receipt target differs from capability target")
    capabilities: dict[str, ProviderCapability] = {}
    for item in inventory.ops:
        if item.opcode != "transpose":
            continue
        try:
            PrefixKvStoreProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
    if not capabilities:
        raise ValidationError("prefix KV store receipt matched no inventory anchor")
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_prefix_kv_store_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: PrefixKvStoreValidationReceipt,
    *,
    target_arch: CudaArch,
) -> PrefixKvStorePartialLowering:
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode()
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("prefix KV store inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("prefix KV store memory plan does not match its schedule")
    if {item.value_id for item in memory_plan.allocations} != set(
        range(len(schedule.values))
    ):
        raise ValidationError("prefix KV store memory plan has incomplete value coverage")

    capabilities = make_prefix_kv_store_capabilities(
        inventory, receipt, target_arch=target_arch
    )
    inventory_by_site = {item.site_id: item for item in inventory.ops}
    ops_by_index = {item.execution_index: item for item in schedule.ops}
    consumers: dict[int, list[ScheduledOp]] = {}
    entry_outputs = {value_id for _, value_id in schedule.entry_outputs}
    state_ids = dict(schedule.states)
    elided = set(memory_plan.elided_ops)
    for op in schedule.ops:
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(op)

    def state_target(op: ScheduledOp) -> int:
        if op.opcode != "state_write" or len(op.inputs) != 1 or op.outputs:
            raise ValidationError("prefix KV store requires unary state writes")
        if len(op.attributes) != 1 or op.attributes[0][0] != "state":
            raise ValidationError("prefix KV store state-write target is invalid")
        state_id = state_ids.get(str(op.attributes[0][1]))
        if state_id is None:
            raise ValidationError("prefix KV store target state is missing")
        state = schedule.values[state_id]
        if state.storage != ValueStorage.STATE or state.type != _STATE_BHSD:
            raise ValidationError("prefix KV store target state has the wrong contract")
        return state_id

    commands: list[PrefixKvStoreLoweredCommand] = []
    fused: set[int] = set()
    for transpose in schedule.ops:
        item = inventory_by_site.get(transpose.site)
        if item is None:
            continue
        try:
            PrefixKvStoreProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        if (
            len(transpose.inputs) != 1
            or len(transpose.outputs) != 1
            or transpose.execution_index not in elided
            or schedule.values[transpose.inputs[0]].type != _VALUE_BSHD
            or schedule.values[transpose.outputs[0]].type != _STATE_BHSD
            or transpose.outputs[0] in entry_outputs
        ):
            raise ValidationError("prefix KV store singleton transpose is not a proven view")
        output_consumers = consumers.get(transpose.outputs[0], [])
        attentions = tuple(
            op
            for op in output_consumers
            if op.opcode == "scaled_dot_product_attention"
        )
        value_writes = tuple(
            op for op in output_consumers if op.opcode == "state_write"
        )
        # The exact singleton signature is shared by prefix K and V. K feeds
        # RoPE and is already an identity view; only V directly feeds the
        # attention/state-write pair owned by this region.
        if not attentions and not value_writes:
            continue
        attention = attentions[0] if len(attentions) == 1 else None
        value_write = value_writes[0] if len(value_writes) == 1 else None
        if (
            attention is None
            or value_write is None
            or len(output_consumers) != 2
            or attention.invocation_id != transpose.invocation_id
            or value_write.invocation_id != transpose.invocation_id
            or len(attention.inputs) != 4
            or len(attention.outputs) != 1
            or attention.inputs[2] != transpose.outputs[0]
            or tuple(schedule.values[value_id].type for value_id in attention.inputs)
            != (_QUERY, _STATE_BHSD, _STATE_BHSD, _MASK)
            or schedule.values[attention.outputs[0]].type != _QUERY
            or attention.attributes != _ATTENTION_ATTRIBUTES
        ):
            raise ValidationError("prefix KV store requires one exact adjacent attention")

        key_id = attention.inputs[1]
        key_value = schedule.values[key_id]
        if key_value.producer is None:
            raise ValidationError("prefix KV store key has no RoPE producer")
        key_rope = ops_by_index[key_value.producer]
        key_consumers = consumers.get(key_id, [])
        key_write = next(
            (op for op in key_consumers if op.opcode == "state_write"), None
        )
        if (
            key_write is None
            or len(key_consumers) != 2
            or attention not in key_consumers
            or key_rope.opcode != "rope_default"
            or key_rope.invocation_id != transpose.invocation_id
            or key_rope.attributes != _ROPE_ATTRIBUTES
            or len(key_rope.inputs) != 2
            or len(key_rope.outputs) != 1
            or key_rope.outputs[0] != key_id
            or schedule.values[key_rope.inputs[1]].type != _POSITIONS
            or key_write.invocation_id != transpose.invocation_id
            or value_write.execution_index != key_write.execution_index + 1
        ):
            raise ValidationError("prefix KV store requires paired K/V state writes")
        key_state = state_target(key_write)
        value_state = state_target(value_write)
        if key_state == value_state:
            raise ValidationError("prefix KV store K/V state targets must be distinct")
        fused_indices = (key_write.execution_index, value_write.execution_index)
        if fused.intersection(fused_indices):
            raise ValidationError("prefix KV store fused regions overlap")
        capability = receipt.capability_for(item)
        commands.append(
            PrefixKvStoreLoweredCommand(
                key_write.execution_index,
                key_write.site,
                fused_indices,
                ProviderCommand(
                    tag=CommandTag.CUDA_KERNEL,
                    provider_id=capability.provider_id,
                    abi_major=capability.abi_major,
                    abi_minor=capability.abi_minor,
                    capability_digest=capability.digest,
                    operands=(
                        CommandOperand(key_id, OperandAccess.READ),
                        CommandOperand(transpose.inputs[0], OperandAccess.READ),
                        CommandOperand(key_state, OperandAccess.WRITE),
                        CommandOperand(value_state, OperandAccess.WRITE),
                    ),
                    payload=receipt.payload().to_bytes(),
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        fused.update(fused_indices)

    if not commands:
        raise ValidationError("prefix KV store found no complete state pair")
    if fused & elided:
        raise ValidationError("prefix KV store fusion overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in fused and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return PrefixKvStorePartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(fused)),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_prefix_kv_store_partial_lowering(
    lowering: PrefixKvStorePartialLowering,
) -> str:
    if not isinstance(lowering, PrefixKvStorePartialLowering):
        raise ValidationError(
            "prefix KV store dump requires a PrefixKvStorePartialLowering"
        )
    return _canonical_json(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"prefix KV store {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or not any(char != "0" for char in value)
    ):
        raise ValidationError(f"prefix KV store {label} must be a SHA-256 digest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
