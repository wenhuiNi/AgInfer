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


FLASHINFER_ATTENTION_PAYLOAD_MAGIC = b"AIFIAT1\0"
FLASHINFER_ATTENTION_SCHEMA_MAJOR = 1
FLASHINFER_ATTENTION_SCHEMA_MINOR = 0
FLASHINFER_ATTENTION_PAYLOAD = struct.Struct(
    "<8sHH" + "I" * 20 + "f" + "Q" * 5 + "20s20s16s"
)
FLASHINFER_ATTENTION_PROVIDER_ID = 3
FLASHINFER_ATTENTION_PROVIDER_ABI_MAJOR = 1
FLASHINFER_ATTENTION_PROVIDER_ABI_MINOR = 0
FLASHINFER_VERSION = "0.6.18"
FLASHINFER_COMMIT = "69ff11fc4954396d98326656dc85debd2223f637"
FLASHINFER_CCCL_COMMIT = "876867684f7fac130e0f5911236e0a92a970d4fd"
FLASHINFER_CUTLASS_COMMIT = "b46b16d003484063bca4ed365e44095c4c6ed633"
FLASHINFER_ATTENTION_IMPLEMENTATION = (
    "flashinfer.fa2.bf16_gqa_dense_bool.attention_transpose.v1"
)
FLASHINFER_ATTENTION_VARIANT = 1
FLASHINFER_PREFIX_ATTENTION_IMPLEMENTATION = (
    "flashinfer.fa2.bf16_gqa_prefix_pad_bool.attention_transpose.v1"
)
FLASHINFER_PREFIX_ATTENTION_VARIANT = 2
FLASHINFER_PREFIX_ATTENTION_RECEIPT_SCHEMA = (
    "aginfer.flashinfer-prefix-attention-validation.v1"
)
FLASHINFER_ATTENTION_RECEIPT_SCHEMA = "aginfer.flashinfer-attention-validation.v1"
FLASHINFER_ATTENTION_PARTIAL_LOWERING_SCHEMA = (
    "aginfer.flashinfer-attention-partial-lowering.v1"
)

assert FLASHINFER_ATTENTION_PAYLOAD.size == 192


def _sig(dtype: str, shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature(dtype, shape, "row_major", "cuda")


@dataclass(frozen=True, slots=True)
class FlashInferAttentionProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("FlashInfer attention exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "FlashInferAttentionProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.ATTENTION
            or item.opcode != "scaled_dot_product_attention"
            or len(item.input_types) != 4
            or len(item.output_types) != 1
        ):
            raise ValidationError("FlashInfer attention requires an attention inventory site")
        expected_inputs = (
            _sig("bf16", (1, 8, 50, 256)),
            _sig("bf16", (1, 1, 1018, 256)),
            _sig("bf16", (1, 1, 1018, 256)),
            _sig("bool", (1, 8, 50, 1018)),
        )
        expected_outputs = (_sig("bf16", (1, 8, 50, 256)),)
        expected_attributes = (
            ("kv_group_size", 8),
            ("mask_fill", "dtype_min"),
            ("scale", 0.0625),
        )
        if (
            item.input_types != expected_inputs
            or item.output_types != expected_outputs
            or item.attributes != expected_attributes
        ):
            raise ValidationError(
                "FlashInfer attention site is outside the BF16 GQA denoise envelope"
            )
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "query": [1, 8, 50, 256],
            "key_value": [1, 1, 1018, 256],
            "mask_2d": [1, 50, 1018],
            "output_bshd": [1, 50, 8, 256],
            "dtype": "bf16",
            "mask_dtype": "bool",
            "scale": 0.0625,
            "kv_group_size": 8,
        }


@dataclass(frozen=True, slots=True)
class FlashInferPrefixAttentionProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("FlashInfer prefix attention exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "FlashInferPrefixAttentionProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.ATTENTION
            or item.opcode != "scaled_dot_product_attention"
            or len(item.input_types) != 4
            or len(item.output_types) != 1
        ):
            raise ValidationError(
                "FlashInfer prefix attention requires an attention inventory site"
            )
        expected_inputs = (
            _sig("bf16", (1, 8, 968, 256)),
            _sig("bf16", (1, 1, 968, 256)),
            _sig("bf16", (1, 1, 968, 256)),
            _sig("bool", (1, 8, 968, 968)),
        )
        expected_outputs = (_sig("bf16", (1, 8, 968, 256)),)
        expected_attributes = (
            ("kv_group_size", 8),
            ("mask_fill", "dtype_min"),
            ("scale", 0.0625),
        )
        if (
            item.input_types != expected_inputs
            or item.output_types != expected_outputs
            or item.attributes != expected_attributes
        ):
            raise ValidationError(
                "FlashInfer attention site is outside the BF16 GQA prefix envelope"
            )
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "query": [1, 8, 968, 256],
            "key_value": [1, 1, 968, 256],
            "pad_mask": [1, 968],
            "output_bshd": [1, 968, 8, 256],
            "dtype": "bf16",
            "mask_dtype": "bool",
            "mask_fill": "dtype_min",
            "scale": 0.0625,
            "kv_group_size": 8,
        }


@dataclass(frozen=True, slots=True)
class FlashInferAttentionPayload:
    problem: FlashInferAttentionProblem

    def __post_init__(self) -> None:
        if not isinstance(self.problem, FlashInferAttentionProblem):
            raise ValidationError("FlashInfer attention payload problem is invalid")

    def to_bytes(self) -> bytes:
        return FLASHINFER_ATTENTION_PAYLOAD.pack(
            FLASHINFER_ATTENTION_PAYLOAD_MAGIC,
            FLASHINFER_ATTENTION_SCHEMA_MAJOR,
            FLASHINFER_ATTENTION_SCHEMA_MINOR,
            int(self.problem.target_arch),
            FLASHINFER_ATTENTION_VARIANT,
            2,  # query dtype: BF16
            2,  # KV dtype: BF16
            2,  # output dtype: BF16
            4,  # mask dtype: BOOL
            1,  # mask layout: dense 2-D QK, head-broadcast
            1,  # output layout: BSHD/QHD
            8,
            1,
            50,
            1018,
            256,
            7,
            1,
            1,
            32,
            4,
            1,
            49152,
            0.0625,
            204800,
            521216,
            521216,
            50900,
            204800,
            bytes.fromhex(FLASHINFER_COMMIT),
            bytes.fromhex(FLASHINFER_CCCL_COMMIT),
            bytes(16),
        )

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "FlashInferAttentionPayload":
        view = memoryview(data)
        if len(view) != FLASHINFER_ATTENTION_PAYLOAD.size:
            raise FormatError("FlashInfer attention payload must have its exact fixed size")
        fields = FLASHINFER_ATTENTION_PAYLOAD.unpack(view)
        expected = cls(FlashInferAttentionProblem(CudaArch.SM120)).to_bytes()
        if bytes(view) != expected:
            if fields[0] != FLASHINFER_ATTENTION_PAYLOAD_MAGIC:
                raise FormatError("FlashInfer attention payload magic is invalid")
            if fields[1] != FLASHINFER_ATTENTION_SCHEMA_MAJOR:
                raise FormatError("FlashInfer attention payload schema is unsupported")
            raise FormatError(
                "FlashInfer attention payload is outside the delivered exact variant"
            )
        return cls(FlashInferAttentionProblem(CudaArch.SM120))


@dataclass(frozen=True, slots=True)
class FlashInferPrefixAttentionPayload:
    problem: FlashInferPrefixAttentionProblem

    def __post_init__(self) -> None:
        if not isinstance(self.problem, FlashInferPrefixAttentionProblem):
            raise ValidationError("FlashInfer prefix attention payload problem is invalid")

    def to_bytes(self) -> bytes:
        return FLASHINFER_ATTENTION_PAYLOAD.pack(
            FLASHINFER_ATTENTION_PAYLOAD_MAGIC,
            FLASHINFER_ATTENTION_SCHEMA_MAJOR,
            FLASHINFER_ATTENTION_SCHEMA_MINOR,
            int(self.problem.target_arch),
            FLASHINFER_PREFIX_ATTENTION_VARIANT,
            2,  # query dtype: BF16
            2,  # KV dtype: BF16
            2,  # output dtype: BF16
            4,  # mask dtype: BOOL
            2,  # mask layout: shared 1-D pad mask
            1,  # output layout: BSHD/QHD
            8,
            1,
            968,
            968,
            256,
            121,
            1,
            1,
            32,
            4,
            1,
            49152,
            0.0625,
            3964928,
            495616,
            495616,
            968,
            3964928,
            bytes.fromhex(FLASHINFER_COMMIT),
            bytes.fromhex(FLASHINFER_CCCL_COMMIT),
            bytes(16),
        )

    @classmethod
    def from_bytes(
        cls, data: bytes | memoryview
    ) -> "FlashInferPrefixAttentionPayload":
        view = memoryview(data)
        if len(view) != FLASHINFER_ATTENTION_PAYLOAD.size:
            raise FormatError(
                "FlashInfer prefix attention payload must have its exact fixed size"
            )
        fields = FLASHINFER_ATTENTION_PAYLOAD.unpack(view)
        expected = cls(FlashInferPrefixAttentionProblem(CudaArch.SM120)).to_bytes()
        if bytes(view) != expected:
            if fields[0] != FLASHINFER_ATTENTION_PAYLOAD_MAGIC:
                raise FormatError("FlashInfer prefix attention payload magic is invalid")
            if fields[1] != FLASHINFER_ATTENTION_SCHEMA_MAJOR:
                raise FormatError(
                    "FlashInfer prefix attention payload schema is unsupported"
                )
            raise FormatError(
                "FlashInfer prefix attention payload is outside the delivered exact variant"
            )
        return cls(FlashInferPrefixAttentionProblem(CudaArch.SM120))


@dataclass(frozen=True, slots=True)
class FlashInferAttentionValidationReceipt:
    problem: FlashInferAttentionProblem
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
    fully_masked_rows: int
    cosine: float
    max_abs: float
    p99_abs: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, FlashInferAttentionProblem):
            raise ValidationError("FlashInfer attention receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if self.normal_launches < 2:
            raise ValidationError("FlashInfer attention receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_output_compared, "full-output comparison"),
        ):
            if value is not True:
                raise ValidationError(f"FlashInfer attention receipt requires {label}")
        if self.fully_masked_rows != 0:
            raise ValidationError(
                "FlashInfer denoise variant requires a proof of no fully-masked rows"
            )
        if (
            not math.isfinite(self.cosine)
            or self.cosine < 0.99999
            or not math.isfinite(self.max_abs)
            or self.max_abs > 0.00025
            or not math.isfinite(self.p99_abs)
            or self.p99_abs > 0.000125
        ):
            raise ValidationError("FlashInfer attention correctness gate failed")
        if self.contains_ptx is not False:
            raise ValidationError("FlashInfer attention implementation must not contain PTX")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = FlashInferAttentionProblem.from_inventory(
            item, target_arch=self.problem.target_arch
        )
        if problem != self.problem:
            raise ValidationError("FlashInfer attention receipt does not cover this problem")
        return ProviderCapability.exact_for(
            item,
            provider_id=FLASHINFER_ATTENTION_PROVIDER_ID,
            abi_major=FLASHINFER_ATTENTION_PROVIDER_ABI_MAJOR,
            abi_minor=FLASHINFER_ATTENTION_PROVIDER_ABI_MINOR,
            provider_version=(
                f"flashinfer={FLASHINFER_VERSION};source={FLASHINFER_COMMIT};"
                f"cccl={FLASHINFER_CCCL_COMMIT};cuda-runtime={self.cuda_runtime_version};"
                f"driver={self.cuda_driver_version}"
            ),
            implementation_id=FLASHINFER_ATTENTION_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": FLASHINFER_ATTENTION_RECEIPT_SCHEMA,
            "problem": self.problem.to_dict(),
            "flashinfer_version": FLASHINFER_VERSION,
            "flashinfer_commit": FLASHINFER_COMMIT,
            "cccl_commit": FLASHINFER_CCCL_COMMIT,
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
            "fully_masked_rows": self.fully_masked_rows,
            "cosine": self.cosine,
            "max_abs": self.max_abs,
            "p99_abs": self.p99_abs,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class FlashInferPrefixAttentionValidationReceipt:
    problem: FlashInferPrefixAttentionProblem
    implementation_sha256: str
    implementation_bytes: int
    cuda_compiler_version: int
    cuda_runtime_version: int
    cuda_driver_version: int
    pad_mask_sha256: str
    valid_tokens: int
    fully_masked_rows: int
    normal_launches: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    capture_matches_normal: bool
    full_output_compared: bool
    finite_mask_rows_compared: bool
    hard_mask_negative_detected: bool
    cosine: float
    max_abs: float
    p99_abs: float
    contains_ptx: bool
    fa3_custom_mask_supported: bool
    fa3_sm120_compiles: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, FlashInferPrefixAttentionProblem):
            raise ValidationError("FlashInfer prefix attention receipt problem is invalid")
        _digest(self.implementation_sha256, "prefix implementation SHA-256")
        _digest(self.pad_mask_sha256, "prefix pad-mask SHA-256")
        for value, label in (
            (self.implementation_bytes, "prefix implementation bytes"),
            (self.cuda_compiler_version, "prefix CUDA compiler version"),
            (self.cuda_runtime_version, "prefix CUDA runtime version"),
            (self.cuda_driver_version, "prefix CUDA driver version"),
            (self.valid_tokens, "prefix valid tokens"),
            (self.fully_masked_rows, "prefix fully-masked rows"),
        ):
            _positive(value, label)
        if self.valid_tokens + self.fully_masked_rows != 968:
            raise ValidationError(
                "FlashInfer prefix attention mask row counts must total 968"
            )
        if self.normal_launches < 2:
            raise ValidationError(
                "FlashInfer prefix attention receipt requires two normal launches"
            )
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.full_output_compared, "full-output comparison"),
            (self.finite_mask_rows_compared, "finite-mask row comparison"),
            (self.hard_mask_negative_detected, "hard-mask negative control"),
        ):
            if value is not True:
                raise ValidationError(
                    f"FlashInfer prefix attention receipt requires {label}"
                )
        if (
            not math.isfinite(self.cosine)
            or self.cosine < 0.99999
            or not math.isfinite(self.max_abs)
            or self.max_abs > 0.00025
            or not math.isfinite(self.p99_abs)
            or self.p99_abs > 0.000125
        ):
            raise ValidationError("FlashInfer prefix attention correctness gate failed")
        if self.contains_ptx is not False:
            raise ValidationError(
                "FlashInfer prefix attention implementation must not contain PTX"
            )
        if self.fa3_custom_mask_supported is not False:
            raise ValidationError(
                "FlashInfer prefix receipt must record the pinned FA3 custom-mask refusal"
            )
        if self.fa3_sm120_compiles is not False:
            raise ValidationError(
                "FlashInfer prefix receipt must record the pinned FA3 SM120 compile blocker"
            )

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        problem = FlashInferPrefixAttentionProblem.from_inventory(
            item, target_arch=self.problem.target_arch
        )
        if problem != self.problem:
            raise ValidationError(
                "FlashInfer prefix attention receipt does not cover this problem"
            )
        return ProviderCapability.exact_for(
            item,
            provider_id=FLASHINFER_ATTENTION_PROVIDER_ID,
            abi_major=FLASHINFER_ATTENTION_PROVIDER_ABI_MAJOR,
            abi_minor=FLASHINFER_ATTENTION_PROVIDER_ABI_MINOR,
            provider_version=(
                f"flashinfer={FLASHINFER_VERSION};source={FLASHINFER_COMMIT};"
                f"cccl={FLASHINFER_CCCL_COMMIT};cutlass={FLASHINFER_CUTLASS_COMMIT};"
                f"cuda-runtime={self.cuda_runtime_version};driver={self.cuda_driver_version}"
            ),
            implementation_id=FLASHINFER_PREFIX_ATTENTION_IMPLEMENTATION,
            implementation_digest=self.implementation_sha256,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": FLASHINFER_PREFIX_ATTENTION_RECEIPT_SCHEMA,
            "problem": self.problem.to_dict(),
            "flashinfer_version": FLASHINFER_VERSION,
            "flashinfer_commit": FLASHINFER_COMMIT,
            "cccl_commit": FLASHINFER_CCCL_COMMIT,
            "cutlass_commit": FLASHINFER_CUTLASS_COMMIT,
            "implementation_sha256": self.implementation_sha256,
            "implementation_bytes": self.implementation_bytes,
            "cuda_compiler_version": self.cuda_compiler_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "pad_mask_sha256": self.pad_mask_sha256,
            "valid_tokens": self.valid_tokens,
            "fully_masked_rows": self.fully_masked_rows,
            "normal_launches": self.normal_launches,
            "normal_repeat_bit_exact": self.normal_repeat_bit_exact,
            "capture_replay_bit_exact": self.capture_replay_bit_exact,
            "capture_matches_normal": self.capture_matches_normal,
            "full_output_compared": self.full_output_compared,
            "finite_mask_rows_compared": self.finite_mask_rows_compared,
            "hard_mask_negative_detected": self.hard_mask_negative_detected,
            "cosine": self.cosine,
            "max_abs": self.max_abs,
            "p99_abs": self.p99_abs,
            "contains_ptx": self.contains_ptx,
            "fa3_custom_mask_supported": self.fa3_custom_mask_supported,
            "fa3_sm120_compiles": self.fa3_sm120_compiles,
        }


@dataclass(frozen=True, slots=True)
class FlashInferAttentionLoweredCommand:
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
class FlashInferAttentionPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[FlashInferAttentionLoweredCommand, ...]
    fused_execution_indices: tuple[int, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": FLASHINFER_ATTENTION_PARTIAL_LOWERING_SCHEMA,
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


def make_flashinfer_attention_capabilities(
    inventory: LoweringInventory,
    receipt: FlashInferAttentionValidationReceipt,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if receipt.problem.target_arch != target_arch:
        raise ValidationError("FlashInfer attention receipt target differs from capability target")
    capabilities: dict[str, ProviderCapability] = {}
    matched = 0
    for item in inventory.ops:
        if item.opcode != "scaled_dot_product_attention":
            continue
        try:
            FlashInferAttentionProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
        matched += 1
    if matched == 0:
        raise ValidationError("FlashInfer attention receipt matched no inventory sites")
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_flashinfer_attention_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: FlashInferAttentionValidationReceipt,
    *,
    target_arch: CudaArch,
) -> FlashInferAttentionPartialLowering:
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode()
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError("FlashInfer attention inventory does not match its schedule")
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError("FlashInfer attention memory plan does not match its schedule")
    if {item.value_id for item in memory_plan.allocations} != set(range(len(schedule.values))):
        raise ValidationError("FlashInfer attention memory plan has incomplete value coverage")

    capabilities = make_flashinfer_attention_capabilities(
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
            FlashInferAttentionProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        supported[op.execution_index] = (op, receipt.capability_for(item))

    commands: list[FlashInferAttentionLoweredCommand] = []
    fused: set[int] = set()
    broadcast_ops: dict[int, ScheduledOp] = {}
    for execution_index, (op, capability) in supported.items():
        if len(op.inputs) != 4 or len(op.outputs) != 1:
            raise ValidationError("FlashInfer attention scheduled arity is invalid")
        mask_value = schedule.values[op.inputs[3]]
        if mask_value.producer is None:
            raise ValidationError("FlashInfer attention mask has no broadcast producer")
        broadcast = ops_by_index[mask_value.producer]
        if (
            broadcast.opcode != "broadcast_in_dim"
            or len(broadcast.inputs) != 1
            or len(broadcast.outputs) != 1
            or broadcast.outputs[0] != op.inputs[3]
            or broadcast.attributes
            != (("broadcast_dimensions", (0, 2, 3)), ("shape", (1, 8, 50, 1018)))
            or schedule.values[broadcast.inputs[0]].type
            != _sig("bool", (1, 50, 1018))
        ):
            raise ValidationError("FlashInfer attention requires the exact head-broadcast mask")
        broadcast_ops[broadcast.execution_index] = broadcast
        if broadcast.outputs[0] in entry_output_ids or op.outputs[0] in entry_output_ids:
            raise ValidationError("FlashInfer fused intermediates must not be public outputs")

        output_consumers = consumers.get(op.outputs[0], [])
        if len(output_consumers) != 1:
            raise ValidationError("FlashInfer attention output must have one transpose consumer")
        transpose = output_consumers[0]
        if (
            transpose.opcode != "transpose"
            or transpose.invocation_id != op.invocation_id
            or transpose.attributes != (("permutation", (0, 2, 1, 3)),)
            or len(transpose.outputs) != 1
            or schedule.values[transpose.outputs[0]].type
            != _sig("bf16", (1, 50, 8, 256))
        ):
            raise ValidationError("FlashInfer attention requires the exact BHSd-to-BSHd transpose")

        payload = (receipt.payload() if isinstance(receipt, BuildBinding)
                   else FlashInferAttentionPayload(receipt.problem)).to_bytes()
        fused_indices = (execution_index, transpose.execution_index)
        commands.append(
            FlashInferAttentionLoweredCommand(
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
                        CommandOperand(op.inputs[0], OperandAccess.READ),
                        CommandOperand(op.inputs[1], OperandAccess.READ),
                        CommandOperand(op.inputs[2], OperandAccess.READ),
                        CommandOperand(broadcast.inputs[0], OperandAccess.READ),
                        CommandOperand(transpose.outputs[0], OperandAccess.WRITE),
                    ),
                    payload=payload,
                    workspace_bytes=capability.workspace_bytes,
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        fused.update(fused_indices)

    for broadcast in broadcast_ops.values():
        if any(consumer.execution_index not in supported for consumer in consumers[broadcast.outputs[0]]):
            raise ValidationError("FlashInfer mask broadcast has an unsupported consumer")
        fused.add(broadcast.execution_index)

    elided = set(memory_plan.elided_ops)
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in fused and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return FlashInferAttentionPartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(fused)),
        tuple(sorted(elided)),
        unhandled,
    )


def make_flashinfer_prefix_attention_capabilities(
    inventory: LoweringInventory,
    receipt: FlashInferPrefixAttentionValidationReceipt,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if receipt.problem.target_arch != target_arch:
        raise ValidationError(
            "FlashInfer prefix attention receipt target differs from capability target"
        )
    capabilities: dict[str, ProviderCapability] = {}
    matched = 0
    for item in inventory.ops:
        if item.opcode != "scaled_dot_product_attention":
            continue
        try:
            FlashInferPrefixAttentionProblem.from_inventory(
                item, target_arch=target_arch
            )
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        capabilities.setdefault(capability.digest, capability)
        matched += 1
    if matched == 0:
        raise ValidationError(
            "FlashInfer prefix attention receipt matched no inventory sites"
        )
    return tuple(sorted(capabilities.values(), key=lambda item: item.digest))


def lower_flashinfer_prefix_attention_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: FlashInferPrefixAttentionValidationReceipt,
    *,
    target_arch: CudaArch,
) -> FlashInferAttentionPartialLowering:
    inventory_sha256 = hashlib.sha256(
        dump_lowering_inventory(inventory).encode()
    ).hexdigest()
    schedule_sha256 = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha256:
        raise ValidationError(
            "FlashInfer prefix attention inventory does not match its schedule"
        )
    if memory_plan.schedule_sha256 != schedule_sha256:
        raise ValidationError(
            "FlashInfer prefix attention memory plan does not match its schedule"
        )
    if {item.value_id for item in memory_plan.allocations} != set(
        range(len(schedule.values))
    ):
        raise ValidationError(
            "FlashInfer prefix attention memory plan has incomplete value coverage"
        )

    capabilities = make_flashinfer_prefix_attention_capabilities(
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
            FlashInferPrefixAttentionProblem.from_inventory(
                item, target_arch=target_arch
            )
        except ValidationError:
            continue
        supported[op.execution_index] = (op, receipt.capability_for(item))

    commands: list[FlashInferAttentionLoweredCommand] = []
    fused: set[int] = set()
    mask_chains: dict[
        int, tuple[ScheduledOp, ScheduledOp, ScheduledOp, ScheduledOp, int]
    ] = {}
    for execution_index, (op, capability) in supported.items():
        if len(op.inputs) != 4 or len(op.outputs) != 1:
            raise ValidationError("FlashInfer prefix attention scheduled arity is invalid")

        head_mask_value = schedule.values[op.inputs[3]]
        if head_mask_value.producer is None:
            raise ValidationError(
                "FlashInfer prefix attention mask has no head-broadcast producer"
            )
        head_broadcast = ops_by_index[head_mask_value.producer]
        if (
            head_broadcast.opcode != "broadcast_in_dim"
            or len(head_broadcast.inputs) != 1
            or len(head_broadcast.outputs) != 1
            or head_broadcast.outputs[0] != op.inputs[3]
            or head_broadcast.attributes
            != (
                ("broadcast_dimensions", (0, 2, 3)),
                ("shape", (1, 8, 968, 968)),
            )
            or schedule.values[head_broadcast.inputs[0]].type
            != _sig("bool", (1, 968, 968))
        ):
            raise ValidationError(
                "FlashInfer prefix attention requires the exact head-broadcast mask"
            )

        mask_2d_value = schedule.values[head_broadcast.inputs[0]]
        if mask_2d_value.producer is None:
            raise ValidationError(
                "FlashInfer prefix attention 2-D mask has no logical-and producer"
            )
        logical_and = ops_by_index[mask_2d_value.producer]
        if (
            logical_and.opcode != "logical_and"
            or logical_and.attributes
            or len(logical_and.inputs) != 2
            or len(logical_and.outputs) != 1
            or logical_and.outputs[0] != head_broadcast.inputs[0]
        ):
            raise ValidationError(
                "FlashInfer prefix attention requires the exact key/query logical-and"
            )

        broadcasts = []
        for value_id in logical_and.inputs:
            value = schedule.values[value_id]
            if value.producer is None:
                raise ValidationError(
                    "FlashInfer prefix attention key/query mask has no producer"
                )
            broadcasts.append(ops_by_index[value.producer])
        by_dimensions = {dict(item.attributes).get("broadcast_dimensions"): item for item in broadcasts}
        key_broadcast = by_dimensions.get((0, 2))
        query_broadcast = by_dimensions.get((0, 1))
        if key_broadcast is None or query_broadcast is None:
            raise ValidationError(
                "FlashInfer prefix attention requires distinct key/query broadcasts"
            )
        for mask_broadcast, dimensions in (
            (key_broadcast, (0, 2)),
            (query_broadcast, (0, 1)),
        ):
            if (
                mask_broadcast.opcode != "broadcast_in_dim"
                or len(mask_broadcast.inputs) != 1
                or len(mask_broadcast.outputs) != 1
                or mask_broadcast.attributes
                != (
                    ("broadcast_dimensions", dimensions),
                    ("shape", (1, 968, 968)),
                )
                or schedule.values[mask_broadcast.inputs[0]].type
                != _sig("bool", (1, 968))
            ):
                raise ValidationError(
                    "FlashInfer prefix attention requires exact pad-mask broadcasts"
                )
        pad_mask_value_id = key_broadcast.inputs[0]
        if query_broadcast.inputs[0] != pad_mask_value_id:
            raise ValidationError(
                "FlashInfer prefix key/query broadcasts must share one pad mask"
            )
        if (
            consumers.get(key_broadcast.outputs[0], []) != [logical_and]
            or consumers.get(query_broadcast.outputs[0], []) != [logical_and]
            or consumers.get(logical_and.outputs[0], []) != [head_broadcast]
        ):
            raise ValidationError(
                "FlashInfer prefix mask intermediates must have exclusive consumers"
            )
        if any(
            value_id in entry_output_ids
            for value_id in (
                key_broadcast.outputs[0],
                query_broadcast.outputs[0],
                logical_and.outputs[0],
                head_broadcast.outputs[0],
                op.outputs[0],
            )
        ):
            raise ValidationError(
                "FlashInfer prefix fused intermediates must not be public outputs"
            )
        mask_chains[head_broadcast.execution_index] = (
            key_broadcast,
            query_broadcast,
            logical_and,
            head_broadcast,
            pad_mask_value_id,
        )

        output_consumers = consumers.get(op.outputs[0], [])
        if len(output_consumers) != 1:
            raise ValidationError(
                "FlashInfer prefix attention output must have one transpose consumer"
            )
        transpose = output_consumers[0]
        if (
            transpose.opcode != "transpose"
            or transpose.invocation_id != op.invocation_id
            or transpose.attributes != (("permutation", (0, 2, 1, 3)),)
            or len(transpose.outputs) != 1
            or schedule.values[transpose.outputs[0]].type
            != _sig("bf16", (1, 968, 8, 256))
        ):
            raise ValidationError(
                "FlashInfer prefix attention requires the exact BHSd-to-BSHd transpose"
            )

        payload = (receipt.payload() if isinstance(receipt, BuildBinding)
                   else FlashInferPrefixAttentionPayload(receipt.problem)).to_bytes()
        fused_indices = (execution_index, transpose.execution_index)
        commands.append(
            FlashInferAttentionLoweredCommand(
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
                        CommandOperand(op.inputs[0], OperandAccess.READ),
                        CommandOperand(op.inputs[1], OperandAccess.READ),
                        CommandOperand(op.inputs[2], OperandAccess.READ),
                        CommandOperand(pad_mask_value_id, OperandAccess.READ),
                        CommandOperand(transpose.outputs[0], OperandAccess.WRITE),
                    ),
                    payload=payload,
                    workspace_bytes=capability.workspace_bytes,
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        fused.update(fused_indices)

    for key_broadcast, query_broadcast, logical_and, head_broadcast, _ in mask_chains.values():
        if any(
            consumer.execution_index not in supported
            for consumer in consumers.get(head_broadcast.outputs[0], [])
        ):
            raise ValidationError(
                "FlashInfer prefix head mask has an unsupported consumer"
            )
        fused.update(
            (
                key_broadcast.execution_index,
                query_broadcast.execution_index,
                logical_and.execution_index,
                head_broadcast.execution_index,
            )
        )

    elided = set(memory_plan.elided_ops)
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in fused and op.execution_index not in elided
    )
    memory_plan_sha256 = hashlib.sha256(
        (_canonical_json(memory_plan.to_dict()) + "\n").encode()
    ).hexdigest()
    return FlashInferAttentionPartialLowering(
        inventory_sha256,
        schedule_sha256,
        memory_plan_sha256,
        capabilities,
        tuple(commands),
        tuple(sorted(fused)),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_flashinfer_attention_partial_lowering(
    lowering: FlashInferAttentionPartialLowering,
) -> str:
    return _canonical_json(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"FlashInfer attention {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or not any(char != "0" for char in value)
    ):
        raise ValidationError(f"FlashInfer attention {label} must be a SHA-256 digest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
