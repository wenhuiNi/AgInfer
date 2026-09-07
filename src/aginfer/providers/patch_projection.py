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
from .cublaslt import CublasLtDType, CublasLtLinearPayload, CublasLtLinearProblem


PATCH_PROJECTION_HEADER_MAGIC = b"AIPPA1\0\0"
PATCH_PROJECTION_HEADER = struct.Struct("<8sHH24I6Q32s4s")
PATCH_PROJECTION_PAYLOAD_SIZE = 384
PATCH_PROJECTION_RECEIPT_SCHEMA = "aginfer.patch-projection-validation.v1"
PATCH_PROJECTION_PARTIAL_LOWERING_SCHEMA = "aginfer.patch-projection-partial-lowering.v1"
PATCH_PROJECTION_IMPLEMENTATION = "aginfer.patch_projection.f32.nchw224.p14.v1"
PATCH_PROJECTION_PROVIDER_ID = 4
PATCH_MATRIX_BYTES = 256 * 588 * 4

assert PATCH_PROJECTION_HEADER.size == 192


def _sig(shape: tuple[int, ...]) -> TensorSignature:
    return TensorSignature("f32", shape, "row_major", "cuda")


_IMAGE = _sig((1, 3, 224, 224))
_WEIGHT = _sig((1152, 3, 14, 14))
_BIAS = _sig((1152,))
_NCHW = _sig((1, 1152, 16, 16))
_NHWC = _sig((1, 16, 16, 1152))
_GEMM_PROBLEM = CublasLtLinearProblem(
    CudaArch.SM120, CublasLtDType.F32, 256, 1152, 588
)


@dataclass(frozen=True, slots=True)
class PatchProjectionProblem:
    target_arch: CudaArch

    def __post_init__(self) -> None:
        if self.target_arch != CudaArch.SM120:
            raise ValidationError("patch projection exact variant requires SM120")

    @classmethod
    def from_inventory(
        cls, item: InventoryOp, *, target_arch: CudaArch
    ) -> "PatchProjectionProblem":
        if (
            not isinstance(item, InventoryOp)
            or item.status != RequirementStatus.REQUIRED
            or item.lowering_kind != LoweringKind.GEMM
            or item.opcode != "conv2d"
            or item.input_types != (_IMAGE, _WEIGHT, _BIAS)
            or item.output_types != (_NCHW,)
            or item.attributes
            != (("pads", (0, 0, 0, 0)), ("strides", (14, 14)))
        ):
            raise ValidationError(
                "patch projection anchor is outside the delivered exact envelope"
            )
        return cls(target_arch)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_arch": self.target_arch.name_string,
            "input": [1, 3, 224, 224],
            "weight": [1152, 3, 14, 14],
            "patch": [14, 14],
            "output": [1, 16, 16, 1152],
        }


@dataclass(frozen=True, slots=True)
class PatchProjectionPayload:
    problem: PatchProjectionProblem
    module_bytes: int
    module_sha256: str
    cublaslt: CublasLtLinearPayload

    def __post_init__(self) -> None:
        if not isinstance(self.problem, PatchProjectionProblem):
            raise ValidationError("patch projection payload problem is invalid")
        _positive(self.module_bytes, "module bytes")
        _digest(self.module_sha256, "module SHA-256")
        if (
            not isinstance(self.cublaslt, CublasLtLinearPayload)
            or self.cublaslt.problem != _GEMM_PROBLEM
        ):
            raise ValidationError("patch projection needs its exact F32 cuBLASLt problem")

    @property
    def workspace_bytes(self) -> int:
        return PATCH_MATRIX_BYTES + self.cublaslt.workspace_bytes

    def to_bytes(self) -> bytes:
        header = PATCH_PROJECTION_HEADER.pack(
            PATCH_PROJECTION_HEADER_MAGIC,
            1,
            0,
            int(self.problem.target_arch),
            1,
            1,
            256,
            1,
            1,
            256,
            1,
            1,
            4,
            4,
            4,
            4,
            256,
            1,
            3,
            224,
            224,
            14,
            14,
            1152,
            16,
            16,
            0,
            602112,
            2709504,
            4608,
            1179648,
            PATCH_MATRIX_BYTES,
            self.module_bytes,
            bytes.fromhex(self.module_sha256),
            bytes(4),
        )
        return header + self.cublaslt.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> "PatchProjectionPayload":
        view = memoryview(data)
        if len(view) != PATCH_PROJECTION_PAYLOAD_SIZE:
            raise FormatError("patch projection payload must have its exact fixed size")
        fields = PATCH_PROJECTION_HEADER.unpack(view[:192])
        fixed = (
            120, 1, 1, 256, 1, 1, 256, 1, 1, 4, 4, 4, 4, 256,
            1, 3, 224, 224, 14, 14, 1152, 16, 16, 0,
        )
        if (
            fields[0] != PATCH_PROJECTION_HEADER_MAGIC
            or fields[1:3] != (1, 0)
            or fields[3:27] != fixed
            or fields[27:32]
            != (602112, 2709504, 4608, 1179648, PATCH_MATRIX_BYTES)
            or fields[32] <= 0
            or not any(fields[33])
            or any(fields[34])
        ):
            raise FormatError("patch projection payload is outside the delivered exact variant")
        try:
            parsed = cls(
                PatchProjectionProblem(CudaArch(fields[3])),
                fields[32],
                fields[33].hex(),
                CublasLtLinearPayload.from_bytes(view[192:]),
            )
        except (ValueError, ValidationError) as exc:
            raise FormatError(str(exc)) from exc
        if bytes(view) != parsed.to_bytes():
            raise FormatError("patch projection payload is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class PatchProjectionValidationReceipt:
    problem: PatchProjectionProblem
    implementation_sha256: str
    implementation_bytes: int
    cuda_compiler_version: int
    cuda_runtime_version: int
    cuda_driver_version: int
    cublaslt_payload: CublasLtLinearPayload
    normal_launches: int
    normal_repeat_bit_exact: bool
    capture_replay_bit_exact: bool
    capture_matches_normal: bool
    full_output_cosine: float
    full_output_max_abs: float
    full_output_p99_abs: float
    patch_order_negative_detected: bool
    racecheck_hazards: int
    latency_ms: float
    original_chain_latency_ms: float
    contains_ptx: bool

    def __post_init__(self) -> None:
        if not isinstance(self.problem, PatchProjectionProblem):
            raise ValidationError("patch projection receipt problem is invalid")
        _digest(self.implementation_sha256, "implementation SHA-256")
        for value, label in (
            (self.implementation_bytes, "implementation bytes"),
            (self.cuda_compiler_version, "CUDA compiler version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.cuda_driver_version, "CUDA driver version"),
        ):
            _positive(value, label)
        if (
            not isinstance(self.cublaslt_payload, CublasLtLinearPayload)
            or self.cublaslt_payload.problem != _GEMM_PROBLEM
        ):
            raise ValidationError("patch projection receipt has the wrong cuBLASLt problem")
        if self.normal_launches < 2:
            raise ValidationError("patch projection receipt requires two normal launches")
        for value, label in (
            (self.normal_repeat_bit_exact, "normal repeat"),
            (self.capture_replay_bit_exact, "capture replay"),
            (self.capture_matches_normal, "capture/normal equality"),
            (self.patch_order_negative_detected, "patch-order negative control"),
        ):
            if value is not True:
                raise ValidationError(f"patch projection receipt requires {label}")
        if (
            not math.isfinite(self.full_output_cosine)
            or self.full_output_cosine < 0.999999
            or not math.isfinite(self.full_output_max_abs)
            or self.full_output_max_abs < 0
            or self.full_output_max_abs > 5e-4
            or not math.isfinite(self.full_output_p99_abs)
            or self.full_output_p99_abs < 0
            or self.full_output_p99_abs > 2e-4
        ):
            raise ValidationError("patch projection full-output evidence misses its gate")
        if self.racecheck_hazards != 0:
            raise ValidationError("patch projection receipt requires zero racecheck hazards")
        if (
            not math.isfinite(self.latency_ms)
            or not math.isfinite(self.original_chain_latency_ms)
            or self.latency_ms <= 0
            or self.original_chain_latency_ms <= self.latency_ms
        ):
            raise ValidationError("patch projection latency evidence is invalid")
        if self.contains_ptx is not False:
            raise ValidationError("patch projection implementation must not contain PTX")

    @property
    def implementation_digest(self) -> str:
        return hashlib.sha256(
            bytes.fromhex(self.implementation_sha256)
            + self.cublaslt_payload.to_bytes()
        ).hexdigest()

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()

    def payload(self) -> PatchProjectionPayload:
        return PatchProjectionPayload(
            self.problem,
            self.implementation_bytes,
            self.implementation_sha256,
            self.cublaslt_payload,
        )

    def capability_for(self, item: InventoryOp) -> ProviderCapability:
        PatchProjectionProblem.from_inventory(item, target_arch=self.problem.target_arch)
        return ProviderCapability.exact_for(
            item,
            provider_id=PATCH_PROJECTION_PROVIDER_ID,
            abi_major=1,
            abi_minor=0,
            provider_version=(
                f"aginfer-aot-cuda=1;cublaslt={self.cublaslt_payload.cublaslt_version};"
                f"cuda-runtime={self.cuda_runtime_version};driver={self.cuda_driver_version}"
            ),
            implementation_id=PATCH_PROJECTION_IMPLEMENTATION,
            implementation_digest=self.implementation_digest,
            target_arch=self.problem.target_arch,
            supports_capture=True,
            workspace_bytes=self.payload().workspace_bytes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PATCH_PROJECTION_RECEIPT_SCHEMA,
            "problem": self.problem.to_dict(),
            "implementation_sha256": self.implementation_sha256,
            "implementation_bytes": self.implementation_bytes,
            "implementation_digest": self.implementation_digest,
            "cuda_compiler_version": self.cuda_compiler_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "cublaslt_payload_sha256": hashlib.sha256(
                self.cublaslt_payload.to_bytes()
            ).hexdigest(),
            "cublaslt_version": self.cublaslt_payload.cublaslt_version,
            "cublaslt_workspace_bytes": self.cublaslt_payload.workspace_bytes,
            "workspace_bytes": self.payload().workspace_bytes,
            "normal_launches": self.normal_launches,
            "normal_repeat_bit_exact": self.normal_repeat_bit_exact,
            "capture_replay_bit_exact": self.capture_replay_bit_exact,
            "capture_matches_normal": self.capture_matches_normal,
            "full_output_cosine": self.full_output_cosine,
            "full_output_max_abs": self.full_output_max_abs,
            "full_output_p99_abs": self.full_output_p99_abs,
            "patch_order_negative_detected": self.patch_order_negative_detected,
            "racecheck_hazards": self.racecheck_hazards,
            "latency_ms": self.latency_ms,
            "original_chain_latency_ms": self.original_chain_latency_ms,
            "speedup": self.original_chain_latency_ms / self.latency_ms,
            "contains_ptx": self.contains_ptx,
        }


@dataclass(frozen=True, slots=True)
class PatchProjectionLoweredCommand:
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
            "workspace_bytes": self.command.workspace_bytes,
            "capture_safe": self.command.capture_safe,
        }


@dataclass(frozen=True, slots=True)
class PatchProjectionPartialLowering:
    inventory_sha256: str
    schedule_sha256: str
    memory_plan_sha256: str
    capabilities: tuple[ProviderCapability, ...]
    commands: tuple[PatchProjectionLoweredCommand, ...]
    fused_execution_indices: tuple[int, ...]
    elided_execution_indices: tuple[int, ...]
    unhandled_execution_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.unhandled_execution_indices

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PATCH_PROJECTION_PARTIAL_LOWERING_SCHEMA,
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
                "workspace_bytes": max(
                    (item.command.workspace_bytes for item in self.commands), default=0
                ),
            },
            "capability_registry_sha256": hashlib.sha256(
                dump_provider_capabilities(self.capabilities).encode()
            ).hexdigest(),
            "commands": [item.to_dict() for item in self.commands],
            "fused_execution_indices": list(self.fused_execution_indices),
            "elided_execution_indices": list(self.elided_execution_indices),
            "unhandled_execution_indices": list(self.unhandled_execution_indices),
        }


def make_patch_projection_capabilities(
    inventory: LoweringInventory,
    receipt: PatchProjectionValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> tuple[ProviderCapability, ...]:
    if not (isinstance(receipt, PatchProjectionValidationReceipt) or is_build_binding(receipt, PatchProjectionProblem)):
        raise ValidationError("patch projection receipt is invalid")
    if receipt.problem.target_arch != target_arch:
        raise ValidationError("patch projection receipt target differs from capability target")
    result: dict[str, ProviderCapability] = {}
    for item in inventory.ops:
        try:
            PatchProjectionProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        capability = receipt.capability_for(item)
        result.setdefault(capability.digest, capability)
    if not result:
        raise ValidationError("patch projection receipt matched no inventory anchor")
    return tuple(sorted(result.values(), key=lambda item: item.digest))


def lower_patch_projection_commands(
    schedule: ExecutionSchedule,
    inventory: LoweringInventory,
    memory_plan: MemoryPlan,
    receipt: PatchProjectionValidationReceipt | BuildBinding,
    *,
    target_arch: CudaArch,
) -> PatchProjectionPartialLowering:
    inventory_sha = hashlib.sha256(dump_lowering_inventory(inventory).encode()).hexdigest()
    schedule_sha = hashlib.sha256(dump_execution_schedule(schedule).encode()).hexdigest()
    if schedule.inventory_sha256 != inventory_sha or memory_plan.schedule_sha256 != schedule_sha:
        raise ValidationError("patch projection stage identities do not match")
    capabilities = make_patch_projection_capabilities(
        inventory, receipt, target_arch=target_arch
    )
    by_site = {item.site_id: item for item in inventory.ops}
    consumers: dict[int, list[ScheduledOp]] = {}
    for op in schedule.ops:
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(op)
    entry_outputs = {value_id for _, value_id in schedule.entry_outputs}
    commands: list[PatchProjectionLoweredCommand] = []
    fused: set[int] = set()
    for conv in schedule.ops:
        item = by_site.get(conv.site)
        if item is None:
            continue
        try:
            PatchProjectionProblem.from_inventory(item, target_arch=target_arch)
        except ValidationError:
            continue
        if len(conv.inputs) != 3 or len(conv.outputs) != 1:
            raise ValidationError("patch projection conv arity is invalid")
        next_ops = consumers.get(conv.outputs[0], [])
        if len(next_ops) != 1:
            raise ValidationError("patch projection conv output is not exclusive")
        transpose = next_ops[0]
        if (
            transpose.invocation_id != conv.invocation_id
            or transpose.opcode != "transpose"
            or transpose.attributes != (("permutation", (0, 2, 3, 1)),)
            or len(transpose.outputs) != 1
            or schedule.values[transpose.outputs[0]].type != _NHWC
            or conv.outputs[0] in entry_outputs
        ):
            raise ValidationError("patch projection output layout chain is not exact")
        indices = (conv.execution_index, transpose.execution_index)
        if fused.intersection(indices):
            raise ValidationError("patch projection regions overlap")
        capability = receipt.capability_for(item)
        commands.append(
            PatchProjectionLoweredCommand(
                conv.execution_index,
                conv.site,
                indices,
                ProviderCommand(
                    CommandTag.CUBLASLT_MATMUL,
                    capability.provider_id,
                    capability.abi_major,
                    capability.abi_minor,
                    capability.digest,
                    tuple(
                        CommandOperand(value_id, OperandAccess.READ)
                        for value_id in conv.inputs
                    )
                    + (CommandOperand(transpose.outputs[0], OperandAccess.WRITE),),
                    receipt.payload().to_bytes(),
                    workspace_offset=0,
                    workspace_bytes=receipt.payload().workspace_bytes,
                    capture_safe=capability.supports_capture,
                ),
            )
        )
        fused.update(indices)
    if not commands:
        raise ValidationError("patch projection found no complete region")
    elided = set(memory_plan.elided_ops)
    if fused & elided:
        raise ValidationError("patch projection overlaps a memory-elided op")
    unhandled = tuple(
        op.execution_index
        for op in schedule.ops
        if op.execution_index not in fused and op.execution_index not in elided
    )
    memory_sha = hashlib.sha256((_canonical(memory_plan.to_dict()) + "\n").encode()).hexdigest()
    return PatchProjectionPartialLowering(
        inventory_sha,
        schedule_sha,
        memory_sha,
        capabilities,
        tuple(commands),
        tuple(sorted(fused)),
        tuple(sorted(elided)),
        unhandled,
    )


def dump_patch_projection_partial_lowering(
    lowering: PatchProjectionPartialLowering,
) -> str:
    return _canonical(lowering.to_dict()) + "\n"


def _positive(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"patch projection {label} must be positive")


def _digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or not any(character != "0" for character in value)
    ):
        raise ValidationError(f"patch projection {label} must be a SHA-256 digest")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
