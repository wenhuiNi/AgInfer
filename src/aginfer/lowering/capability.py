from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ..errors import ValidationError
from ..schema import CudaArch
from .inventory import (
    InventoryOp,
    LoweringInventory,
    LoweringKind,
    RequirementStatus,
    TensorSignature,
)


PROVIDER_RESOLUTION_SCHEMA = "aginfer.provider-resolution.v2"


@dataclass(frozen=True, slots=True)
class ProviderCapability:
    provider_id: int
    abi_major: int
    abi_minor: int
    provider_version: str
    implementation_id: str
    implementation_digest: str
    target_arch: CudaArch
    lowering_kind: LoweringKind
    opcode: str
    input_types: tuple[TensorSignature, ...]
    output_types: tuple[TensorSignature, ...]
    attributes: tuple[tuple[str, object], ...]
    supports_capture: bool
    workspace_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.provider_id, int)
            or isinstance(self.provider_id, bool)
            or not 0 < self.provider_id <= 2**32 - 1
        ):
            raise ValidationError("provider capability provider_id must be a positive uint32")
        for value, label, minimum in (
            (self.abi_major, "abi_major", 1),
            (self.abi_minor, "abi_minor", 0),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= 2**32 - 1
            ):
                raise ValidationError(f"provider capability {label} is out of range")
        _text(self.provider_version, "provider_version")
        _text(self.implementation_id, "implementation_id")
        _digest(self.implementation_digest, "implementation_digest")
        _text(self.opcode, "opcode")
        if not isinstance(self.target_arch, CudaArch):
            raise ValidationError("provider capability target_arch must be a CudaArch")
        if self.lowering_kind not in {
            LoweringKind.MEMORY,
            LoweringKind.GEMM,
            LoweringKind.ATTENTION,
            LoweringKind.AOT_CUDA,
        }:
            raise ValidationError("provider capability lowering_kind must name a runtime provider class")
        if not all(isinstance(item, TensorSignature) for item in self.input_types + self.output_types):
            raise ValidationError("provider capability tensor signatures are invalid")
        if any(item.device != "cuda" for item in self.input_types + self.output_types):
            raise ValidationError("provider capability only accepts exact CUDA tensor signatures")
        keys = [key for key, _ in self.attributes]
        if any(not isinstance(key, str) or not key or "\0" in key for key in keys):
            raise ValidationError("provider capability attributes contain an invalid name")
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValidationError("provider capability attributes must be uniquely and canonically sorted")
        if not isinstance(self.supports_capture, bool):
            raise ValidationError("provider capability supports_capture must be boolean")
        if (
            not isinstance(self.workspace_bytes, int)
            or isinstance(self.workspace_bytes, bool)
            or not 0 <= self.workspace_bytes <= 2**64 - 1
        ):
            raise ValidationError("provider capability workspace_bytes must be a uint64")

    @classmethod
    def exact_for(
        cls,
        item: InventoryOp,
        *,
        provider_id: int,
        abi_major: int,
        abi_minor: int,
        provider_version: str,
        implementation_id: str,
        implementation_digest: str,
        target_arch: CudaArch,
        supports_capture: bool,
        workspace_bytes: int,
    ) -> "ProviderCapability":
        if item.status != RequirementStatus.REQUIRED:
            raise ValidationError("provider capability can only be built for a required inventory site")
        return cls(
            provider_id=provider_id,
            abi_major=abi_major,
            abi_minor=abi_minor,
            provider_version=provider_version,
            implementation_id=implementation_id,
            implementation_digest=implementation_digest,
            target_arch=target_arch,
            lowering_kind=item.lowering_kind,
            opcode=item.opcode,
            input_types=item.input_types,
            output_types=item.output_types,
            attributes=item.attributes,
            supports_capture=supports_capture,
            workspace_bytes=workspace_bytes,
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "abi_major": self.abi_major,
            "abi_minor": self.abi_minor,
            "provider_version": self.provider_version,
            "implementation_id": self.implementation_id,
            "implementation_digest": self.implementation_digest,
            "target_arch": self.target_arch.name_string,
            "lowering_kind": self.lowering_kind.value,
            "opcode": self.opcode,
            "input_types": [item.to_dict() for item in self.input_types],
            "output_types": [item.to_dict() for item in self.output_types],
            "attributes": {key: value for key, value in self.attributes},
            "supports_capture": self.supports_capture,
            "workspace_bytes": self.workspace_bytes,
        }


@dataclass(frozen=True, slots=True)
class ProviderReceipt:
    site: str
    executions: int
    capability_digest: str
    provider_id: int
    abi_major: int
    abi_minor: int
    provider_version: str
    implementation_id: str
    implementation_digest: str
    supports_capture: bool
    workspace_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "executions": self.executions,
            "capability_digest": self.capability_digest,
            "provider_id": self.provider_id,
            "abi_major": self.abi_major,
            "abi_minor": self.abi_minor,
            "provider_version": self.provider_version,
            "implementation_id": self.implementation_id,
            "implementation_digest": self.implementation_digest,
            "supports_capture": self.supports_capture,
            "workspace_bytes": self.workspace_bytes,
        }


@dataclass(frozen=True, slots=True)
class ResolutionIssue:
    site: str
    opcode: str
    executions: int
    kind: str
    detail: str
    candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "opcode": self.opcode,
            "executions": self.executions,
            "kind": self.kind,
            "detail": self.detail,
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True, slots=True)
class ProviderResolution:
    inventory_sha256: str
    program_sha256: str
    target_arch: CudaArch
    receipts: tuple[ProviderReceipt, ...]
    issues: tuple[ResolutionIssue, ...]

    @property
    def complete(self) -> bool:
        return not self.issues

    def require_complete(self) -> None:
        if not self.issues:
            return
        expanded = sum(item.executions for item in self.issues)
        first = self.issues[0]
        raise ValidationError(
            f"provider resolution incomplete: {len(self.issues)} static sites/{expanded} executions; "
            f"first {first.site} {first.kind}: {first.detail}"
        )

    def to_dict(self) -> dict[str, object]:
        issue_kinds = Counter(item.kind for item in self.issues)
        return {
            "schema": PROVIDER_RESOLUTION_SCHEMA,
            "inventory_sha256": self.inventory_sha256,
            "program_sha256": self.program_sha256,
            "target_arch": self.target_arch.name_string,
            "complete": self.complete,
            "summary": {
                "resolved_static_sites": len(self.receipts),
                "resolved_executions": sum(item.executions for item in self.receipts),
                "issue_static_sites": len(self.issues),
                "issue_executions": sum(item.executions for item in self.issues),
                "issues_by_kind": dict(sorted(issue_kinds.items())),
            },
            "receipts": [item.to_dict() for item in self.receipts],
            "issues": [item.to_dict() for item in self.issues],
        }


def resolve_provider_capabilities(
    inventory: LoweringInventory,
    capabilities: Iterable[ProviderCapability],
    *,
    target_arch: CudaArch,
    strict: bool = True,
) -> ProviderResolution:
    if not isinstance(inventory, LoweringInventory):
        raise ValidationError("provider resolution requires a LoweringInventory")
    if not isinstance(target_arch, CudaArch):
        raise ValidationError("provider resolution target_arch must be a CudaArch")
    if not isinstance(strict, bool):
        raise ValidationError("provider resolution strict must be boolean")
    capability_records = tuple(capabilities)
    if not all(isinstance(item, ProviderCapability) for item in capability_records):
        raise ValidationError("provider capability registry contains an invalid record")
    digests = [item.digest for item in capability_records]
    if len(digests) != len(set(digests)):
        raise ValidationError("provider capability registry contains a duplicate record")
    ordered_capabilities = tuple(sorted(capability_records, key=lambda item: item.digest))

    receipts: list[ProviderReceipt] = []
    issues: list[ResolutionIssue] = []
    for item in inventory.ops:
        if item.status == RequirementStatus.META:
            continue
        if item.status == RequirementStatus.BLOCKED:
            issues.append(
                ResolutionIssue(
                    item.site_id,
                    item.opcode,
                    item.executions,
                    "inventory_blocked",
                    item.blocker or "inventory site is blocked",
                )
            )
            continue
        matches = tuple(
            capability
            for capability in ordered_capabilities
            if _matches(item, capability, target_arch)
        )
        if not matches:
            issues.append(
                ResolutionIssue(
                    item.site_id,
                    item.opcode,
                    item.executions,
                    "unresolved",
                    _requirement_detail(item, target_arch),
                )
            )
            continue
        if len(matches) > 1:
            issues.append(
                ResolutionIssue(
                    item.site_id,
                    item.opcode,
                    item.executions,
                    "ambiguous",
                    "multiple exact provider capabilities match",
                    tuple(capability.digest for capability in matches),
                )
            )
            continue
        capability = matches[0]
        receipts.append(
            ProviderReceipt(
                site=item.site_id,
                executions=item.executions,
                capability_digest=capability.digest,
                provider_id=capability.provider_id,
                abi_major=capability.abi_major,
                abi_minor=capability.abi_minor,
                provider_version=capability.provider_version,
                implementation_id=capability.implementation_id,
                implementation_digest=capability.implementation_digest,
                supports_capture=capability.supports_capture,
                workspace_bytes=capability.workspace_bytes,
            )
        )

    inventory_dump = _canonical_json(inventory.to_dict()) + "\n"
    resolution = ProviderResolution(
        inventory_sha256=hashlib.sha256(inventory_dump.encode("utf-8")).hexdigest(),
        program_sha256=inventory.program_sha256,
        target_arch=target_arch,
        receipts=tuple(receipts),
        issues=tuple(issues),
    )
    if strict:
        resolution.require_complete()
    return resolution


def dump_provider_capabilities(capabilities: Iterable[ProviderCapability]) -> str:
    records = tuple(capabilities)
    if not all(isinstance(item, ProviderCapability) for item in records):
        raise ValidationError("provider capability dump contains an invalid record")
    digests = [item.digest for item in records]
    if len(digests) != len(set(digests)):
        raise ValidationError("provider capability dump contains a duplicate record")
    payload = {
        "schema": "aginfer.provider-capabilities.v2",
        "capabilities": [
            {"digest": item.digest, **item.to_dict()}
            for item in sorted(records, key=lambda item: item.digest)
        ],
    }
    return _canonical_json(payload) + "\n"


def dump_provider_resolution(resolution: ProviderResolution) -> str:
    if not isinstance(resolution, ProviderResolution):
        raise ValidationError("provider resolution dump requires a ProviderResolution")
    return _canonical_json(resolution.to_dict()) + "\n"


def _matches(
    item: InventoryOp,
    capability: ProviderCapability,
    target_arch: CudaArch,
) -> bool:
    return (
        capability.target_arch == target_arch
        and capability.lowering_kind == item.lowering_kind
        and capability.opcode == item.opcode
        and capability.input_types == item.input_types
        and capability.output_types == item.output_types
        and capability.attributes == item.attributes
    )


def _requirement_detail(item: InventoryOp, target_arch: CudaArch) -> str:
    inputs = ",".join(_signature_text(value) for value in item.input_types)
    outputs = ",".join(_signature_text(value) for value in item.output_types)
    attributes = _canonical_json({key: value for key, value in item.attributes})
    return (
        f"needs exact {item.lowering_kind.value}/{item.opcode} capability for "
        f"{target_arch.name_string}; inputs=[{inputs}] outputs=[{outputs}] attrs={attributes}"
    )


def _signature_text(signature: TensorSignature) -> str:
    shape = "x".join(str(value) for value in signature.shape)
    return f"{signature.dtype}[{shape}]<{signature.layout},{signature.device}>"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or "\0" in value or len(value.encode("utf-8")) > 4096:
        raise ValidationError(f"provider capability {label} must be bounded non-empty text")


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationError(f"provider capability {label} must be a canonical SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValidationError(
            f"provider capability {label} must be a canonical SHA-256 digest"
        ) from exc
    if len(decoded) != 32 or not any(decoded) or value != decoded.hex():
        raise ValidationError(f"provider capability {label} must be a canonical SHA-256 digest")
