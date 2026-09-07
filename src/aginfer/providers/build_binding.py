"""Explicit, unvalidated build descriptions, separate from measured receipts.

A binding permits compilation of a candidate. It never asserts correctness,
capture support, timing, or that a kernel was executed. Existing validation
receipts keep their stronger contracts.
"""
from dataclasses import dataclass
import hashlib

from ..errors import ValidationError
from ..lowering.capability import ProviderCapability


@dataclass(frozen=True, slots=True)
class BuildBinding:
    problems: tuple
    payloads: tuple
    provider_id: int = 2

    def __post_init__(self):
        if (not isinstance(self.problems, tuple) or not self.problems
                or not isinstance(self.payloads, tuple)
                or len(self.problems) != len(self.payloads)
                or len(set(self.problems)) != len(self.problems)
                or self.provider_id not in (2, 4, 5)):
            raise ValidationError("invalid candidate build binding")
        for problem, payload in zip(self.problems, self.payloads):
            if type(problem) is not type(self.problem) or problem.target_arch != self.target_arch:
                raise ValidationError("build binding mixes problem families or targets")
            # Every payload must survive its public, strict binary decoder.
            if type(payload).from_bytes(payload.to_bytes()) != payload:
                raise ValidationError("candidate payload is not canonical")
            if hasattr(payload, "problem") and payload.problem != problem:
                raise ValidationError("candidate problem differs from payload")
            from .rounded_attention import RoundedAttentionPayload
            from .flashinfer_attention import FlashInferAttentionProblem, FlashInferPrefixAttentionProblem
            from .aot_vision_attention import VisionAttentionProblem
            if isinstance(payload, RoundedAttentionPayload):
                expected = {1: FlashInferAttentionProblem, 2: FlashInferPrefixAttentionProblem,
                            3: VisionAttentionProblem, 4: VisionAttentionProblem}[payload.variant]
                if self.provider_id != 5 or type(problem) is not expected or payload.target_arch != self.target_arch:
                    raise ValidationError("materialized attention boundary differs from executable form")
            elif self.provider_id != (4 if payload.to_bytes()[:8] == b"AIPPA1\0\0" else 2):
                raise ValidationError("candidate provider ID differs from payload")
            if payload.module_bytes != self.module_bytes or payload.module_sha256 != self.module_sha256:
                raise ValidationError("build binding mixes module identities")

    @property
    def problem(self):
        return self.problems[0]

    @property
    def target_arch(self):
        return self.problem.target_arch

    @property
    def module_bytes(self):
        return self.payloads[0].module_bytes

    @property
    def module_sha256(self):
        return self.payloads[0].module_sha256

    def payload(self):
        if len(self.payloads) != 1:
            raise ValidationError("group binding requires an exact problem")
        return self.payloads[0]

    def capability_for(self, item):
        problem = type(self.problem).from_inventory(item, target_arch=self.target_arch)
        try:
            payload = self.payloads[self.problems.index(problem)]
        except ValueError as exc:
            raise ValidationError("candidate binding does not cover this exact problem") from exc
        return build_capability(item, payload, self.provider_id, self.target_arch)


def build_capability(item, payload, provider_id, target_arch):
    encoded = payload.to_bytes()
    return ProviderCapability.exact_for(
        item, provider_id=provider_id, abi_major=1, abi_minor=0,
        provider_version="candidate-build.v1",
        implementation_id="candidate." + encoded[:8].rstrip(b"\0").decode("ascii"),
        implementation_digest=hashlib.sha256(encoded).hexdigest(),
        target_arch=target_arch, supports_capture=False,
        workspace_bytes=getattr(payload, "workspace_bytes", 0),
    )


def is_build_binding(value, problem_type):
    return isinstance(value, BuildBinding) and all(type(p) is problem_type for p in value.problems)


@dataclass(frozen=True, slots=True)
class LinearBuildBinding:
    payload: object

    def __post_init__(self):
        from .cublaslt import CublasLtLinearPayload
        if not isinstance(self.payload, CublasLtLinearPayload):
            raise ValidationError("linear build binding needs a cuBLASLt payload")
        if CublasLtLinearPayload.from_bytes(self.payload.to_bytes()) != self.payload:
            raise ValidationError("linear candidate payload is noncanonical")

    def capability_for(self, item):
        from .cublaslt import CublasLtLinearProblem
        arch = self.payload.problem.target_arch
        if CublasLtLinearProblem.from_inventory(item, target_arch=arch) != self.payload.problem:
            raise ValidationError("linear binding does not cover this exact problem")
        return build_capability(item, self.payload, 1, arch)
