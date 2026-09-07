"""Offline container, executable, payload and build-identity verification."""
import hashlib
import json

from .identity import digest, file_identity
from .pipeline import validate_kernel_record
from .. import providers as p
from ..aim import AimReader
from ..errors import FormatError, ValidationError
from ..executable import parse_executable_plan, EXECUTABLE_PLAN_MAGIC
from ..lowering.command import CommandTag
from ..providers.rounded_attention import RoundedAttentionPayload
from ..providers.projection_split import ProjectionSplitPayload
from ..providers.gelu_mul import GeluMulPayload
from ..providers.state_update import StateUpdatePayload


def decode_command_payload(command):
    kinds = {
        1: (p.CublasLtLinearPayload,),
        2: (p.CudaKernelPayload, p.LayerNormPayload, p.RmsNormPayload, p.RopePayload,
            p.AdaptiveRmsNormPayload, p.KvPackPayload, p.PrefixKvStorePayload,
            p.PrefixInputPayload, p.SuffixMetadataPayload, p.TimeEmbeddingPayload,
            p.ActionSlicePayload, p.VisionAttentionPayload, ProjectionSplitPayload, StateUpdatePayload, GeluMulPayload),
        3: (p.FlashInferAttentionPayload, p.FlashInferPrefixAttentionPayload),
        4: (p.PatchProjectionPayload,),
        5: (RoundedAttentionPayload,),
    }
    matches = []
    for cls in kinds.get(command.provider_id, ()):
        try:
            matches.append(cls.from_bytes(command.payload))
        except (FormatError, ValidationError, ValueError):
            continue
    if len(matches) != 1:
        raise ValidationError("unknown, malformed or ambiguous provider payload")
    payload = matches[0]
    attention = isinstance(payload, (p.VisionAttentionPayload, p.FlashInferAttentionPayload,
        p.FlashInferPrefixAttentionPayload, RoundedAttentionPayload))
    expected_tag = CommandTag.ATTENTION if attention else CommandTag.CUBLASLT_MATMUL if command.provider_id in (1, 4) else CommandTag.CUDA_KERNEL
    if command.tag != expected_tag or (command.abi_major, command.abi_minor) != (1, 0):
        raise ValidationError("command provider ABI/tag differs from executable form")
    if command.workspace_bytes != getattr(payload, "workspace_bytes", 0):
        raise ValidationError("command workspace differs from provider payload")
    return payload


def verify_artifact(path, *, require_build_record=False):
    info = AimReader.read(path)
    build = info.manifest.get("build")
    if require_build_record and build is None:
        raise ValidationError("artifact has no compiler build record")
    if build is not None:
        if (not isinstance(build, dict) or build.get("schema") != "aginfer.candidate-build.v1"
                or info.manifest.get("build_sha256") != digest(build)):
            raise ValidationError("compiler build identity is invalid")
        if len(info.variants) != 1:
            raise ValidationError("candidate build currently requires one exact variant")
        compiler = build.get("compiler", {})
        if (not isinstance(compiler, dict) or not isinstance(compiler.get("files"), dict)
                or not compiler["files"] or compiler.get("sha256") != digest(compiler["files"])):
            raise ValidationError("compiler source identity is invalid")
        if not isinstance(build.get("capabilities"), dict):
            raise ValidationError("compiler capability registry is invalid")
    results = []
    with open(path, "rb") as f:
        for variant in info.variants:
            if variant.plan.size > 512 * 1024 * 1024 or variant.kernels.size > 256 * 1024 * 1024:
                raise ValidationError("executable metadata exceeds verification limit")
            f.seek(variant.plan.offset)
            data = f.read(variant.plan.size)
            if not data.startswith(EXECUTABLE_PLAN_MAGIC):
                raise ValidationError("verify requires a fully parsed execution-plan v2")
            plan = parse_executable_plan(data)
            if plan.target_arch != variant.arch or plan.weights_bytes != variant.weights.size:
                raise ValidationError("AIM variant differs from executable plan")
            if build is not None:
                f.seek(variant.kernels.offset)
                module = f.read(variant.kernels.size)
                validate_kernel_record(build.get("kernel_build"), module)
                if "algorithm_selection" in build:
                    from .selection import validate_selection_report
                    validate_selection_report(build["algorithm_selection"], build.get("fixed_algorithms"),
                        module, build.get("program_sha256"))
                if (build.get("plan_sha256") != hashlib.sha256(data).hexdigest()
                        or build.get("command_stream_sha256") != plan.command_stream_sha256
                        or build.get("schedule_sha256") != plan.schedule_sha256
                        or build.get("memory_sha256") != plan.memory_plan_sha256
                        or build.get("weights") != {"bytes": variant.weights.size, "sha256": variant.weights.sha256.hex()}):
                    raise ValidationError("compiler build does not bind its executable sections")
                if "constant_folding" in build:
                    from .constant_folding import validate_folding_report
                    fold = build["constant_folding"]
                    validate_folding_report(fold, plan)
                    computed_digest = hashlib.sha256()
                    for entry in fold["values"]:
                        value = plan.values[entry["value_id"]]
                        f.seek(variant.weights.offset + value.offset)
                        computed_digest.update(f.read(value.byte_size))
                    if computed_digest.hexdigest() != fold["output_sha256"]:
                        raise ValidationError("folded constant payload digest differs from evaluator output")
            decoded = {}
            used = set()
            for command in plan.command_stream.commands:
                key = (command.provider_id, command.tag, command.abi_major, command.abi_minor, command.workspace_bytes, command.payload)
                if key not in decoded:
                    decoded[key] = decode_command_payload(command)
                payload = decoded[key]
                if hasattr(payload, "module_sha256") and (payload.module_sha256 != variant.kernels.sha256.hex() or payload.module_bytes != variant.kernels.size):
                    raise ValidationError("command references a different AOT module")
                arch = payload.target_arch if isinstance(payload, RoundedAttentionPayload) else payload.problem.target_arch
                if arch != variant.arch:
                    raise ValidationError("provider payload target differs from AIM variant")
                if build is not None:
                    cap = build.get("capabilities", {}).get(command.capability_digest)
                    if not isinstance(cap, dict):
                        raise ValidationError("command lacks its capability build record")
                    cap_hash = hashlib.sha256(json.dumps(cap, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
                    if (cap_hash != command.capability_digest
                            or cap.get("implementation_digest") != hashlib.sha256(command.payload).hexdigest()
                            or cap.get("provider_id") != command.provider_id
                            or cap.get("abi_major") != command.abi_major
                            or cap.get("abi_minor") != command.abi_minor
                            or cap.get("target_arch") != variant.arch.name_string
                            or cap.get("supports_capture") is not command.capture_safe
                            or cap.get("workspace_bytes") != command.workspace_bytes):
                        raise ValidationError("capability does not bind the command payload/contract")
                    used.add(command.capability_digest)
            if build is not None and used != set(build.get("capabilities", {})):
                raise ValidationError("build contains missing or unused capabilities")
            results.append({"arch": variant.arch.name_string, "commands": len(plan.command_stream.commands),
                "inputs": sum(int(x.kind) == 1 for x in plan.ports),
                "outputs": sum(int(x.kind) == 2 for x in plan.ports),
                "arena_bytes": plan.arena_bytes, "workspace_bytes": plan.workspace_bytes,
                "providers": {str(x.provider_id): x.command_count for x in plan.providers}})
    return {"status": "structurally_verified", "file": file_identity(path),
        "aim_checksum": info.file_sha256, "build_sha256": info.manifest.get("build_sha256"),
        "numerical_validation": "not_performed_by_verify", "variants": results}
