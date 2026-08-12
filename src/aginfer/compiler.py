from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import create_adapter
from .checkpoint import Checkpoint, collect_metadata, load_checkpoint, pack_weights
from .aim import AimInfo, AimWriter, VariantPayload
from .errors import ValidationError
from .profiles import load_profile
from .schema import RUNTIME_ABI, CudaArch, Platform, ToolchainRange, validate_target

@dataclass(frozen=True)
class CompileOptions:
    source: str
    revision: str | None
    adapter: str
    platform: Platform
    cuda_arches: tuple[CudaArch, ...]
    profile_path: Path
    artifact_dir: Path
    output: Path
    offline: bool = False
    backbone: str | None = None


def compile_model(options: CompileOptions) -> AimInfo:
    if options.output.suffix.lower() != ".aim":
        raise ValidationError("output filename must use the .aim extension")
    try:
        validate_target(options.platform, list(options.cuda_arches))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    profile = load_profile(options.profile_path)
    checkpoint = load_checkpoint(options.source, options.revision, options.offline)
    adapter = create_adapter(options.adapter)
    result = adapter.build(checkpoint, profile)

    backbone_checkpoint: Checkpoint | None = None
    all_tensors = list(checkpoint.tensors)
    if options.adapter == "lingbot_vla_1":
        if not options.backbone:
            raise ValidationError("lingbot_vla_1 requires --backbone so deployment has no QWEN25_PATH dependency")
        backbone_checkpoint = load_checkpoint(options.backbone, None, options.offline)
        duplicates = {tensor.name for tensor in all_tensors} & {tensor.name for tensor in backbone_checkpoint.tensors}
        renamed = []
        for tensor in backbone_checkpoint.tensors:
            if tensor.name in duplicates:
                tensor = type(tensor)(
                    f"backbone.{tensor.name}", tensor.dtype, tensor.shape, tensor.source, tensor.source_offset, tensor.byte_length
                )
            renamed.append(tensor)
        all_tensors.extend(renamed)

    weight_blob, tensor_table = pack_weights(all_tensors)
    variants: list[VariantPayload] = []
    tactic_summaries: dict[str, Any] = {}
    for arch in options.cuda_arches:
        artifact = options.artifact_dir / arch.name_string
        kernel_path = artifact / "kernels.cubin"
        plan_path = artifact / "plan.json"
        tactic_path = artifact / "tactics.json"
        kernels = _read_required(kernel_path, f"precompiled {arch.name_string} kernel bundle")
        plan = _load_object(plan_path, "execution plan")
        tactics = _load_object(tactic_path, "tactic database")
        _validate_plan(plan, arch)
        _validate_tactics(tactics, arch)
        tactic_summaries[arch.name_string] = {
            "database_sha256": _sha256_file(tactic_path),
            "entry_count": len(tactics["tactics"]),
        }
        variants.append(VariantPayload(arch, kernels, weight_blob, _canonical_json(plan)))

    toolchain = _toolchain_from_artifacts(options.artifact_dir)
    manifest = {
        "model_family": result.model_family,
        "adapter": options.adapter,
        "adapter_version": result.adapter_version,
        "source": options.source,
        "source_revision": checkpoint.revision,
        "backbone_revision": backbone_checkpoint.revision if backbone_checkpoint else None,
        "platform": options.platform.triple,
        "cuda_arches": [arch.name_string for arch in options.cuda_arches],
        "runtime_abi": RUNTIME_ABI,
        "toolchain": toolchain.to_dict(),
        "checkpoint_dtypes": sorted({tensor.dtype for tensor in all_tensors}),
        "strict_dtype_preservation": True,
        "tf32_default": False,
        "batch": 1,
        "profiles": profile["profiles"],
        "default_denoising": profile.get("default_denoising", {"steps": 1}),
        "inputs": result.inputs,
        "outputs": result.outputs,
        "checkpoint_metadata": collect_metadata(checkpoint.root),
        "tactics": tactic_summaries,
        "deployment_features": {
            "ptx": False,
            "jit": False,
            "autotune": False,
            "runtime_weight_repacking": False,
            "cpu_fallback": False,
        },
    }
    tensors = {"alignment": 256, "count": len(tensor_table), "items": tensor_table}
    return AimWriter.write(
        options.output,
        platform=options.platform,
        manifest=manifest,
        graph=result.graph,
        tensors=tensors,
        variants=variants,
    )


def _toolchain_from_artifacts(root: Path) -> ToolchainRange:
    value = _load_object(root / "toolchain.json", "toolchain compatibility file")
    try:
        result = ToolchainRange.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid toolchain.json: {exc}") from exc
    if result.cuda_runtime_min <= 0 or result.cuda_runtime_max < result.cuda_runtime_min:
        raise ValidationError("toolchain CUDA Runtime range is invalid")
    if min(result.cuda_driver_min, result.cublaslt_abi, result.cudnn_abi) <= 0:
        raise ValidationError("toolchain ABI values must be positive")
    return result


def _validate_plan(plan: dict[str, Any], arch: CudaArch) -> None:
    if plan.get("cuda_arch") != arch.name_string:
        raise ValidationError(f"execution plan target must be exactly {arch.name_string}")
    required = ("arena_bytes", "workspace_bytes", "shape_dispatch", "cuda_graph_templates")
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValidationError(f"{arch.name_string} execution plan is missing: {', '.join(missing)}")
    if int(plan["arena_bytes"]) < 0 or int(plan["workspace_bytes"]) < 0:
        raise ValidationError("execution plan memory sizes cannot be negative")
    if not isinstance(plan["shape_dispatch"], list) or not plan["shape_dispatch"]:
        raise ValidationError("execution plan needs a non-empty shape dispatch table")


def _validate_tactics(value: dict[str, Any], arch: CudaArch) -> None:
    if value.get("cuda_arch") != arch.name_string:
        raise ValidationError(f"tactic database target must be exactly {arch.name_string}")
    tactics = value.get("tactics")
    if not isinstance(tactics, list) or not tactics:
        raise ValidationError(f"no verified tactics available for {arch.name_string}")
    for index, tactic in enumerate(tactics):
        if not isinstance(tactic, dict) or not tactic.get("op") or not tactic.get("algorithm"):
            raise ValidationError(f"invalid tactic entry {index} for {arch.name_string}")


def _read_required(path: Path, label: str) -> bytes:
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not value:
        raise ValidationError(f"empty {label}: {path}")
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must contain a JSON object: {path}")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
