from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .errors import ValidationError
from .source_manifest import AssetRole, SourceAsset
from .source_package import SourcePackage


@dataclass(frozen=True, slots=True)
class FeatureContract:
    name: str
    kind: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProcessorStepContract:
    index: int
    registry_name: str
    config_json: str
    state_asset_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessorPipelineContract:
    namespace: str
    name: str
    source_asset_id: str
    steps: tuple[ProcessorStepContract, ...]


@dataclass(frozen=True, order=True, slots=True)
class ExternalAssetRequirement:
    kind: str
    locator: str
    requested_by: str


@dataclass(frozen=True, slots=True)
class SourceAssetContract:
    source_type: str
    model_config_asset_id: str
    input_features: tuple[FeatureContract, ...]
    output_features: tuple[FeatureContract, ...]
    pipelines: tuple[ProcessorPipelineContract, ...]
    external_requirements: tuple[ExternalAssetRequirement, ...]
    config_json: str


def inspect_source_asset_contract(source: SourcePackage) -> SourceAssetContract:
    """Parse model IO and processor assets without importing their framework."""

    metadata = source.assets.select(role=AssetRole.METADATA.value)
    parsed: list[tuple[SourceAsset, Any]] = []
    for asset in metadata:
        if not asset.path.endswith(".json") or asset.index_path is not None:
            continue
        parsed.append((asset, source.assets.read_json(asset.asset_id)))

    model_candidates = [
        (asset, data)
        for asset, data in parsed
        if asset.namespace == "model"
        and isinstance(data, dict)
        and ("input_features" in data or "output_features" in data)
    ]
    if len(model_candidates) != 1:
        raise ValidationError(
            f"source must contain exactly one feature-bearing model config, found {len(model_candidates)}"
        )
    model_asset, model_config = model_candidates[0]
    source_type = model_config.get("type", model_config.get("model_type"))
    if not isinstance(source_type, str) or not source_type:
        raise ValidationError("feature-bearing model config requires a non-empty type or model_type")
    input_features = _parse_features(model_config.get("input_features"), "input_features")
    output_features = _parse_features(model_config.get("output_features"), "output_features")

    pipelines: list[ProcessorPipelineContract] = []
    requirements: set[ExternalAssetRequirement] = set()
    for asset, data in parsed:
        structural_pipeline = (
            isinstance(data, dict)
            and isinstance(data.get("name"), str)
            and isinstance(data.get("steps"), list)
        )
        if asset.namespace not in {"preprocessor", "postprocessor"} and not structural_pipeline:
            continue
        if not isinstance(data, dict):
            raise ValidationError(f"processor asset {asset.path} must contain an object")
        pipeline, pipeline_requirements = _parse_pipeline(source, asset, data)
        pipelines.append(pipeline)
        requirements.update(pipeline_requirements)

    return SourceAssetContract(
        source_type=source_type,
        model_config_asset_id=model_asset.asset_id,
        input_features=input_features,
        output_features=output_features,
        pipelines=tuple(sorted(pipelines, key=lambda item: (item.namespace, item.name))),
        external_requirements=tuple(sorted(requirements)),
        config_json=_canonical_json(model_config),
    )


def _parse_features(value: Any, field: str) -> tuple[FeatureContract, ...]:
    if not isinstance(value, dict) or not value:
        raise ValidationError(f"model config {field} must be a non-empty object")
    result: list[FeatureContract] = []
    for name, spec in sorted(value.items()):
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise ValidationError(f"model config {field} contains an invalid feature")
        kind = spec.get("type")
        shape = spec.get("shape")
        if not isinstance(kind, str) or not kind:
            raise ValidationError(f"feature {name} requires a non-empty type")
        if not isinstance(shape, list) or not shape:
            raise ValidationError(f"feature {name} requires a non-empty shape")
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape):
            raise ValidationError(f"feature {name} shape must contain positive integers")
        unknown = set(spec) - {"type", "shape"}
        if unknown:
            raise ValidationError(f"feature {name} has unsupported fields: {sorted(unknown)}")
        result.append(FeatureContract(name, kind, tuple(shape)))
    return tuple(result)


def _parse_pipeline(
    source: SourcePackage,
    asset: SourceAsset,
    data: dict[str, Any],
) -> tuple[ProcessorPipelineContract, tuple[ExternalAssetRequirement, ...]]:
    unknown = set(data) - {"name", "steps"}
    if unknown:
        raise ValidationError(f"processor asset {asset.path} has unsupported fields: {sorted(unknown)}")
    name = data.get("name")
    steps_data = data.get("steps")
    if not isinstance(name, str) or not name or not isinstance(steps_data, list):
        raise ValidationError(f"processor asset {asset.path} requires a name and steps array")

    steps: list[ProcessorStepContract] = []
    requirements: list[ExternalAssetRequirement] = []
    for index, value in enumerate(steps_data):
        if not isinstance(value, dict):
            raise ValidationError(f"processor step {asset.path}[{index}] must be an object")
        unknown_step = set(value) - {"registry_name", "config", "state_file"}
        if unknown_step:
            raise ValidationError(
                f"processor step {asset.path}[{index}] has unsupported fields: {sorted(unknown_step)}"
            )
        registry_name = value.get("registry_name")
        config = value.get("config")
        if not isinstance(registry_name, str) or not registry_name or not isinstance(config, dict):
            raise ValidationError(f"processor step {asset.path}[{index}] requires registry_name and config")
        state_asset_id = _resolve_state_asset(source, asset, value.get("state_file"), index)
        steps.append(ProcessorStepContract(index, registry_name, _canonical_json(config), state_asset_id))

        tokenizer_name = config.get("tokenizer_name")
        if tokenizer_name is not None:
            if not isinstance(tokenizer_name, str) or not tokenizer_name:
                raise ValidationError(f"processor step {asset.path}[{index}] has an invalid tokenizer_name")
            requirements.append(
                ExternalAssetRequirement("tokenizer", tokenizer_name, f"{asset.asset_id}#step={index}")
            )

    return (
        ProcessorPipelineContract(asset.namespace, name, asset.asset_id, tuple(steps)),
        tuple(requirements),
    )


def _resolve_state_asset(
    source: SourcePackage,
    pipeline_asset: SourceAsset,
    value: Any,
    step_index: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValidationError(f"processor step {pipeline_asset.path}[{step_index}] has an invalid state_file")
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts or "\\" in value:
        raise ValidationError(f"processor step {pipeline_asset.path}[{step_index}] has an unsafe state_file")
    state_path = (PurePosixPath(pipeline_asset.path).parent / logical).as_posix()
    state_asset = source.assets.asset_by_path(state_path)
    valid_roles = {AssetRole.PREPROCESSOR_STATE.value, AssetRole.POSTPROCESSOR_STATE.value}
    if state_asset.namespace != pipeline_asset.namespace or state_asset.role not in valid_roles:
        raise ValidationError(
            f"processor state asset does not match pipeline namespace: {pipeline_asset.path}[{step_index}]"
        )
    return state_asset.asset_id


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"processor asset contains a non-canonical JSON value: {exc}") from exc
