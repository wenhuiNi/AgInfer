from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..constant_store import ConstantCoverage, ConstantKey, CoverageRecord
from ..errors import ValidationError
from ..frontend import FrontendOutput
from ..ir import (
    DType,
    Device,
    Function,
    Op,
    Program,
    Region,
    State,
    StateAccess,
    TensorType,
    Value,
    attributes,
)
from ..processor_assets import ExternalAssetRequirement, SourceAssetContract, inspect_source_asset_contract
from ..source_package import SourcePackage


@dataclass(frozen=True, slots=True)
class TensorExpectation:
    key: ConstantKey
    region: str
    dtype: str
    shape: tuple[int, ...]
    disposition: str = "consumed"
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Pi05SourceInventory:
    recipe_id: str
    tensor_count: int
    consumed_count: int
    ignored_count: int
    region_counts: tuple[tuple[str, int], ...]
    coverage: tuple[CoverageRecord, ...]
    external_requirements: tuple[ExternalAssetRequirement, ...]


class Pi05SourceFrontend:
    """Framework-free importer for the locked PI0.5 tensor-only profile."""

    frontend_id = "pi05.lerobot.flow_matching.v1"

    def import_program(self, source: SourcePackage) -> FrontendOutput:
        contract = inspect_source_asset_contract(source)
        inventory = audit_pi05_source(source, contract)
        try:
            config = json.loads(contract.config_json)
        except json.JSONDecodeError as exc:
            raise AssertionError("validated PI0.5 source contract lost canonical JSON") from exc
        program = pi05_inference_program(config)
        coverage = ConstantCoverage(source.constants)
        for record in inventory.coverage:
            if record.disposition == "consumed":
                coverage.consume(record.key.namespace, record.key.name, consumer=record.detail)
            elif record.disposition == "ignored":
                coverage.ignore(record.key.namespace, record.key.name, reason=record.detail)
            else:
                raise AssertionError(f"unknown PI0.5 coverage disposition: {record.disposition}")
        return FrontendOutput.finalize(program, source, coverage)


class _RegionBuilder:
    def __init__(self) -> None:
        self.ops: list[Op] = []

    def emit(
        self,
        opcode: str,
        inputs: tuple[Value, ...],
        output_id: str,
        output_type: TensorType,
        **op_attributes: object,
    ) -> Value:
        output = Value(output_id, output_type)
        self.ops.append(
            Op(
                opcode,
                tuple(value.value_id for value in inputs),
                (output,),
                attributes(**op_attributes),
            )
        )
        return output

    def effect(self, opcode: str, inputs: tuple[Value, ...], **op_attributes: object) -> None:
        self.ops.append(
            Op(
                opcode,
                tuple(value.value_id for value in inputs),
                (),
                attributes(**op_attributes),
            )
        )


def pi05_vision_program(config: dict[str, Any]) -> Program:
    """Build the exact single-image PI0.5 vision encoder and projector ProgramIR.

    Source constants keep their checkpoint dtype at ``constant_ref``. The reference
    LeRobot implementation loads the complete vision tower and multimodal projector
    in FP32, so BF16 source constants are explicitly cast before use.
    """

    expectations = {
        item.key.name: item
        for item in pi05_tensor_expectations(config)
        if item.region in {"vision_encoder", "multimodal_projector"}
    }
    if len(expectations) != 439:
        raise AssertionError(f"PI0.5 vision recipe expected 439 constants, got {len(expectations)}")

    device = Device.CUDA
    builder = _RegionBuilder()

    def tensor(dtype: DType, shape: tuple[int, ...]) -> TensorType:
        return TensorType(dtype, shape, device=device)

    def source(name: str, value_id: str) -> Value:
        try:
            expectation = expectations[name]
        except KeyError as exc:
            raise AssertionError(f"PI0.5 vision recipe references an undeclared constant: {name}") from exc
        dtype = {"F32": DType.F32, "BF16": DType.BF16}.get(expectation.dtype)
        if dtype is None:
            raise AssertionError(f"PI0.5 vision constant has unsupported dtype: {name}")
        referenced = builder.emit(
            "constant_ref",
            (),
            f"{value_id}.source",
            tensor(dtype, expectation.shape),
            namespace="model",
            name=name,
        )
        if dtype == DType.F32:
            return referenced
        return builder.emit(
            "cast",
            (referenced,),
            value_id,
            tensor(DType.F32, expectation.shape),
            dtype=DType.F32.value,
        )

    vision_prefix = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    image = Value("image", tensor(DType.F32, (1, 3, 224, 224)))
    patch_weight = source(
        f"{vision_prefix}.embeddings.patch_embedding.weight",
        "vision.patch.weight",
    )
    patch_bias = source(
        f"{vision_prefix}.embeddings.patch_embedding.bias",
        "vision.patch.bias",
    )
    patches_nchw = builder.emit(
        "conv2d",
        (image, patch_weight, patch_bias),
        "vision.patch.nchw",
        tensor(DType.F32, (1, 1152, 16, 16)),
        pads=(0, 0, 0, 0),
        strides=(14, 14),
    )
    patches_nhwc = builder.emit(
        "transpose",
        (patches_nchw,),
        "vision.patch.nhwc",
        tensor(DType.F32, (1, 16, 16, 1152)),
        permutation=(0, 2, 3, 1),
    )
    hidden = builder.emit(
        "reshape",
        (patches_nhwc,),
        "vision.patch.tokens",
        tensor(DType.F32, (1, 256, 1152)),
        shape=(1, 256, 1152),
    )
    positions = source(
        f"{vision_prefix}.embeddings.position_embedding.weight",
        "vision.position.weight",
    )
    positions_batched = builder.emit(
        "broadcast_in_dim",
        (positions,),
        "vision.position.batched",
        tensor(DType.F32, (1, 256, 1152)),
        shape=(1, 256, 1152),
        broadcast_dimensions=(1, 2),
    )
    hidden = builder.emit(
        "add",
        (hidden, positions_batched),
        "vision.position.output",
        tensor(DType.F32, (1, 256, 1152)),
    )

    mask_scalar = builder.emit(
        "constant",
        (),
        "vision.attention.mask.scalar",
        tensor(DType.BOOL, (1,)),
        value=(True,),
    )
    attention_mask = builder.emit(
        "broadcast_in_dim",
        (mask_scalar,),
        "vision.attention.mask",
        tensor(DType.BOOL, (1, 16, 256, 256)),
        shape=(1, 16, 256, 256),
        broadcast_dimensions=(0,),
    )

    for layer in range(27):
        source_prefix = f"{vision_prefix}.encoder.layers.{layer}"
        value_prefix = f"vision.layer_{layer:02d}"
        residual = hidden
        norm1_weight = source(f"{source_prefix}.layer_norm1.weight", f"{value_prefix}.norm1.weight")
        norm1_bias = source(f"{source_prefix}.layer_norm1.bias", f"{value_prefix}.norm1.bias")
        normalized = builder.emit(
            "layer_norm",
            (hidden, norm1_weight, norm1_bias),
            f"{value_prefix}.norm1.output",
            tensor(DType.F32, (1, 256, 1152)),
            epsilon=1e-6,
        )

        projected: dict[str, Value] = {}
        for projection in ("q", "k", "v"):
            weight = source(
                f"{source_prefix}.self_attn.{projection}_proj.weight",
                f"{value_prefix}.{projection}.weight",
            )
            bias = source(
                f"{source_prefix}.self_attn.{projection}_proj.bias",
                f"{value_prefix}.{projection}.bias",
            )
            linear = builder.emit(
                "linear",
                (normalized, weight, bias),
                f"{value_prefix}.{projection}.linear",
                tensor(DType.F32, (1, 256, 1152)),
            )
            shaped = builder.emit(
                "reshape",
                (linear,),
                f"{value_prefix}.{projection}.bshd",
                tensor(DType.F32, (1, 256, 16, 72)),
                shape=(1, 256, 16, 72),
            )
            projected[projection] = builder.emit(
                "transpose",
                (shaped,),
                f"{value_prefix}.{projection}.bhsd",
                tensor(DType.F32, (1, 16, 256, 72)),
                permutation=(0, 2, 1, 3),
            )

        attended = builder.emit(
            "scaled_dot_product_attention",
            (projected["q"], projected["k"], projected["v"], attention_mask),
            f"{value_prefix}.attention.bhsd",
            tensor(DType.F32, (1, 16, 256, 72)),
            kv_group_size=1,
            scale=1.0 / math.sqrt(72.0),
            mask_fill="dtype_min",
        )
        attended_bshd = builder.emit(
            "transpose",
            (attended,),
            f"{value_prefix}.attention.bshd",
            tensor(DType.F32, (1, 256, 16, 72)),
            permutation=(0, 2, 1, 3),
        )
        attended_flat = builder.emit(
            "reshape",
            (attended_bshd,),
            f"{value_prefix}.attention.flat",
            tensor(DType.F32, (1, 256, 1152)),
            shape=(1, 256, 1152),
        )
        output_weight = source(
            f"{source_prefix}.self_attn.out_proj.weight",
            f"{value_prefix}.attention.output.weight",
        )
        output_bias = source(
            f"{source_prefix}.self_attn.out_proj.bias",
            f"{value_prefix}.attention.output.bias",
        )
        attention_output = builder.emit(
            "linear",
            (attended_flat, output_weight, output_bias),
            f"{value_prefix}.attention.output",
            tensor(DType.F32, (1, 256, 1152)),
        )
        hidden = builder.emit(
            "add",
            (residual, attention_output),
            f"{value_prefix}.attention.residual",
            tensor(DType.F32, (1, 256, 1152)),
        )

        residual = hidden
        norm2_weight = source(f"{source_prefix}.layer_norm2.weight", f"{value_prefix}.norm2.weight")
        norm2_bias = source(f"{source_prefix}.layer_norm2.bias", f"{value_prefix}.norm2.bias")
        normalized = builder.emit(
            "layer_norm",
            (hidden, norm2_weight, norm2_bias),
            f"{value_prefix}.norm2.output",
            tensor(DType.F32, (1, 256, 1152)),
            epsilon=1e-6,
        )
        fc1_weight = source(f"{source_prefix}.mlp.fc1.weight", f"{value_prefix}.mlp.fc1.weight")
        fc1_bias = source(f"{source_prefix}.mlp.fc1.bias", f"{value_prefix}.mlp.fc1.bias")
        expanded = builder.emit(
            "linear",
            (normalized, fc1_weight, fc1_bias),
            f"{value_prefix}.mlp.fc1.output",
            tensor(DType.F32, (1, 256, 4304)),
        )
        activated = builder.emit(
            "gelu",
            (expanded,),
            f"{value_prefix}.mlp.activation",
            tensor(DType.F32, (1, 256, 4304)),
            approximation="tanh",
        )
        fc2_weight = source(f"{source_prefix}.mlp.fc2.weight", f"{value_prefix}.mlp.fc2.weight")
        fc2_bias = source(f"{source_prefix}.mlp.fc2.bias", f"{value_prefix}.mlp.fc2.bias")
        contracted = builder.emit(
            "linear",
            (activated, fc2_weight, fc2_bias),
            f"{value_prefix}.mlp.fc2.output",
            tensor(DType.F32, (1, 256, 1152)),
        )
        hidden = builder.emit(
            "add",
            (residual, contracted),
            f"{value_prefix}.mlp.residual",
            tensor(DType.F32, (1, 256, 1152)),
        )

    post_weight = source(f"{vision_prefix}.post_layernorm.weight", "vision.post_norm.weight")
    post_bias = source(f"{vision_prefix}.post_layernorm.bias", "vision.post_norm.bias")
    normalized = builder.emit(
        "layer_norm",
        (hidden, post_weight, post_bias),
        "vision.post_norm.output",
        tensor(DType.F32, (1, 256, 1152)),
        epsilon=1e-6,
    )
    projector_prefix = "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
    projector_weight = source(f"{projector_prefix}.weight", "vision.projector.weight")
    projector_bias = source(f"{projector_prefix}.bias", "vision.projector.bias")
    output = builder.emit(
        "linear",
        (normalized, projector_weight, projector_bias),
        "vision.projector.output",
        tensor(DType.F32, (1, 256, 2048)),
    )
    return Program(
        functions=(Function("pi05_vision_projector", (image,), (output.value_id,), Region(tuple(builder.ops))),),
        entry="pi05_vision_projector",
    )


def pi05_prefix_program(config: dict[str, Any]) -> Program:
    """Build the exact PI0.5 language-prefix prefill ProgramIR.

    The function consumes the three projected image-token sequences and tokenized
    language input. It returns the final prefix hidden states and materializes the
    rotary key/value cache as 36 explicit read-write states.
    """

    expectations = {
        item.key.name: item
        for item in pi05_tensor_expectations(config)
        if item.region == "language_model"
    }
    if len(expectations) != 164:
        raise AssertionError(f"PI0.5 prefix recipe expected 164 constants, got {len(expectations)}")

    device = Device.CUDA
    builder = _RegionBuilder()

    def tensor(dtype: DType, shape: tuple[int, ...]) -> TensorType:
        return TensorType(dtype, shape, device=device)

    def source(name: str, value_id: str) -> Value:
        try:
            expectation = expectations[name]
        except KeyError as exc:
            raise AssertionError(f"PI0.5 prefix recipe references an undeclared constant: {name}") from exc
        dtype = {"F32": DType.F32, "BF16": DType.BF16}.get(expectation.dtype)
        if dtype is None:
            raise AssertionError(f"PI0.5 prefix constant has unsupported dtype: {name}")
        return builder.emit(
            "constant_ref",
            (),
            value_id,
            tensor(dtype, expectation.shape),
            namespace="model",
            name=name,
        )

    images = tuple(
        Value(f"image_{index}", tensor(DType.F32, (1, 256, 2048))) for index in range(3)
    )
    image_masks = tuple(
        Value(f"image_mask_{index}", tensor(DType.BOOL, (1,))) for index in range(3)
    )
    tokens = Value("tokens", tensor(DType.I32, (1, 200)))
    token_mask = Value("token_mask", tensor(DType.BOOL, (1, 200)))

    model_prefix = "model.paligemma_with_expert.paligemma"
    embedding = source(f"{model_prefix}.lm_head.weight", "prefix.language.embedding.weight")
    language_unscaled = builder.emit(
        "gather",
        (embedding, tokens),
        "prefix.language.embedding.unscaled",
        tensor(DType.BF16, (1, 200, 2048)),
    )
    language_scale_scalar = builder.emit(
        "constant",
        (),
        "prefix.language.embedding.scale.scalar",
        tensor(DType.BF16, (1,)),
        value=(math.sqrt(2048.0),),
    )
    language_scale = builder.emit(
        "broadcast_in_dim",
        (language_scale_scalar,),
        "prefix.language.embedding.scale",
        tensor(DType.BF16, (1, 200, 2048)),
        shape=(1, 200, 2048),
        broadcast_dimensions=(0,),
    )
    language_bf16 = builder.emit(
        "mul",
        (language_unscaled, language_scale),
        "prefix.language.embedding.bf16",
        tensor(DType.BF16, (1, 200, 2048)),
    )
    language = builder.emit(
        "cast",
        (language_bf16,),
        "prefix.language.embedding.f32",
        tensor(DType.F32, (1, 200, 2048)),
        dtype=DType.F32.value,
    )
    prefix_f32 = builder.emit(
        "concat",
        images + (language,),
        "prefix.embedding.f32",
        tensor(DType.F32, (1, 968, 2048)),
        axis=1,
    )

    expanded_image_masks = tuple(
        builder.emit(
            "broadcast_in_dim",
            (image_mask,),
            f"prefix.image_mask_{index}.tokens",
            tensor(DType.BOOL, (1, 256)),
            shape=(1, 256),
            broadcast_dimensions=(0,),
        )
        for index, image_mask in enumerate(image_masks)
    )
    pad_mask = builder.emit(
        "concat",
        expanded_image_masks + (token_mask,),
        "prefix.pad_mask",
        tensor(DType.BOOL, (1, 968)),
        axis=1,
    )
    key_mask = builder.emit(
        "broadcast_in_dim",
        (pad_mask,),
        "prefix.attention.key_mask",
        tensor(DType.BOOL, (1, 968, 968)),
        shape=(1, 968, 968),
        broadcast_dimensions=(0, 2),
    )
    query_mask = builder.emit(
        "broadcast_in_dim",
        (pad_mask,),
        "prefix.attention.query_mask",
        tensor(DType.BOOL, (1, 968, 968)),
        shape=(1, 968, 968),
        broadcast_dimensions=(0, 1),
    )
    attention_mask_2d = builder.emit(
        "logical_and",
        (key_mask, query_mask),
        "prefix.attention.mask_2d",
        tensor(DType.BOOL, (1, 968, 968)),
    )
    attention_mask = builder.emit(
        "broadcast_in_dim",
        (attention_mask_2d,),
        "prefix.attention.mask",
        tensor(DType.BOOL, (1, 8, 968, 968)),
        shape=(1, 8, 968, 968),
        broadcast_dimensions=(0, 2, 3),
    )

    pad_mask_i32 = builder.emit(
        "cast",
        (pad_mask,),
        "prefix.pad_mask.i32",
        tensor(DType.I32, (1, 968)),
        dtype=DType.I32.value,
    )
    cumulative_positions = builder.emit(
        "cumulative_sum",
        (pad_mask_i32,),
        "prefix.position.cumulative",
        tensor(DType.I32, (1, 968)),
        axis=1,
    )
    negative_one = builder.emit(
        "constant",
        (),
        "prefix.position.negative_one",
        tensor(DType.I32, (1,)),
        value=(-1,),
    )
    negative_ones = builder.emit(
        "broadcast_in_dim",
        (negative_one,),
        "prefix.position.negative_ones",
        tensor(DType.I32, (1, 968)),
        shape=(1, 968),
        broadcast_dimensions=(0,),
    )
    positions = builder.emit(
        "add",
        (cumulative_positions, negative_ones),
        "prefix.position_ids",
        tensor(DType.I32, (1, 968)),
    )

    hidden = builder.emit(
        "cast",
        (prefix_f32,),
        "prefix.embedding.bf16",
        tensor(DType.BF16, (1, 968, 2048)),
        dtype=DType.BF16.value,
    )
    zero = builder.emit(
        "constant",
        (),
        "prefix.zero.bf16",
        tensor(DType.BF16, (1,)),
        value=(0.0,),
    )
    zero_2048 = builder.emit(
        "broadcast_in_dim",
        (zero,),
        "prefix.zero.2048",
        tensor(DType.BF16, (2048,)),
        shape=(2048,),
        broadcast_dimensions=(0,),
    )
    zero_256 = builder.emit(
        "broadcast_in_dim",
        (zero,),
        "prefix.zero.256",
        tensor(DType.BF16, (256,)),
        shape=(256,),
        broadcast_dimensions=(0,),
    )
    zero_16384 = builder.emit(
        "broadcast_in_dim",
        (zero,),
        "prefix.zero.16384",
        tensor(DType.BF16, (16384,)),
        shape=(16384,),
        broadcast_dimensions=(0,),
    )
    one = builder.emit(
        "constant",
        (),
        "prefix.one.f32",
        tensor(DType.F32, (1,)),
        value=(1.0,),
    )
    ones_2048 = builder.emit(
        "broadcast_in_dim",
        (one,),
        "prefix.ones.2048",
        tensor(DType.F32, (2048,)),
        shape=(2048,),
        broadcast_dimensions=(0,),
    )

    language_prefix = f"{model_prefix}.model.language_model"
    states: list[State] = []
    for layer in range(18):
        source_prefix = f"{language_prefix}.layers.{layer}"
        value_prefix = f"prefix.layer_{layer:02d}"
        residual = hidden
        norm1_delta = source(
            f"{source_prefix}.input_layernorm.weight",
            f"{value_prefix}.norm1.delta",
        )
        norm1_weight = builder.emit(
            "add",
            (norm1_delta, ones_2048),
            f"{value_prefix}.norm1.weight",
            tensor(DType.F32, (2048,)),
        )
        normalized = builder.emit(
            "rms_norm",
            (hidden, norm1_weight),
            f"{value_prefix}.norm1.output",
            tensor(DType.BF16, (1, 968, 2048)),
            epsilon=1e-6,
        )

        projected: dict[str, Value] = {}
        for projection, width, heads in (("q", 2048, 8), ("k", 256, 1), ("v", 256, 1)):
            weight = source(
                f"{source_prefix}.self_attn.{projection}_proj.weight",
                f"{value_prefix}.{projection}.weight",
            )
            bias = zero_2048 if width == 2048 else zero_256
            linear = builder.emit(
                "linear",
                (normalized, weight, bias),
                f"{value_prefix}.{projection}.linear",
                tensor(DType.BF16, (1, 968, width)),
            )
            shaped = builder.emit(
                "reshape",
                (linear,),
                f"{value_prefix}.{projection}.bshd",
                tensor(DType.BF16, (1, 968, heads, 256)),
                shape=(1, 968, heads, 256),
            )
            projected[projection] = builder.emit(
                "transpose",
                (shaped,),
                f"{value_prefix}.{projection}.bhsd",
                tensor(DType.BF16, (1, heads, 968, 256)),
                permutation=(0, 2, 1, 3),
            )

        query = builder.emit(
            "rope_default",
            (projected["q"], positions),
            f"{value_prefix}.q.rope",
            tensor(DType.BF16, (1, 8, 968, 256)),
            pairing="split_half",
            theta=10000.0,
            frequency_dtype=DType.BF16.value,
        )
        key = builder.emit(
            "rope_default",
            (projected["k"], positions),
            f"{value_prefix}.k.rope",
            tensor(DType.BF16, (1, 1, 968, 256)),
            pairing="split_half",
            theta=10000.0,
            frequency_dtype=DType.BF16.value,
        )
        key_state = f"prefix.layer_{layer:02d}.key"
        value_state = f"prefix.layer_{layer:02d}.value"
        state_type = tensor(DType.BF16, (1, 1, 968, 256))
        states.extend(
            (
                State(key_state, state_type, StateAccess.READ_WRITE),
                State(value_state, state_type, StateAccess.READ_WRITE),
            )
        )
        builder.effect("state_write", (key,), state=key_state)
        builder.effect("state_write", (projected["v"],), state=value_state)

        attended = builder.emit(
            "scaled_dot_product_attention",
            (query, key, projected["v"], attention_mask),
            f"{value_prefix}.attention.bhsd",
            tensor(DType.BF16, (1, 8, 968, 256)),
            kv_group_size=8,
            scale=1.0 / math.sqrt(256.0),
            mask_fill="dtype_min",
        )
        attended_bshd = builder.emit(
            "transpose",
            (attended,),
            f"{value_prefix}.attention.bshd",
            tensor(DType.BF16, (1, 968, 8, 256)),
            permutation=(0, 2, 1, 3),
        )
        attended_flat = builder.emit(
            "reshape",
            (attended_bshd,),
            f"{value_prefix}.attention.flat",
            tensor(DType.BF16, (1, 968, 2048)),
            shape=(1, 968, 2048),
        )
        output_weight = source(
            f"{source_prefix}.self_attn.o_proj.weight",
            f"{value_prefix}.attention.output.weight",
        )
        attention_output = builder.emit(
            "linear",
            (attended_flat, output_weight, zero_2048),
            f"{value_prefix}.attention.output",
            tensor(DType.BF16, (1, 968, 2048)),
        )
        hidden = builder.emit(
            "add",
            (residual, attention_output),
            f"{value_prefix}.attention.residual",
            tensor(DType.BF16, (1, 968, 2048)),
        )

        residual = hidden
        norm2_delta = source(
            f"{source_prefix}.post_attention_layernorm.weight",
            f"{value_prefix}.norm2.delta",
        )
        norm2_weight = builder.emit(
            "add",
            (norm2_delta, ones_2048),
            f"{value_prefix}.norm2.weight",
            tensor(DType.F32, (2048,)),
        )
        normalized = builder.emit(
            "rms_norm",
            (hidden, norm2_weight),
            f"{value_prefix}.norm2.output",
            tensor(DType.BF16, (1, 968, 2048)),
            epsilon=1e-6,
        )
        gate_weight = source(
            f"{source_prefix}.mlp.gate_proj.weight",
            f"{value_prefix}.mlp.gate.weight",
        )
        gate = builder.emit(
            "linear",
            (normalized, gate_weight, zero_16384),
            f"{value_prefix}.mlp.gate",
            tensor(DType.BF16, (1, 968, 16384)),
        )
        activated = builder.emit(
            "gelu",
            (gate,),
            f"{value_prefix}.mlp.activation",
            tensor(DType.BF16, (1, 968, 16384)),
            approximation="tanh",
        )
        up_weight = source(
            f"{source_prefix}.mlp.up_proj.weight",
            f"{value_prefix}.mlp.up.weight",
        )
        up = builder.emit(
            "linear",
            (normalized, up_weight, zero_16384),
            f"{value_prefix}.mlp.up",
            tensor(DType.BF16, (1, 968, 16384)),
        )
        gated = builder.emit(
            "mul",
            (activated, up),
            f"{value_prefix}.mlp.gated",
            tensor(DType.BF16, (1, 968, 16384)),
        )
        down_weight = source(
            f"{source_prefix}.mlp.down_proj.weight",
            f"{value_prefix}.mlp.down.weight",
        )
        mlp_output = builder.emit(
            "linear",
            (gated, down_weight, zero_2048),
            f"{value_prefix}.mlp.output",
            tensor(DType.BF16, (1, 968, 2048)),
        )
        hidden = builder.emit(
            "add",
            (residual, mlp_output),
            f"{value_prefix}.mlp.residual",
            tensor(DType.BF16, (1, 968, 2048)),
        )

    final_norm_delta = source(f"{language_prefix}.norm.weight", "prefix.final_norm.delta")
    final_norm_weight = builder.emit(
        "add",
        (final_norm_delta, ones_2048),
        "prefix.final_norm.weight",
        tensor(DType.F32, (2048,)),
    )
    output = builder.emit(
        "rms_norm",
        (hidden, final_norm_weight),
        "prefix.output",
        tensor(DType.BF16, (1, 968, 2048)),
        epsilon=1e-6,
    )
    function_inputs = images + image_masks + (tokens, token_mask)
    return Program(
        functions=(
            Function(
                "pi05_prefix",
                function_inputs,
                (output.value_id, pad_mask.value_id),
                Region(tuple(builder.ops)),
            ),
        ),
        entry="pi05_prefix",
        states=tuple(states),
    )


def pi05_denoise_step_program(config: dict[str, Any]) -> Program:
    """Build one PI0.5 action-expert denoising step over a materialized prefix cache."""

    _require_value(config, "max_action_dim", 32)
    _require_value(config, "chunk_size", 50)
    expectations = {
        item.key.name: item
        for item in pi05_tensor_expectations(config)
        if item.region in {"action_projection", "action_expert"}
    }
    if len(expectations) != 208:
        raise AssertionError(f"PI0.5 denoise recipe expected 208 constants, got {len(expectations)}")
    minimum_period = _positive_number(config, "min_period")
    maximum_period = _positive_number(config, "max_period")
    if minimum_period > maximum_period:
        raise ValidationError("PI0.5 recipe requires min_period <= max_period")

    device = Device.CUDA
    builder = _RegionBuilder()

    def tensor(dtype: DType, shape: tuple[int, ...]) -> TensorType:
        return TensorType(dtype, shape, device=device)

    def source(name: str, value_id: str) -> Value:
        try:
            expectation = expectations[name]
        except KeyError as exc:
            raise AssertionError(f"PI0.5 denoise recipe references an undeclared constant: {name}") from exc
        dtype = {"F32": DType.F32, "BF16": DType.BF16}.get(expectation.dtype)
        if dtype is None:
            raise AssertionError(f"PI0.5 denoise constant has unsupported dtype: {name}")
        return builder.emit(
            "constant_ref",
            (),
            value_id,
            tensor(dtype, expectation.shape),
            namespace="model",
            name=name,
        )

    latent = Value("latent", tensor(DType.F32, (1, 50, 32)))
    timestep = Value("timestep", tensor(DType.F32, (1,)))
    prefix_pad_mask = Value("prefix_pad_mask", tensor(DType.BOOL, (1, 968)))

    action_weight = source("model.action_in_proj.weight", "denoise.action_input.weight")
    action_bias = source("model.action_in_proj.bias", "denoise.action_input.bias")
    action_embedding_f32 = builder.emit(
        "linear",
        (latent, action_weight, action_bias),
        "denoise.action_embedding.f32",
        tensor(DType.F32, (1, 50, 1024)),
    )
    hidden = builder.emit(
        "cast",
        (action_embedding_f32,),
        "denoise.action_embedding.bf16",
        tensor(DType.BF16, (1, 50, 1024)),
        dtype=DType.BF16.value,
    )

    time_embedding = builder.emit(
        "sinusoidal_embedding",
        (timestep,),
        "denoise.time.sinusoidal",
        tensor(DType.F32, (1, 1024)),
        dimension=1024,
        min_period=minimum_period,
        max_period=maximum_period,
    )
    time_in_weight = source("model.time_mlp_in.weight", "denoise.time.input.weight")
    time_in_bias = source("model.time_mlp_in.bias", "denoise.time.input.bias")
    time_hidden = builder.emit(
        "linear",
        (time_embedding, time_in_weight, time_in_bias),
        "denoise.time.input",
        tensor(DType.F32, (1, 1024)),
    )
    time_hidden = builder.emit(
        "silu",
        (time_hidden,),
        "denoise.time.input.activation",
        tensor(DType.F32, (1, 1024)),
    )
    time_out_weight = source("model.time_mlp_out.weight", "denoise.time.output.weight")
    time_out_bias = source("model.time_mlp_out.bias", "denoise.time.output.bias")
    condition = builder.emit(
        "linear",
        (time_hidden, time_out_weight, time_out_bias),
        "denoise.time.output",
        tensor(DType.F32, (1, 1024)),
    )
    condition = builder.emit(
        "silu",
        (condition,),
        "denoise.time.condition",
        tensor(DType.F32, (1, 1024)),
    )

    true_value = builder.emit(
        "constant",
        (),
        "denoise.true",
        tensor(DType.BOOL, (1,)),
        value=(True,),
    )
    suffix_pad_mask = builder.emit(
        "broadcast_in_dim",
        (true_value,),
        "denoise.suffix.pad_mask",
        tensor(DType.BOOL, (1, 50)),
        shape=(1, 50),
        broadcast_dimensions=(0,),
    )
    prefix_key_mask = builder.emit(
        "broadcast_in_dim",
        (prefix_pad_mask,),
        "denoise.attention.prefix_mask",
        tensor(DType.BOOL, (1, 50, 968)),
        shape=(1, 50, 968),
        broadcast_dimensions=(0, 2),
    )
    suffix_key_mask = builder.emit(
        "broadcast_in_dim",
        (true_value,),
        "denoise.attention.suffix_mask",
        tensor(DType.BOOL, (1, 50, 50)),
        shape=(1, 50, 50),
        broadcast_dimensions=(0,),
    )
    attention_mask_2d = builder.emit(
        "concat",
        (prefix_key_mask, suffix_key_mask),
        "denoise.attention.mask_2d",
        tensor(DType.BOOL, (1, 50, 1018)),
        axis=2,
    )
    attention_mask = builder.emit(
        "broadcast_in_dim",
        (attention_mask_2d,),
        "denoise.attention.mask",
        tensor(DType.BOOL, (1, 8, 50, 1018)),
        shape=(1, 8, 50, 1018),
        broadcast_dimensions=(0, 2, 3),
    )

    prefix_pad_i32 = builder.emit(
        "cast",
        (prefix_pad_mask,),
        "denoise.prefix_pad.i32",
        tensor(DType.I32, (1, 968)),
        dtype=DType.I32.value,
    )
    prefix_offset = builder.emit(
        "reduce_sum",
        (prefix_pad_i32,),
        "denoise.prefix_offset",
        tensor(DType.I32, (1,)),
        axes=(1,),
        keepdims=False,
    )
    prefix_offsets = builder.emit(
        "broadcast_in_dim",
        (prefix_offset,),
        "denoise.prefix_offsets",
        tensor(DType.I32, (1, 50)),
        shape=(1, 50),
        broadcast_dimensions=(0,),
    )
    suffix_pad_i32 = builder.emit(
        "cast",
        (suffix_pad_mask,),
        "denoise.suffix_pad.i32",
        tensor(DType.I32, (1, 50)),
        dtype=DType.I32.value,
    )
    suffix_cumulative = builder.emit(
        "cumulative_sum",
        (suffix_pad_i32,),
        "denoise.suffix_position.cumulative",
        tensor(DType.I32, (1, 50)),
        axis=1,
    )
    positions_one_based = builder.emit(
        "add",
        (prefix_offsets, suffix_cumulative),
        "denoise.suffix_position.one_based",
        tensor(DType.I32, (1, 50)),
    )
    negative_one = builder.emit(
        "constant",
        (),
        "denoise.negative_one",
        tensor(DType.I32, (1,)),
        value=(-1,),
    )
    negative_ones = builder.emit(
        "broadcast_in_dim",
        (negative_one,),
        "denoise.negative_ones",
        tensor(DType.I32, (1, 50)),
        shape=(1, 50),
        broadcast_dimensions=(0,),
    )
    positions = builder.emit(
        "add",
        (positions_one_based, negative_ones),
        "denoise.position_ids",
        tensor(DType.I32, (1, 50)),
    )

    zero_bf16 = builder.emit(
        "constant",
        (),
        "denoise.zero.bf16",
        tensor(DType.BF16, (1,)),
        value=(0.0,),
    )
    zero_biases = {
        width: builder.emit(
            "broadcast_in_dim",
            (zero_bf16,),
            f"denoise.zero.{width}",
            tensor(DType.BF16, (width,)),
            shape=(width,),
            broadcast_dimensions=(0,),
        )
        for width in (256, 1024, 2048, 4096)
    }
    one_f32 = builder.emit(
        "constant",
        (),
        "denoise.one.f32",
        tensor(DType.F32, (1,)),
        value=(1.0,),
    )
    ones_1024 = builder.emit(
        "broadcast_in_dim",
        (one_f32,),
        "denoise.ones.1024",
        tensor(DType.F32, (1024,)),
        shape=(1024,),
        broadcast_dimensions=(0,),
    )
    ones_tokens = builder.emit(
        "broadcast_in_dim",
        (one_f32,),
        "denoise.ones.tokens",
        tensor(DType.F32, (1, 50, 1024)),
        shape=(1, 50, 1024),
        broadcast_dimensions=(0,),
    )

    def adaptive_norm(
        value: Value,
        dense_prefix: str,
        value_prefix: str,
    ) -> tuple[Value, Value]:
        value_f32 = builder.emit(
            "cast",
            (value,),
            f"{value_prefix}.input.f32",
            tensor(DType.F32, (1, 50, 1024)),
            dtype=DType.F32.value,
        )
        normalized_f32 = builder.emit(
            "rms_norm",
            (value_f32, ones_1024),
            f"{value_prefix}.normalized.f32",
            tensor(DType.F32, (1, 50, 1024)),
            epsilon=1e-6,
        )
        dense_weight = source(f"{dense_prefix}.dense.weight", f"{value_prefix}.dense.weight")
        dense_bias = source(f"{dense_prefix}.dense.bias", f"{value_prefix}.dense.bias")
        modulation = builder.emit(
            "linear",
            (condition, dense_weight, dense_bias),
            f"{value_prefix}.modulation",
            tensor(DType.F32, (1, 3072)),
        )
        components: dict[str, Value] = {}
        for index, component in enumerate(("scale", "shift", "gate")):
            sliced = builder.emit(
                "slice",
                (modulation,),
                f"{value_prefix}.{component}.flat",
                tensor(DType.F32, (1, 1024)),
                axis=1,
                start=index * 1024,
                stop=(index + 1) * 1024,
            )
            components[component] = builder.emit(
                "broadcast_in_dim",
                (sliced,),
                f"{value_prefix}.{component}",
                tensor(DType.F32, (1, 50, 1024)),
                shape=(1, 50, 1024),
                broadcast_dimensions=(0, 2),
            )
        scale = builder.emit(
            "add",
            (ones_tokens, components["scale"]),
            f"{value_prefix}.scale_plus_one",
            tensor(DType.F32, (1, 50, 1024)),
        )
        scaled = builder.emit(
            "mul",
            (normalized_f32, scale),
            f"{value_prefix}.scaled",
            tensor(DType.F32, (1, 50, 1024)),
        )
        shifted = builder.emit(
            "add",
            (scaled, components["shift"]),
            f"{value_prefix}.shifted",
            tensor(DType.F32, (1, 50, 1024)),
        )
        normalized = builder.emit(
            "cast",
            (shifted,),
            f"{value_prefix}.output",
            tensor(DType.BF16, (1, 50, 1024)),
            dtype=DType.BF16.value,
        )
        gate = builder.emit(
            "cast",
            (components["gate"],),
            f"{value_prefix}.gate.bf16",
            tensor(DType.BF16, (1, 50, 1024)),
            dtype=DType.BF16.value,
        )
        return normalized, gate

    expert_prefix = "model.paligemma_with_expert.gemma_expert.model"
    state_type = tensor(DType.BF16, (1, 1, 968, 256))
    states = tuple(
        State(f"prefix.layer_{layer:02d}.{kind}", state_type, StateAccess.READ_WRITE)
        for layer in range(18)
        for kind in ("key", "value")
    )
    for layer in range(18):
        source_prefix = f"{expert_prefix}.layers.{layer}"
        value_prefix = f"denoise.layer_{layer:02d}"
        residual = hidden
        normalized, attention_gate = adaptive_norm(
            hidden,
            f"{source_prefix}.input_layernorm",
            f"{value_prefix}.norm1",
        )

        projected: dict[str, Value] = {}
        for projection, width, heads in (("q", 2048, 8), ("k", 256, 1), ("v", 256, 1)):
            weight = source(
                f"{source_prefix}.self_attn.{projection}_proj.weight",
                f"{value_prefix}.{projection}.weight",
            )
            linear = builder.emit(
                "linear",
                (normalized, weight, zero_biases[width]),
                f"{value_prefix}.{projection}.linear",
                tensor(DType.BF16, (1, 50, width)),
            )
            shaped = builder.emit(
                "reshape",
                (linear,),
                f"{value_prefix}.{projection}.bshd",
                tensor(DType.BF16, (1, 50, heads, 256)),
                shape=(1, 50, heads, 256),
            )
            projected[projection] = builder.emit(
                "transpose",
                (shaped,),
                f"{value_prefix}.{projection}.bhsd",
                tensor(DType.BF16, (1, heads, 50, 256)),
                permutation=(0, 2, 1, 3),
            )

        query = builder.emit(
            "rope_default",
            (projected["q"], positions),
            f"{value_prefix}.q.rope",
            tensor(DType.BF16, (1, 8, 50, 256)),
            pairing="split_half",
            theta=10000.0,
            frequency_dtype=DType.BF16.value,
        )
        key = builder.emit(
            "rope_default",
            (projected["k"], positions),
            f"{value_prefix}.k.rope",
            tensor(DType.BF16, (1, 1, 50, 256)),
            pairing="split_half",
            theta=10000.0,
            frequency_dtype=DType.BF16.value,
        )
        prefix_key = builder.emit(
            "state_read",
            (),
            f"{value_prefix}.prefix_key",
            state_type,
            state=f"prefix.layer_{layer:02d}.key",
        )
        prefix_value = builder.emit(
            "state_read",
            (),
            f"{value_prefix}.prefix_value",
            state_type,
            state=f"prefix.layer_{layer:02d}.value",
        )
        full_key = builder.emit(
            "concat",
            (prefix_key, key),
            f"{value_prefix}.key.full",
            tensor(DType.BF16, (1, 1, 1018, 256)),
            axis=2,
        )
        full_value = builder.emit(
            "concat",
            (prefix_value, projected["v"]),
            f"{value_prefix}.value.full",
            tensor(DType.BF16, (1, 1, 1018, 256)),
            axis=2,
        )
        attended = builder.emit(
            "scaled_dot_product_attention",
            (query, full_key, full_value, attention_mask),
            f"{value_prefix}.attention.bhsd",
            tensor(DType.BF16, (1, 8, 50, 256)),
            kv_group_size=8,
            scale=1.0 / math.sqrt(256.0),
            mask_fill="dtype_min",
        )
        attended_bshd = builder.emit(
            "transpose",
            (attended,),
            f"{value_prefix}.attention.bshd",
            tensor(DType.BF16, (1, 50, 8, 256)),
            permutation=(0, 2, 1, 3),
        )
        attended_flat = builder.emit(
            "reshape",
            (attended_bshd,),
            f"{value_prefix}.attention.flat",
            tensor(DType.BF16, (1, 50, 2048)),
            shape=(1, 50, 2048),
        )
        output_weight = source(
            f"{source_prefix}.self_attn.o_proj.weight",
            f"{value_prefix}.attention.output.weight",
        )
        attention_output = builder.emit(
            "linear",
            (attended_flat, output_weight, zero_biases[1024]),
            f"{value_prefix}.attention.output",
            tensor(DType.BF16, (1, 50, 1024)),
        )
        gated_attention = builder.emit(
            "mul",
            (attention_output, attention_gate),
            f"{value_prefix}.attention.gated",
            tensor(DType.BF16, (1, 50, 1024)),
        )
        hidden = builder.emit(
            "add",
            (residual, gated_attention),
            f"{value_prefix}.attention.residual",
            tensor(DType.BF16, (1, 50, 1024)),
        )

        residual = hidden
        normalized, mlp_gate = adaptive_norm(
            hidden,
            f"{source_prefix}.post_attention_layernorm",
            f"{value_prefix}.norm2",
        )
        gate_weight = source(
            f"{source_prefix}.mlp.gate_proj.weight",
            f"{value_prefix}.mlp.gate.weight",
        )
        gate = builder.emit(
            "linear",
            (normalized, gate_weight, zero_biases[4096]),
            f"{value_prefix}.mlp.gate",
            tensor(DType.BF16, (1, 50, 4096)),
        )
        activated = builder.emit(
            "gelu",
            (gate,),
            f"{value_prefix}.mlp.activation",
            tensor(DType.BF16, (1, 50, 4096)),
            approximation="tanh",
        )
        up_weight = source(
            f"{source_prefix}.mlp.up_proj.weight",
            f"{value_prefix}.mlp.up.weight",
        )
        up = builder.emit(
            "linear",
            (normalized, up_weight, zero_biases[4096]),
            f"{value_prefix}.mlp.up",
            tensor(DType.BF16, (1, 50, 4096)),
        )
        gated_mlp = builder.emit(
            "mul",
            (activated, up),
            f"{value_prefix}.mlp.gated",
            tensor(DType.BF16, (1, 50, 4096)),
        )
        down_weight = source(
            f"{source_prefix}.mlp.down_proj.weight",
            f"{value_prefix}.mlp.down.weight",
        )
        mlp_output = builder.emit(
            "linear",
            (gated_mlp, down_weight, zero_biases[1024]),
            f"{value_prefix}.mlp.output",
            tensor(DType.BF16, (1, 50, 1024)),
        )
        gated_mlp_output = builder.emit(
            "mul",
            (mlp_output, mlp_gate),
            f"{value_prefix}.mlp.output.gated",
            tensor(DType.BF16, (1, 50, 1024)),
        )
        hidden = builder.emit(
            "add",
            (residual, gated_mlp_output),
            f"{value_prefix}.mlp.residual",
            tensor(DType.BF16, (1, 50, 1024)),
        )

    normalized, _ = adaptive_norm(
        hidden,
        f"{expert_prefix}.norm",
        "denoise.final_norm",
    )
    normalized_f32 = builder.emit(
        "cast",
        (normalized,),
        "denoise.final_norm.f32",
        tensor(DType.F32, (1, 50, 1024)),
        dtype=DType.F32.value,
    )
    output_weight = source("model.action_out_proj.weight", "denoise.action_output.weight")
    output_bias = source("model.action_out_proj.bias", "denoise.action_output.bias")
    output = builder.emit(
        "linear",
        (normalized_f32, output_weight, output_bias),
        "denoise.velocity",
        tensor(DType.F32, (1, 50, 32)),
    )
    return Program(
        functions=(
            Function(
                "pi05_denoise_step",
                (latent, timestep, prefix_pad_mask),
                (output.value_id,),
                Region(tuple(builder.ops)),
            ),
        ),
        entry="pi05_denoise_step",
        states=states,
    )


def pi05_inference_program(config: dict[str, Any]) -> Program:
    """Build the fixed-profile PI0.5 tensor-only inference ProgramIR."""

    _require_value(config, "num_inference_steps", 10)
    vision_program = pi05_vision_program(config)
    prefix_program = pi05_prefix_program(config)
    denoise_program = pi05_denoise_step_program(config)
    if prefix_program.states != denoise_program.states:
        raise AssertionError("PI0.5 prefix and denoise cache declarations diverged")

    device = Device.CUDA
    builder = _RegionBuilder()

    def tensor(dtype: DType, shape: tuple[int, ...]) -> TensorType:
        return TensorType(dtype, shape, device=device)

    def call(callee: str, inputs: tuple[Value, ...], outputs: tuple[Value, ...]) -> None:
        builder.ops.append(
            Op(
                "call",
                tuple(value.value_id for value in inputs),
                outputs,
                attributes(callee=callee, repeat=1),
            )
        )

    images = tuple(
        Value(f"image_{index}", tensor(DType.F32, (1, 3, 224, 224))) for index in range(3)
    )
    image_masks = tuple(
        Value(f"image_mask_{index}", tensor(DType.BOOL, (1,))) for index in range(3)
    )
    tokens = Value("tokens", tensor(DType.I32, (1, 200)))
    token_mask = Value("token_mask", tensor(DType.BOOL, (1, 200)))
    initial_noise = Value("initial_noise", tensor(DType.F32, (1, 50, 32)))

    projected_images: list[Value] = []
    for index, image in enumerate(images):
        projected = Value(f"infer.image_{index}.projected", tensor(DType.F32, (1, 256, 2048)))
        call("pi05_vision_projector", (image,), (projected,))
        projected_images.append(projected)

    prefix_output = Value("infer.prefix.output", tensor(DType.BF16, (1, 968, 2048)))
    prefix_pad_mask = Value("infer.prefix.pad_mask", tensor(DType.BOOL, (1, 968)))
    call(
        "pi05_prefix",
        tuple(projected_images) + image_masks + (tokens, token_mask),
        (prefix_output, prefix_pad_mask),
    )

    step_size = builder.emit(
        "constant",
        (),
        "infer.euler.dt.scalar",
        tensor(DType.F32, (1,)),
        value=(-0.1,),
    )
    step_sizes = builder.emit(
        "broadcast_in_dim",
        (step_size,),
        "infer.euler.dt",
        tensor(DType.F32, (1, 50, 32)),
        shape=(1, 50, 32),
        broadcast_dimensions=(0,),
    )
    latent = initial_noise
    timesteps = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
    for step, time in enumerate(timesteps):
        timestep = builder.emit(
            "constant",
            (),
            f"infer.step_{step:02d}.timestep",
            tensor(DType.F32, (1,)),
            value=(time,),
        )
        velocity = Value(
            f"infer.step_{step:02d}.velocity",
            tensor(DType.F32, (1, 50, 32)),
        )
        call("pi05_denoise_step", (latent, timestep, prefix_pad_mask), (velocity,))
        delta = builder.emit(
            "mul",
            (velocity, step_sizes),
            f"infer.step_{step:02d}.delta",
            tensor(DType.F32, (1, 50, 32)),
        )
        latent = builder.emit(
            "add",
            (latent, delta),
            f"infer.step_{step:02d}.latent",
            tensor(DType.F32, (1, 50, 32)),
        )

    action = builder.emit(
        "slice",
        (latent,),
        "infer.action",
        tensor(DType.F32, (1, 50, 7)),
        axis=2,
        start=0,
        stop=7,
    )
    entry_inputs = images + image_masks + (tokens, token_mask, initial_noise)
    entry = Function("pi05_infer", entry_inputs, (action.value_id,), Region(tuple(builder.ops)))
    return Program(
        functions=(
            entry,
            vision_program.functions[0],
            prefix_program.functions[0],
            denoise_program.functions[0],
        ),
        entry="pi05_infer",
        states=prefix_program.states,
    )


def pi05_tensor_expectations(config: dict[str, Any]) -> tuple[TensorExpectation, ...]:
    """Generate the exact source-tensor contract for the first PI0.5 recipe."""

    _require_value(config, "paligemma_variant", "gemma_2b")
    _require_value(config, "action_expert_variant", "gemma_300m")
    _require_value(config, "dtype", "bfloat16")
    _require_value(config, "image_resolution", [224, 224])
    action_dimension = _positive_int(config, "max_action_dim")
    _positive_int(config, "chunk_size")
    _positive_int(config, "num_inference_steps")

    values: dict[str, TensorExpectation] = {}

    def add(
        name: str,
        region: str,
        dtype: str,
        shape: tuple[int, ...],
        *,
        disposition: str = "consumed",
        detail: str | None = None,
    ) -> None:
        key = ConstantKey("model", name)
        if name in values:
            raise AssertionError(f"duplicate generated PI0.5 expectation: {name}")
        values[name] = TensorExpectation(
            key,
            region,
            dtype,
            shape,
            disposition,
            detail or f"{region}:{name}",
        )

    for projection, output_width, input_width in (
        ("action_in_proj", 1024, action_dimension),
        ("action_out_proj", action_dimension, 1024),
        ("time_mlp_in", 1024, 1024),
        ("time_mlp_out", 1024, 1024),
    ):
        add(f"model.{projection}.weight", "action_projection", "F32", (output_width, input_width))
        add(f"model.{projection}.bias", "action_projection", "F32", (output_width,))

    expert_prefix = "model.paligemma_with_expert.gemma_expert"
    add(
        f"{expert_prefix}.lm_head.weight",
        "expert_lm_head",
        "BF16",
        (257152, 1024),
        disposition="ignored",
        detail="inference returns expert hidden states and uses action_out_proj",
    )
    for layer in range(18):
        prefix = f"{expert_prefix}.model.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            add(f"{prefix}.{norm}.dense.weight", "action_expert", "F32", (3072, 1024))
            add(f"{prefix}.{norm}.dense.bias", "action_expert", "F32", (3072,))
        for projection, shape in (
            ("q_proj", (2048, 1024)),
            ("k_proj", (256, 1024)),
            ("v_proj", (256, 1024)),
            ("o_proj", (1024, 2048)),
        ):
            add(f"{prefix}.self_attn.{projection}.weight", "action_expert", "BF16", shape)
        for projection, shape in (
            ("gate_proj", (4096, 1024)),
            ("up_proj", (4096, 1024)),
            ("down_proj", (1024, 4096)),
        ):
            add(f"{prefix}.mlp.{projection}.weight", "action_expert", "BF16", shape)
    add(f"{expert_prefix}.model.norm.dense.weight", "action_expert", "F32", (3072, 1024))
    add(f"{expert_prefix}.model.norm.dense.bias", "action_expert", "F32", (3072,))

    paligemma_prefix = "model.paligemma_with_expert.paligemma"
    add(
        f"{paligemma_prefix}.lm_head.weight",
        "language_model",
        "BF16",
        (257152, 2048),
        detail="tied language token embedding source",
    )
    language_prefix = f"{paligemma_prefix}.model.language_model"
    for layer in range(18):
        prefix = f"{language_prefix}.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            add(f"{prefix}.{norm}.weight", "language_model", "F32", (2048,))
        for projection, shape in (
            ("q_proj", (2048, 2048)),
            ("k_proj", (256, 2048)),
            ("v_proj", (256, 2048)),
            ("o_proj", (2048, 2048)),
        ):
            add(f"{prefix}.self_attn.{projection}.weight", "language_model", "BF16", shape)
        for projection, shape in (
            ("gate_proj", (16384, 2048)),
            ("up_proj", (16384, 2048)),
            ("down_proj", (2048, 16384)),
        ):
            add(f"{prefix}.mlp.{projection}.weight", "language_model", "BF16", shape)
    add(f"{language_prefix}.norm.weight", "language_model", "F32", (2048,))

    vision_prefix = f"{paligemma_prefix}.model.vision_tower.vision_model"
    add(f"{vision_prefix}.embeddings.patch_embedding.weight", "vision_encoder", "F32", (1152, 3, 14, 14))
    add(f"{vision_prefix}.embeddings.patch_embedding.bias", "vision_encoder", "F32", (1152,))
    add(f"{vision_prefix}.embeddings.position_embedding.weight", "vision_encoder", "F32", (256, 1152))
    for layer in range(27):
        prefix = f"{vision_prefix}.encoder.layers.{layer}"
        for norm in ("layer_norm1", "layer_norm2"):
            add(f"{prefix}.{norm}.weight", "vision_encoder", "BF16", (1152,))
            add(f"{prefix}.{norm}.bias", "vision_encoder", "BF16", (1152,))
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            add(f"{prefix}.self_attn.{projection}.weight", "vision_encoder", "BF16", (1152, 1152))
            add(f"{prefix}.self_attn.{projection}.bias", "vision_encoder", "BF16", (1152,))
        add(f"{prefix}.mlp.fc1.weight", "vision_encoder", "BF16", (4304, 1152))
        add(f"{prefix}.mlp.fc1.bias", "vision_encoder", "BF16", (4304,))
        add(f"{prefix}.mlp.fc2.weight", "vision_encoder", "BF16", (1152, 4304))
        add(f"{prefix}.mlp.fc2.bias", "vision_encoder", "BF16", (1152,))
    add(f"{vision_prefix}.post_layernorm.weight", "vision_encoder", "BF16", (1152,))
    add(f"{vision_prefix}.post_layernorm.bias", "vision_encoder", "BF16", (1152,))

    projector_prefix = f"{paligemma_prefix}.model.multi_modal_projector.linear"
    add(f"{projector_prefix}.weight", "multimodal_projector", "BF16", (2048, 1152))
    add(f"{projector_prefix}.bias", "multimodal_projector", "BF16", (2048,))

    result = tuple(sorted(values.values(), key=lambda item: item.key))
    if len(result) != 812:
        raise AssertionError(f"PI0.5 expectation generator produced {len(result)} tensors instead of 812")
    return result


def audit_pi05_source(
    source: SourcePackage,
    contract: SourceAssetContract | None = None,
) -> Pi05SourceInventory:
    """Validate PI0.5 source identity and close its constant coverage ledger."""

    selected_contract = inspect_source_asset_contract(source) if contract is None else contract
    if selected_contract.source_type != "pi05":
        raise ValidationError(f"PI0.5 recipe refuses source type {selected_contract.source_type!r}")
    try:
        config = json.loads(selected_contract.config_json)
    except json.JSONDecodeError as exc:
        raise AssertionError("validated source asset contract lost canonical JSON") from exc
    expected = {item.key: item for item in pi05_tensor_expectations(config)}
    actual_model = {key for key in source.constants.keys if key.namespace == "model"}
    missing = sorted(set(expected) - actual_model)
    unexpected = sorted(actual_model - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(key.name for key in missing[:5]))
        if unexpected:
            details.append("unexpected=" + ", ".join(key.name for key in unexpected[:5]))
        raise ValidationError("PI0.5 model tensor inventory mismatch: " + "; ".join(details))

    coverage = ConstantCoverage(source.constants)
    regions: Counter[str] = Counter()
    for key, item in expected.items():
        tensor = source.constants.tensor(key.namespace, key.name)
        if tensor.dtype != item.dtype or tuple(tensor.shape) != item.shape:
            raise ValidationError(
                f"PI0.5 tensor contract mismatch for {key.name}: "
                f"expected {item.dtype}{item.shape}, got {tensor.dtype}{tuple(tensor.shape)}"
            )
        regions[item.region] += 1
        if item.disposition == "ignored":
            coverage.ignore(key.namespace, key.name, reason=item.detail)
        else:
            coverage.consume(key.namespace, key.name, consumer=item.detail)

    for key in source.constants.keys:
        if key.namespace == "model":
            continue
        regions[f"{key.namespace}_state"] += 1
        coverage.consume(key.namespace, key.name, consumer=f"{key.namespace}_processor_state")
    coverage.require_complete()
    records = coverage.records
    ignored_count = sum(record.disposition == "ignored" for record in records)
    return Pi05SourceInventory(
        recipe_id="pi05.lerobot.flow_matching.v1",
        tensor_count=len(records),
        consumed_count=len(records) - ignored_count,
        ignored_count=ignored_count,
        region_counts=tuple(sorted(regions.items())),
        coverage=records,
        external_requirements=selected_contract.external_requirements,
    )


def _require_value(config: dict[str, Any], name: str, expected: object) -> None:
    if config.get(name) != expected:
        raise ValidationError(f"PI0.5 recipe requires {name}={expected!r}, got {config.get(name)!r}")


def _positive_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"PI0.5 recipe requires positive integer {name}")
    return value


def _positive_number(config: dict[str, Any], name: str) -> float:
    value = config.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValidationError(f"PI0.5 recipe requires positive finite number {name}")
    return float(value)
