"""Structural lowering for the currently delivered native executable forms.

This module knows exact tensor problems, not model names or checkpoint slots.
It creates candidate capabilities; it does not manufacture validation receipts.
"""
from dataclasses import dataclass, replace
import hashlib

from .. import providers as p
from ..errors import ValidationError
from ..lowering import (build_execution_schedule, build_lowering_inventory,
    build_memory_plan, build_literal_materialization, assemble_command_stream,
    placements_from_partial_lowering)
from ..lowering.memory import replan_memory_for_commands
from ..providers.build_binding import BuildBinding, LinearBuildBinding
from ..providers.rounded_attention import RoundedAttentionPayload
from ..providers.projection_split import lower_projection_splits
from ..providers.state_update import lower_state_updates
from ..schema import CudaArch


@dataclass(frozen=True)
class FixedAlgorithms:
    linear: tuple
    patch: object
    attention: tuple

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"schema", "linear", "patch", "attention"} or data["schema"] != "aginfer.fixed-algorithms.v1":
            raise ValidationError("unsupported fixed algorithm selection")
        if (not isinstance(data["linear"], list) or not data["linear"]
                or len(data["linear"]) > 256 or not isinstance(data["attention"], list)
                or len(data["attention"]) != 3):
            raise ValidationError("invalid fixed algorithm selection counts")
        def decode(value, kind):
            if not isinstance(value, str) or len(value) > 4096:
                raise ValidationError("invalid fixed algorithm payload")
            try:
                payload = kind.from_bytes(bytes.fromhex(value))
            except (ValueError, TypeError) as exc:
                raise ValidationError("invalid fixed algorithm encoding") from exc
            if payload.to_bytes().hex() != value:
                raise ValidationError("fixed algorithm payload is noncanonical")
            return payload
        linear = tuple(decode(x, p.CublasLtLinearPayload) for x in data["linear"])
        patch = decode(data["patch"], p.CublasLtLinearPayload)
        attention = tuple(decode(x, RoundedAttentionPayload) for x in data["attention"])
        if len({x.problem for x in linear}) != len(linear):
            raise ValidationError("duplicate linear algorithm selection")
        if {x.variant for x in attention} != {1, 2, 4}:
            raise ValidationError("candidate attention selection requires variants 1, 2, 4")
        if any(x.cublaslt_version != 120803 for x in linear + (patch,) + attention):
            raise ValidationError("fixed algorithms require the delivered cuBLASLt version")
        if any(x.problem.target_arch != CudaArch.SM120 for x in linear + (patch,)):
            raise ValidationError("fixed algorithm target differs from SM120")
        return cls(linear, patch, tuple(sorted(attention, key=lambda x: x.variant)))


def lower_native_program(program, cubin: bytes, algorithms: FixedAlgorithms):
    arch = CudaArch.SM120
    digest = hashlib.sha256(cubin).hexdigest()
    if any(x.module_bytes != len(cubin) or x.module_sha256 != digest for x in algorithms.attention):
        raise ValidationError("attention algorithms belong to a different CUBIN")
    inventory = build_lowering_inventory(program)
    schedule = build_execution_schedule(program)
    resident = any(op.opcode == "state_update" for op in schedule.ops)
    memory = build_memory_plan(schedule)

    def binding(problem, payload_type, provider_id=2):
        return BuildBinding((problem,), (payload_type(problem, len(cubin), digest),), provider_id)

    def elementwise(problem_type, opcodes):
        problems = sorted({problem_type.from_inventory(x, target_arch=arch)
            for x in inventory.ops if x.opcode in opcodes}, key=lambda x: (int(x.kernel_id), x.numel))
        payloads = tuple(p.CudaKernelPayload.for_problem(x, module_bytes=len(cubin), module_sha256=digest) for x in problems)
        return BuildBinding(tuple(problems), payloads)

    cast = elementwise(p.AotCastProblem, {"cast"})
    pointwise = elementwise(p.AotPointwiseProblem, {"add", "mul"})
    activation_group = elementwise(p.AotActivationProblem, {"gelu", "silu"})
    activation = tuple(BuildBinding((x,), (y,)) for x, y in zip(activation_group.problems, activation_group.payloads))
    linear = tuple(LinearBuildBinding(x) for x in algorithms.linear)
    specs = (
        ("layer_norm", p.LayerNormProblem(arch), p.LayerNormPayload, p.lower_layer_norm_commands),
        ("adaptive", p.AdaptiveRmsNormProblem(arch), p.AdaptiveRmsNormPayload, p.lower_adaptive_rms_norm_commands),
        ("kv_pack", p.KvPackProblem(arch), p.KvPackPayload, p.lower_kv_pack_commands),
        ("kv_store", p.PrefixKvStoreProblem(arch), p.PrefixKvStorePayload, p.lower_prefix_kv_store_commands),
        ("suffix", p.SuffixMetadataProblem(arch), p.SuffixMetadataPayload, p.lower_suffix_metadata_commands),
        ("prefix", p.PrefixInputProblem(arch), p.PrefixInputPayload, p.lower_prefix_input_commands),
        ("time", p.TimeEmbeddingProblem(arch), p.TimeEmbeddingPayload, p.lower_time_embedding_commands),
        ("slice", p.ActionSliceProblem(arch), p.ActionSlicePayload, p.lower_action_slice_commands),
    )
    rms = tuple(binding(p.RmsNormProblem(arch, variant), p.RmsNormPayload) for variant in p.RmsNormVariant)
    rope = tuple(binding(p.RopeProblem(arch, variant), p.RopePayload) for variant in p.RopeVariant)
    patch_problem = p.PatchProjectionProblem(arch)
    patch = BuildBinding((patch_problem,), (p.PatchProjectionPayload(patch_problem, len(cubin), digest, algorithms.patch),), 4)
    # Reuse the existing shape/dataflow matchers. Their boundary contracts are
    # shared by FlashInfer and materialized forms; no old provider plan is made.
    attention = (
        (p.FlashInferAttentionProblem(arch), p.lower_flashinfer_attention_commands),
        (p.FlashInferPrefixAttentionProblem(arch), p.lower_flashinfer_prefix_attention_commands),
        (p.VisionAttentionProblem(arch), p.lower_vision_attention_commands),
    )

    def lower(memory):
        def call(fn, description):
            return fn(schedule, inventory, memory, description, target_arch=arch)
        lowered = {
            "linear": call(p.lower_cublaslt_linear_commands, linear),
            "cast": call(p.lower_aot_cast_commands, cast),
            "pointwise": call(p.lower_aot_pointwise_commands, pointwise),
            "activation": call(p.lower_aot_activation_commands, activation),
            "rms": call(p.lower_rms_norm_commands, rms),
            "rope": call(p.lower_rope_commands, rope),
            "patch": call(p.lower_patch_projection_commands, patch),
        }
        for name, problem, payload_type, fn in specs:
            if resident and name in {"kv_pack", "kv_store"}:
                continue
            lowered[name] = call(fn, binding(problem, payload_type))
        if resident:
            lowered["state_update"] = lower_state_updates(schedule, len(cubin), digest)
        split = lower_projection_splits(schedule, len(cubin), digest)
        if split.commands:
            lowered["projection_split"] = split
        for index, ((problem, fn), payload) in enumerate(zip(attention, algorithms.attention)):
            lowered[f"attention_{index}"] = call(fn, BuildBinding((problem,), (payload,), 5))
        superseded = set().union(*(set(lowered[name].fused_execution_indices) for name in ("adaptive", "suffix", "prefix")))
        for name in ("cast", "pointwise", "rms"):
            lowered[name] = replace(lowered[name], commands=tuple(x for x in lowered[name].commands if x.execution_index not in superseded))
        placements = tuple(x for result in lowered.values() for x in placements_from_partial_lowering(schedule, result))
        owners = {}
        for name, result in lowered.items():
            for command in result.commands:
                if command.execution_index in owners:
                    raise ValidationError(f"duplicate lowering anchor {command.execution_index}: {owners[command.execution_index]} / {name}")
                owners[command.execution_index] = name
        return lowered, placements

    lowered, placements = lower(memory)
    covered = {x for placement in placements for x in placement.covered_execution_indices}
    literals = build_literal_materialization(schedule, inventory, excluded_execution_indices=covered)
    memory = build_memory_plan(schedule, literal_materialization=literals)
    lowered, placements = lower(memory)
    memory = replan_memory_for_commands(schedule, memory, placements)
    commands = assemble_command_stream(schedule, memory, placements, target_arch=arch)
    capabilities = {cap.digest: cap for result in lowered.values() for cap in result.capabilities}
    used = {cmd.capability_digest for cmd in commands.commands}
    for cmd in commands.commands:
        cap = capabilities[cmd.capability_digest]
        if (cap.implementation_digest != hashlib.sha256(cmd.payload).hexdigest()
                or cap.supports_capture != cmd.capture_safe or cap.provider_id != cmd.provider_id):
            raise ValidationError("candidate command differs from its exact capability")
    return inventory, schedule, memory, commands, literals, tuple(capabilities[key] for key in sorted(used))
