from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from aginfer.constant_store import ConstantKey
from aginfer.errors import ValidationError
from aginfer.ir import DType, Device, StateAccess, dump_program, verify_program
from aginfer.recipes.pi05 import (
    Pi05SourceFrontend,
    audit_pi05_source,
    pi05_denoise_step_program,
    pi05_inference_program,
    pi05_prefix_program,
    pi05_tensor_expectations,
    pi05_vision_program,
)
from aginfer.source_manifest import AssetRole, SourceAsset
from aginfer.source_package import open_source_package
from tests.helpers import write_safetensors


def _config() -> dict[str, object]:
    return {
        "type": "pi05",
        "input_features": {"image": {"type": "VISUAL", "shape": [3, 224, 224]}},
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "dtype": "bfloat16",
        "image_resolution": [224, 224],
        "max_action_dim": 32,
        "chunk_size": 50,
        "num_inference_steps": 10,
        "min_period": 0.004,
        "max_period": 4.0,
    }


class _FakePi05Constants:
    def __init__(self, config: dict[str, object], *, wrong_shape: str | None = None) -> None:
        self._tensors = {}
        for item in pi05_tensor_expectations(config):
            shape = (1,) if item.key.name == wrong_shape else item.shape
            self._tensors[item.key] = SimpleNamespace(dtype=item.dtype, shape=list(shape))

    @property
    def keys(self) -> tuple[ConstantKey, ...]:
        return tuple(sorted(self._tensors))

    def tensor(self, namespace: str, name: str) -> object:
        try:
            return self._tensors[ConstantKey(namespace, name)]
        except KeyError as exc:
            raise ValidationError(f"constant not found: {namespace}:{name}") from exc


class _FakePi05Assets:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.config_asset = SourceAsset(
            "model:config.json",
            AssetRole.METADATA.value,
            "model",
            "config.json",
            1,
        )

    def select(self, *, role: str | None = None, namespace: str | None = None) -> tuple[SourceAsset, ...]:
        if role not in (None, AssetRole.METADATA.value) or namespace not in (None, "model"):
            return ()
        return (self.config_asset,)

    def read_json(self, asset_id: str) -> dict[str, object]:
        if asset_id != self.config_asset.asset_id:
            raise ValidationError(f"source asset not found: {asset_id}")
        return self.config


def _fake_pi05_source(
    config: dict[str, object],
    *,
    wrong_shape: str | None = None,
) -> object:
    return SimpleNamespace(
        constants=_FakePi05Constants(config, wrong_shape=wrong_shape),
        assets=_FakePi05Assets(config),
    )


class Pi05RecipeTests(unittest.TestCase):
    def test_expectations_cover_exact_regions_and_explicit_unused_head(self) -> None:
        expectations = pi05_tensor_expectations(_config())
        counts = Counter(item.region for item in expectations)
        self.assertEqual(
            counts,
            {
                "action_projection": 8,
                "expert_lm_head": 1,
                "action_expert": 200,
                "language_model": 164,
                "vision_encoder": 437,
                "multimodal_projector": 2,
            },
        )
        ignored = [item for item in expectations if item.disposition == "ignored"]
        self.assertEqual(len(ignored), 1)
        self.assertTrue(ignored[0].key.name.endswith("gemma_expert.lm_head.weight"))
        action_input = next(item for item in expectations if item.key.name == "model.action_in_proj.weight")
        self.assertEqual((action_input.dtype, action_input.shape), ("F32", (1024, 32)))

    def test_recipe_refuses_unverified_variants_and_incomplete_inventory(self) -> None:
        config = _config()
        config["paligemma_variant"] = "gemma_300m"
        with self.assertRaisesRegex(ValidationError, "paligemma_variant"):
            pi05_tensor_expectations(config)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
            write_safetensors(root / "model.safetensors", {"model.action_in_proj.weight": ("F16", [1], b"ab")})
            source = open_source_package(str(root), offline=True)
            with self.assertRaisesRegex(ValidationError, "tensor inventory mismatch"):
                audit_pi05_source(source)

    def test_vision_program_covers_exact_slots_and_topology(self) -> None:
        program = pi05_vision_program(_config())
        verify_program(program)
        function = program.functions[0]
        counts = Counter(op.opcode for op in function.body.ops)
        self.assertEqual(counts["constant_ref"], 439)
        self.assertEqual(counts["cast"], 436)
        self.assertEqual(counts["conv2d"], 1)
        self.assertEqual(counts["scaled_dot_product_attention"], 27)
        self.assertEqual(counts["layer_norm"], 55)
        self.assertEqual(counts["linear"], 163)
        self.assertEqual(counts["gelu"], 27)
        self.assertEqual(counts["add"], 55)
        self.assertEqual(counts["reshape"], 109)
        self.assertEqual(counts["transpose"], 109)
        self.assertEqual(counts["broadcast_in_dim"], 2)
        self.assertEqual(function.inputs[0].type.dtype, DType.F32)
        self.assertEqual(function.inputs[0].type.shape, (1, 3, 224, 224))
        self.assertEqual(function.inputs[0].type.device, Device.CUDA)
        output_type = function.body.ops[-1].outputs[0].type
        self.assertEqual(output_type.dtype, DType.F32)
        self.assertEqual(output_type.shape, (1, 256, 2048))
        self.assertEqual(output_type.device, Device.CUDA)

        expected_names = {
            item.key.name
            for item in pi05_tensor_expectations(_config())
            if item.region in {"vision_encoder", "multimodal_projector"}
        }
        referenced_names = [
            str(op.attribute("name")) for op in function.body.ops if op.opcode == "constant_ref"
        ]
        self.assertEqual(set(referenced_names), expected_names)
        self.assertEqual(len(referenced_names), len(set(referenced_names)))
        self.assertNotIn("pi05", {op.opcode for op in function.body.ops})

    def test_vision_program_dump_is_deterministic_and_config_is_fail_closed(self) -> None:
        first = pi05_vision_program(_config())
        second = pi05_vision_program(_config())
        self.assertEqual(dump_program(first), dump_program(second))
        config = _config()
        config["image_resolution"] = [256, 256]
        with self.assertRaisesRegex(ValidationError, "image_resolution"):
            pi05_vision_program(config)

    def test_prefix_program_covers_exact_slots_topology_and_cache(self) -> None:
        program = pi05_prefix_program(_config())
        verify_program(program)
        function = program.functions[0]
        counts = Counter(op.opcode for op in function.body.ops)
        self.assertEqual(
            counts,
            {
                "constant_ref": 164,
                "constant": 4,
                "broadcast_in_dim": 12,
                "gather": 1,
                "cast": 3,
                "concat": 2,
                "logical_and": 1,
                "cumulative_sum": 1,
                "add": 74,
                "rms_norm": 37,
                "linear": 126,
                "reshape": 72,
                "transpose": 72,
                "rope_default": 36,
                "state_write": 36,
                "scaled_dot_product_attention": 18,
                "gelu": 18,
                "mul": 19,
            },
        )

        expected_names = {
            item.key.name
            for item in pi05_tensor_expectations(_config())
            if item.region == "language_model"
        }
        referenced_names = [
            str(op.attribute("name")) for op in function.body.ops if op.opcode == "constant_ref"
        ]
        self.assertEqual(set(referenced_names), expected_names)
        self.assertEqual(len(referenced_names), len(set(referenced_names)))

        language_scale = next(
            op
            for op in function.body.ops
            if op.outputs and op.outputs[0].value_id == "prefix.language.embedding.scale.scalar"
        )
        self.assertEqual(language_scale.outputs[0].type.dtype, DType.BF16)
        self.assertEqual(language_scale.attribute("value"), (2048.0**0.5,))
        language_mul = next(
            op
            for op in function.body.ops
            if op.outputs and op.outputs[0].value_id == "prefix.language.embedding.bf16"
        )
        self.assertEqual(language_mul.opcode, "mul")

        self.assertEqual(len(program.states), 36)
        for layer in range(18):
            for suffix in ("key", "value"):
                state = program.states[2 * layer + (suffix == "value")]
                self.assertEqual(state.name, f"prefix.layer_{layer:02d}.{suffix}")
                self.assertEqual(state.access, StateAccess.READ_WRITE)
                self.assertEqual(state.type.dtype, DType.BF16)
                self.assertEqual(state.type.shape, (1, 1, 968, 256))
                self.assertEqual(state.type.device, Device.CUDA)

        attentions = [
            op for op in function.body.ops if op.opcode == "scaled_dot_product_attention"
        ]
        self.assertTrue(
            all(
                op.attribute("kv_group_size") == 8
                and op.attribute("scale") == 0.0625
                and op.inputs[3] == "prefix.attention.mask"
                for op in attentions
            )
        )
        ropes = [op for op in function.body.ops if op.opcode == "rope_default"]
        self.assertTrue(
            all(
                op.attribute("pairing") == "split_half"
                and op.attribute("theta") == 10000.0
                and op.attribute("frequency_dtype") == "bf16"
                and op.inputs[1] == "prefix.position_ids"
                for op in ropes
            )
        )
        output = function.body.ops[-1].outputs[0]
        self.assertEqual(output.type.dtype, DType.BF16)
        self.assertEqual(output.type.shape, (1, 968, 2048))
        self.assertEqual(output.type.device, Device.CUDA)
        self.assertEqual(function.outputs, ("prefix.output", "prefix.pad_mask"))
        self.assertNotIn("pi05", {op.opcode for op in function.body.ops})

    def test_prefix_program_dump_is_deterministic_and_config_is_fail_closed(self) -> None:
        first = pi05_prefix_program(_config())
        second = pi05_prefix_program(_config())
        self.assertEqual(dump_program(first), dump_program(second))
        config = _config()
        config["num_inference_steps"] = 0
        with self.assertRaisesRegex(ValidationError, "num_inference_steps"):
            pi05_prefix_program(config)

    def test_denoise_step_covers_exact_slots_topology_and_prefix_cache(self) -> None:
        program = pi05_denoise_step_program(_config())
        verify_program(program)
        function = program.functions[0]
        counts = Counter(op.opcode for op in function.body.ops)
        self.assertEqual(counts["constant_ref"], 208)
        self.assertEqual(counts["state_read"], 36)
        self.assertEqual(counts["scaled_dot_product_attention"], 18)
        self.assertEqual(counts["rope_default"], 36)
        self.assertEqual(counts["rms_norm"], 37)
        self.assertEqual(counts["sinusoidal_embedding"], 1)
        self.assertEqual(counts["silu"], 2)
        self.assertEqual(counts["slice"], 111)
        self.assertEqual(counts["linear"], 167)

        expected_names = {
            item.key.name
            for item in pi05_tensor_expectations(_config())
            if item.region in {"action_projection", "action_expert"}
        }
        referenced_names = [
            str(op.attribute("name")) for op in function.body.ops if op.opcode == "constant_ref"
        ]
        self.assertEqual(set(referenced_names), expected_names)
        self.assertEqual(len(referenced_names), len(set(referenced_names)))

        self.assertEqual(len(program.states), 36)
        self.assertTrue(all(state.access == StateAccess.READ_WRITE for state in program.states))
        self.assertTrue(
            all(
                state.type.dtype == DType.BF16
                and state.type.shape == (1, 1, 968, 256)
                and state.type.device == Device.CUDA
                for state in program.states
            )
        )
        state_reads = [op for op in function.body.ops if op.opcode == "state_read"]
        self.assertEqual(
            [str(op.attribute("state")) for op in state_reads],
            [state.name for state in program.states],
        )

        attentions = [
            op for op in function.body.ops if op.opcode == "scaled_dot_product_attention"
        ]
        self.assertTrue(
            all(
                op.attribute("kv_group_size") == 8
                and op.attribute("scale") == 0.0625
                and op.inputs[3] == "denoise.attention.mask"
                for op in attentions
            )
        )
        self.assertTrue(
            all(
                op.attribute("pairing") == "split_half"
                and op.attribute("theta") == 10000.0
                and op.attribute("frequency_dtype") == "bf16"
                for op in function.body.ops
                if op.opcode == "rope_default"
            )
        )
        self.assertEqual(
            tuple(value.value_id for value in function.inputs),
            ("latent", "timestep", "prefix_pad_mask"),
        )
        output = function.body.ops[-1].outputs[0]
        self.assertEqual(output.type.dtype, DType.F32)
        self.assertEqual(output.type.shape, (1, 50, 32))
        self.assertEqual(output.type.device, Device.CUDA)
        self.assertNotIn("pi05", {op.opcode for op in function.body.ops})

    def test_denoise_step_dump_is_deterministic_and_periods_fail_closed(self) -> None:
        first = pi05_denoise_step_program(_config())
        second = pi05_denoise_step_program(_config())
        self.assertEqual(dump_program(first), dump_program(second))
        config = _config()
        config["max_period"] = 0.001
        with self.assertRaisesRegex(ValidationError, "min_period <= max_period"):
            pi05_denoise_step_program(config)

    def test_inference_program_closes_model_slots_and_fixed_euler_schedule(self) -> None:
        program = pi05_inference_program(_config())
        verify_program(program)
        self.assertEqual(
            [function.name for function in program.functions],
            ["pi05_infer", "pi05_vision_projector", "pi05_prefix", "pi05_denoise_step"],
        )
        entry = program.functions[0]
        counts = Counter(op.opcode for op in entry.body.ops)
        self.assertEqual(
            counts,
            {
                "call": 14,
                "constant": 11,
                "broadcast_in_dim": 1,
                "mul": 10,
                "add": 10,
                "slice": 1,
            },
        )
        callees = [str(op.attribute("callee")) for op in entry.body.ops if op.opcode == "call"]
        self.assertEqual(callees[:4], ["pi05_vision_projector"] * 3 + ["pi05_prefix"])
        self.assertEqual(callees[4:], ["pi05_denoise_step"] * 10)
        timestep_values = [
            op.attribute("value")
            for op in entry.body.ops
            if op.opcode == "constant" and op.outputs[0].value_id.endswith("timestep")
        ]
        self.assertEqual(
            timestep_values,
            [(value,) for value in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)],
        )
        step_size = next(
            op for op in entry.body.ops if op.outputs and op.outputs[0].value_id == "infer.euler.dt.scalar"
        )
        self.assertEqual(step_size.attribute("value"), (-0.1,))
        final_slice = entry.body.ops[-1]
        self.assertEqual(final_slice.opcode, "slice")
        self.assertEqual(
            (final_slice.attribute("axis"), final_slice.attribute("start"), final_slice.attribute("stop")),
            (2, 0, 7),
        )
        self.assertEqual(final_slice.outputs[0].type.dtype, DType.F32)
        self.assertEqual(final_slice.outputs[0].type.shape, (1, 50, 7))

        consumed = {
            item.key.name for item in pi05_tensor_expectations(_config()) if item.disposition == "consumed"
        }
        referenced = [
            str(op.attribute("name"))
            for function in program.functions
            for op in function.body.ops
            if op.opcode == "constant_ref"
        ]
        self.assertEqual(len(referenced), 811)
        self.assertEqual(set(referenced), consumed)
        self.assertEqual(len(referenced), len(set(referenced)))

        prefix_writes = [
            str(op.attribute("state"))
            for op in program.functions[2].body.ops
            if op.opcode == "state_write"
        ]
        denoise_reads = [
            str(op.attribute("state"))
            for op in program.functions[3].body.ops
            if op.opcode == "state_read"
        ]
        self.assertEqual(prefix_writes, [state.name for state in program.states])
        self.assertEqual(denoise_reads, prefix_writes)
        self.assertNotIn("pi05", {op.opcode for function in program.functions for op in function.body.ops})

    def test_inference_program_dump_is_deterministic_and_profile_is_fail_closed(self) -> None:
        first = pi05_inference_program(_config())
        second = pi05_inference_program(_config())
        self.assertEqual(dump_program(first), dump_program(second))
        config = _config()
        config["num_inference_steps"] = 9
        with self.assertRaisesRegex(ValidationError, "num_inference_steps"):
            pi05_inference_program(config)

        config = _config()
        config["chunk_size"] = 48
        with self.assertRaisesRegex(ValidationError, "chunk_size"):
            pi05_denoise_step_program(config)

    def test_source_frontend_builds_stable_program_and_complete_coverage(self) -> None:
        source = _fake_pi05_source(_config())
        frontend = Pi05SourceFrontend()
        first = frontend.import_program(source)  # type: ignore[arg-type]
        second = frontend.import_program(source)  # type: ignore[arg-type]
        self.assertEqual(frontend.frontend_id, "pi05.lerobot.flow_matching.v1")
        self.assertEqual(dump_program(first.program), dump_program(second.program))
        self.assertEqual(first.coverage, second.coverage)
        self.assertIs(first.constants, source.constants)
        self.assertIs(first.assets, source.assets)
        self.assertEqual(len(first.coverage), 812)
        self.assertEqual(Counter(record.disposition for record in first.coverage), {"consumed": 811, "ignored": 1})

    def test_source_frontend_refuses_wrong_type_and_tensor_contract(self) -> None:
        config = _config()
        config["type"] = "other"
        with self.assertRaisesRegex(ValidationError, "refuses source type"):
            Pi05SourceFrontend().import_program(_fake_pi05_source(config))  # type: ignore[arg-type]

        bad_name = "model.action_in_proj.weight"
        with self.assertRaisesRegex(ValidationError, "tensor contract mismatch"):
            Pi05SourceFrontend().import_program(  # type: ignore[arg-type]
                _fake_pi05_source(_config(), wrong_shape=bad_name)
            )


if __name__ == "__main__":
    unittest.main()
